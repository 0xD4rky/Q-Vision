import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
import math
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3", 
        device_map="auto", 
        torch_dtype=torch.float16
    )

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def fp4_quantize(tensor, num_exp_bits=2, num_mantissa_bits=1):
    """
    Simple 4-bit floating point quantization with 1 sign bit, 
    configurable exponent bits, and remaining bits for mantissa
    """
    assert num_exp_bits + num_mantissa_bits == 3, "Need 3 bits for exp + mantissa in 4-bit float"
    
    original_shape = tensor.shape
    tensor_flat = tensor.reshape(-1).clone()
    
    max_abs = tensor_flat.abs().max()
    if max_abs == 0:
        return torch.zeros_like(tensor), torch.zeros_like(tensor)
    
    num_exp_values = 2 ** num_exp_bits
    num_mantissa_values = 2 ** num_mantissa_bits
    
    quantized = torch.zeros_like(tensor_flat)

    signs = torch.sign(tensor_flat)
    abs_values = tensor_flat.abs()
    
    fp4_values = []
    for e in range(num_exp_values):
        for m in range(num_mantissa_values):
            val = 2**e * (1 + m/num_mantissa_values)
            fp4_values.append(val)
    
    max_fp4 = max(fp4_values)
    fp4_values = [v * max_abs / max_fp4 for v in fp4_values]
    fp4_values = torch.tensor(fp4_values, device=device)
    fp4_values.to(device)

    for i in range(tensor_flat.shape[0]):
        if tensor_flat[i] == 0:
            quantized[i] = 0
            continue
            
        abs_val = abs_values[i]
        diffs = torch.abs(fp4_values - abs_val)
        idx = torch.argmin(diffs)
        quantized[i] = signs[i] * fp4_values[idx]
    
    quantized = quantized.reshape(original_shape)
    quantization_error = (tensor - quantized)**2
    
    return quantized, quantization_error

def nf4_quantize(tensor):
    """
    4-bit Normal Float quantization as used in QLoRA
    """

    # NF4 quantization levels - these are based on the normal distribution
    # Values from the QLoRA paper, representing standard deviations from the mean
    nf4_values = torch.tensor([
        -1.7, -1.3, -1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0, 1.3, 1.7, 2.1, 2.5, 2.9, 3.3
    ], device=device)

    original_shape = tensor.shape
    tensor_flat = tensor.reshape(-1).clone

    mean = tensor_flat.mean()
    std = tensor_flat.std()

    if std == 0:
        return tensor, torch.zeros_like(tensor)
    
    normalized = (tensor_flat - mean) / std
    quantized = torch.zeros_like(tensor_flat) # initializing the quantized tensor

    for i in range(tensor_flat.shape[0]):
        diffs = torch.abs(nf4_values - normalized[i])
        idx = torch.argmin(diffs)
        quantized[i] = (mean + std).to(device) * nf4_values[idx]

    quantized = quantized.reshape(original_shape)
    quantization_error = (tensor - quantized)**2
    
    return quantized, quantization_error


def load_weight_sample():
    print("Loading model (this might take a while)...")
    model_config = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    linear_weight = model_config.get_submodule("model.layers.0.self_attn.q_proj").weight
    
    print(f"Extracted weight tensor with shape: {linear_weight.shape}")
    return linear_weight


def compare_quantization_methods(weight_tensor):
    """
    Compare NF4 and FP4 quantization on a weight tensor
    """
    # Run both quantization methods
    print("Running NF4 quantization...")
    nf4_quantized, nf4_error = nf4_quantize(weight_tensor)
    
    print("Running FP4 quantization...")
    fp4_quantized, fp4_error = fp4_quantize(weight_tensor)
    
    # Calculate error metrics
    nf4_mse = nf4_error.mean().item()
    fp4_mse = fp4_error.mean().item()
    
    nf4_max_error = nf4_error.max().item()
    fp4_max_error = fp4_error.max().item()
    
    print("\nQuantization Error Comparison:")
    print(f"NF4 Mean Squared Error: {nf4_mse:.8f}")
    print(f"FP4 Mean Squared Error: {fp4_mse:.8f}")
    print(f"Error Reduction Ratio (FP4/NF4): {fp4_mse/nf4_mse:.2f}x")
    
    print(f"\nNF4 Maximum Error: {nf4_max_error:.8f}")
    print(f"FP4 Maximum Error: {fp4_max_error:.8f}")
    
    # Visualize weight distribution and errors
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Original weight distribution
    plt.subplot(2, 2, 1)
    plt.hist(weight_tensor.flatten().cpu().numpy(), bins=100)
    plt.title("Original Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    
    # Plot 2: Quantization Error Distribution
    plt.subplot(2, 2, 2)
    plt.hist(nf4_error.flatten().cpu().numpy(), bins=100, alpha=0.5, label="NF4")
    plt.hist(fp4_error.flatten().cpu().numpy(), bins=100, alpha=0.5, label="FP4")
    plt.title("Quantization Error Distribution")
    plt.xlabel("Squared Error")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Plot 3: Correlation between weight magnitude and error
    plt.subplot(2, 2, 3)
    abs_weights = weight_tensor.abs().flatten().cpu().numpy()
    nf4_errors = nf4_error.flatten().cpu().numpy()
    fp4_errors = fp4_error.flatten().cpu().numpy()
    
    # Sample points for clarity
    sample_indices = np.random.choice(len(abs_weights), min(10000, len(abs_weights)), replace=False)
    
    plt.scatter(abs_weights[sample_indices], nf4_errors[sample_indices], alpha=0.5, label="NF4", s=2)
    plt.scatter(abs_weights[sample_indices], fp4_errors[sample_indices], alpha=0.5, label="FP4", s=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Error vs Weight Magnitude")
    plt.xlabel("Absolute Weight Value (log scale)")
    plt.ylabel("Squared Error (log scale)")
    plt.legend()
    
    # Plot 4: Value preservation comparison
    plt.subplot(2, 2, 4)
    # Sample points for clarity
    sample_indices = np.random.choice(len(weight_tensor.flatten()), min(1000, len(weight_tensor.flatten())), replace=False)
    original_sampled = weight_tensor.flatten()[sample_indices].cpu().numpy()
    nf4_sampled = nf4_quantized.flatten()[sample_indices].cpu().numpy()
    fp4_sampled = fp4_quantized.flatten()[sample_indices].cpu().numpy()
    
    plt.scatter(original_sampled, nf4_sampled, alpha=0.5, label="NF4", s=10)
    plt.scatter(original_sampled, fp4_sampled, alpha=0.5, label="FP4", s=10)
    plt.plot([original_sampled.min(), original_sampled.max()], 
             [original_sampled.min(), original_sampled.max()], 'k--')
    plt.title("Original vs Quantized Values")
    plt.xlabel("Original Weight")
    plt.ylabel("Quantized Weight")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("quantization_comparison.png")
    plt.close()
    
    return {
        "nf4_mse": nf4_mse,
        "fp4_mse": fp4_mse,
        "error_reduction": fp4_mse/nf4_mse
    }

if __name__ == "__main__":
    weight_tensor = load_weight_sample()
    weight_tensor.to(device)
    results = compare_quantization_methods(weight_tensor)
    
    print("\nConclusion:")
    print(f"NF4 achieves {results['error_reduction']:.2f}x lower quantization error than FP4")
    print("This demonstrates NF4's superior ability to represent neural network weights")