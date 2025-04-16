import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_integration import load_llm
import torch
import torch.nn as nn
import numpy as np
from utils import load_weights, calibration_data, compute_activations
from tqdm import tqdm

class GPTQ:

    def __init__(self, bits=4, block_size=32, group_size=128):
        self.bits = bits
        self.block_size = block_size
        self.group_size = group_size
        self.maxq = 2**bits - 1

    def quantize_weights(self, weights, activations):
        """
        Quantize weights with block-diagonal Hessian and group-wise processing.
        """
        quantized_model = {}

        for name, weight in tqdm(weights.items(), desc="Quantizing layers"):
            weight = weight.float()
            n_out, n_in = weight.shape

            if (n_in % self.group_size) != 0:
                pad_size = self.group_size - (n_in % self.group_size)
                weight = nn.functional.pad(weight, (0, pad_size))
                n_in = weight.shape[1]
            
            num_groups = n_in // self.group_size
            weight_groups = weight.view(n_out, num_groups, self.group_size)
            activation = activations[name].float().view(-1, n_in)
            activation_groups = activation.view(-1, num_groups, self.group_size)

            weight_quant = torch.zeros_like(weight_groups, dtype=torch.int8)
            scales = torch.zeros(n_out, num_groups, device=weight.device, dtype=weight.dtype)            
            weight_copy = weight_groups.clone()

            for g in tqdm(range(num_groups), desc=f"Layer {name}: Processing groups", leave=False):
                activation_g = activation_groups[:, g, :]
                # Compute the group-specific Hessian
                hessian_g = torch.matmul(activation_g.T, activation_g) + 1e-6 * torch.eye(self.group_size, device=weight.device)
                
                if self.block_size < self.group_size:
                    eigvals, eigvecs = torch.linalg.eigh(hessian_g)
                    top_k = self.block_size
                    hessian_g_approx = torch.matmul(eigvecs[:, -top_k:], torch.diag(eigvals[-top_k:]) @ eigvecs[:, -top_k:].T)
                    hessian_g_inv = torch.linalg.pinv(hessian_g_approx)
                else:
                    hessian_g_inv = torch.linalg.inv(hessian_g)
                
                weight_g = weight_copy[:, g, :]
                # Compute scale per output channel (each row)
                scale = weight_g.abs().max(dim=1, keepdim=True)[0] / self.maxq
                scales[:, g] = scale.squeeze()
                
                errors = torch.zeros_like(weight_g)
                for i in range(self.group_size):
                    w_col = weight_g[:, i] - errors[:, i] 
                    q_col = torch.clamp(torch.round(w_col / scale.squeeze()), -self.maxq, self.maxq)
                    weight_quant[:, g, i] = q_col.to(torch.int8)
                    
                    delta = w_col - q_col * scale.squeeze()
                    
                    if i < self.group_size - 1:
                        hessian_slice = hessian_g_inv[i, i+1:]
                        for j in range(self.group_size - i - 1):
                            errors[:, i+1+j] -= delta * hessian_slice[j]
                
                if g < num_groups - 1:
                    quant_weight_g = weight_quant[:, g, :].float() * scale
                    delta_weight_g = weight_g - quant_weight_g
                    
                    for next_g in range(g+1, num_groups):
                        activation_next = activation_groups[:, next_g, :]
                        correction = torch.mm(
                            torch.mm(delta_weight_g, activation_g.T),
                            torch.mm(activation_next, hessian_g_inv)
                        )
                        weight_copy[:, next_g, :] -= correction
            
            # Reshape quantized weights to match original weight dimensions
            original_shape = weights[name].shape[1]
            W_quant = weight_quant.view(n_out, -1)[:, :original_shape]
            
            quantized_model[name] = {
                "weights": W_quant, 
                "scales": scales
            }
        
        return quantized_model
    
    def apply_to_model(self, model, quantized_weights):
        """Apply quantized weights back to the model."""
        for name, module in tqdm(model.named_modules(), desc="Applying quantized weights"):
            if name in quantized_weights:
                q_data = quantized_weights[name]
                # q_data["weights"]: shape [n_out, original_shape]
                # q_data["scales"]: shape [n_out, num_groups]
                n_out, num_groups = q_data["scales"].shape
                group_size = self.group_size
                # Expand scales to match the weight matrix's second dimension.
                # This repeats each scale factor for the columns in its group.
                scale_expanded = q_data["scales"].unsqueeze(2)  \
                    .expand(n_out, num_groups, group_size)        \
                    .reshape(n_out, num_groups * group_size)
                # Module weight's second dimension gives the original shape.
                original_shape = module.weight.shape[1]
                scale_expanded = scale_expanded[:, :original_shape]
                # Multiply element-wise to get the scaled quantized weight.
                W_q = q_data["weights"].float() * scale_expanded
                module.weight.data = W_q.view_as(module.weight).to(module.weight.dtype)

def quantize_model(model, input_ids, bits=4, group_size=128, block_size=32):
    """Full quantization pipeline."""
    print("Loading weights...")
    weights = load_weights(model)
    
    print("Computing activations...")
    activations = compute_activations(model, input_ids)
    
    print(f"Starting quantization: {bits}-bit, group size {group_size}, block size {block_size}")
    gptq = GPTQ(bits=bits, group_size=group_size, block_size=block_size)
    quantized_weights = gptq.quantize_weights(weights, activations)
    
    print("Applying quantized weights to model...")
    gptq.apply_to_model(model, quantized_weights)
    
    return model
    
if __name__ == "__main__":
    from utils import calibration_data
    print("Loading model...")
    model, tokenizer = load_llm("meta-llama/Llama-3.2-1B", "mps")
    
    print("Preparing calibration data...")
    calib_data = calibration_data(tokenizer)
    
    print("Starting quantization process...")
    quantized_model = quantize_model(model, calib_data)
    torch.save(quantized_model.state_dict(), "/teamspace/studios/this_studio")
    print("Model quantized successfully!")
