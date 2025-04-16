import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.nn import CrossEntropyLoss
from utils import measure_memory, calibration_data

# Replace relative import with absolute import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_integration import load_llm

from quantizer import quantize_model


torch.manual_seed(42)
np.random.seed(42)

def run_inference(model, tokenizer, prompt, max_tokens=50):
    """Run inference and measure latency, memory, and output."""
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # Warm-up without CUDA-specific code
    _ = pipe("Warm-up", max_new_tokens=5, do_sample=False)
    
    # Remove CUDA synchronization that doesn't exist on MPS
    start_time = time.time()
    output = pipe(prompt, max_new_tokens=max_tokens, do_sample=False, return_full_text=False)[0]["generated_text"]
    # No synchronization needed for MPS
    
    latency = (time.time() - start_time) * 1000 / max_tokens  # ms/token
    memory = measure_memory()
    
    return output, latency, memory

def compute_perplexity(model, tokenizer, text, max_length=512):
    """Compute perplexity on a given text."""
    model.eval()
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
    input_ids = encodings["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :].contiguous()  # Exclude last token's prediction
        labels = input_ids[:, 1:].contiguous()  # Shifted targets
        loss = CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
    return torch.exp(loss).item()


def load_and_quantize_custom_model(model_name, tokenizer, calib_data):
    """Load and quantize the custom model."""
    model, _ = load_llm(model_name)
    print("Quantizing custom model with EnhancedGPTQ...")
    start_time = time.time()
    quantized_model = quantize_model(model, calib_data, bits=4, group_size=128, block_size=32)
    quant_time = time.time() - start_time
    print(f"Quantization completed in {quant_time:.2f} seconds")
    save_path = "/teamspace/studios/this_studio/quantized_model.pt"
    torch.save(quantized_model.state_dict(), save_path)
    print(f"Quantized state dict saved to {save_path}")    
    return quantized_model

def load_bloke_model(model_name):
    """Load TheBloke's pre-quantized model."""
    model, tokenizer = load_llm(model_name)
    return model, tokenizer

def compare_models():
    """Compare custom GPTQ model vs. TheBloke's model."""
    # Define test parameters
    prompt = "Tell me a short story about a brave adventurer in a mystical forest."
    eval_text = (
        "In the heart of the mystical forest, where ancient trees whispered secrets of old, "
        "a brave adventurer named Elara set forth on a quest to find the lost Crystal of Dawn."
    )  

    _, tokenizer = load_llm("Qwen/Qwen2.5-1.5B-Instruct")
    
    print("Loading calibration data from Tiny Shakespeare...")
    calib_data = calibration_data(tokenizer, num_samples=128, seq_len=512)
    print(f"Calibration data shape: {calib_data.shape}")
    
    if os.path.exists("/teamspace/studios/this_studio/quantized_model.pt"):

        custom_model, _ = load_llm("meta-llama/Llama-3.2-1B")
        state_dict = torch.load("/teamspace/studios/this_studio/quantized_model.pt", map_location="cuda:0")
        new_state_dict = {}
        prefix = "model."
        for k,v in state_dict.items():
            new_k = k if k.startswith(prefix) else prefix+k
            new_state_dict[new_k] = v
        original_state_dict = custom_model.state_dict()
        original_state_dict.update(new_state_dict)
        custom_model.load_state_dict(original_state_dict)

    else:

        custom_model = load_and_quantize_custom_model("meta-llama/Llama-3.2-1B", tokenizer, calib_data)
        
    bloke_model, _ = load_bloke_model("feelconstantfear/Llama-3.2-1B-QPTQ")
    
    # Run inference on both models
    print("\nRunning inference on custom model...")
    custom_output, custom_latency, custom_memory = run_inference(custom_model, tokenizer, prompt)
    print("Running inference on Qwen's quantized model...")
    bloke_output, bloke_latency, bloke_memory = run_inference(bloke_model, tokenizer, prompt)
    
    print("\nComputing perplexity...")
    custom_ppl = compute_perplexity(custom_model, tokenizer, eval_text)
    bloke_ppl = compute_perplexity(bloke_model, tokenizer, eval_text)
    
    # Print results
    print("\n=== Performance Comparison ===")
    print("\nCustom GPTQ Model:")
    print(f"Output: {custom_output}")
    print(f"Latency: {custom_latency:.2f} ms/token")
    print(f"Memory Usage: {custom_memory:.2f} GB")
    print(f"Perplexity: {custom_ppl:.2f}")
    
    print("\nTheBloke's GPTQ Model:")
    print(f"Output: {bloke_output}")
    print(f"Latency: {bloke_latency:.2f} ms/token")
    print(f"Memory Usage: {bloke_memory:.2f} GB")
    print(f"Perplexity: {bloke_ppl:.2f}")
    
    # Summary analysis
    print("\n=== Summary ===")
    if custom_ppl < bloke_ppl:
        print("Custom model has better perplexity (lower is better).")
    else:
        print("Qwen quantized model has better perplexity.")
    print(f"Latency difference: {custom_latency - bloke_latency:.2f} ms/token (positive means custom is slower)")
    print(f"Memory difference: {custom_memory - bloke_memory:.2f} GB (positive means custom uses more)")

if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")
    
    print(f"Current date: March 31, 2025\n")
    
    compare_models()