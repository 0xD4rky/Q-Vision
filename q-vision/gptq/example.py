import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.nn import CrossEntropyLoss
from utils import measure_memory, calibration_data
from ..hf_integration import load_llm
from quantizer import quantize_model


torch.manual_seed(42)
np.random.seed(42)

def run_inference(model, tokenizer, prompt, max_tokens=50):
    """Run inference and measure latency, memory, and output."""
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device= "mps")
    
    _ = pipe("Warm-up", max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    start_time = time.time()
    output = pipe(prompt, max_new_tokens=max_tokens, do_sample=False, return_full_text=False)[0]["generated_text"]
    torch.cuda.synchronize()  
    latency = (time.time() - start_time) * 1000 / max_tokens  # ms/token
    memory = measure_memory()
    
    return output, latency, memory

def compute_perplexity(model, tokenizer, text, max_length=512):
    """Compute perplexity on a given text."""
    model.eval()
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to("mps")
    input_ids = encodings["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :].contiguous()  # Exclude last token's prediction
        labels = input_ids[:, 1:].contiguous()  # Shifted targets
        loss = CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
    return torch.exp(loss).item()


def load_and_quantize_custom_model(model_name, tokenizer, calib_data):
    """Load and quantize the custom model."""
    model, _ = load_llm(model_name, dtype=torch.float16)
    print("Quantizing custom model with EnhancedGPTQ...")
    start_time = time.time()
    quantized_model = quantize_model(model, calib_data, bits=4, group_size=128, block_size=32)
    quant_time = time.time() - start_time
    print(f"Quantization completed in {quant_time:.2f} seconds")
    return quantized_model