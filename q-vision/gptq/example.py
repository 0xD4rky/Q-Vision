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
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    
    _ = pipe("Warm-up", max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    start_time = time.time()
    output = pipe(prompt, max_new_tokens=max_tokens, do_sample=False, return_full_text=False)[0]["generated_text"]
    torch.cuda.synchronize()  
    latency = (time.time() - start_time) * 1000 / max_tokens  # ms/token
    memory = measure_memory()
    
    return output, latency, memory