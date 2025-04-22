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