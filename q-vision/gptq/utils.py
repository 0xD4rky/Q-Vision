import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np

# Replace relative import with absolute import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_integration import load_llm 

model, tokenizer = load_llm('meta-llama/Llama-3.2-1B', 'mps')

def load_weights(model):

    """
    Load all the linear layer weights of the model
    """

    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights[name] = module.weight.data.clone().to("mps")
    return weights

def calibration_data(tokenizer, dataset_name="karpathy/tiny_shakespeare", num_samples=128, seq_len=512):

    """
    The main aim of this function is load and prepare data for caliberation
    """

    dataset = load_dataset(dataset_name, split = "train")
    text = dataset["text"][0] # since the Tiny Shakespeare has one large text entry

    tokenized = tokenizer(
        text, 
        return_tensors = "pt",
        truncation = False,
        add_special_tokens = False
        )

    input_ids = tokenized["input_ids"].squeeze(0)

    total_tokens = input_ids.size(0)
    if total_tokens < seq_len:
        raise ValueError(f"Dataset is too small ({total_tokens} tokens) for seq_len={seq_len}")
    
    start_indices = np.random.randint(0, total_tokens - seq_len + 1, size=num_samples)
    calibration_data = torch.stack([input_ids[i:i+seq_len] for i in start_indices]).to("cuda")
    
    return calibration_data

def compute_activations(model, input_ids):

    """
    Function to compute the activations of model using the caliberation data
    """

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = input[0].detach().clone()
        return activations
    
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(hook_fn(name))
            handles.append(handle)

    with torch.no_grad():
        batch_size = 16
        for i in range(0, input_ids.size(0), batch_size):
            batch = input_ids[i:i+batch_size]
            model(batch)
    
    for handle in handles:
        handle.remove()
    
    return activations

def measure_memory():
    """Measure GPU memory usage in GB."""
    return torch.cuda.memory_allocated() / 1024**3


if __name__ == "__main__":

    weights = load_weights(model)
    print(f"Extracted {len(weights)} weight matrices")
    calib_data = calibration_data(tokenizer)
    print(f"Calibration data shape: {calib_data.shape}")
    activations = compute_activations(model, calib_data)
    print(f"Captured activations for {len(activations)} layers")