import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np

# Replace relative import with absolute import
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_integration import load_llm 

model, tokenizer = load_llm('meta-llama/Llama-3.2-1B', 'cuda')

def load_weights(model):

    """
    Load all the linear layer weights of the model
    """

    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights[name] = module.weight.data.clone().to("cuda")
    return weights

def calibration_data(tokenizer, num_samples=128, seq_len=512, dataset_name="karpathy/tiny_shakespeare"):
    """
    Load calibration data from a dataset and tokenize it.
    Returns a tensor of shape [num_samples, seq_len] with token IDs.
    """
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
    
    text = dataset["text"][0]  # since the Tiny Shakespeare has one large text entry

    # Use a smaller chunk of text to avoid the sequence length issue
    # Take the first 100K characters to avoid the token limit issue
    truncated_text = text[:100000]
    
    tokenized = tokenizer(
        truncated_text, 
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False
    )

    input_ids = tokenized["input_ids"].squeeze(0)

    total_tokens = input_ids.size(0)
    if total_tokens < seq_len:
        raise ValueError(f"Dataset is too small ({total_tokens} tokens) for seq_len={seq_len}")
    
    start_indices = np.random.randint(0, total_tokens - seq_len + 1, size=num_samples)
    
    # Use "mps" instead of "cuda" for Mac with Apple Silicon
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calibration_data = torch.stack([input_ids[i:i+seq_len] for i in start_indices]).to(device)
    
    return calibration_data

def compute_activations(model, input_ids):
    """
    Function to compute the activations of model using the caliberation data
    """

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = input[0].detach().clone()
        return hook  # Return the hook function, not the activations dict
    
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
    if torch.cuda.is_available():
        # On macOS with MPS, we don't have direct memory measurement API
        # Return an estimated value or 0
        return 0.0  # Placeholder
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    else:
        return 0.0


if __name__ == "__main__":

    weights = load_weights(model)
    print(f"Extracted {len(weights)} weight matrices")
    calib_data = calibration_data(tokenizer)
    print(f"Calibration data shape: {calib_data.shape}")
    activations = compute_activations(model, calib_data)
    print(f"Captured activations for {len(activations)} layers")