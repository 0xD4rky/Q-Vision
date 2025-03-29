import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np


from ..hf_integration import *  

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

def caliberation_data(tokenizer, dataset_name="karpathy/tiny_shakespeare", num_samples=128, seq_len=512):

    """
    The main aim of this function is load and prepare data for caliberation
    """

    dataset = load_dataset(dataset_name, split = "train")
    text = dataset["text"][0] # Tiny Shakespeare has one large text entry

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