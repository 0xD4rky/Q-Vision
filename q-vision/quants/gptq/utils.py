import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np


from ...hf_integration import *  

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

