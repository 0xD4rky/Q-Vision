import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_integration import load_llm
import torch
import torch.nn as nn
import numpy as np
from utils import load_weights, calibration_data, compute_activations

class GPTQ:

    def __init__(self, bits = 4, block_size = 32, group_size = 128):

        self.bits = bits
        self.block_size = block_size
        self.group_size = group_size
        self.maxq = 2 ** bits - 1

        # group size is the size of hessian blocks

    def quantize_weights(self, weights, activations):

        """
        Quantize weights with block-diagonal Hessian and group-wise processing.
        """

        quantized_model = {}

        for name, weight in weights.items():
            
            weight = weight.float()
            n_out, n_in = weight.shape

            if (n_in % self.group_size) != 0 :

                pad_size = self.group_size - (n_in % self.group_size)
                weight = nn.functional.pad(weight, (0,pad_size))
                n_in = weight.shape[1]

            
            num_groups = n_in // self.group_size
            weight_groups = weight.view(n_out, num_groups, self.group_size)
            activation = activations[name].float().view(-1, n_in)
            activation_groups = activation.view(-1, num_groups, self.group_size)

            ### INITIALIZING QUANTIZATION WEIGHTS AND SCALES

            weight_quant = torch.zeros_like(weight_groups, dtype = torch.int8)
            scales = torch.zeros(n_out, num_groups, device=weight.device, dtype=weight.dtype)            
            weight_copy = weight_groups.clone()

            for g in range(num_groups):
                activation_g = activation_groups[:, g, :]  # Shape: (B*seq_len, group_size)
                hessian_g = torch.matmul(activation_g.T, activation_g) + 1e-6 * torch.eye(self.group_size, device=W.device)
                
                # Low-rank approximation (optional, for speed)
                if self.block_size < self.group_size:
                    eigvals, eigvecs = torch.linalg.eigh(hessian_g)
                    top_k = self.block_size
                    hessian_g_approx = torch.matmul(eigvecs[:, -top_k:], torch.diag(eigvals[-top_k:]) @ eigvecs[:, -top_k:].T)
                    hessian_g_inv = torch.linalg.pinv(hessian_g_approx)
                else:
                    hessian_g_inv = torch.linalg.inv(hessian_g)
                
                weight_g = weight_copy[:, g, :]  # Shape: (n_out, group_size)
                scale = weight_g.abs().max(dim=1, keepdim=True)[0] / self.maxq
                scales[:, g] = scale.squeeze()
                
                errors = torch.zeros_like(weight_g)
                for i in range(self.group_size):
                    w_col = weight_g[:, i] - errors[:, i] 
                    q_col = torch.clamp(torch.round(w_col / scale), -self.maxq, self.maxq)
                    W_quant[:, g, i] = q_col.to(torch.int8)
                    delta = w_col - q_col * scale.squeeze()
                    # Update errors for remaining columns
                    if i < self.group_size - 1:
                        errors[:, i+1:] -= torch.outer(delta, hessian_g_inv[i, i+1:].squeeze())
                
                # Lazy update: Apply to remaining groups (simplified here)
                if g < num_groups - 1:
                    delta_weight_g = (weight_g - W_quant[:, g, :] * scale) @ activation_g.T
                    weight_copy[:, g+1:, :] -= (delta_weight_g @ activation_groups[:, g+1:, :].transpose(-1, -2)) @ hessian_g_inv
                
            # Reshape and store
            W_quant = W_quant.view(n_out, -1)[:, :weights[name].shape[1]]
            scales = scales[:, :num_groups]
            quantized_model[name] = {"weights": W_quant, "scales": scales}
        
        return quantized_model
    
    def apply_to_model(self, model, quantized_weights):
        """Apply quantized weights back to the model."""

        for name, module in model.named_modules():
            if name in quantized_weights:
                q_data = quantized_weights[name]
                W_q = q_data["weights"].float() * q_data["scales"]
                module.weight.data = W_q.view_as(module.weight).to(module.weight.dtype)

    
def quantize_model(model, input_ids, bits=4, group_size=128, block_size=32):
    """Full quantization pipeline."""
    weights = load_weights(model)
    activations = compute_activations(model, input_ids)
    
    gptq = GPTQ(bits=bits, group_size=group_size, block_size=block_size)
    quantized_weights = gptq.quantize_weights(weights, activations)
    gptq.apply_to_model(model, quantized_weights)
    return model
    
if __name__ == "__main__":
    from utils import calibration_data
    model, tokenizer = load_llm("meta-llama/Llama-3.2-1B", "mps")
    calib_data = calibration_data(tokenizer)
    quantized_model = quantize_model(model, calib_data)
    print("Model quantized successfully!")