import torch
import torch.nn as nn
import numpy as np
from utils import load_weights, caliberation_data

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
            scales = torch.zeros(n_out, num_groups, dtype = weight.device)
            weight_copy = weight_groups.clone()

            