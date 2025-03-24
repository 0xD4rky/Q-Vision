import torch
import torch.nn as nn

def capture_activations(model, caliberation_inputs):

    """
    Runs a forward pass using calibration_inputs and captures
    the (input, output) activations for every nn.Linear layer in the model.
    
    Args:
        model: The PyTorch model.
        calibration_inputs: A dict (as returned by a tokenizer) to feed into the model.
    
    Returns:
        activation_data: A dict mapping layer names to a dict with keys "input" and "output".
                         The shapes are:
                            - "input": [N, in_features]
                            - "output": [N, out_features]
    """


    activation_data = {}

    def get_hook(name):

        def hook(module, inputs, output):

            activation_data[name] = {
                "input": inputs[0].detach().cpu(),
                "output": output.detach().cpu()
            }

        return hook
