import torch
from transformers import AutoModel, AutoTokenizer


def load_llm(
        model_name : str,
        device : str
):
    """
    Load an LLM from huggingface
    """

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype = torch.float32
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model,tokenizer

def save_llm_locally(model, tokenizer, save_path: str):
    """
    Save the quantized model + tokenizer in a local directory (Hugging Face format).
    """
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)