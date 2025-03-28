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
        torch_dtype = torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
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


def push_to_hub(local_repo: str, repo_id: str):
    """
    Push the local model folder to the Hugging Face Hub.
    """
    from huggingface_hub import HfApi, create_repo, upload_folder

    api = HfApi()
    create_repo(repo_id, exist_ok=True)
    upload_folder(
        folder_path=local_repo,
        repo_id=repo_id,
        commit_message="Add quantized model"
    )