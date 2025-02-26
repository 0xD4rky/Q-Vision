import multiprocessing
import os

import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
    set_seed,
)
from trl import SFTTrainer

def main(
    model_id="HuggingFaceTB/SmolLM2-1.7B",
    dataset_name="HuggingFaceTB/smoltalk",
    subset="data/python",
    split="train",
    dataset_text_field="content",
    
    max_seq_length=2048,
    max_steps=1000,
    micro_batch_size=1,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    bf16=True,
    use_bnb=False,
    attention_dropout=0.1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    
    seed=0,
    output_dir="finetune_smollm2_python",
    num_proc=None,
    save_merged_model=True,
    push_to_hub=True,
    repo_id="SmolLM2-1.7B-finetune",
    ):
    """
    Fine-tune a language model using the parameters provided.
    
    This function handles the entire fine-tuning process including:
    - Loading the model and dataset
    - Setting up LoRA configuration
    - Training with SFTTrainer
    - Saving the model and optionally merging LoRA weights
    - Pushing to Hugging Face Hub if requested
    """