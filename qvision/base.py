import argparse
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

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_agrument("--model_id", type = str, default = "HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-smol")
    parser.add_argument("--subset", type=str, default="data/python")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="content")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--use_bnb", type=bool, default=False)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="finetune_smollm2_python")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--save_merged_model", type=bool, default=True)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--repo_id", type=str, default="SmolLM2-1.7B-finetune")

    return parser

def main():

    lora_config = LoraConfig(
        r = 16,
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    bnb_config = None

    if args.use_bnb == True:

        bnb_config = BitsAndBytesConfig(
            load_in_4bits = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16
        )
    


if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok = True)
    logging.set_verbosity_error()
    main(args)
