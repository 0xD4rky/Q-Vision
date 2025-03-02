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

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    bnb_config = None

    if use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bits=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=attention_dropout,
        trust_remote_code=True
    )

    data = load_dataset(
        dataset_name,
        data_dir=subset,
        split=split,
        token=token,
        num_proc=num_proc if num_proc else multiprocessing.cpu_count(),
        trust_remote_code=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=weight_decay,
            bf16=bf16,
            logging_strategy="steps",
            logging_steps=10,
            output_dir=output_dir,
            optim="paged_adamw_8bit",
            seed=seed,
            run_name=f"train-{model_id.split('/')[-1]}",
            report_to="wandb",
        ),
        peft_config=lora_config,
    )

    print("Training...")
    trainer.train()

    print("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(output_dir, "final_checkpoint/"))
    
    if save_merged_model:
        del model
        if is_torch_xpu_available():
            torch.xpu.empty_cache()
        elif is_torch_npu_available():
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)

        if push_to_hub:
            model.push_to_hub(repo_id, "Upload model")
    
    print("Training Done! 💥")