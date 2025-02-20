# Q-Vision

FINE TUNING THE BASE MODEL:

`cd qvision`

```
accelerate launch qvision/base.py \
    --model_id "HuggingFaceTB/SmolLM2-1.7B" \
    --dataset_name "HuggingFaceTB/smoltalk" \
    --subset "data/python" \
    --dataset_text_field "content" \
    --split "train" \
    --max_seq_length 2048 \
    --max_steps 5000 \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --num_proc "$(sysctl -n hw.ncpu)"
```
