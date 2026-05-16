"""
Qwen2-VL-2B fine-tuning on ChartQA with LoRA/PiSSA/KaSA init.

For Phase 6 VLM evaluation.
Simplified: subsample train data to 10% for feasibility.
Eval: relaxed accuracy on validation set.
"""

import argparse
import json
import os
import re
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceM4/ChartQA")
    parser.add_argument("--init_lora_weights", type=str, default="True",
                        help="'True' for LoRA, 'pissa', 'kasa' (kasa uses fork)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--train_fraction", type=float, default=0.10,
                        help="Fraction of ChartQA train to use (default 10% = 2830 examples)")
    parser.add_argument("--eval_samples", type=int, default=500,
                        help="Number of val samples to evaluate (default 500)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def relaxed_correctness(pred: str, gold: str, max_relative_error: float = 0.05) -> bool:
    """ChartQA relaxed accuracy: exact match for text, 5% tolerance for numbers."""
    pred = pred.strip().lower()
    gold = gold.strip().lower()

    # Try numeric comparison first
    def _to_float(s):
        s = re.sub(r'[%$,]', '', s).strip()
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    p_num = _to_float(pred)
    g_num = _to_float(gold)
    if p_num is not None and g_num is not None:
        if g_num == 0:
            return p_num == 0
        return abs(p_num - g_num) / abs(g_num) <= max_relative_error

    # Else exact string match
    return pred == gold


def format_chat(processor, example):
    """Format a ChartQA example as Qwen2-VL chat messages."""
    query = example["query"]
    answer = example["label"][0] if isinstance(example["label"], list) else example["label"]
    image = example["image"]

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer}
        ]}
    ]
    return messages


def prepare_inputs_for_training(processor, example, device):
    """Prepare tokenized inputs with labels masked on prompt."""
    messages = format_chat(processor, example)

    # Full text with assistant response
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Prompt text only
    prompt_msgs = messages[:-1]
    prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    image = example["image"].convert("RGB")

    inputs = processor(text=[full_text], images=[image], return_tensors="pt", padding=True)
    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)

    # Mask prompt tokens in labels
    labels = inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    inputs["labels"] = labels
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def evaluate(model, processor, eval_dataset, device, max_new_tokens=50):
    """Run generation on eval set and compute relaxed accuracy."""
    model.eval()
    correct = 0
    total = 0

    for example in tqdm(eval_dataset, desc="Evaluating"):
        messages = format_chat(processor, example)[:-1]  # drop assistant
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = example["image"].convert("RGB")
        gold = example["label"][0] if isinstance(example["label"], list) else example["label"]

        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(device)

        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        pred = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        correct += int(relaxed_correctness(pred, gold))
        total += 1

    return correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Init weights string
    if args.init_lora_weights == "True":
        init_weights = True
    elif args.init_lora_weights == "False":
        init_weights = False
    else:
        init_weights = args.init_lora_weights

    print(f"Loading {args.model_name_or_path} in {dtype}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, device_map=device
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # Freeze vision tower
    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False

    # LoRA config
    target_modules = args.target_modules.split(",")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=init_weights,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    print(f"Loading {args.dataset_name}...")
    full_train = load_dataset(args.dataset_name, split="train")
    if args.train_fraction < 1.0:
        n = int(len(full_train) * args.train_fraction)
        full_train = full_train.shuffle(seed=args.seed).select(range(n))
        print(f"Using {n}/{len(full_train) if args.train_fraction == 1.0 else int(n/args.train_fraction)} train samples")

    eval_ds = load_dataset(args.dataset_name, split="val")
    if args.eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.eval_samples, len(eval_ds))))

    # Manual training loop (simpler than Trainer for VLM)
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    n_steps = len(full_train) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * n_steps), num_training_steps=n_steps
    )

    print(f"Training: {len(full_train)} examples, {args.num_train_epochs} epochs, ~{n_steps} optimizer steps")

    step = 0
    model.train()
    for epoch in range(args.num_train_epochs):
        print(f"\n=== Epoch {epoch+1}/{args.num_train_epochs} ===")
        full_train_shuffled = full_train.shuffle(seed=args.seed + epoch)

        accum_loss = 0
        accum_count = 0
        pbar = tqdm(full_train_shuffled, desc=f"Epoch {epoch+1}")
        for i, example in enumerate(pbar):
            try:
                inputs = prepare_inputs_for_training(processor, example, device)
                outputs = model(**inputs)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                accum_loss += loss.item() * args.gradient_accumulation_steps
                accum_count += 1

                if (i + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    if step % 20 == 0:
                        pbar.set_postfix({"loss": f"{accum_loss/accum_count:.3f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})
                        accum_loss = 0
                        accum_count = 0
            except Exception as e:
                print(f"Skipping sample {i} due to: {e}")
                continue

    # Eval
    print(f"\n=== Evaluating on {len(eval_ds)} samples ===")
    acc = evaluate(model, processor, eval_ds, device)
    print(f"\nRelaxed Accuracy: {acc:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "dataset": "chartqa",
        "method": args.init_lora_weights,
        "rank": args.lora_r,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "eval_samples": len(eval_ds),
        "best_metric": acc,
        "metric_name": "relaxed_accuracy",
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
