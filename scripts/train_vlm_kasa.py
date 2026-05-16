"""
Qwen2-VL-2B fine-tuning on ChartQA with KaSA (uses PEFT fork via PYTHONPATH).

Adds KaSA auxiliary loss on lora_diag + orthogonality of A/B.
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

# Import relaxed_correctness and format helpers from train_vlm
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_vlm import (
    relaxed_correctness, format_chat,
    prepare_inputs_for_training, evaluate
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceM4/ChartQA")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--train_fraction", type=float, default=0.10)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--gemma", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def kasa_aux_loss(model, beta=1e-4, gamma=1e-3, device='cuda'):
    """KaSA auxiliary loss: L2 on diag + Frobenius on A^T A - I, B B^T - I."""
    l2_loss = 0.0
    l3_loss = 0.0
    block_num = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_diag' in name:
            block_num += 1
            l2_loss += torch.sum(param ** 2)
        elif 'lora_A' in name or 'lora_B' in name:
            if 'lora_A' in name:
                mm = torch.matmul(param.T, param)
            else:
                mm = torch.matmul(param, param.T)
            I = torch.eye(mm.size(0), device=device, dtype=mm.dtype)
            l3_loss += torch.norm(mm - I, p='fro')
    if block_num == 0:
        return torch.tensor(0.0, device=device)
    return (beta * l2_loss + gamma * l3_loss) / block_num


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"Loading {args.model_name_or_path} in {dtype}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, device_map=device
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # Freeze vision tower
    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False

    # KaSA config (fork injects lora_diag automatically)
    target_modules = args.target_modules.split(",")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Print trainable (with KaSA counting)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.3f}%)")

    # Dataset
    print(f"Loading {args.dataset_name}...")
    full_train = load_dataset(args.dataset_name, split="train")
    if args.train_fraction < 1.0:
        n = int(len(full_train) * args.train_fraction)
        full_train = full_train.shuffle(seed=args.seed).select(range(n))

    eval_ds = load_dataset(args.dataset_name, split="val")
    if args.eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.eval_samples, len(eval_ds))))

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    n_steps = len(full_train) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * n_steps), num_training_steps=n_steps
    )

    print(f"Training: {len(full_train)} examples, {args.num_train_epochs} epochs, ~{n_steps} steps")

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
                loss = outputs.loss + kasa_aux_loss(model, args.beta, args.gemma, device)
                loss = loss / args.gradient_accumulation_steps
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
                print(f"Skipping sample {i}: {e}")
                continue

    # Eval
    print(f"\n=== Evaluating on {len(eval_ds)} samples ===")
    acc = evaluate(model, processor, eval_ds, device)
    print(f"\nRelaxed Accuracy: {acc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "dataset": "chartqa",
        "method": "kasa",
        "rank": args.lora_r,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "eval_samples": len(eval_ds),
        "best_metric": acc,
        "metric_name": "relaxed_accuracy",
        "beta": args.beta,
        "gemma": args.gemma,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
