"""
Test 2: SVD-only ablation.

Applies KaSA's SVD weight surgery to base weights (removes bottom-r singular
components) but then uses STANDARD LoRA on top (no lora_diag, no aux loss).

Tests whether SVD surgery alone — without diag or orthogonality penalty —
provides the LR robustness that KaSA exhibits.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
import evaluate
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="cola")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--head_lr", type=float, default=4e-4)
    p.add_argument("--module_lr", type=float, default=4e-4)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def apply_svd_surgery(model, r, target_names=("query", "value")):
    """Mirror of KaSA's SVD surgery from layer.py:132-138.
    For each target nn.Linear, replace W ← U_top[in-r] @ diag(S_top) @ V^T_top.
    Effectively removes the bottom r singular components from W."""
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(tn in name for tn in target_names):
            continue
        # Skip classifier and LM head
        if 'classifier' in name or 'lm_head' in name or 'embeddings' in name:
            continue

        with torch.no_grad():
            W = module.weight.data
            dtype = W.dtype
            W_f32 = W.float()
            in_features = W.shape[1]
            svd_rank = in_features - r  # Keep top (in_features - r) components
            U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
            U_p, S_p, Vh_p = U[:, :svd_rank], S[:svd_rank], Vh[:svd_rank, :]
            W_new = (U_p @ torch.diag(S_p) @ Vh_p).to(dtype)
            module.weight.data.copy_(W_new)
        count += 1
    print(f"SVD surgery applied to {count} layers")
    return count


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = args.task
    num_labels = 2

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding_side="right")
    datasets_dict = load_dataset("glue", task)
    metric = evaluate.load("glue", task)

    def tok(ex):
        if task == 'cola':
            return tokenizer(ex["sentence"], truncation=True, max_length=args.max_length)
        return tokenizer(ex["sentence1"], ex["sentence2"], truncation=True, max_length=args.max_length)

    rc = ["idx", "sentence"] if task == "cola" else ["idx", "sentence1", "sentence2"]
    tokenized = datasets_dict.map(tok, batched=True, remove_columns=rc)
    tokenized = tokenized.rename_column("label", "labels")

    def collate(ex):
        return tokenizer.pad(ex, padding="longest", return_tensors="pt")

    train_dl = DataLoader(tokenized["train"], shuffle=True, collate_fn=collate, batch_size=args.bs)
    eval_dl = DataLoader(tokenized["validation"], shuffle=False, collate_fn=collate, batch_size=args.bs)

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    # Apply SVD surgery to base weights BEFORE adding LoRA adapters
    apply_svd_surgery(model, args.lora_r, target_names=("query", "value"))

    # Standard LoRA (no aux loss)
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["query", "value"],
        bias="none", task_type="SEQ_CLS", inference_mode=False,
        init_lora_weights=True,
    )
    model = get_peft_model(model, peft_cfg)
    model.to(device)

    head_ids = list(map(id, model.classifier.parameters()))
    other = filter(lambda p: id(p) not in head_ids, model.parameters())
    opt = AdamW([
        {"params": model.classifier.parameters(), "lr": args.head_lr},
        {"params": other, "lr": args.module_lr},
    ], weight_decay=0.0)

    total_steps = len(train_dl) * args.num_epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps
    )

    metric_key = "matthews_correlation" if task == "cola" else "accuracy"
    best_metric = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(train_dl, desc=f"E{epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss  # No aux loss — pure LoRA on SVD-surgery base
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()

        model.eval()
        for batch in eval_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                out = model(**batch)
            preds = out.logits.argmax(-1)
            metric.add_batch(predictions=preds, references=batch["labels"])
        m = metric.compute()
        score = m[metric_key]
        best_metric = max(best_metric, score)
        print(f"Epoch {epoch}: {metric_key}={score:.4f} (best={best_metric:.4f})")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({
            "task": task, "method": "svd_only",
            "rank": args.lora_r, "seed": args.seed,
            "head_lr": args.head_lr, "module_lr": args.module_lr,
            "best_metric": best_metric, "metric_name": metric_key,
        }, f, indent=2)
    print(f"Done. Best {metric_key}: {best_metric:.4f}")


if __name__ == "__main__":
    main()
