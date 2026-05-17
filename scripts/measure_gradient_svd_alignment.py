"""Measure task-gradient alignment with pretrained SVD directions.

For each target pretrained weight W0 = U S V.T and accumulated task gradient G,
compute alignment_i = |u_i.T @ G @ v_i| and correlate it with singular values.
"""

import argparse
import csv
import json
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def task_num_labels(task):
    if task == "stsb":
        return 1
    if task == "mnli":
        return 3
    return 2


def tokenize_function(task, tokenizer, max_length):
    def _tokenize(examples):
        if task in ("sst2", "cola"):
            return tokenizer(examples["sentence"], truncation=True, max_length=max_length)
        if task == "qnli":
            return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=max_length)
        if task == "qqp":
            return tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=max_length)
        if task == "mnli":
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=max_length)
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)

    return _tokenize


def remove_columns_for_task(task):
    if task in ("sst2", "cola"):
        return ["idx", "sentence"]
    if task == "qnli":
        return ["idx", "question", "sentence"]
    if task == "qqp":
        return ["idx", "question1", "question2"]
    if task == "mnli":
        return ["idx", "premise", "hypothesis"]
    return ["idx", "sentence1", "sentence2"]


def selected_modules(model, target_modules):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in target_modules:
            yield name, module


def alignment_metrics(weight, gradient, rank):
    svd_device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = weight.detach().float().to(svd_device)
    gradient = gradient.detach().float().to(svd_device)
    U, singular_values, Vh = torch.linalg.svd(weight, full_matrices=False)
    V = Vh.transpose(0, 1)
    C = U.transpose(0, 1) @ gradient @ V
    alignment = C.diag().abs()
    total_alignment = alignment.sum().clamp_min(1e-12)

    rho = spearmanr(singular_values.detach().cpu().numpy(), alignment.detach().cpu().numpy()).correlation
    if rho != rho:
        rho = 0.0

    k = min(rank, len(alignment))
    topk_mass = float((alignment[:k].sum() / total_alignment).cpu())
    bottomk_mass = float((alignment[-k:].sum() / total_alignment).cpu())
    topk_mean = float(alignment[:k].mean().cpu())
    bottomk_mean = float(alignment[-k:].mean().cpu())
    top_bottom_ratio = topk_mean / max(bottomk_mean, 1e-12)

    return {
        "spearman_sigma_alignment": float(rho),
        "gradient_norm_fro": float(gradient.norm().cpu()),
        "weight_norm_fro": float(weight.norm().cpu()),
        "alignment_sum": float(alignment.sum().cpu()),
        "topk_alignment_mass": topk_mass,
        "bottomk_alignment_mass": bottomk_mass,
        "topk_alignment_mean": topk_mean,
        "bottomk_alignment_mean": bottomk_mean,
        "top_bottom_alignment_ratio": top_bottom_ratio,
        "top_singular_value": float(singular_values[0].cpu()),
        "top_alignment": float(alignment[0].cpu()),
        "max_alignment": float(alignment.max().cpu()),
    }


def quantile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def summarize(rows):
    out = {"num_modules": len(rows)}
    keys = [
        "spearman_sigma_alignment",
        "gradient_norm_fro",
        "alignment_sum",
        "topk_alignment_mass",
        "bottomk_alignment_mass",
        "topk_alignment_mean",
        "bottomk_alignment_mean",
        "top_bottom_alignment_ratio",
    ]
    for key in keys:
        values = [float(row[key]) for row in rows]
        out[f"mean_{key}"] = sum(values) / len(values) if values else 0.0
    spearman_values = [float(row["spearman_sigma_alignment"]) for row in rows]
    out["median_spearman_sigma_alignment"] = quantile(spearman_values, 0.5)
    out["q1_spearman_sigma_alignment"] = quantile(spearman_values, 0.25)
    out["q3_spearman_sigma_alignment"] = quantile(spearman_values, 0.75)
    out["iqr_spearman_sigma_alignment"] = (
        out["q3_spearman_sigma_alignment"] - out["q1_spearman_sigma_alignment"]
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--target_modules", type=str, default="query,value")
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    target_modules = parse_csv(args.target_modules)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    padding_side = "left" if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")) else "right"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets_dict = load_dataset("glue", args.task)
    _ = evaluate.load("glue", args.task)
    tokenized = datasets_dict.map(
        tokenize_function(args.task, tokenizer, args.max_length),
        batched=True,
        remove_columns=remove_columns_for_task(args.task),
    )
    tokenized = tokenized.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_loader = DataLoader(tokenized["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.bs)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=task_num_labels(args.task),
        return_dict=True,
    )
    selected = list(selected_modules(model, target_modules))
    if not selected:
        raise RuntimeError(f"No target modules found for {target_modules}")

    for param in model.parameters():
        param.requires_grad = False
    for _, module in selected:
        module.weight.requires_grad = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    accumulated = {name: torch.zeros_like(module.weight.detach().cpu(), dtype=torch.float32) for name, module in selected}
    batches_seen = 0
    for batch in tqdm(train_loader, total=args.num_batches, desc=f"Gradient {args.task}"):
        if batches_seen >= args.num_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()

        for name, module in selected:
            if module.weight.grad is not None:
                accumulated[name] += module.weight.grad.detach().float().cpu()
                module.weight.grad = None
        batches_seen += 1

    rows = []
    selected_by_name = dict(selected)
    for name, gradient in accumulated.items():
        metrics = alignment_metrics(selected_by_name[name].weight, gradient, args.rank)
        rows.append({"module": name, "batches_seen": batches_seen, **metrics})

    per_layer_path = output_dir / "per_layer.csv"
    with per_layer_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "model_name_or_path": args.model_name_or_path,
        "task": args.task,
        "target_modules": target_modules,
        "num_batches": args.num_batches,
        "batches_seen": batches_seen,
        "batch_size": args.bs,
        "rank": args.rank,
        "seed": args.seed,
        **summarize(rows),
    }
    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Per-layer metrics saved to {per_layer_path}")


if __name__ == "__main__":
    main()
