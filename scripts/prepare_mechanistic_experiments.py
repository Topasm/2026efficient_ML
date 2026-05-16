"""Prepare cached assets for the KaSA mechanistic experiments.

This downloads/caches the model, tokenizer, GLUE datasets, GLUE metrics, and
tokenized datasets used by the planned experiments. It also performs a small
batch sanity check so training failures are less likely to be caused by setup.
"""

import argparse
import json
import os
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


SUPPORTED_TASKS = ("cola", "rte", "mrpc", "sst2", "stsb", "qqp", "mnli", "qnli")


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
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                max_length=max_length,
            )
        if task == "qqp":
            return tokenizer(
                examples["question1"],
                examples["question2"],
                truncation=True,
                max_length=max_length,
            )
        if task == "mnli":
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                max_length=max_length,
            )
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=max_length,
        )

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


def validation_split_for_task(task):
    return "validation_matched" if task == "mnli" else "validation"


def tensor_shapes(batch):
    shapes = {}
    for key, value in batch.items():
        if hasattr(value, "shape"):
            shapes[key] = list(value.shape)
    return shapes


def inspect_kasa(kasa_dir):
    path = Path(kasa_dir).expanduser().resolve()
    peft_src = path / "peft" / "src"
    return {
        "path": str(path),
        "exists": path.exists(),
        "peft_src_exists": peft_src.exists(),
        "recommended_pythonpath": f"{path}:{peft_src}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--tasks", type=str, default="cola,rte,mrpc,sst2")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/kasa_mechanistic/prepare",
    )
    parser.add_argument(
        "--kasa_dir",
        type=str,
        default=os.environ.get("KASA_DIR", "KaSA"),
    )
    args = parser.parse_args()

    tasks = parse_csv(args.tasks)
    unknown_tasks = sorted(set(tasks) - set(SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported GLUE tasks: {unknown_tasks}")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    padding_side = "left" if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")) else "right"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side=padding_side,
        cache_dir=args.cache_dir,
    )
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_cache_checks = {}
    for num_labels in sorted({task_num_labels(task) for task in tasks}):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            return_dict=True,
            cache_dir=args.cache_dir,
        )
        model_cache_checks[str(num_labels)] = {
            "class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }
        del model

    task_summaries = {}
    for task in tasks:
        print(f"\n=== Preparing GLUE/{task} ===")
        datasets_dict = load_dataset("glue", task, cache_dir=args.cache_dir)
        metric = evaluate.load("glue", task, cache_dir=args.cache_dir)

        tokenized = datasets_dict.map(
            tokenize_function(task, tokenizer, args.max_length),
            batched=True,
            remove_columns=remove_columns_for_task(task),
            desc=f"Tokenizing {task}",
        )
        if "label" in tokenized["train"].column_names:
            tokenized = tokenized.rename_column("label", "labels")

        def collate_fn(examples):
            return tokenizer.pad(examples, padding="longest", return_tensors="pt")

        train_loader = DataLoader(tokenized["train"], collate_fn=collate_fn, batch_size=args.bs)
        eval_loader = DataLoader(
            tokenized[validation_split_for_task(task)],
            collate_fn=collate_fn,
            batch_size=args.bs,
        )
        train_batch = next(iter(train_loader))
        eval_batch = next(iter(eval_loader))

        split_sizes = {split: len(dataset) for split, dataset in datasets_dict.items()}
        task_summary = {
            "split_sizes": split_sizes,
            "raw_columns": {split: dataset.column_names for split, dataset in datasets_dict.items()},
            "tokenized_columns": {split: dataset.column_names for split, dataset in tokenized.items()},
            "metric_module": getattr(metric, "name", "glue"),
            "train_batch_shapes": tensor_shapes(train_batch),
            "eval_batch_shapes": tensor_shapes(eval_batch),
        }
        task_summaries[task] = task_summary
        print(json.dumps(task_summary, indent=2))

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "tasks": tasks,
        "max_length": args.max_length,
        "batch_size": args.bs,
        "seed": args.seed,
        "cache_dir": args.cache_dir,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_cache_checks": model_cache_checks,
        "tasks_prepared": task_summaries,
        "kasa": inspect_kasa(args.kasa_dir),
    }

    summary_path = output_dir / "prepare_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPreparation summary saved to {summary_path}")
    if not summary["kasa"]["peft_src_exists"]:
        print("\nKaSA checkout was not found. Full KaSA runs still need KASA_DIR or ./KaSA.")
        print("Expected PEFT fork path:", summary["kasa"]["recommended_pythonpath"])


if __name__ == "__main__":
    main()
