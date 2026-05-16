"""
Unified training script for HF PEFT methods (LoRA, PiSSA, CorDA, EVA)
on GLUE tasks with RoBERTa-base.

IMPORTANT: This script uses a custom training loop identical to KaSA's main.py
for fair comparison (same optimizer, lr schedule, evaluation protocol).

Used by Phase 2, 3, 4 shell scripts.
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftType
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--task", type=str, required=True,
                        choices=["cola", "rte", "mrpc", "sst2", "stsb", "qqp", "mnli", "qnli"])
    parser.add_argument("--init_lora_weights", type=str, default="True",
                        help="LoRA init: 'True' (vanilla), 'pissa', 'corda', 'eva'")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--head_lr", type=float, default=4e-4)
    parser.add_argument("--module_lr", type=float, default=4e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (for Phase 3)")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    return parser.parse_args()


def main():
    args = parse_args()
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')

    # Parse init_lora_weights
    if args.init_lora_weights == "True":
        init_weights = True
    elif args.init_lora_weights == "False":
        init_weights = False
    else:
        init_weights = args.init_lora_weights  # "pissa", "corda", "eva"

    # Basic config (match KaSA's main.py)
    torch.manual_seed(args.seed)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = args.task

    if task == "stsb":
        num_labels = 1
    elif task == "mnli":
        num_labels = 3
    else:
        num_labels = 2

    # Tokenizer (match KaSA's setup)
    if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset and metric
    datasets_dict = load_dataset("glue", task)
    metric = evaluate.load("glue", task)

    # Tokenize (match KaSA's tokenize_function exactly)
    def tokenize_function(examples):
        if task in ('sst2', 'cola'):
            return tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)
        elif task == 'qnli':
            return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=args.max_length)
        elif task == 'qqp':
            return tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=args.max_length)
        elif task == 'mnli':
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=args.max_length)
        else:
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_length)

    # Remove columns (match KaSA)
    if task in ('sst2', 'cola'):
        remove_cols = ["idx", "sentence"]
    elif task == 'qnli':
        remove_cols = ["idx", "question", "sentence"]
    elif task == 'qqp':
        remove_cols = ["idx", "question1", "question2"]
    elif task == 'mnli':
        remove_cols = ["idx", "premise", "hypothesis"]
    else:
        remove_cols = ["idx", "sentence1", "sentence2"]

    tokenized_datasets = datasets_dict.map(tokenize_function, batched=True, remove_columns=remove_cols)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Data fraction subsampling (Phase 3)
    train_dataset = tokenized_datasets["train"]
    if args.data_fraction < 1.0:
        n_total = len(train_dataset)
        n = max(int(n_total * args.data_fraction), 1)
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(n))
        print(f"[Data fraction] Using {n}/{n_total} training samples ({args.data_fraction*100:.1f}%)")
        # Need to re-set the format
        tokenized_datasets["train"] = train_dataset

    # DataLoader (match KaSA's collate_fn exactly)
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation" if task != "mnli" else "validation_matched"],
        shuffle=False, collate_fn=collate_fn, batch_size=args.bs
    )
    if task == "mnli":
        eval_dataloader_mismatched = DataLoader(
            tokenized_datasets["validation_mismatched"],
            shuffle=False, collate_fn=collate_fn, batch_size=args.bs
        )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, return_dict=True
    )

    # PEFT config
    # KaSA targets query+value by default; match that for all methods
    target_modules = ["query", "value"]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="SEQ_CLS",
        inference_mode=False,
        init_lora_weights=init_weights,
    )

    # CorDA requires preprocessing before get_peft_model
    if init_weights == "corda":
        from peft.tuners.lora.corda import preprocess_corda
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        def run_model_for_corda():
            """Run a few batches for CorDA covariance estimation."""
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(train_dataloader):
                    if i >= 10:  # 10 batches is enough for covariance
                        break
                    batch = {k: v.to(device) for k, v in batch.items()}
                    model(**batch)

        preprocess_corda(model, peft_config, run_model=run_model_for_corda)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Optimizer with separate head/module lr (match KaSA exactly)
    head_param = list(map(id, model.classifier.parameters()))
    others_param = filter(lambda p: id(p) not in head_param, model.parameters())
    optimizer = AdamW([
        {"params": model.classifier.parameters(), "lr": args.head_lr},
        {"params": others_param, "lr": args.module_lr}
    ], weight_decay=args.weight_decay)

    # LR scheduler (match KaSA)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * len(train_dataloader) * args.num_epochs),
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    # Training loop (match KaSA's loop exactly, but no auxiliary loss)
    acc_list = []
    model.to(device)

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # No auxiliary loss for baselines (that's the whole point)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluation (match KaSA exactly)
        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Eval {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if task != "stsb" else outputs.logits.squeeze(-1)
            references = batch["labels"]
            metric.add_batch(predictions=predictions, references=references)

        eval_metric = metric.compute()

        if task == "mnli":
            for step, batch in enumerate(tqdm(eval_dataloader_mismatched, desc=f"Eval-mm {epoch}")):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                references = batch["labels"]
                metric.add_batch(predictions=predictions, references=references)
            eval_metric_mismatched = metric.compute()

        # Track the best validation metric across epochs.
        if task == "stsb":
            acc_list.append(eval_metric['pearson'])
            print(f"epoch {epoch}: {eval_metric}, current_best_pearson: {max(acc_list)}, train_loss: {loss.item()}")
        elif task == 'cola':
            acc_list.append(eval_metric['matthews_correlation'])
            print(f"epoch {epoch}: {eval_metric}, current_best_corr: {max(acc_list)}, train_loss: {loss.item()}")
        elif task == 'mnli':
            acc_list.append((eval_metric['accuracy'] + eval_metric_mismatched['accuracy']) / 2)
            print(f"epoch {epoch}: matched={eval_metric}, mismatched={eval_metric_mismatched}, current_best_acc: {max(acc_list)}, train_loss: {loss.item()}")
        else:
            acc_list.append(eval_metric['accuracy'])
            print(f"epoch {epoch}: {eval_metric}, current_best_acc: {max(acc_list)}, train_loss: {loss.item()}")

    # Save results
    best_metric = max(acc_list)
    metric_name = {
        "cola": "matthews_correlation",
        "stsb": "pearson",
    }.get(task, "accuracy")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    results = {
        "task": task,
        "method": args.init_lora_weights,
        "rank": args.lora_r,
        "seed": args.seed,
        "data_fraction": args.data_fraction,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "all_epochs": acc_list,
        "adapter_dir": args.output_dir,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Final Best {metric_name}: {best_metric:.4f} ===")
    print(f"Results saved to: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
