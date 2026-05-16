"""
KaSA training with data fraction support.
Based on KaSA/main.py but adds --data_fraction and --output_dir for Phase 3/4.
Uses the KaSA PEFT fork (with lora_diag and SVD reparameterization).
"""

import os
import time
import json
import torch
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from peft import (
    get_peft_model,
    LoraConfig,
    PeftType,
    LoraModel,
)
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
parser.add_argument("--task", type=str, default="cola")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.0)
parser.add_argument("--head_lr", type=float, default=4e-4)
parser.add_argument("--module_lr", type=float, default=4e-4)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warmup_ratio", type=float, default=0.06)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--beta", type=float, default=1e-4)
parser.add_argument("--gemma", type=float, default=1e-3)
parser.add_argument("--data_fraction", type=float, default=1.0)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

for arg, value in vars(args).items():
    print(f'{arg}: {value}')

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
task = args.task

if task == "stsb":
    num_labels = 1
elif task == "mnli":
    num_labels = 3
else:
    num_labels = 2

peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="SEQ_CLS",
    inference_mode=False,
)

if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets_dict = load_dataset("glue", task)
metric = evaluate.load("glue", task)

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

# Data fraction subsampling
train_data = tokenized_datasets["train"]
if args.data_fraction < 1.0:
    n_total = len(train_data)
    n = max(int(n_total * args.data_fraction), 1)
    train_data = train_data.shuffle(seed=args.seed).select(range(n))
    print(f"[Data fraction] Using {n}/{n_total} training samples ({args.data_fraction*100:.1f}%)")

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

train_dataloader = DataLoader(train_data, shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
eval_dataloader = DataLoader(
    tokenized_datasets["validation" if task != "mnli" else "validation_matched"],
    shuffle=False, collate_fn=collate_fn, batch_size=args.bs
)

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels, return_dict=True)
model = get_peft_model(model, peft_config)

# Print trainable params (match KaSA's counting)
trainable_params = 0
all_param = 0
for name, param in model.named_parameters():
    if 'lora_diag' in name:
        all_param += int(param.numel())
    elif 'classifier' not in name:
        all_param += param.numel()
    if param.requires_grad and 'classifier' not in name:
        trainable_params += param.numel() if 'lora_diag' not in name else int(param.numel())
print(f'trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {trainable_params/all_param}')

head_param = list(map(id, model.classifier.parameters()))
others_param = filter(lambda p: id(p) not in head_param, model.parameters())
optimizer = AdamW([
    {"params": model.classifier.parameters(), "lr": args.head_lr},
    {"params": others_param, "lr": args.module_lr}
], weight_decay=args.weight_decay)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(args.warmup_ratio * len(train_dataloader) * args.num_epochs),
    num_training_steps=len(train_dataloader) * args.num_epochs,
)

# Auxiliary loss (identical to KaSA main.py)
def loss_fn(model, beta=0.01, gamma=0.01, device='cuda'):
    l2_loss = 0.0
    l3_loss = 0.0
    block_num = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_diag' in name:
                block_num += 1
                diag_norm = torch.sum(param ** 2)
                l2_loss += diag_norm
            elif 'lora_A' in name or 'lora_B' in name:
                if 'lora_A' in name:
                    matmul_result = torch.matmul(param.T, param)
                else:
                    matmul_result = torch.matmul(param, param.T)
                I = torch.eye(matmul_result.size(0), device=device)
                diff_I = matmul_result - I
                matrix_loss = torch.norm(diff_I, p='fro')
                l3_loss += matrix_loss
    auxi_loss = (beta * l2_loss + gamma * l3_loss) / block_num
    return auxi_loss

acc_list = []
model.to(device)
for epoch in range(args.num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss += loss_fn(model, args.beta, args.gemma, device)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Eval {epoch}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if task != "stsb" else outputs.logits.squeeze(-1)
        references = batch["labels"]
        metric.add_batch(predictions=predictions, references=references)

    eval_metric = metric.compute()

    if task == "stsb":
        acc_list.append(eval_metric['pearson'])
        print(f"epoch {epoch}: {eval_metric}, current_best_pearson: {max(acc_list)}, train_loss: {loss.item()}")
    elif task == 'cola':
        acc_list.append(eval_metric['matthews_correlation'])
        print(f"epoch {epoch}: {eval_metric}, current_best_corr: {max(acc_list)}, train_loss: {loss.item()}")
    else:
        acc_list.append(eval_metric['accuracy'])
        print(f"epoch {epoch}: {eval_metric}, current_best_acc: {max(acc_list)}, train_loss: {loss.item()}")

# Save results
best_metric = max(acc_list)
metric_name = {"cola": "matthews_correlation", "stsb": "pearson"}.get(task, "accuracy")

os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
results = {
    "task": task,
    "method": "kasa",
    "rank": args.lora_r,
    "seed": args.seed,
    "data_fraction": args.data_fraction,
    "best_metric": best_metric,
    "metric_name": metric_name,
    "beta": args.beta,
    "gemma": args.gemma,
    "all_epochs": acc_list,
    "adapter_dir": args.output_dir,
}
with open(os.path.join(args.output_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== Final Best {metric_name}: {best_metric:.4f} ===")
