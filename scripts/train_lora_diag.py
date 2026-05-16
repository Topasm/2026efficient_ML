"""
Train a rank-wise weighted LoRA baseline on GLUE.

This is the key ablation for separating KaSA's SVD story from the optimization
effect of a learnable diagonal in the low-rank update:

    Delta W = scaling * B diag(d) A

The script intentionally matches scripts/train_hf_peft.py as closely as
possible while avoiding KaSA's SVD surgery and auxiliary losses.
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
from datasets import load_dataset
import evaluate
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


class LoRADiagLinear(nn.Module):
    def __init__(
        self,
        base_layer,
        r,
        lora_alpha,
        lora_dropout=0.0,
        diag_init="ones",
        diag_trainable=True,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRADiagLinear expects nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, r))
        diag = self._init_diag(diag_init)
        self.lora_diag = nn.Parameter(diag, requires_grad=diag_trainable)
        self.reset_parameters()

    def _init_diag(self, diag_init):
        dtype = self.base_layer.weight.dtype
        device = self.base_layer.weight.device
        if diag_init == "ones":
            return torch.ones(self.r, dtype=dtype, device=device)

        with torch.no_grad():
            singular_values = torch.linalg.svdvals(self.base_layer.weight.detach().float())[: self.r]
            if diag_init == "svd_norm":
                singular_values = singular_values / singular_values.mean().clamp_min(1e-12)
            elif diag_init != "svd":
                raise ValueError(f"Unknown diag_init: {diag_init}")
            return singular_values.to(dtype=dtype, device=device)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.base_layer(x)
        lora = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
        lora = lora * self.lora_diag
        lora = lora @ self.lora_B.transpose(0, 1)
        return result + lora * self.scaling


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "yes", "y"):
        return True
    if value in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["cola", "rte", "mrpc", "sst2", "stsb", "qqp", "mnli", "qnli"],
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", type=str, default="query,value")
    parser.add_argument("--diag_init", type=str, default="ones", choices=["ones", "svd", "svd_norm"])
    parser.add_argument("--diag_trainable", type=parse_bool, default=True)
    parser.add_argument("--head_lr", type=float, default=4e-4)
    parser.add_argument("--module_lr", type=float, default=4e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    return parser.parse_args()


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


def set_child_module(root, module_name, new_module):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora_diag(model, target_modules, r, alpha, dropout, diag_init, diag_trainable):
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in target_modules:
            replacements.append((name, module))

    for name, module in replacements:
        set_child_module(
            model,
            name,
            LoRADiagLinear(
                module,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                diag_init=diag_init,
                diag_trainable=diag_trainable,
            ),
        )

    if not replacements:
        raise ValueError(f"No target Linear modules found for {target_modules}")
    return [name for name, _ in replacements]


def iter_lora_diag_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRADiagLinear):
            yield name, module


def save_lora_diag_adapter(model, output_dir, args, injected_modules):
    adapter_state = {}
    diag_norms = {}
    for name, module in iter_lora_diag_modules(model):
        adapter_state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        adapter_state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
        adapter_state[f"{name}.lora_diag"] = module.lora_diag.detach().cpu()
        diag_norms[name] = float(module.lora_diag.detach().float().norm().cpu())

    payload = {
        "state_dict": adapter_state,
        "config": {
            "method": "lora_diag",
            "model_name_or_path": args.model_name_or_path,
            "target_modules": parse_csv(args.target_modules),
            "injected_modules": injected_modules,
            "rank": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "diag_init": args.diag_init,
            "diag_trainable": args.diag_trainable,
            "task": args.task,
        },
        "diag_norms": diag_norms,
    }
    torch.save(payload, os.path.join(output_dir, "adapter_model.pt"))
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(payload["config"], f, indent=2)
    return diag_norms


def main():
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    torch.manual_seed(args.seed)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = args.task
    target_modules = parse_csv(args.target_modules)

    if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets_dict = load_dataset("glue", task)
    metric = evaluate.load("glue", task)

    tokenized_datasets = datasets_dict.map(
        tokenize_function(task, tokenizer, args.max_length),
        batched=True,
        remove_columns=remove_columns_for_task(task),
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    train_dataset = tokenized_datasets["train"]
    if args.data_fraction < 1.0:
        n_total = len(train_dataset)
        n = max(int(n_total * args.data_fraction), 1)
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(n))
        print(f"[Data fraction] Using {n}/{n_total} training samples ({args.data_fraction*100:.1f}%)")
        tokenized_datasets["train"] = train_dataset

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation" if task != "mnli" else "validation_matched"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.bs,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=task_num_labels(task),
        return_dict=True,
    )
    for param in model.parameters():
        param.requires_grad = False

    injected_modules = inject_lora_diag(
        model,
        target_modules=target_modules,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        diag_init=args.diag_init,
        diag_trainable=args.diag_trainable,
    )
    print(f"Injected LoRA+diag modules: {len(injected_modules)}")

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {trainable_params / all_params:.6f}")

    head_params = list(model.classifier.parameters()) if hasattr(model, "classifier") else []
    head_param_ids = {id(param) for param in head_params}
    other_params = [param for param in model.parameters() if param.requires_grad and id(param) not in head_param_ids]
    optimizer = AdamW(
        [
            {"params": head_params, "lr": args.head_lr},
            {"params": other_params, "lr": args.module_lr},
        ],
        weight_decay=args.weight_decay,
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * len(train_dataloader) * args.num_epochs),
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    acc_list = []
    model.to(device)
    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for batch in tqdm(eval_dataloader, desc=f"Eval {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if task != "stsb" else outputs.logits.squeeze(-1)
            references = batch["labels"]
            metric.add_batch(predictions=predictions, references=references)

        eval_metric = metric.compute()
        if task == "stsb":
            score = eval_metric["pearson"]
            print(f"epoch {epoch}: {eval_metric}, current_best_pearson: {max(acc_list + [score])}, train_loss: {loss.item()}")
        elif task == "cola":
            score = eval_metric["matthews_correlation"]
            print(f"epoch {epoch}: {eval_metric}, current_best_corr: {max(acc_list + [score])}, train_loss: {loss.item()}")
        else:
            score = eval_metric["accuracy"]
            print(f"epoch {epoch}: {eval_metric}, current_best_acc: {max(acc_list + [score])}, train_loss: {loss.item()}")
        acc_list.append(score)

    best_metric = max(acc_list)
    metric_name = {"cola": "matthews_correlation", "stsb": "pearson"}.get(task, "accuracy")

    os.makedirs(args.output_dir, exist_ok=True)
    diag_norms = save_lora_diag_adapter(model, args.output_dir, args, injected_modules)
    results = {
        "task": task,
        "method": "lora_diag",
        "rank": args.lora_r,
        "seed": args.seed,
        "data_fraction": args.data_fraction,
        "head_lr": args.head_lr,
        "module_lr": args.module_lr,
        "diag_init": args.diag_init,
        "diag_trainable": args.diag_trainable,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "all_epochs": acc_list,
        "adapter_path": os.path.join(args.output_dir, "adapter_model.pt"),
        "diag_norms": diag_norms,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Final Best {metric_name}: {best_metric:.4f} ===")
    print(f"Results saved to: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
