"""Train a rank-wise weighted LoRA baseline with rank-space rotation.

This tests whether the off-diagonal/mixing structure observed in the pretrained
SVD frame can help beyond diagonal rank-wise weighting:

    Delta W = scaling * B diag(sigma) R A

The default rotation is a Cayley transform of a trainable skew-symmetric rank
matrix, so R is orthogonal and initialized to identity.
"""

import argparse
import json
import math
import os

import evaluate
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from train_lora_diag import (
    parse_bool,
    parse_csv,
    remove_columns_for_task,
    set_child_module,
    task_num_labels,
    tokenize_function,
)


class LoRADiagRotLinear(nn.Module):
    def __init__(
        self,
        base_layer,
        r,
        lora_alpha,
        lora_dropout=0.0,
        diag_init="ones",
        diag_trainable=True,
        rotation_type="cayley",
        rotation_order="diag_rot",
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRADiagRotLinear expects nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rotation_type = rotation_type
        self.rotation_order = rotation_order
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, r))
        self.lora_diag = nn.Parameter(self._init_diag(diag_init), requires_grad=diag_trainable)
        if rotation_type == "cayley":
            self.lora_rot_raw = nn.Parameter(torch.zeros(r, r, dtype=base_layer.weight.dtype))
        elif rotation_type == "linear":
            self.lora_rot_raw = nn.Parameter(torch.eye(r, dtype=base_layer.weight.dtype))
        else:
            raise ValueError(f"Unknown rotation_type: {rotation_type}")
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

    def rotation_matrix(self):
        if self.rotation_type == "linear":
            return self.lora_rot_raw
        skew = self.lora_rot_raw - self.lora_rot_raw.transpose(0, 1)
        eye = torch.eye(self.r, dtype=skew.dtype, device=skew.device)
        return torch.linalg.solve(eye + skew, eye - skew)

    def rotation_orth_error(self):
        rot = self.rotation_matrix().float()
        eye = torch.eye(self.r, dtype=rot.dtype, device=rot.device)
        return torch.norm(rot.transpose(0, 1) @ rot - eye, p="fro")

    def forward(self, x):
        result = self.base_layer(x)
        hidden = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
        rot = self.rotation_matrix()
        if self.rotation_order == "diag_rot":
            hidden = hidden @ rot.transpose(0, 1)
            hidden = hidden * self.lora_diag
        elif self.rotation_order == "rot_diag":
            hidden = hidden * self.lora_diag
            hidden = hidden @ rot.transpose(0, 1)
        else:
            raise ValueError(f"Unknown rotation_order: {self.rotation_order}")
        hidden = hidden @ self.lora_B.transpose(0, 1)
        return result + hidden * self.scaling


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
    parser.add_argument("--diag_l2_beta", type=float, default=0.0)
    parser.add_argument("--rotation_type", type=str, default="cayley", choices=["cayley", "linear"])
    parser.add_argument("--rotation_order", type=str, default="diag_rot", choices=["diag_rot", "rot_diag"])
    parser.add_argument("--rot_orth_beta", type=float, default=0.0)
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


def inject_lora_diag_rot(model, target_modules, args):
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in target_modules:
            replacements.append((name, module))

    for name, module in replacements:
        set_child_module(
            model,
            name,
            LoRADiagRotLinear(
                module,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                diag_init=args.diag_init,
                diag_trainable=args.diag_trainable,
                rotation_type=args.rotation_type,
                rotation_order=args.rotation_order,
            ),
        )

    if not replacements:
        raise ValueError(f"No target Linear modules found for {target_modules}")
    return [name for name, _ in replacements]


def iter_lora_diag_rot_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRADiagRotLinear):
            yield name, module


def diag_l2_loss(model):
    loss = 0.0
    count = 0
    for _, module in iter_lora_diag_rot_modules(model):
        if module.lora_diag.requires_grad:
            loss = loss + torch.sum(module.lora_diag.float() ** 2)
            count += 1
    return loss / count if count else None


def rotation_orth_loss(model):
    loss = 0.0
    count = 0
    for _, module in iter_lora_diag_rot_modules(model):
        loss = loss + module.rotation_orth_error()
        count += 1
    return loss / count if count else None


def save_adapter(model, output_dir, args, injected_modules):
    adapter_state = {}
    diag_norms = {}
    rot_orth_errors = {}
    for name, module in iter_lora_diag_rot_modules(model):
        adapter_state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        adapter_state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
        adapter_state[f"{name}.lora_diag"] = module.lora_diag.detach().cpu()
        adapter_state[f"{name}.lora_rot"] = module.rotation_matrix().detach().cpu()
        diag_norms[name] = float(module.lora_diag.detach().float().norm().cpu())
        rot_orth_errors[name] = float(module.rotation_orth_error().detach().cpu())

    payload = {
        "state_dict": adapter_state,
        "config": {
            "method": "lora_diag_rot",
            "model_name_or_path": args.model_name_or_path,
            "target_modules": parse_csv(args.target_modules),
            "injected_modules": injected_modules,
            "rank": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "diag_init": args.diag_init,
            "diag_trainable": args.diag_trainable,
            "diag_l2_beta": args.diag_l2_beta,
            "rotation_type": args.rotation_type,
            "rotation_order": args.rotation_order,
            "rot_orth_beta": args.rot_orth_beta,
            "task": args.task,
        },
        "diag_norms": diag_norms,
        "rot_orth_errors": rot_orth_errors,
    }
    torch.save(payload, os.path.join(output_dir, "adapter_model.pt"))
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(payload["config"], f, indent=2)
    return diag_norms, rot_orth_errors


def main():
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    torch.manual_seed(args.seed)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_modules = parse_csv(args.target_modules)

    padding_side = "left" if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")) else "right"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets_dict = load_dataset("glue", args.task)
    metric = evaluate.load("glue", args.task)
    tokenized_datasets = datasets_dict.map(
        tokenize_function(args.task, tokenizer, args.max_length),
        batched=True,
        remove_columns=remove_columns_for_task(args.task),
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
        tokenized_datasets["validation" if args.task != "mnli" else "validation_matched"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.bs,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=task_num_labels(args.task),
        return_dict=True,
    )
    for param in model.parameters():
        param.requires_grad = False

    injected_modules = inject_lora_diag_rot(model, target_modules, args)
    print(f"Injected LoRA+diag+rot modules: {len(injected_modules)}")

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
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if args.diag_l2_beta > 0:
                regularizer = diag_l2_loss(model)
                if regularizer is not None:
                    loss = loss + args.diag_l2_beta * regularizer
            if args.rot_orth_beta > 0:
                regularizer = rotation_orth_loss(model)
                if regularizer is not None:
                    loss = loss + args.rot_orth_beta * regularizer
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for batch in tqdm(eval_dataloader, desc=f"Eval {epoch}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if args.task != "stsb" else outputs.logits.squeeze(-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        eval_metric = metric.compute()
        if args.task == "stsb":
            score = eval_metric["pearson"]
            print(f"epoch {epoch}: {eval_metric}, current_best_pearson: {max(acc_list + [score])}, train_loss: {loss.item()}")
        elif args.task == "cola":
            score = eval_metric["matthews_correlation"]
            print(f"epoch {epoch}: {eval_metric}, current_best_corr: {max(acc_list + [score])}, train_loss: {loss.item()}")
        else:
            score = eval_metric["accuracy"]
            print(f"epoch {epoch}: {eval_metric}, current_best_acc: {max(acc_list + [score])}, train_loss: {loss.item()}")
        acc_list.append(score)

    best_metric = max(acc_list)
    metric_name = {"cola": "matthews_correlation", "stsb": "pearson"}.get(args.task, "accuracy")

    os.makedirs(args.output_dir, exist_ok=True)
    diag_norms, rot_orth_errors = save_adapter(model, args.output_dir, args, injected_modules)
    results = {
        "task": args.task,
        "method": "lora_diag_rot",
        "rank": args.lora_r,
        "seed": args.seed,
        "data_fraction": args.data_fraction,
        "head_lr": args.head_lr,
        "module_lr": args.module_lr,
        "diag_init": args.diag_init,
        "diag_trainable": args.diag_trainable,
        "diag_l2_beta": args.diag_l2_beta,
        "rotation_type": args.rotation_type,
        "rotation_order": args.rotation_order,
        "rot_orth_beta": args.rot_orth_beta,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "all_epochs": acc_list,
        "adapter_path": os.path.join(args.output_dir, "adapter_model.pt"),
        "diag_norms": diag_norms,
        "rot_orth_errors": rot_orth_errors,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Final Best {metric_name}: {best_metric:.4f} ===")
    print(f"Results saved to: {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
