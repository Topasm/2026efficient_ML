"""Analyze learned adapter updates in the pretrained SVD frame.

For each target module, reconstruct Delta W and compute:

    C = U.T @ Delta W @ V

where W0 = U S V.T is the pretrained weight. Diagonal energy measures
singular-direction scaling; off-diagonal energy measures mixing/rotation.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def task_num_labels(task):
    if task == "stsb":
        return 1
    if task == "mnli":
        return 3
    return 2


def get_module(root, module_name):
    module = root
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def effective_rank(matrix):
    singular_values = torch.linalg.svdvals(matrix.float())
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probs = singular_values / total
    entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum()
    return float(entropy.exp().cpu())


def svd_frame_metrics(weight, delta, rank):
    weight = weight.detach().float().cpu()
    delta = delta.detach().float().cpu()
    U, singular_values, Vh = torch.linalg.svd(weight, full_matrices=False)
    V = Vh.transpose(0, 1)
    C = U.transpose(0, 1) @ delta @ V

    total_energy = float((C * C).sum().cpu())
    diag = torch.diag(C)
    diag_energy = float((diag * diag).sum().cpu())
    off_energy = max(total_energy - diag_energy, 0.0)
    if total_energy <= 0:
        r_diag = 0.0
        r_off = 0.0
    else:
        r_diag = diag_energy / total_energy
        r_off = off_energy / total_energy

    k = min(rank, C.shape[0], C.shape[1])
    C_k = C[:k, :k]
    C_k_energy = float((C_k * C_k).sum().cpu())
    diag_k = torch.diag(C_k)
    diag_k_energy = float((diag_k * diag_k).sum().cpu())
    off_k_energy = max(C_k_energy - diag_k_energy, 0.0)
    delta_energy = float((delta * delta).sum().cpu())
    if C_k_energy <= 0:
        r_diag_k = 0.0
        r_off_k = 0.0
    else:
        r_diag_k = diag_k_energy / C_k_energy
        r_off_k = off_k_energy / C_k_energy
    eta_k = C_k_energy / delta_energy if delta_energy > 0 else 0.0

    return {
        "delta_norm_fro": float(delta.norm().cpu()),
        "w0_norm_fro": float(weight.norm().cpu()),
        "C_norm_fro": math.sqrt(total_energy),
        "diag_energy": diag_energy,
        "offdiag_energy": off_energy,
        "R_diag": r_diag,
        "R_off": r_off,
        "rank_k": k,
        "C_k_norm_fro": math.sqrt(C_k_energy),
        "subspace_energy_k": C_k_energy,
        "eta_k": eta_k,
        "diag_energy_k": diag_k_energy,
        "offdiag_energy_k": off_k_energy,
        "R_diag_k": r_diag_k,
        "R_off_k": r_off_k,
        "effective_rank": effective_rank(delta),
        "top_singular_value_w0": float(singular_values[0].cpu()) if len(singular_values) else 0.0,
    }


def load_lora_diag_updates(base_model, adapter_dir, target_modules):
    payload = torch.load(Path(adapter_dir) / "adapter_model.pt", map_location="cpu")
    state = payload["state_dict"]
    config = payload["config"]
    rank = int(config["rank"])
    alpha = float(config["lora_alpha"])
    scaling = alpha / rank

    updates = []
    module_names = config.get("injected_modules") or [
        key[: -len(".lora_A")] for key in state if key.endswith(".lora_A")
    ]
    for name in module_names:
        if name.split(".")[-1] not in target_modules:
            continue
        A = state[f"{name}.lora_A"].float()
        B = state[f"{name}.lora_B"].float()
        diag = state[f"{name}.lora_diag"].float()
        delta = (B @ torch.diag(diag) @ A) * scaling
        base_module = get_module(base_model, name)
        updates.append(
            {
                "module": name,
                "weight": base_module.weight,
                "delta": delta,
                "rank": rank,
                "scaling": scaling,
                "has_diag": True,
                "diag_norm": float(diag.norm().cpu()),
            }
        )
    return updates


def maybe_get_default(mapping_or_module):
    if hasattr(mapping_or_module, "__getitem__"):
        try:
            return mapping_or_module["default"]
        except Exception:
            pass
    return mapping_or_module


def get_peft_diag(module):
    if not hasattr(module, "lora_diag"):
        return None
    diag_obj = maybe_get_default(module.lora_diag)
    if isinstance(diag_obj, torch.nn.Parameter):
        return diag_obj
    if torch.is_tensor(diag_obj):
        return diag_obj
    if hasattr(diag_obj, "weight"):
        return diag_obj.weight
    return None


def load_peft_updates(model_name_or_path, adapter_dir, task, target_modules):
    from peft import PeftModel

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=task_num_labels(task),
        return_dict=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    updates = []
    for name, module in model.named_modules():
        short_name = name
        for prefix in ("base_model.model.", "model."):
            if short_name.startswith(prefix):
                short_name = short_name[len(prefix) :]
        if short_name.split(".")[-1] not in target_modules:
            continue
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        A_module = maybe_get_default(module.lora_A)
        B_module = maybe_get_default(module.lora_B)
        A = A_module.weight.detach().float().cpu()
        B = B_module.weight.detach().float().cpu()
        scaling = module.scaling["default"] if hasattr(module, "scaling") else 1.0
        diag = get_peft_diag(module)
        if diag is None:
            delta = (B @ A) * scaling
            has_diag = False
            diag_norm = 0.0
        else:
            diag = diag.detach().float().cpu().flatten()
            delta = (B @ torch.diag(diag) @ A) * scaling
            has_diag = True
            diag_norm = float(diag.norm().cpu())

        base_layer = module.base_layer if hasattr(module, "base_layer") else None
        if base_layer is None or not hasattr(base_layer, "weight"):
            continue
        updates.append(
            {
                "module": short_name,
                "weight": base_layer.weight,
                "delta": delta,
                "rank": A.shape[0],
                "scaling": float(scaling),
                "has_diag": has_diag,
                "diag_norm": diag_norm,
            }
        )
    return updates


def aggregate(rows):
    numeric_keys = [
        "delta_norm_fro",
        "w0_norm_fro",
        "C_norm_fro",
        "diag_energy",
        "offdiag_energy",
        "R_diag",
        "R_off",
        "rank_k",
        "C_k_norm_fro",
        "subspace_energy_k",
        "eta_k",
        "diag_energy_k",
        "offdiag_energy_k",
        "R_diag_k",
        "R_off_k",
        "effective_rank",
        "top_singular_value_w0",
        "diag_norm",
    ]
    out = {"num_modules": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        out[f"mean_{key}"] = sum(values) / len(values) if values else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["lora", "svd_only", "lora_diag", "lora_diag_l2", "lora_diag_rot", "kasa", "kasa_noaux", "peft"],
    )
    parser.add_argument("--task", type=str, default="cola")
    parser.add_argument("--target_modules", type=str, default="query,value")
    parser.add_argument("--rank", type=int, default=8, help="Top-k SVD subspace rank for eta_k/R_off_k")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    target_modules = parse_csv(args.target_modules)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method in ("lora_diag", "lora_diag_l2", "lora_diag_rot"):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=task_num_labels(args.task),
            return_dict=True,
        )
        updates = load_lora_diag_updates(base_model, args.adapter_dir, target_modules)
    else:
        updates = load_peft_updates(args.model_name_or_path, args.adapter_dir, args.task, target_modules)

    if not updates:
        raise RuntimeError(f"No adapter updates found in {args.adapter_dir}")

    rows = []
    for update in updates:
        metrics = svd_frame_metrics(update["weight"], update["delta"], args.rank)
        row = {
            "module": update["module"],
            "rank": update["rank"],
            "scaling": update["scaling"],
            "has_diag": update["has_diag"],
            "diag_norm": update["diag_norm"],
            **metrics,
        }
        rows.append(row)

    per_layer_path = output_dir / "per_layer.csv"
    with per_layer_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_dir": args.adapter_dir,
        "method": args.method,
        "task": args.task,
        "target_modules": target_modules,
        "rank_k": args.rank,
        **aggregate(rows),
    }
    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Per-layer metrics saved to {per_layer_path}")


if __name__ == "__main__":
    main()
