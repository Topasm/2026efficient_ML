"""Measure SVD basis stability under small perturbations.

This supports Experiment 4 in KASA_MECHANISTIC_EXPERIMENT_PLAN.md. It measures
the spectral gap at rank k and the perturbation-induced subspace distance for
target pretrained weights.
"""

import argparse
import csv
import json
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


def projection_distance(A, B):
    PA = A @ A.transpose(0, 1)
    PB = B @ B.transpose(0, 1)
    return float((PA - PB).norm(p="fro").cpu())


def selected_linear_weights(model, target_modules):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in target_modules:
            yield name, module.weight.detach().float().cpu()


def summarize(rows):
    out = {"num_rows": len(rows)}
    for key in ("spectral_gap", "d_U", "d_V"):
        values = [float(row[key]) for row in rows]
        out[f"mean_{key}"] = sum(values) / len(values) if values else 0.0
        out[f"max_{key}"] = max(values) if values else 0.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--task", type=str, default="cola")
    parser.add_argument("--target_modules", type=str, default="query,value")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--epsilons", type=str, default="1e-5,1e-4,1e-3")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    target_modules = parse_csv(args.target_modules)
    epsilons = [float(value) for value in parse_csv(args.epsilons)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=task_num_labels(args.task),
        return_dict=True,
    )

    rows = []
    for module_name, weight in selected_linear_weights(model, target_modules):
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        if args.rank >= len(S):
            raise ValueError(f"rank={args.rank} is too large for {module_name} with {len(S)} singular values")
        spectral_gap = float(((S[args.rank - 1] - S[args.rank]) / S[0].clamp_min(1e-12)).cpu())
        U_k = U[:, : args.rank]
        V_k = Vh.transpose(0, 1)[:, : args.rank]
        w_norm = weight.norm(p="fro").clamp_min(1e-12)

        for epsilon in epsilons:
            for trial in range(args.trials):
                noise = torch.randn_like(weight)
                scaled_noise = epsilon * w_norm * noise / noise.norm(p="fro").clamp_min(1e-12)
                perturbed = weight + scaled_noise
                U_hat, _, Vh_hat = torch.linalg.svd(perturbed, full_matrices=False)
                U_hat_k = U_hat[:, : args.rank]
                V_hat_k = Vh_hat.transpose(0, 1)[:, : args.rank]
                rows.append(
                    {
                        "module": module_name,
                        "rank": args.rank,
                        "epsilon": epsilon,
                        "trial": trial,
                        "spectral_gap": spectral_gap,
                        "d_U": projection_distance(U_k, U_hat_k),
                        "d_V": projection_distance(V_k, V_hat_k),
                        "weight_norm_fro": float(w_norm.cpu()),
                        "sigma_1": float(S[0].cpu()),
                        "sigma_k": float(S[args.rank - 1].cpu()),
                        "sigma_k_plus_1": float(S[args.rank].cpu()),
                    }
                )

    per_layer_path = output_dir / "per_layer.csv"
    with per_layer_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "model_name_or_path": args.model_name_or_path,
        "task": args.task,
        "target_modules": target_modules,
        "rank": args.rank,
        "epsilons": epsilons,
        "trials": args.trials,
        "seed": args.seed,
        **summarize(rows),
    }
    with (output_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Per-layer metrics saved to {per_layer_path}")


if __name__ == "__main__":
    main()
