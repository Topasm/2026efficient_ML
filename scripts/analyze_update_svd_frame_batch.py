"""Fast batched SVD-frame analysis for trained adapter runs.

The single-run analyzer reloads RoBERTa and recomputes pretrained SVD bases for
every adapter. This batch variant computes each pretrained SVD basis once, then
reads adapter tensors directly from disk and reuses the cached bases.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from analyze_update_svd_frame import aggregate, parse_csv


ADAPTER_SUFFIXES = (
    (".lora_A.default.weight", "A"),
    (".lora_B.default.weight", "B"),
    (".lora_diag.default.weight", "diag"),
    (".lora_diag.default", "diag"),
    (".lora_rot.default.weight", "rot"),
    (".lora_rot.default", "rot"),
    (".lora_A.weight", "A"),
    (".lora_B.weight", "B"),
    (".lora_diag.weight", "diag"),
    (".lora_rot.weight", "rot"),
    (".lora_A", "A"),
    (".lora_B", "B"),
    (".lora_diag", "diag"),
    (".lora_rot", "rot"),
)


def normalize_module_name(name):
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def task_from_run_id(run_id):
    return run_id.split("_lr", 1)[0]


def choose_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def collect_adapter_tensors(state_dict, target_modules):
    grouped = defaultdict(dict)
    for key, value in state_dict.items():
        for suffix, kind in ADAPTER_SUFFIXES:
            if not key.endswith(suffix):
                continue
            module_name = normalize_module_name(key[: -len(suffix)])
            if module_name.split(".")[-1] in target_modules:
                grouped[module_name][kind] = value.detach().float().cpu()
            break

    factors = []
    for module_name, parts in sorted(grouped.items()):
        if "A" not in parts or "B" not in parts:
            continue
        factors.append(
            {
                "module": module_name,
                "A": parts["A"],
                "B": parts["B"],
                "diag": parts.get("diag"),
                "rot": parts.get("rot"),
            }
        )
    return factors


def load_torch_factors(adapter_dir, target_modules):
    payload = torch.load(Path(adapter_dir) / "adapter_model.pt", map_location="cpu")
    config = payload["config"]
    factors = collect_adapter_tensors(payload["state_dict"], target_modules)
    for factor in factors:
        factor["rotation_order"] = config.get("rotation_order", "diag_rot")
    rank = int(config.get("rank", config.get("r", 0)))
    alpha = float(config.get("lora_alpha", rank or 1))
    return factors, alpha


def load_safetensor_factors(adapter_dir, target_modules):
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    with (adapter_dir / "adapter_config.json").open() as f:
        config = json.load(f)
    factors = collect_adapter_tensors(state, target_modules)
    alpha = float(config.get("lora_alpha", 1.0))
    return factors, alpha


def load_adapter_factors(method, adapter_dir, target_modules):
    if method in ("lora_diag", "lora_diag_l2"):
        return load_torch_factors(adapter_dir, target_modules)
    return load_safetensor_factors(adapter_dir, target_modules)


def build_svd_cache(model_name_or_path, target_modules, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=2,
        return_dict=True,
    )
    model.eval()

    cache = {}
    with torch.inference_mode():
        for name, module in model.named_modules():
            if name.split(".")[-1] not in target_modules:
                continue
            if not hasattr(module, "weight"):
                continue
            weight = module.weight.detach().float().to(device)
            U, singular_values, Vh = torch.linalg.svd(weight, full_matrices=False)
            cache[name] = {
                "U": U,
                "V": Vh.transpose(0, 1),
                "w0_norm_fro": float(weight.norm().detach().cpu()),
                "top_singular_value_w0": float(singular_values[0].detach().cpu())
                if len(singular_values)
                else 0.0,
            }
    return cache


def effective_rank_from_factors(left, right):
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    _, left_r = torch.linalg.qr(left, mode="reduced")
    _, right_r = torch.linalg.qr(right.transpose(0, 1), mode="reduced")
    singular_values = torch.linalg.svdvals(left_r @ right_r.transpose(0, 1))
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probs = singular_values / total
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().detach().cpu())


def svd_frame_metrics_from_cache(cache_entry, A, B, diag, rot, rotation_order, scaling, rank, device):
    A = A.to(device)
    B = B.to(device)
    rot = rot.to(device) if rot is not None else None
    if diag is None:
        diag_values = None
        has_diag = False
        diag_norm = 0.0
    else:
        diag_values = diag.to(device).flatten()
        has_diag = True
        diag_norm = float(diag_values.norm().detach().cpu())

    if rot is None:
        if diag_values is None:
            left = B * scaling
            right = A
        else:
            left = B * diag_values.reshape(1, -1) * scaling
            right = A
    elif rotation_order == "rot_diag":
        left = (B @ rot) * scaling
        right = A if diag_values is None else diag_values.reshape(-1, 1) * A
    else:
        left = B * scaling if diag_values is None else B * diag_values.reshape(1, -1) * scaling
        right = rot @ A

    delta = left @ right
    C = cache_entry["U"].transpose(0, 1) @ delta @ cache_entry["V"]

    total_energy = float((C * C).sum().detach().cpu())
    diag_values = torch.diag(C)
    diag_energy = float((diag_values * diag_values).sum().detach().cpu())
    off_energy = max(total_energy - diag_energy, 0.0)
    if total_energy <= 0:
        r_diag = 0.0
        r_off = 0.0
    else:
        r_diag = diag_energy / total_energy
        r_off = off_energy / total_energy

    k = min(rank, C.shape[0], C.shape[1])
    C_k = C[:k, :k]
    C_k_energy = float((C_k * C_k).sum().detach().cpu())
    diag_k = torch.diag(C_k)
    diag_k_energy = float((diag_k * diag_k).sum().detach().cpu())
    off_k_energy = max(C_k_energy - diag_k_energy, 0.0)
    delta_energy = float((delta * delta).sum().detach().cpu())
    if C_k_energy <= 0:
        r_diag_k = 0.0
        r_off_k = 0.0
    else:
        r_diag_k = diag_k_energy / C_k_energy
        r_off_k = off_k_energy / C_k_energy

    return {
        "rank": int(A.shape[0]),
        "scaling": float(scaling),
        "has_diag": has_diag,
        "diag_norm": diag_norm,
        "delta_norm_fro": float(delta.norm().detach().cpu()),
        "w0_norm_fro": cache_entry["w0_norm_fro"],
        "C_norm_fro": math.sqrt(total_energy),
        "diag_energy": diag_energy,
        "offdiag_energy": off_energy,
        "R_diag": r_diag,
        "R_off": r_off,
        "rank_k": k,
        "C_k_norm_fro": math.sqrt(C_k_energy),
        "subspace_energy_k": C_k_energy,
        "eta_k": C_k_energy / delta_energy if delta_energy > 0 else 0.0,
        "diag_energy_k": diag_k_energy,
        "offdiag_energy_k": off_k_energy,
        "R_diag_k": r_diag_k,
        "R_off_k": r_off_k,
        "effective_rank": effective_rank_from_factors(left, right),
        "top_singular_value_w0": cache_entry["top_singular_value_w0"],
    }


def write_json(path, data):
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2)


def analyze_run(args, svd_cache, method, run_dir, output_dir):
    task = task_from_run_id(run_dir.name)
    factors, alpha = load_adapter_factors(method, run_dir, args.target_modules)
    if not factors:
        raise RuntimeError(f"No adapter tensors found in {run_dir}")

    rows = []
    with torch.inference_mode():
        for factor in factors:
            module_name = factor["module"]
            if module_name not in svd_cache:
                continue
            rank = int(factor["A"].shape[0])
            scaling = alpha / rank if rank else 1.0
            metrics = svd_frame_metrics_from_cache(
                svd_cache[module_name],
                factor["A"],
                factor["B"],
                factor["diag"],
                factor.get("rot"),
                factor.get("rotation_order", "diag_rot"),
                scaling,
                args.rank,
                args.device,
            )
            rows.append({"module": module_name, **metrics})

    if not rows:
        raise RuntimeError(f"No target module updates matched pretrained SVD cache in {run_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_layer_path = output_dir / "per_layer.csv"
    with per_layer_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "model_name_or_path": args.model_name_or_path,
        "adapter_dir": str(run_dir),
        "method": method,
        "task": task,
        "target_modules": args.target_modules,
        "rank_k": args.rank,
        **aggregate(rows),
    }
    write_json(output_dir / "results.json", results)
    (output_dir / "status.txt").write_text("0\n")
    return results


def iter_ablation_runs(root, methods):
    exp3_root = root / "exp3_weighted_rank_ablation"
    for method in methods:
        method_root = exp3_root / method
        for result_path in sorted(method_root.glob("*/results.json")):
            run_dir = result_path.parent
            yield method, run_dir, root / "exp2_update_frame" / f"{method}_{run_dir.name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/kasa_mechanistic/full")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--methods", type=str, default="lora,svd_only,lora_diag,lora_diag_l2,lora_diag_rot,kasa_noaux,kasa")
    parser.add_argument("--target_modules", type=str, default="query,value")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    args.root = Path(args.root)
    args.methods = parse_csv(args.methods)
    args.target_modules = parse_csv(args.target_modules)
    args.device = choose_device(args.device)

    print(f"Building pretrained SVD cache on {args.device}...", flush=True)
    svd_cache = build_svd_cache(args.model_name_or_path, args.target_modules, args.device)
    print(f"Cached {len(svd_cache)} SVD bases.", flush=True)

    runs = list(iter_ablation_runs(args.root, args.methods))
    if args.limit:
        runs = runs[: args.limit]

    completed = 0
    skipped = 0
    failed = 0
    for index, (method, run_dir, output_dir) in enumerate(runs, start=1):
        if (output_dir / "results.json").exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(runs)}] SKIP {output_dir}", flush=True)
            continue
        try:
            print(f"[{index}/{len(runs)}] START {output_dir}", flush=True)
            analyze_run(args, svd_cache, method, run_dir, output_dir)
            completed += 1
            print(f"[{index}/{len(runs)}] DONE {output_dir}", flush=True)
        except Exception as exc:
            failed += 1
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "status.txt").write_text("1\n")
            (output_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            print(f"[{index}/{len(runs)}] FAIL {output_dir}: {exc}", flush=True)

    summary = {"completed": completed, "skipped": skipped, "failed": failed, "total": len(runs)}
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
