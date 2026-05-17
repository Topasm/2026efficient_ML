"""Collect report-ready data files for the KaSA mechanistic experiments."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def lr_from_run_id(run_id):
    marker = "_lr"
    if marker not in run_id or "_s" not in run_id:
        return None
    try:
        return float(run_id.split(marker, 1)[1].split("_s", 1)[0])
    except ValueError:
        return None


def collect_ablation(root):
    rows = []
    exp_root = Path(root) / "exp3_weighted_rank_ablation"
    for path in sorted(exp_root.glob("*/*/results.json")):
        data = load_json(path)
        method = path.parents[1].name
        run_id = path.parent.name
        head_lr = data.get("head_lr")
        module_lr = data.get("module_lr")
        parsed_lr = lr_from_run_id(run_id)
        row = {
            "method_group": method,
            "run_id": run_id,
            "results_path": str(path),
            "output_dir": str(path.parent),
            "task": data.get("task"),
            "method": data.get("method"),
            "rank": data.get("rank"),
            "seed": data.get("seed"),
            "data_fraction": data.get("data_fraction"),
            "head_lr": head_lr if head_lr is not None else parsed_lr,
            "module_lr": module_lr if module_lr is not None else parsed_lr,
            "best_metric": data.get("best_metric"),
            "metric_name": data.get("metric_name"),
            "num_epochs_recorded": len(data.get("all_epochs", [])),
            "final_metric": data.get("all_epochs", [None])[-1] if data.get("all_epochs") else None,
        }
        rows.append(row)
    return rows


def collect_flat_results(root, exp_name):
    rows = []
    exp_root = Path(root) / exp_name
    for path in sorted(exp_root.glob("*/results.json")):
        data = load_json(path)
        row = {
            "run_id": path.parent.name,
            "results_path": str(path),
            **{key: value for key, value in data.items() if not isinstance(value, (dict, list))},
        }
        if "target_modules" in data:
            row["target_modules"] = ",".join(data["target_modules"])
        rows.append(row)
    return rows


def aggregate_ablation(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row.get("task"),
            row.get("method_group"),
            row.get("head_lr"),
            row.get("metric_name"),
        )
        groups[key].append(row)

    out = []
    for (task, method, lr, metric_name), items in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        out.append(
            {
                "task": task,
                "method_group": method,
                "lr": lr,
                "metric_name": metric_name,
                "num_runs": len(items),
                "mean_best_metric": mean([item.get("best_metric") for item in items]),
                "mean_final_metric": mean([item.get("final_metric") for item in items]),
                "seeds": ",".join(str(item.get("seed")) for item in items),
            }
        )
    return out


def aggregate_update(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("task"), row.get("method"), row.get("target_modules"))
        groups[key].append(row)

    out = []
    for (task, method, target_modules), items in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        out.append(
            {
                "task": task,
                "method": method,
                "target_modules": target_modules,
                "num_runs": len(items),
                "mean_R_diag": mean([item.get("mean_R_diag") for item in items]),
                "mean_R_off": mean([item.get("mean_R_off") for item in items]),
                "mean_eta_k": mean([item.get("mean_eta_k") for item in items]),
                "mean_R_diag_k": mean([item.get("mean_R_diag_k") for item in items]),
                "mean_R_off_k": mean([item.get("mean_R_off_k") for item in items]),
                "mean_effective_rank": mean([item.get("mean_effective_rank") for item in items]),
                "mean_delta_norm_fro": mean([item.get("mean_delta_norm_fro") for item in items]),
            }
        )
    return out


def aggregate_gradient(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("task"), row.get("target_modules"))
        groups[key].append(row)

    out = []
    for (task, target_modules), items in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        out.append(
            {
                "task": task,
                "target_modules": target_modules,
                "num_runs": len(items),
                "mean_spearman_sigma_alignment": mean([item.get("mean_spearman_sigma_alignment") for item in items]),
                "mean_median_spearman_sigma_alignment": mean(
                    [item.get("median_spearman_sigma_alignment") for item in items]
                ),
                "mean_iqr_spearman_sigma_alignment": mean(
                    [item.get("iqr_spearman_sigma_alignment") for item in items]
                ),
                "mean_topk_alignment_mass": mean([item.get("mean_topk_alignment_mass") for item in items]),
                "mean_bottomk_alignment_mass": mean([item.get("mean_bottomk_alignment_mass") for item in items]),
                "mean_topk_alignment_mean": mean([item.get("mean_topk_alignment_mean") for item in items]),
                "mean_bottomk_alignment_mean": mean([item.get("mean_bottomk_alignment_mean") for item in items]),
                "mean_top_bottom_alignment_ratio": mean(
                    [item.get("mean_top_bottom_alignment_ratio") for item in items]
                ),
            }
        )
    return out


def write_index(root, paths, counts):
    path = Path(root) / "summaries" / "REPORT_DATA_INDEX.md"
    lines = [
        "# KaSA Mechanistic Report Data Index",
        "",
        "This directory contains report-ready outputs for the four mechanistic experiments.",
        "",
        "## Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Files", ""])
    for label, file_path in paths.items():
        lines.append(f"- {label}: `{file_path}`")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/kasa_mechanistic/full")
    args = parser.parse_args()

    root = Path(args.root)
    summary_dir = root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    ablation = collect_ablation(root)
    update = collect_flat_results(root, "exp2_update_frame")
    gradient = collect_flat_results(root, "exp1_gradient_alignment")
    stability = collect_flat_results(root, "exp4_svd_stability")

    paths = {
        "ablation_runs": str(summary_dir / "ablation_runs.csv"),
        "ablation_by_task_method_lr": str(summary_dir / "ablation_by_task_method_lr.csv"),
        "update_frame_runs": str(summary_dir / "update_frame_runs.csv"),
        "update_frame_by_task_method": str(summary_dir / "update_frame_by_task_method.csv"),
        "gradient_alignment_runs": str(summary_dir / "gradient_alignment_runs.csv"),
        "gradient_alignment_by_task": str(summary_dir / "gradient_alignment_by_task.csv"),
        "svd_stability_runs": str(summary_dir / "svd_stability_runs.csv"),
    }

    write_csv(paths["ablation_runs"], ablation)
    write_csv(paths["ablation_by_task_method_lr"], aggregate_ablation(ablation))
    write_csv(paths["update_frame_runs"], update)
    write_csv(paths["update_frame_by_task_method"], aggregate_update(update))
    write_csv(paths["gradient_alignment_runs"], gradient)
    write_csv(paths["gradient_alignment_by_task"], aggregate_gradient(gradient))
    write_csv(paths["svd_stability_runs"], stability)

    counts = {
        "ablation result files": len(ablation),
        "update-frame result files": len(update),
        "gradient-alignment result files": len(gradient),
        "svd-stability result files": len(stability),
    }
    write_index(root, paths, counts)
    print(json.dumps({"root": str(root), "counts": counts, "files": paths}, indent=2))


if __name__ == "__main__":
    main()
