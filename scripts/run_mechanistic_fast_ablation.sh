#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

read -r -a TASKS_ARR <<< "${TASKS:-cola rte mrpc sst2}"
read -r -a LRS_ARR <<< "${LRS:-3e-4 3e-3}"
read -r -a SEEDS_ARR <<< "${SEEDS:-0}"

RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-32}"
DATA_FRACTION="${DATA_FRACTION:-1.0}"
OUT_ROOT="${OUT_ROOT:-outputs/kasa_mechanistic/exp3_weighted_rank_ablation}"
KASA_ROOT="${KASA_DIR:-$PROJECT_DIR/KaSA}"
KASA_PYTHONPATH="$KASA_ROOT:$KASA_ROOT/peft/src"

mkdir -p "$OUT_ROOT"

run_or_skip() {
    local out="$1"
    shift
    if [ -f "$out/results.json" ]; then
        echo "[SKIP] $out"
        return 0
    fi
    mkdir -p "$out"
    echo "[$(date '+%H:%M:%S')] $out"
    "$@" 2>&1 | tee "$out/run.log"
}

for task in "${TASKS_ARR[@]}"; do
    for lr in "${LRS_ARR[@]}"; do
        for seed in "${SEEDS_ARR[@]}"; do
            common_args=(
                --task "$task"
                --num_epochs "$EPOCHS"
                --data_fraction "$DATA_FRACTION"
                --lora_r "$RANK"
                --lora_alpha "$ALPHA"
                --head_lr "$lr"
                --module_lr "$lr"
                --bs "$BS"
                --seed "$seed"
            )

            run_or_skip "$OUT_ROOT/lora/${task}_lr${lr}_s${seed}" \
                uv run python scripts/train_hf_peft.py \
                "${common_args[@]}" \
                --init_lora_weights True \
                --output_dir "$OUT_ROOT/lora/${task}_lr${lr}_s${seed}"

            run_or_skip "$OUT_ROOT/svd_only/${task}_lr${lr}_s${seed}" \
                uv run python scripts/train_svd_only.py \
                "${common_args[@]}" \
                --output_dir "$OUT_ROOT/svd_only/${task}_lr${lr}_s${seed}"

            run_or_skip "$OUT_ROOT/lora_diag/${task}_lr${lr}_s${seed}" \
                uv run python scripts/train_lora_diag.py \
                "${common_args[@]}" \
                --diag_init ones \
                --diag_trainable true \
                --output_dir "$OUT_ROOT/lora_diag/${task}_lr${lr}_s${seed}"

            if [ -d "$KASA_ROOT/peft/src" ]; then
                run_or_skip "$OUT_ROOT/kasa/${task}_lr${lr}_s${seed}" \
                    env PYTHONPATH="$KASA_PYTHONPATH" uv run python scripts/train_kasa_fraction.py \
                    "${common_args[@]}" \
                    --lora_dropout 0.0 \
                    --output_dir "$OUT_ROOT/kasa/${task}_lr${lr}_s${seed}"
            else
                echo "[SKIP] KaSA checkout not found at $KASA_ROOT"
            fi
        done
    done
done

echo "Fast ablation runs complete under $OUT_ROOT"
