#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

read -r -a TASKS_ARR <<< "${TASKS:-cola rte mrpc sst2}"
read -r -a LRS_ARR <<< "${LRS:-1e-4 3e-4 1e-3 3e-3}"
read -r -a SEEDS_ARR <<< "${SEEDS:-0 42 123}"

RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
EPOCHS="${EPOCHS:-20}"
BS="${BS:-32}"
DATA_FRACTION="${DATA_FRACTION:-1.0}"
MAX_JOBS="${MAX_JOBS:-3}"
DIAG_L2_BETA="${DIAG_L2_BETA:-0.0}"
ROTATION_TYPE="${ROTATION_TYPE:-cayley}"
ROTATION_ORDER="${ROTATION_ORDER:-diag_rot}"
ROT_ORTH_BETA="${ROT_ORTH_BETA:-0.0}"
OUT_ROOT="${OUT_ROOT:-outputs/kasa_mechanistic/full}"

EXP2_ROOT="$OUT_ROOT/exp2_update_frame"
EXP3_ROOT="$OUT_ROOT/exp3_weighted_rank_ablation"
SUMMARY_ROOT="$OUT_ROOT/summaries"

mkdir -p "$EXP2_ROOT" "$EXP3_ROOT/lora_diag_rot" "$SUMMARY_ROOT"

RUN_FAILURES=0

worker_count() {
    jobs -pr | wc -l
}

wait_for_slot() {
    while [ "$(worker_count)" -ge "$MAX_JOBS" ]; do
        if ! wait -n; then
            RUN_FAILURES=$((RUN_FAILURES + 1))
        fi
    done
}

run_job() {
    local out="$1"
    shift
    if [ -f "$out/results.json" ]; then
        echo "[SKIP] $out"
        return 0
    fi
    mkdir -p "$out"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $out"
    set +e
    "$@" > "$out/run.log" 2>&1
    local status=$?
    set -e
    echo "$status" > "$out/status.txt"
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL $out status=$status"
        tail -40 "$out/run.log" || true
        return "$status"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE $out"
}

launch_job() {
    wait_for_slot
    run_job "$@" &
}

echo "=== Rotation ablation: LoRA+diag+rot ==="
echo "OUT_ROOT=$OUT_ROOT"
echo "rotation_type=$ROTATION_TYPE rotation_order=$ROTATION_ORDER diag_l2_beta=$DIAG_L2_BETA rot_orth_beta=$ROT_ORTH_BETA"

for task in "${TASKS_ARR[@]}"; do
    for lr in "${LRS_ARR[@]}"; do
        for seed in "${SEEDS_ARR[@]}"; do
            out="$EXP3_ROOT/lora_diag_rot/${task}_lr${lr}_s${seed}"
            launch_job "$out" \
                uv run python scripts/train_lora_diag_rot.py \
                --task "$task" \
                --num_epochs "$EPOCHS" \
                --data_fraction "$DATA_FRACTION" \
                --lora_r "$RANK" \
                --lora_alpha "$ALPHA" \
                --head_lr "$lr" \
                --module_lr "$lr" \
                --bs "$BS" \
                --seed "$seed" \
                --diag_init ones \
                --diag_trainable true \
                --diag_l2_beta "$DIAG_L2_BETA" \
                --rotation_type "$ROTATION_TYPE" \
                --rotation_order "$ROTATION_ORDER" \
                --rot_orth_beta "$ROT_ORTH_BETA" \
                --output_dir "$out"
        done
    done
done

while [ "$(worker_count)" -gt 0 ]; do
    if ! wait -n; then
        RUN_FAILURES=$((RUN_FAILURES + 1))
    fi
done

echo "=== Rotation ablation: SVD-frame analysis ==="
mkdir -p "$EXP2_ROOT/_rotation_batch"
if ! uv run python scripts/analyze_update_svd_frame_batch.py \
    --root "$OUT_ROOT" \
    --model_name_or_path roberta-base \
    --methods lora_diag_rot \
    --target_modules query,value \
    --rank "$RANK" \
    --device auto > "$EXP2_ROOT/_rotation_batch/run.log" 2>&1; then
    RUN_FAILURES=$((RUN_FAILURES + 1))
    echo "Rotation update-frame batch failed. See $EXP2_ROOT/_rotation_batch/run.log"
    tail -80 "$EXP2_ROOT/_rotation_batch/run.log" || true
fi

echo "=== Rotation ablation: summarize report data ==="
uv run python scripts/summarize_mechanistic_results.py --root "$OUT_ROOT" > "$SUMMARY_ROOT/summarize.log" 2>&1

if [ "$RUN_FAILURES" -ne 0 ]; then
    echo "Completed with $RUN_FAILURES failed jobs."
    exit 1
fi

echo "Rotation ablation complete. Updated summary data is under $SUMMARY_ROOT"
