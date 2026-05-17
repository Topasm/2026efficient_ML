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
MAX_JOBS="${MAX_JOBS:-4}"
NUM_GRAD_BATCHES="${NUM_GRAD_BATCHES:-16}"
STABILITY_TRIALS="${STABILITY_TRIALS:-5}"
GPU_SAMPLE_SECONDS="${GPU_SAMPLE_SECONDS:-5}"
DIAG_L2_BETA="${DIAG_L2_BETA:-1e-4}"
OUT_ROOT="${OUT_ROOT:-outputs/kasa_mechanistic/full}"
KASA_ROOT="${KASA_DIR:-$PROJECT_DIR/KaSA}"
KASA_PYTHONPATH="$KASA_ROOT:$KASA_ROOT/peft/src"

EXP1_ROOT="$OUT_ROOT/exp1_gradient_alignment"
EXP2_ROOT="$OUT_ROOT/exp2_update_frame"
EXP3_ROOT="$OUT_ROOT/exp3_weighted_rank_ablation"
EXP4_ROOT="$OUT_ROOT/exp4_svd_stability"
SUMMARY_ROOT="$OUT_ROOT/summaries"

mkdir -p "$EXP1_ROOT" "$EXP2_ROOT" "$EXP3_ROOT" "$EXP4_ROOT" "$SUMMARY_ROOT"

cat > "$OUT_ROOT/run_config.json" <<JSON
{
  "tasks": "${TASKS_ARR[*]}",
  "lrs": "${LRS_ARR[*]}",
  "seeds": "${SEEDS_ARR[*]}",
  "rank": "$RANK",
  "alpha": "$ALPHA",
  "epochs": "$EPOCHS",
  "batch_size": "$BS",
  "data_fraction": "$DATA_FRACTION",
  "max_jobs": "$MAX_JOBS",
  "methods": "lora svd_only lora_diag lora_diag_l2 kasa_noaux kasa",
  "num_gradient_batches": "$NUM_GRAD_BATCHES",
  "stability_trials": "$STABILITY_TRIALS",
  "diag_l2_beta": "$DIAG_L2_BETA",
  "out_root": "$OUT_ROOT",
  "kasa_root": "$KASA_ROOT"
}
JSON

start_gpu_monitor() {
    local out="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; skipping GPU monitor"
        return 0
    fi
    (
        echo "timestamp,memory_used_mib,memory_total_mib,utilization_gpu_pct,power_draw_w,temperature_c"
        while true; do
            ts="$(date +%s.%N)"
            vals="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits | head -1)"
            printf '%s,%s\n' "$ts" "$vals"
            sleep "$GPU_SAMPLE_SECONDS"
        done
    ) > "$out" &
    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    if [ "${GPU_MONITOR_PID:-}" != "" ]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}

RUN_FAILURES=0
GPU_MONITOR_PID=""
trap stop_gpu_monitor EXIT
start_gpu_monitor "$OUT_ROOT/gpu_usage.csv"

worker_count() {
    if [ "${GPU_MONITOR_PID:-}" = "" ]; then
        jobs -pr | wc -l
    else
        jobs -pr | awk -v monitor="$GPU_MONITOR_PID" '$1 != monitor { count++ } END { print count + 0 }'
    fi
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

echo "=== Experiment 0: prepare caches ==="
uv run python scripts/prepare_mechanistic_experiments.py \
    --model_name_or_path roberta-base \
    --tasks "$(IFS=,; echo "${TASKS_ARR[*]}")" \
    --bs "$BS" \
    --max_length 512 \
    --output_dir "$OUT_ROOT/prepare" > "$OUT_ROOT/prepare.log" 2>&1

echo "=== Experiment 3: weighted-rank ablation ==="
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

            launch_job "$EXP3_ROOT/lora/${task}_lr${lr}_s${seed}" \
                uv run python scripts/train_hf_peft.py \
                "${common_args[@]}" \
                --init_lora_weights True \
                --output_dir "$EXP3_ROOT/lora/${task}_lr${lr}_s${seed}"

            launch_job "$EXP3_ROOT/svd_only/${task}_lr${lr}_s${seed}" \
                uv run python scripts/train_svd_only.py \
                "${common_args[@]}" \
                --output_dir "$EXP3_ROOT/svd_only/${task}_lr${lr}_s${seed}"

            launch_job "$EXP3_ROOT/lora_diag/${task}_lr${lr}_s${seed}" \
                uv run python scripts/train_lora_diag.py \
                "${common_args[@]}" \
                --diag_init ones \
                --diag_trainable true \
                --output_dir "$EXP3_ROOT/lora_diag/${task}_lr${lr}_s${seed}"

            launch_job "$EXP3_ROOT/lora_diag_l2/${task}_lr${lr}_s${seed}" \
                uv run python scripts/train_lora_diag.py \
                "${common_args[@]}" \
                --diag_init ones \
                --diag_trainable true \
                --diag_l2_beta "$DIAG_L2_BETA" \
                --output_dir "$EXP3_ROOT/lora_diag_l2/${task}_lr${lr}_s${seed}"

            if [ -d "$KASA_ROOT/peft/src" ]; then
                launch_job "$EXP3_ROOT/kasa_noaux/${task}_lr${lr}_s${seed}" \
                    env PYTHONPATH="$KASA_PYTHONPATH" uv run python scripts/train_kasa_fraction.py \
                    "${common_args[@]}" \
                    --lora_dropout 0.0 \
                    --beta 0.0 \
                    --gemma 0.0 \
                    --output_dir "$EXP3_ROOT/kasa_noaux/${task}_lr${lr}_s${seed}"

                launch_job "$EXP3_ROOT/kasa/${task}_lr${lr}_s${seed}" \
                    env PYTHONPATH="$KASA_PYTHONPATH" uv run python scripts/train_kasa_fraction.py \
                    "${common_args[@]}" \
                    --lora_dropout 0.0 \
                    --output_dir "$EXP3_ROOT/kasa/${task}_lr${lr}_s${seed}"
            else
                echo "[SKIP] KaSA checkout not found at $KASA_ROOT"
            fi
        done
    done
done

while [ "$(worker_count)" -gt 0 ]; do
    if ! wait -n; then
        RUN_FAILURES=$((RUN_FAILURES + 1))
    fi
done

echo "=== Experiment 2: SVD-frame update analysis ==="
mkdir -p "$EXP2_ROOT/_batch"
if ! uv run python scripts/analyze_update_svd_frame_batch.py \
    --root "$OUT_ROOT" \
    --model_name_or_path roberta-base \
    --methods lora,svd_only,lora_diag,lora_diag_l2,kasa_noaux,kasa \
    --target_modules query,value \
    --rank "$RANK" \
    --device auto > "$EXP2_ROOT/_batch/run.log" 2>&1; then
    RUN_FAILURES=$((RUN_FAILURES + 1))
    echo "Experiment 2 batch analysis failed. See $EXP2_ROOT/_batch/run.log"
    tail -80 "$EXP2_ROOT/_batch/run.log" || true
fi

echo "=== Experiment 1: gradient-SVD alignment ==="
for task in "${TASKS_ARR[@]}"; do
    for seed in "${SEEDS_ARR[@]}"; do
        launch_job "$EXP1_ROOT/${task}_s${seed}" \
            uv run python scripts/measure_gradient_svd_alignment.py \
            --model_name_or_path roberta-base \
            --task "$task" \
            --target_modules query,value \
            --num_batches "$NUM_GRAD_BATCHES" \
            --bs "$BS" \
            --rank "$RANK" \
            --seed "$seed" \
            --output_dir "$EXP1_ROOT/${task}_s${seed}"
    done
done

while [ "$(worker_count)" -gt 0 ]; do
    if ! wait -n; then
        RUN_FAILURES=$((RUN_FAILURES + 1))
    fi
done

echo "=== Experiment 4: SVD basis stability ==="
run_job "$EXP4_ROOT/roberta_base_r${RANK}" \
    uv run python scripts/measure_svd_basis_stability.py \
    --model_name_or_path roberta-base \
    --target_modules query,value \
    --rank "$RANK" \
    --epsilons 1e-5,1e-4,1e-3 \
    --trials "$STABILITY_TRIALS" \
    --seed 0 \
    --output_dir "$EXP4_ROOT/roberta_base_r${RANK}" || RUN_FAILURES=$((RUN_FAILURES + 1))

echo "=== Summarize report data ==="
uv run python scripts/summarize_mechanistic_results.py --root "$OUT_ROOT" > "$SUMMARY_ROOT/summarize.log" 2>&1

if [ "$RUN_FAILURES" -ne 0 ]; then
    echo "Completed with $RUN_FAILURES failed jobs. See run.log/status.txt files."
    exit 1
fi

echo "All mechanistic experiments completed. Report data is under $SUMMARY_ROOT"
