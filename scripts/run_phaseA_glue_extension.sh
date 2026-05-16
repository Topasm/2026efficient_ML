#!/bin/bash
# Phase A: Extend head-to-head comparison to SST-2, STS-B, QNLI
# Per-task HPs from KaSA paper run scripts
# 3 tasks × 3 methods × 3 seeds = 27 runs

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/phaseA_glue_ext"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)

# Per-task HPs from KaSA paper (roberta_base runs)
declare -A EPOCHS
declare -A BS
declare -A LR
declare -A BETA
declare -A GEMMA

# SST-2: 67K samples, bs=128, 20 epochs (shorter to match ~100-epoch wall-clock of CoLA)
EPOCHS[sst2]=20; BS[sst2]=128; LR[sst2]=5e-4; BETA[sst2]=0.0001; GEMMA[sst2]=0.001

# STS-B: 5.7K samples, bs=32, 100 epochs, regression
EPOCHS[stsb]=100; BS[stsb]=32; LR[stsb]=3e-4; BETA[stsb]=0.0001; GEMMA[stsb]=0.00001

# QNLI: 105K samples, bs=32, 20 epochs (shorter)
EPOCHS[qnli]=20; BS[qnli]=32; LR[qnli]=4e-4; BETA[qnli]=0.01; GEMMA[qnli]=0.00001

TASKS=("sst2" "stsb" "qnli")

echo "=== Phase A: GLUE extension (SST-2, STS-B, QNLI) ==="

for task in "${TASKS[@]}"; do
    E=${EPOCHS[$task]}; B=${BS[$task]}; L=${LR[$task]}
    BT=${BETA[$task]}; GM=${GEMMA[$task]}
    echo ""
    echo "--- Task: $task (epochs=$E, bs=$B, lr=$L, beta=$BT, gemma=$GM) ---"

    for seed in "${SEEDS[@]}"; do

        # LoRA
        out="$LOG_DIR/lora_${task}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] LoRA $task s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --task "$task" --init_lora_weights True \
                --lora_r 8 --lora_alpha 16 \
                --head_lr "$L" --module_lr "$L" \
                --num_epochs $E --bs $B --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/lora_${task}_s${seed}.log"
        fi

        # PiSSA
        out="$LOG_DIR/pissa_${task}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] PiSSA $task s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --task "$task" --init_lora_weights pissa \
                --lora_r 8 --lora_alpha 16 \
                --head_lr "$L" --module_lr "$L" \
                --num_epochs $E --bs $B --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/pissa_${task}_s${seed}.log"
        fi

        # KaSA
        out="$LOG_DIR/kasa_${task}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] KaSA $task s=$seed"
            PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python \
                "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
                --task "$task" \
                --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
                --num_epochs $E --bs $B --max_length 512 \
                --head_lr "$L" --module_lr "$L" \
                --beta "$BT" --gemma "$GM" \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/kasa_${task}_s${seed}.log"
        fi
    done
done

echo "=== Phase A complete: 27 runs ==="
