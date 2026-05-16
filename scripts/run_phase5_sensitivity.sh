#!/bin/bash
# Phase 5: Beta/Gamma Sensitivity
# CoLA only, r=8
# (a) beta=1e-4 fixed, gamma sweep x 3 seeds = 15
# (b) gamma=1e-3 fixed, beta sweep x 3 seeds = 15
# Total: 30 runs

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/outputs/phase5"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$PROJECT_DIR/KaSA/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)
TASK="cola"
RANK=8
ALPHA=16
EPOCHS=100
BATCH=32
HEAD_LR=4e-4
MODULE_LR=4e-4

SWEEP_VALUES=(1e-5 1e-4 1e-3 1e-2 1e-1)

echo "=== Phase 5: Beta/Gamma Sensitivity ==="

# --- (a) Gamma sweep, beta fixed at 1e-4 (CoLA default) ---
BETA_FIXED="1e-4"
echo ""
echo "--- (a) Gamma sweep (beta=$BETA_FIXED) ---"

for gamma in "${SWEEP_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="$LOG_DIR/gamma_sweep_g${gamma}_s${seed}"
        if [ -f "$out/results.json" ]; then
            echo "[SKIP] gamma=$gamma seed=$seed"
            continue
        fi
        echo "[$(date '+%H:%M:%S')] KaSA | beta=$BETA_FIXED gamma=$gamma | seed=$seed"
        PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
            --model_name_or_path roberta-base \
            --task "$TASK" \
            --lora_r $RANK --lora_alpha $ALPHA --lora_dropout 0.0 \
            --num_epochs $EPOCHS --bs $BATCH --max_length 512 \
            --head_lr $HEAD_LR --module_lr $MODULE_LR \
            --beta "$BETA_FIXED" --gemma "$gamma" \
            --weight_decay 0.0 --warmup_ratio 0.06 \
            --seed "$seed" --output_dir "$out" \
            2>&1 | tee "$LOG_DIR/gamma_sweep_g${gamma}_s${seed}.log"
    done
done

# --- (b) Beta sweep, gamma fixed at 1e-3 (CoLA default) ---
GAMMA_FIXED="1e-3"
echo ""
echo "--- (b) Beta sweep (gamma=$GAMMA_FIXED) ---"

for beta in "${SWEEP_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="$LOG_DIR/beta_sweep_b${beta}_s${seed}"
        if [ -f "$out/results.json" ]; then
            echo "[SKIP] beta=$beta seed=$seed"
            continue
        fi
        echo "[$(date '+%H:%M:%S')] KaSA | beta=$beta gamma=$GAMMA_FIXED | seed=$seed"
        PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
            --model_name_or_path roberta-base \
            --task "$TASK" \
            --lora_r $RANK --lora_alpha $ALPHA --lora_dropout 0.0 \
            --num_epochs $EPOCHS --bs $BATCH --max_length 512 \
            --head_lr $HEAD_LR --module_lr $MODULE_LR \
            --beta "$beta" --gemma "$GAMMA_FIXED" \
            --weight_decay 0.0 --warmup_ratio 0.06 \
            --seed "$seed" --output_dir "$out" \
            2>&1 | tee "$LOG_DIR/beta_sweep_b${beta}_s${seed}.log"
    done
done

echo ""
echo "=== Phase 5 Complete ==="
echo "Total runs: 30"
