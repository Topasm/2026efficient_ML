#!/bin/bash
# Phase 6: VLM Extension (Qwen2-VL-2B on ChartQA)
# Simplified: 10% train fraction, 500 val samples, 3 seeds x 3 methods = 9 runs
# Target: ~3-4 hours total

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/phase6"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)
MODEL="Qwen/Qwen2-VL-2B-Instruct"
DATASET="HuggingFaceM4/ChartQA"
RANK=8
ALPHA=16
EPOCHS=3
BATCH=2
GRAD_ACCUM=16
LR=2e-5
TRAIN_FRAC=0.10
EVAL_SAMPLES=500

echo "=== Phase 6: VLM Extension ==="
echo "Model: $MODEL"
echo "Train fraction: $TRAIN_FRAC, Eval: $EVAL_SAMPLES samples"
echo ""

# --- LoRA ---
for seed in "${SEEDS[@]}"; do
    out="$LOG_DIR/lora_chartqa_s${seed}"
    if [ -f "$out/results.json" ]; then echo "[SKIP] LoRA s=$seed"; continue; fi
    echo "[$(date '+%H:%M:%S')] LoRA | seed=$seed"
    CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_vlm.py" \
        --model_name_or_path "$MODEL" --dataset_name "$DATASET" \
        --init_lora_weights True \
        --lora_r $RANK --lora_alpha $ALPHA \
        --target_modules "q_proj,v_proj" \
        --learning_rate $LR --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $BATCH \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --train_fraction $TRAIN_FRAC --eval_samples $EVAL_SAMPLES \
        --seed "$seed" --bf16 --output_dir "$out" \
        2>&1 | tee "$LOG_DIR/lora_chartqa_s${seed}.log"
done

# --- PiSSA ---
for seed in "${SEEDS[@]}"; do
    out="$LOG_DIR/pissa_chartqa_s${seed}"
    if [ -f "$out/results.json" ]; then echo "[SKIP] PiSSA s=$seed"; continue; fi
    echo "[$(date '+%H:%M:%S')] PiSSA | seed=$seed"
    CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_vlm.py" \
        --model_name_or_path "$MODEL" --dataset_name "$DATASET" \
        --init_lora_weights pissa \
        --lora_r $RANK --lora_alpha $ALPHA \
        --target_modules "q_proj,v_proj" \
        --learning_rate $LR --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $BATCH \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --train_fraction $TRAIN_FRAC --eval_samples $EVAL_SAMPLES \
        --seed "$seed" --bf16 --output_dir "$out" \
        2>&1 | tee "$LOG_DIR/pissa_chartqa_s${seed}.log"
done

# --- KaSA (PEFT fork) ---
for seed in "${SEEDS[@]}"; do
    out="$LOG_DIR/kasa_chartqa_s${seed}"
    if [ -f "$out/results.json" ]; then echo "[SKIP] KaSA s=$seed"; continue; fi
    echo "[$(date '+%H:%M:%S')] KaSA | seed=$seed"
    PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_vlm_kasa.py" \
        --model_name_or_path "$MODEL" --dataset_name "$DATASET" \
        --lora_r $RANK --lora_alpha $ALPHA \
        --target_modules "q_proj,v_proj" \
        --learning_rate $LR --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $BATCH \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --train_fraction $TRAIN_FRAC --eval_samples $EVAL_SAMPLES \
        --beta 1e-4 --gemma 1e-3 \
        --seed "$seed" --bf16 --output_dir "$out" \
        2>&1 | tee "$LOG_DIR/kasa_chartqa_s${seed}.log"
done

echo ""
echo "=== Phase 6 Complete: 9 runs ==="
