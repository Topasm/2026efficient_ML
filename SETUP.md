# Setup

This project uses `uv` for Python environment management.

## 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell if `uv` is not found after installation.

## 2. Create the Environment

For GLUE/RoBERTa experiments:

```bash
uv sync
```

For VLM experiments as well:

```bash
uv sync --extra vlm
```

Activate the environment:

```bash
source .venv/bin/activate
```

The run scripts will automatically use `.venv/bin/activate` when it exists.

## 3. Optional CUDA Wheel Selection

If the default PyTorch wheel does not match the machine's CUDA setup, reinstall PyTorch inside the `uv` environment with the CUDA wheel you need.

Example for CUDA 12.8:

```bash
uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then verify:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. KaSA Dependency

KaSA experiments require the KaSA repository because those runs use its modified PEFT fork.

Default expected location:

```bash
git clone https://github.com/juyongjiang/KaSA.git KaSA
```

Or point to an existing checkout:

```bash
export KASA_DIR=/path/to/KaSA
```

## 5. Smoke Tests

Check imports:

```bash
uv run python - <<'PY'
import torch
import transformers
import datasets
import evaluate
import peft
print("ok")
print("cuda:", torch.cuda.is_available())
PY
```

Check script syntax:

```bash
for f in scripts/run_*.sh; do bash -n "$f"; done
uv run python -m py_compile scripts/*.py
```

## 6. Mechanistic Experiment Preparation

For the KaSA mechanistic experiments, prepare GLUE data, metrics, tokenizer, and `roberta-base` caches with:

```bash
uv run python scripts/prepare_mechanistic_experiments.py \
  --model_name_or_path roberta-base \
  --tasks cola,rte,mrpc,sst2 \
  --bs 32 \
  --max_length 512 \
  --output_dir outputs/kasa_mechanistic/prepare
```

Then run `scripts/run_mechanistic_fast_ablation.sh` for a short grid or `scripts/run_mechanistic_full_experiments.sh` for full report-data collection.

## 7. Run

Examples:

```bash
bash scripts/run_phase2_headtohead.sh
bash scripts/run_phase3_data_scaling.sh
bash scripts/run_lr_sensitivity.sh
```

Generated files go under `outputs/`, which is ignored by git.
