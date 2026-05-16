# Efficient ML Experiment Scripts

Source code for running PEFT adaptation experiments with LoRA, PiSSA, EVA, and KaSA-style variants.

This repository tracks code only. Generated outputs are written under `outputs/` and ignored by git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

KaSA runs require a local KaSA checkout that includes the modified PEFT fork:

```bash
git clone https://github.com/juyongjiang/KaSA.git KaSA
```

Alternatively, set `KASA_DIR` before running scripts:

```bash
export KASA_DIR=/path/to/KaSA
```

## Examples

```bash
bash scripts/run_phase2_headtohead.sh
bash scripts/run_phase3_data_scaling.sh
bash scripts/run_lr_sensitivity.sh
```

You can override the virtualenv path with:

```bash
export VENV=/path/to/venv/bin/activate
```
