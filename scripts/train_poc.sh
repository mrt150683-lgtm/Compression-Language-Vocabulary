#!/usr/bin/env bash
set -euo pipefail

# --- Paths (adjust if needed)
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1

# --- Create cache dir
mkdir -p "$HF_HOME"

# --- Install deps (CUDA build assumed on an A100 image)
python -m pip install --upgrade pip
pip install "torch==2.4.0" --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-cuda.txt || true
# safety pins (common):
pip install transformers==4.44.2 accelerate==0.34.2 peft==0.12.0 datasets==2.21.0 einops==0.8.0 faiss-gpu==1.8.0.post2

# --- Quick health
python - <<'PY'
import torch; print("CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count(), "Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

# --- Accelerate config (non-interactive sane defaults)
accelerate config default

# --- Kick training (dry-run toggle via DRY=1)
CFG="configs/train_qwen2_7b_clv.yaml"
if [[ "${DRY:-0}" == "1" ]]; then
  EXTRA="--dry_run"
else
  EXTRA=""
fi

python training/train_clv_lora.py \
  --config "$CFG" \
  $EXTRA \
  --report_to wandb \
  --wandb_project clv-poc \
  --wandb_run_name qwen2-7b-k8192
