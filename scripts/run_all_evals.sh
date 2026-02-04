#!/usr/bin/env bash
set -euo pipefail

# Run all evaluations: Execute full evaluation suite
#
# ⚠️ GPU WARNING: Most evaluations require GPU for reasonable performance
# Some may run on CPU but will be very slow.
#
# Evaluations:
# - Perplexity (Wikitext-103)
# - LongBench tasks
# - SCROLLS tasks
# - Needle-in-a-Haystack
# - Latency/Cost

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running CLV-Lang evaluation suite..."
echo "⚠️  WARNING: Evaluations require GPU for reasonable performance."

# TODO: Check for GPU availability
# TODO: Warn if no GPU (but allow --force-cpu flag)

# Create reports directory
mkdir -p reports/plots

# Detect device
DEVICE="cuda"
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    DEVICE="cpu"
    echo "⚠️  No GPU detected, using CPU (will be slow)"
fi

# Get checkpoint path (use most recent or specified)
CKPT="${1:-runs/qwen2_7b_clv_k8192}"
if [ ! -d "$CKPT" ]; then
    echo "⚠️  Checkpoint not found: $CKPT"
    echo "   Usage: $0 [checkpoint_path]"
    echo "   Default: runs/qwen2_7b_clv_k8192"
    exit 1
fi

echo "Using checkpoint: $CKPT"
echo "Device: $DEVICE"
echo ""

# Run perplexity evaluation (baseline + CLV)
# Use lossless mode to match training config (lossless: true)
echo "Running perplexity evaluation (baseline + CLV lossless)..."
python eval/eval_perplexity.py \
  --base-model-name Qwen/Qwen2-7B-Instruct \
  --clv-adapter "$CKPT/lora_adapter" \
  --phrase-index data/phrase_index.jsonl \
  --use-lossless \
  --dataset wikitext \
  --output-dir reports \
  --device "$DEVICE" \
  --num-samples 1000 || echo "⚠️  Perplexity evaluation failed"

echo ""

# Run Needle-in-a-Haystack evaluation (baseline + CLV lossless)
echo "Running Needle-in-a-Haystack evaluation (baseline + lossless CLV)..."
python eval/eval_needle.py \
  --base-model-name Qwen/Qwen2-7B-Instruct \
  --clv-adapter "$CKPT/lora_adapter" \
  --phrase-index data/phrase_index.jsonl \
  --use-lossless \
  --output-dir reports \
  --device "$DEVICE" \
  --num-trials 20 || echo "⚠️  Needle evaluation failed"

echo ""

# Run latency/cost evaluation (baseline + CLV lossless)
echo "Running latency/cost evaluation (baseline + lossless CLV)..."
python eval/eval_latency_cost.py \
  --base-model-name Qwen/Qwen2-7B-Instruct \
  --clv-adapter "$CKPT/lora_adapter" \
  --phrase-index data/phrase_index.jsonl \
  --use-lossless \
  --output-dir reports \
  --device "$DEVICE" \
  --num-texts 10 \
  --num-runs 5 || echo "⚠️  Latency evaluation failed"

echo ""

# Run LongBench evaluation (optional - requires dataset setup)
# Uncomment and configure when LongBench datasets are available
# if [ -f "configs/eval_longbench.yaml" ]; then
#     echo "Running LongBench evaluation..."
#     python eval/eval_longbench.py \
#       --base-model-name Qwen/Qwen2-7B-Instruct \
#       --clv-adapter "$CKPT/lora_adapter" \
#       --clv-map artifacts/clv_map.json \
#       --output-dir reports \
#       --device "$DEVICE" || echo "⚠️  LongBench evaluation failed"
#     echo ""
# fi

# Run SCROLLS evaluation (optional - requires dataset setup)
# Uncomment and configure when SCROLLS datasets are available
# if [ -f "configs/eval_scrolls.yaml" ]; then
#     echo "Running SCROLLS evaluation..."
#     python eval/eval_scrolls.py \
#       --base-model-name Qwen/Qwen2-7B-Instruct \
#       --clv-adapter "$CKPT/lora_adapter" \
#       --clv-map artifacts/clv_map.json \
#       --output-dir reports \
#       --device "$DEVICE" || echo "⚠️  SCROLLS evaluation failed"
#     echo ""
# fi

# Aggregate results into summary report
echo "Generating summary report..."
python -c "
import json
from pathlib import Path
import glob

reports_dir = Path('reports')
json_files = list(reports_dir.glob('*.json'))

summary = {
    'evaluations': {},
    'timestamp': __import__('datetime').datetime.now().isoformat()
}

for json_file in json_files:
    if json_file.name == 'summary.json':
        continue
    try:
        with open(json_file) as f:
            data = json.load(f)
            summary['evaluations'][json_file.stem] = data
    except Exception as e:
        print(f'⚠️  Could not load {json_file}: {e}')

with open(reports_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'✓ Summary report saved to: {reports_dir / \"summary.json\"}')
" || echo "⚠️  Summary generation failed"

echo ""
echo "All evaluations complete!"
echo "Results saved to: reports/"
echo ""
echo "Summary: reports/summary.json"

