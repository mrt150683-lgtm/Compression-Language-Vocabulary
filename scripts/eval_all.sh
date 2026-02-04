#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-runs/qwen2_7b_clv_k8192/last}"
IDX="data/phrase_index.jsonl"
MAP="artifacts/clv_map.json"

# Perplexity (lossless on small slice)
python eval/eval_perplexity.py \
  --model_path "$CKPT" \
  --phrase_index "$IDX" \
  --mode lossless \
  --limit 2000 \
  --out reports/ppl_lossless.json

# Needle retrieval under long contexts (lossless)
python eval/eval_needle.py \
  --model_path "$CKPT" \
  --phrase_index "$IDX" \
  --mode lossless \
  --needles 128 \
  --depth 8192 \
  --out reports/needle_8192.json

# Latency + token cost comparison (baseline vs CLV)
python eval/eval_latency_cost.py \
  --model_path "$CKPT" \
  --phrase_index "$IDX" \
  --mode lossless \
  --prompts eval/prompts/sanity_20.jsonl \
  --out reports/latency_cost_lossless.json

# Optional: codebook (lossy) comparison
python eval/eval_latency_cost.py \
  --model_path "$CKPT" \
  --clv_map "$MAP" \
  --mode codebook \
  --prompts eval/prompts/sanity_20.jsonl \
  --out reports/latency_cost_codebook.json
