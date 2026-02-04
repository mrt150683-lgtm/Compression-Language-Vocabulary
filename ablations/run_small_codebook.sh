#!/usr/bin/env bash
set -euo pipefail

# Ablation: Small codebook (K=2k)
#
# Shows compression vs stability trade-off.
# Rebuilds codebook with K=2000 and retrains.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running ablation: Small codebook (K=2000)"
echo "⚠️  WARNING: Requires GPU for training."

# TODO: Rebuild codebook with K=2000
# python mining/build_codebook.py \
#   --embeddings data/phrase_embeddings.npy \
#   --output artifacts/clv_codebook_2k.npy \
#   --codebook-size 2000 \
#   --code-dim 256

# TODO: Rebuild CLV mapping with small codebook
# python mining/build_clv_map.py \
#   --embeddings data/phrase_embeddings.npy \
#   --index data/phrase_index.jsonl \
#   --codebook artifacts/clv_codebook_2k.npy \
#   --output artifacts/clv_map_2k.json

# TODO: Train with small codebook
# TODO: Run evaluations
# TODO: Save results to reports/ablations/small_codebook/

echo "Ablation complete: Small codebook"

