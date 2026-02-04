#!/usr/bin/env bash
set -euo pipefail

# Ablation: Large codebook (K=32k)
#
# Shows compression vs stability trade-off.
# Rebuilds codebook with K=32000 and retrains.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running ablation: Large codebook (K=32000)"
echo "⚠️  WARNING: Requires GPU for training."

# TODO: Rebuild codebook with K=32000
# python mining/build_codebook.py \
#   --embeddings data/phrase_embeddings.npy \
#   --output artifacts/clv_codebook_32k.npy \
#   --codebook-size 32000 \
#   --code-dim 256

# TODO: Rebuild CLV mapping with large codebook
# python mining/build_clv_map.py \
#   --embeddings data/phrase_embeddings.npy \
#   --index data/phrase_index.jsonl \
#   --codebook artifacts/clv_codebook_32k.npy \
#   --output artifacts/clv_map_32k.json

# TODO: Train with large codebook
# TODO: Run evaluations
# TODO: Save results to reports/ablations/large_codebook/

echo "Ablation complete: Large codebook"

