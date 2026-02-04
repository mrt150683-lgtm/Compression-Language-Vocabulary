#!/usr/bin/env bash
set -euo pipefail

# Ablation: No VQ - phrase replacement only
#
# Shows performance degradation → proves codebook helps.
# Runs training and evaluation without VQ module, using only
# phrase replacement with CLV tokens.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running ablation: No VQ (phrase replacement only)"
echo "⚠️  WARNING: Requires GPU for training."

# TODO: Modify training config to disable VQ module
# TODO: Run training without VQ
# TODO: Run evaluations
# TODO: Save results to reports/ablations/no_vq/

echo "Ablation complete: No VQ"

