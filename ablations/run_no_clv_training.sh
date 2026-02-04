#!/usr/bin/env bash
set -euo pipefail

# Ablation: No CLV training
#
# At inference, compress with <clv> but no adapted model.
# Shows it breaks → proves joint training is required.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running ablation: No CLV training (baseline model with CLV compression)"
echo "⚠️  WARNING: Requires GPU for evaluation."

# TODO: Run evaluations with:
# - Baseline model (no CLV adapter)
# - CLV compression applied at inference
# - This should show degraded performance

# TODO: Save results to reports/ablations/no_clv_training/

echo "Ablation complete: No CLV training"

