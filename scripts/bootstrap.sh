#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script: Setup environment and download dependencies
# 
# Responsibilities:
# - Create venv (if not under Docker)
# - Install deps (CPU-safe if no GPU)
# - Download base model (small for local, large for remote GPU)
# - Set up dataset code paths (full downloads behind confirmation)
# - Print: model paths, dataset summaries

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "CLV-Lang Bootstrap Script"
echo "=========================================="
echo ""

# Detect if running inside Docker
IN_DOCKER=false
if [ -f /.dockerenv ] || grep -q docker /proc/self/cgroup 2>/dev/null; then
    IN_DOCKER=true
    echo "✓ Detected: Running inside Docker container"
else
    echo "✓ Detected: Running on local machine"
fi
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        echo "✓ Python $PYTHON_VERSION found (3.10+ required)"
        PYTHON_CMD=python3
    else
        echo "✗ Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    echo "✗ Python 3 not found"
    exit 1
fi
echo ""

# Create virtual environment (only if not in Docker)
if [ "$IN_DOCKER" = false ]; then
    VENV_PATH="$PROJECT_ROOT/.venv"
    if [ ! -d "$VENV_PATH" ]; then
        echo "Creating Python virtual environment..."
        "$PYTHON_CMD" -m venv "$VENV_PATH"
        echo "✓ Virtual environment created at $VENV_PATH"
    else
        echo "✓ Virtual environment already exists at $VENV_PATH"
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
    echo ""
fi

# Upgrade pip
echo "Upgrading pip..."
"$PYTHON_CMD" -m pip install --quiet --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Detect GPU availability
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
        echo "✓ GPU detected (nvidia-smi available)"
    else
        echo "⚠ GPU driver found but nvidia-smi failed (may be CPU-only mode)"
    fi
else
    echo "⚠ No GPU detected (nvidia-smi not available) - using CPU-only packages"
fi
echo ""

# Install PyTorch (CPU or GPU based on availability)
echo "Installing PyTorch..."
if [ "$HAS_GPU" = true ]; then
    echo "  → Installing PyTorch with CUDA support..."
    "$PYTHON_CMD" -m pip install --quiet \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
else
    echo "  → Installing PyTorch CPU-only version..."
    "$PYTHON_CMD" -m pip install --quiet \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
fi
echo "✓ PyTorch installed"
echo ""

# Install core ML libraries
echo "Installing core ML libraries..."
"$PYTHON_CMD" -m pip install --quiet \
    transformers \
    accelerate \
    datasets \
    peft \
    bitsandbytes
echo "✓ Core ML libraries installed"
echo ""

# Install flash-attn (only if GPU available, optional)
if [ "$HAS_GPU" = true ]; then
    echo "Installing flash-attn (GPU required)..."
    if "$PYTHON_CMD" -m pip install --quiet flash-attn --no-build-isolation 2>/dev/null; then
        echo "✓ flash-attn installed"
    else
        echo "⚠ flash-attn build failed (optional, continuing without it)"
    fi
else
    echo "⚠ Skipping flash-attn (GPU required, not available)"
fi
echo ""

# Install additional dependencies
echo "Installing additional dependencies..."
if [ "$HAS_GPU" = true ]; then
    "$PYTHON_CMD" -m pip install --quiet faiss-gpu
else
    "$PYTHON_CMD" -m pip install --quiet faiss-cpu
fi

"$PYTHON_CMD" -m pip install --quiet \
    scikit-learn \
    einops \
    wandb
echo "✓ Additional dependencies installed"
echo ""

# Verify installations
echo "Verifying installations..."
"$PYTHON_CMD" -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ✗ PyTorch not found"
"$PYTHON_CMD" -c "import transformers; print(f'  Transformers: {transformers.__version__}')" 2>/dev/null || echo "  ✗ Transformers not found"
"$PYTHON_CMD" -c "import datasets; print(f'  Datasets: {datasets.__version__}')" 2>/dev/null || echo "  ✗ Datasets not found"
echo ""

# Create data directories
echo "Setting up data directories..."
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/artifacts"
mkdir -p "$PROJECT_ROOT/reports"
echo "✓ Data directories created"
echo ""

# Download base model
echo "=========================================="
echo "Model Download"
echo "=========================================="
echo ""

MODEL_NAME=""
MODEL_PATH=""

# For local CPU testing, use small model; for remote GPU, use large model
if [ "$HAS_GPU" = false ]; then
    echo "CPU-only mode detected: Using small model for local testing"
    MODEL_NAME="Qwen/Qwen2-0.5B-Instruct"
    MODEL_PATH="$PROJECT_ROOT/data/models/qwen2-0.5b-instruct"
    echo ""
    echo "Note: For remote GPU training, use Qwen2-7B-Instruct instead."
    echo "      See configs/base_qwen2_7b.yaml for 7B model configuration."
    echo ""
else
    echo "GPU detected: You can use either small or large model"
    echo ""
    read -p "Use small model (Qwen2-0.5B) for testing? [y/N]: " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        MODEL_NAME="Qwen/Qwen2-0.5B-Instruct"
        MODEL_PATH="$PROJECT_ROOT/data/models/qwen2-0.5b-instruct"
    else
        MODEL_NAME="Qwen/Qwen2-7B-Instruct"
        MODEL_PATH="$PROJECT_ROOT/data/models/qwen2-7b-instruct"
        echo "⚠ Large model selected (7B). This will take significant time and disk space."
    fi
fi

echo ""
echo "Downloading model: $MODEL_NAME"
echo "  → Target path: $MODEL_PATH"
echo ""

if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
    "$PYTHON_CMD" -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
model_name = '$MODEL_NAME'
model_path = '$MODEL_PATH'
print(f'Downloading {model_name}...')
os.makedirs(model_path, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path)
print(f'✓ Model downloaded to {model_path}')
" || {
    echo "✗ Model download failed"
    exit 1
}
else
    echo "✓ Model already exists at $MODEL_PATH"
fi
echo ""

# Dataset setup (code paths only, downloads behind confirmation)
echo "=========================================="
echo "Dataset Setup"
echo "=========================================="
echo ""

echo "Setting up dataset code paths..."
echo ""

# LongBench setup
LONGBENCH_PATH="$PROJECT_ROOT/data/longbench"
mkdir -p "$LONGBENCH_PATH"
echo "✓ LongBench path: $LONGBENCH_PATH"
echo "  → Full download: Use --download-datasets flag or run manually"
echo ""

# SCROLLS setup
SCROLLS_PATH="$PROJECT_ROOT/data/scrolls"
mkdir -p "$SCROLLS_PATH"
echo "✓ SCROLLS path: $SCROLLS_PATH"
echo "  → Full download: Use --download-datasets flag or run manually"
echo ""

# Wikitext-103 setup
WIKITEXT_PATH="$PROJECT_ROOT/data/wikitext"
mkdir -p "$WIKITEXT_PATH"
echo "✓ Wikitext-103 path: $WIKITEXT_PATH"
echo "  → Full download: Use --download-datasets flag or run manually"
echo ""

# Check if --download-datasets flag is provided
DOWNLOAD_DATASETS=false
for arg in "$@"; do
    if [ "$arg" = "--download-datasets" ]; then
        DOWNLOAD_DATASETS=true
        break
    fi
done

if [ "$DOWNLOAD_DATASETS" = true ]; then
    echo "=========================================="
    echo "Downloading Full Datasets"
    echo "=========================================="
    echo ""
    echo "⚠ This will download large datasets. Continue? [y/N]"
    read -p "> " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Downloading LongBench..."
        "$PYTHON_CMD" -c "
from datasets import load_dataset
import os
longbench_path = '$LONGBENCH_PATH'
os.makedirs(longbench_path, exist_ok=True)
# TODO: Load LongBench dataset
print('LongBench download initiated...')
print('Note: LongBench may require manual setup. See data/README_DATA.md')
" || echo "⚠ LongBench download may require manual setup"
        
        echo ""
        echo "Downloading SCROLLS..."
        "$PYTHON_CMD" -c "
from datasets import load_dataset
import os
scrolls_path = '$SCROLLS_PATH'
os.makedirs(scrolls_path, exist_ok=True)
# TODO: Load SCROLLS dataset
print('SCROLLS download initiated...')
print('Note: SCROLLS may require manual setup. See data/README_DATA.md')
" || echo "⚠ SCROLLS download may require manual setup"
        
        echo ""
        echo "Downloading Wikitext-103..."
        "$PYTHON_CMD" -c "
from datasets import load_dataset
import os
wikitext_path = '$WIKITEXT_PATH'
os.makedirs(wikitext_path, exist_ok=True)
try:
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', cache_dir=wikitext_path)
    print('✓ Wikitext-103 downloaded')
except Exception as e:
    print(f'⚠ Wikitext-103 download failed: {e}')
" || echo "⚠ Wikitext-103 download failed"
        
        echo ""
    else
        echo "Skipping full dataset downloads."
        echo "Run with --download-datasets flag to download later."
    fi
else
    echo "Skipping full dataset downloads (use --download-datasets to enable)."
    echo ""
fi

# Print summary
echo "=========================================="
echo "Bootstrap Summary"
echo "=========================================="
echo ""
echo "Environment:"
if [ "$IN_DOCKER" = true ]; then
    echo "  ✓ Running in Docker"
else
    echo "  ✓ Local machine (venv at $VENV_PATH)"
fi
echo "  ✓ Python: $PYTHON_VERSION"
if [ "$HAS_GPU" = true ]; then
    echo "  ✓ GPU: Available"
else
    echo "  ⚠ GPU: Not available (CPU-only mode)"
fi
echo ""
echo "Model:"
echo "  ✓ Model: $MODEL_NAME"
echo "  ✓ Path: $MODEL_PATH"
echo ""
echo "Datasets:"
echo "  ✓ LongBench: $LONGBENCH_PATH"
echo "  ✓ SCROLLS: $SCROLLS_PATH"
echo "  ✓ Wikitext-103: $WIKITEXT_PATH"
echo ""
echo "Next Steps:"
echo "  1. Review configs/ for training/evaluation settings"
echo "  2. For remote GPU training, use: ./scripts/train_poc.sh"
echo "  3. For local testing, use small models and CPU-compatible scripts"
echo ""
if [ "$IN_DOCKER" = false ]; then
    echo "Note: Activate virtual environment with:"
    echo "  source $VENV_PATH/bin/activate"
    echo ""
fi
echo "=========================================="
echo "Bootstrap complete!"
echo "=========================================="
