# CLV-Lang Dockerfile
# Deterministic environment for remote GPU runtime (A100 80GB)
# Base: Ubuntu 22.04 + CUDA 12.x + Python 3.10

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3 1

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
# Note: Using CUDA 12.1 compatible build
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    peft \
    bitsandbytes

# Install flash-attn
# WARNING: flash-attn build is environment-specific and may fail on some systems
# If build fails, you may need to adjust CUDA/compiler versions or skip this package
# Alternative: Use pre-built wheels if available for your CUDA version
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || \
    echo "WARNING: flash-attn build failed. You may need to install manually or use alternative."

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    scikit-learn \
    faiss-gpu \
    einops \
    wandb

# Verify installations
RUN python3 --version && \
    pip3 list | grep -E "(torch|transformers|accelerate|datasets|peft|bitsandbytes|einops|wandb)"

# Default command (shell)
CMD ["/bin/bash"]
