#!/bin/bash

set -e  # exit on error

echo "========================================"
echo "  Azure ML VM Setup Script Started"
echo "========================================"

# ---------------------------------------------------------
# 1. Update System
# ---------------------------------------------------------
echo "[1/8] Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential unzip htop screen tmux

# ---------------------------------------------------------
# 2. Install Miniconda (if not already installed)
# ---------------------------------------------------------
if [ ! -d "$HOME/miniconda3" ]; then
    echo "[2/8] Installing Miniconda..."
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
else
    echo "[2/8] Miniconda already installed. Skipping."
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
fi

source ~/.bashrc

# ---------------------------------------------------------
# 3. Create Conda Environment
# ---------------------------------------------------------
echo "[3/8] Creating conda environment 'food-exp'..."
conda env remove -n food-exp -y >/dev/null 2>&1 || true
conda create -n food-exp python=3.11 -y

# Activate env
source ~/.bashrc
conda activate food-exp

# ---------------------------------------------------------
# 4. Install PyTorch (GPU CUDA)
# ---------------------------------------------------------
echo "[4/8] Installing PyTorch with CUDA support..."
# Automatically detects GPU and installs CUDA build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------
# 5. Install HuggingFace, Azure SDK, wandb, and utilities
# ---------------------------------------------------------
echo "[5/8] Installing common ML libraries..."
pip install \
    datasets \
    transformers \
    accelerate \
    timm \
    scikit-learn \
    matplotlib \
    opencv-python \
    pillow \
    tqdm \
    wandb \
    azure-storage-blob \
    azure-identity \
    python-dotenv

# ---------------------------------------------------------
# 6. Install AzCopy (for Blob uploads/downloads)
# ---------------------------------------------------------
echo "[6/8] Installing AzCopy..."
cd /tmp
wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
tar -xvf azcopy.tar.gz
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/local/bin/
azcopy --version

# ---------------------------------------------------------
# 7. Install NVIDIA Drivers Checker (optional safety)
# ---------------------------------------------------------
echo "[7/8] Checking GPU availability..."
nvidia-smi || echo "Warning: NVIDIA GPU not detected yet. Ensure VM is GPU-enabled."

# ---------------------------------------------------------
# 8. Final Summary
# ---------------------------------------------------------
echo "======================================================"
echo "   Setup Complete!"
echo "   Command to follow to start working:"
echo "      chmod +x setup.sh"
echo "      bash setup.sh"
echo "  To start working, run:"
echo "       source ~/.bashrc"
echo "       conda activate food-exp"
echo "======================================================"

