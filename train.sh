#!/bin/bash
# Cross-platform Linux training script with auto requirements

# 1. Detect CUDA or ROCm
GPU_FRAMEWORK="cpu"
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU detected. Using CUDA..."
    GPU_FRAMEWORK="cuda"
elif command -v rocminfo &> /dev/null; then
    echo "[INFO] AMD GPU detected. Using ROCm..."
    GPU_FRAMEWORK="rocm"
else
    echo "[INFO] No GPU detected. Using CPU."
fi

# 2. Create virtual environment
python3 -m venv venv
source ./venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install pipreqs to auto-generate requirements
pip install pipreqs

# 5. Auto-generate requirements.txt
pipreqs . --force

# 6. Install GPU-compatible torch if GPU exists
if [ "$GPU_FRAMEWORK" = "cuda" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [ "$GPU_FRAMEWORK" = "rocm" ]; then
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.4
fi

# 7. Install remaining dependencies from generated requirements.txt
pip install -r requirements.txt

# 8. Run cleanDataset.py and trainGPT.py
python cleanDataset.py
python trainGPT.py

echo "[INFO] Training finished!"
