@echo off
REM Detect CUDA or ROCm
set GPU_FRAMEWORK=cpu
where nvidia-smi >nul 2>nul
IF %ERRORLEVEL%==0 (
    echo NVIDIA GPU detected. Using CUDA...
    set GPU_FRAMEWORK=cuda
) ELSE (
    where rocminfo >nul 2>nul
    IF %ERRORLEVEL%==0 (
        echo AMD GPU detected. Using ROCm...
        set GPU_FRAMEWORK=rocm
    ) ELSE (
        echo No GPU detected. Using CPU.
    )
)

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install pipreqs to auto-generate requirements
pip install pipreqs

REM Auto-generate requirements.txt
pipreqs . --force

REM Install GPU-compatible torch if GPU exists
IF "%GPU_FRAMEWORK%"=="cuda" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) ELSE IF "%GPU_FRAMEWORK%"=="rocm" (
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.4
)

REM Install remaining dependencies from generated requirements.txt
pip install -r requirements.txt

REM Run scripts
python cleanDataset.py
python trainGPT.py

echo Training finished!
pause
