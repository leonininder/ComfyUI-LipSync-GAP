@echo off
setlocal enabledelayedexpansion
echo ===============================================================================
echo Geekatplay Studio - ComfyUI-LipSync-GAP Installer
echo ===============================================================================

REM Attempt to find Python
set "PYTHON_EXEC=python"

REM Check 1: ComfyUI Portable (../../python_embeded/python.exe)
if exist "..\..\python_embeded\python.exe" (
    set "PYTHON_EXEC=..\..\python_embeded\python.exe"
    echo [INFO] Found ComfyUI Portable Python: !PYTHON_EXEC!
) else if exist "..\..\..\python_embeded\python.exe" (
    set "PYTHON_EXEC=..\..\..\python_embeded\python.exe"
    echo [INFO] Found ComfyUI Portable Python: !PYTHON_EXEC!
) else (
    REM Check 2: System Python
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        echo [INFO] Found System Python.
    ) else (
        echo [ERROR] Python not found in 'python_embeded' or PATH.
        echo Please make sure you are running this from within ComfyUI or have Python installed.
        pause
        exit /b 1
    )
)

REM 1. Install Dependencies
echo [INFO] Installing Python dependencies with "!PYTHON_EXEC!"...
"!PYTHON_EXEC!" -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM 2. Download Models (If possible)
echo.
echo [INFO] Checking for models...
set "MODEL_DIR=..\..\models\latentsync"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%MODEL_DIR%\whisper" mkdir "%MODEL_DIR%\whisper"

echo.
echo IMPORTANT: LatentSync models might require manual download.
echo Models will be placed in: ComfyUI\models\latentsync
echo.

"!PYTHON_EXEC!" -c "from huggingface_hub import snapshot_download; print('Downloading UNet...'); snapshot_download(repo_id='ByteDance/LatentSync', allow_patterns=['latentsync_unet.pt'], local_dir=r'%MODEL_DIR%', local_dir_use_symlinks=False); print('Downloading Whisper...'); snapshot_download(repo_id='ByteDance/LatentSync', allow_patterns=['whisper/tiny.pt'], local_dir=r'%MODEL_DIR%', local_dir_use_symlinks=False)" 2>nul
if !errorlevel! neq 0 (
    echo [WARNING] Automatic download failed. 
    echo Please download 'latentsync_unet.pt' and 'whisper/tiny.pt' manually.
    echo Place them in: !MODEL_DIR!
) else (
    echo [SUCCESS] Models downloaded.
    echo [INFO] The downloaded 'latentsync_unet.pt' supports FP16, FP32, and FP8 via the Node settings.
)

REM 3. Download Mediapipe Weights (Required for new API)
echo.
echo [INFO] Downloading Mediapipe Face Landmarker...
set "WEIGHTS_DIR=latentsync\weights"
if not exist "%WEIGHTS_DIR%" mkdir "%WEIGHTS_DIR%"
"!PYTHON_EXEC!" -c "import urllib.request, os; url='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'; dest=os.path.join(r'%WEIGHTS_DIR%', 'face_landmarker.task'); print(f'Downloading to {dest}...'); urllib.request.urlretrieve(url, dest) if not os.path.exists(dest) else print('File already exists.')"


echo.
echo [INFO] ComfyUI-LipSync-GAP setup complete. Please restart ComfyUI.
echo ===============================================================================
pause
