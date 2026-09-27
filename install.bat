@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
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

REM Verify existing assets or download to temporary files before atomic publication.
"!PYTHON_EXEC!" asset_integrity.py --comfy-root "..\.." --download
if !errorlevel! neq 0 (
    echo [ERROR] Asset verification failed. Setup is incomplete. Existing corrupt files are preserved.
    pause
    exit /b 1
)
echo [INFO] Dependencies and pinned assets installed. Run preflight.py with your selected VAE before inference.
pause
