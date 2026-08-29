@echo off
REM Starts the two-process receipt demo in separate windows.
REM
REM   window 1: Surya OCR   (SuryaOCR venv, CPU by default)
REM   window 2: extraction  (this venv, GPU) + the phone web page
REM
REM Surya and this project pin incompatible transformers versions, so they cannot share one
REM interpreter -- hence two venvs and two processes talking over localhost.
REM
REM To put OCR on the GPU instead (~3.5s vs ~20s per receipt, costs ~3.6GB VRAM):
REM   set OCR_DEVICE=cuda  before running this, or edit the default below.

setlocal
if "%OCR_DEVICE%"=="" set OCR_DEVICE=cpu

echo Starting Surya OCR service on %OCR_DEVICE% ...
start "Surya OCR (%OCR_DEVICE%)" cmd /k "cd /d D:\Documents\SuryaOCR && set TORCH_DEVICE=%OCR_DEVICE% && .venv\Scripts\python.exe D:\Documents\train\ocr_service.py"

echo Waiting for the OCR service to load ...
timeout /t 12 /nobreak >nul

echo Starting extraction server on GPU ...
start "Receipt extraction (GPU)" cmd /k "cd /d D:\Documents\train && .venv\Scripts\python.exe demo_server.py"

echo.
echo Both windows are starting. The extraction window prints the URL to open on your phone.
echo Close either window to stop that service.
endlocal
