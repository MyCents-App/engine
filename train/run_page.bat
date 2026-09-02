@echo off
REM Local-only: the phone-facing HTML page (demo_server.py), for eyeballing the pipeline
REM over Wi-Fi. The deployed engine is the API in run_demo.bat -- this is a dev tool.
REM
REM Binds 8080 so it can run alongside the API on 8000, but note it loads its OWN copy of
REM the model into VRAM. Fine for a look; don't leave both running.
REM
REM Requires the Surya OCR service to already be up (start run_demo.bat first, or run
REM ocr_service.py by hand).

setlocal
set ROOT=%~dp0
start "Receipt phone page (dev)" cmd /k "cd /d %ROOT% && .venv\Scripts\python.exe demo_server.py --port 8080"
echo Phone page starting on port 8080 -- it prints the LAN address to open on a phone.
endlocal
