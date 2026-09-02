@echo off
REM Starts the receipt engine -- both services plus a public tunnel -- in separate windows.
REM
REM   window 1: Surya OCR      (SuryaOCR venv, GPU by default)  127.0.0.1:8001, private
REM   window 2: extraction API (this venv, GPU)                 0.0.0.0:8000
REM   window 3: public HTTPS tunnel (Cloudflare) -> prints the URL to share
REM
REM Surya and this project pin incompatible transformers versions, so they cannot share one
REM interpreter -- hence two venvs and two processes talking over localhost.
REM
REM OCR runs on the GPU by default: ~3.4s/receipt vs ~20s on CPU. Measured with both
REM models resident, the pair uses ~4.8GB of the 12GB card, so they fit comfortably.
REM   set OCR_DEVICE=cpu  before running this to free VRAM at the cost of latency.
REM
REM For the phone-facing HTML page instead of the API, use run_page.bat (it binds a
REM different port, but loads a SECOND copy of the model -- don't run both routinely).

setlocal
if "%OCR_DEVICE%"=="" set OCR_DEVICE=cuda

REM Paths derive from this file's own location (%~dp0 ends with a backslash), so the
REM pair of project folders can be moved without editing this script.
set ROOT=%~dp0
set SURYA=%ROOT%..\SuryaOCR

REM The shared secret clients send as X-API-Key. Kept in .env (gitignored) rather than
REM here, so this file stays safe to commit. Regenerate with:
REM   .venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
if not exist "%ROOT%.env" (
    echo ERROR: %ROOT%.env is missing -- it must define ENGINE_API_KEY.
    exit /b 1
)
for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ROOT%.env") do set "%%A=%%B"
if "%ENGINE_API_KEY%"=="" (
    echo ERROR: ENGINE_API_KEY is empty. The tunnel makes this service public -- never
    echo run it unauthenticated. Put a key in %ROOT%.env
    exit /b 1
)

echo Starting Surya OCR service on %OCR_DEVICE% ...
start "Surya OCR (%OCR_DEVICE%)" cmd /k "cd /d %SURYA% && set TORCH_DEVICE=%OCR_DEVICE%&& .venv\Scripts\python.exe %ROOT%ocr_service.py"

echo Waiting for the OCR service to load ...
timeout /t 12 /nobreak >nul

REM All configuration is environment driven -- no host paths inside the code -- so the
REM same app/ runs here and anywhere else it is ever deployed.
echo Starting extraction API on GPU ...
start "Receipt extraction API (GPU)" cmd /k "cd /d %ROOT% && set ENGINE_CHECKPOINT=%ROOT%checkpoints\qwen3.5-2b-qlora\checkpoint-550&& set OCR_URL=http://127.0.0.1:8001/ocr&& set ENGINE_API_KEY=%ENGINE_API_KEY%&& .venv\Scripts\python.exe -m uvicorn app.api:app --host 0.0.0.0 --port 8000"

REM Only port 8000 is exposed. The Surya service on 8001 stays bound to 127.0.0.1 and is
REM never tunneled -- the outside world cannot reach it at all.
REM
REM The URL is newly generated on every run, so read it from this window each time. The
REM tunnel comes up before the model finishes loading; requests in that gap return an
REM error until window 2 logs "warm-up generation done".
echo Starting public tunnel ...
start "Public tunnel (Cloudflare)" cmd /k cloudflared tunnel --url http://localhost:8000

echo.
echo All three windows are starting.
echo   - window 2 is ready when it logs "warm-up generation done"
echo   - window 3 prints the public https://...trycloudflare.com URL to share
echo.
echo Send your frontend dev BOTH of these:
echo   URL:     the https://...trycloudflare.com address from window 3
echo   API key: the ENGINE_API_KEY value in .env
echo.
echo Check readiness yourself first:  curl http://localhost:8000/ready
echo Closing a window stops that service.
endlocal
