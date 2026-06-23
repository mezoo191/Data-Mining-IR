@echo off
REM ============================================================
REM  News Search Engine - one-shot setup + launch (Windows)
REM
REM  Usage:
REM    run.bat            BEST MODE (default): sample dataset WITH BERT
REM    run.bat lite       Sample dataset, no BERT (fastest)
REM    run.bat full       Full ~210k dataset WITH BERT (best at scale, slow build)
REM    run.bat full lite  Full dataset, no BERT
REM
REM  Best mode (BERT) is always the default. Pass "lite" (or "nobert") to opt out.
REM  The browser opens automatically once the server is ready.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Best mode = BERT enabled. It stays ON by default and is only turned off when
REM the user explicitly opts out with "lite" / "nobert".
set USE_BERT=1
set USE_FULL=0
for %%A in (%*) do (
  if /I "%%A"=="lite" set USE_BERT=0
  if /I "%%A"=="nobert" set USE_BERT=0
  if /I "%%A"=="bert" set USE_BERT=1
  if /I "%%A"=="full" set USE_FULL=1
)

REM --- 0. Prerequisites --------------------------------------
where python >nul 2>nul || (echo [error] Python not found. Install Python 3.9+ from python.org & goto :error)
where npm >nul 2>nul
if errorlevel 1 (
  echo [error] Node.js / npm not found - the web UI cannot be built without it.
  echo         Install Node.js LTS from https://nodejs.org/ then run this again.
  goto :error
)

REM --- 1. Stop any previous server still holding port 8000 ---
REM Match the local listening socket precisely (":8000 " with a trailing space)
REM so we don't kill unrelated ports like :58000 or remote :8000 connections.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 " ^| findstr LISTENING') do (
  echo [setup] Stopping previous server on port 8000 ^(PID %%P^)...
  taskkill /F /PID %%P >nul 2>nul
)

REM --- 2. Python virtual environment -------------------------
if not exist .venv (
  echo [setup] Creating virtual environment...
  python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat

REM --- 3. Python dependencies --------------------------------
echo [setup] Installing Python dependencies...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt || goto :error
if "%USE_BERT%"=="1" (
  echo [setup] Installing BERT dependencies ^(large download, one time^)...
  pip install -q -r requirements-bert.txt || goto :error
)

REM --- 4. Choose dataset -------------------------------------
set DATA_ARG=
if "%USE_FULL%"=="1" (
  set DATASET=
  if exist data\News_Category_Dataset_v3.json set DATASET=data\News_Category_Dataset_v3.json
  if exist News_Category_Dataset_v3.json set DATASET=News_Category_Dataset_v3.json
  if "!DATASET!"=="" (
    echo [setup] Full dataset not found - downloading automatically...
    python scripts\download_data.py || goto :error
    if exist data\News_Category_Dataset_v3.json set DATASET=data\News_Category_Dataset_v3.json
  )
  if "!DATASET!"=="" (
    echo [error] Full dataset still missing after download.
    goto :error
  )
  echo [setup] Using full dataset: !DATASET!
  set DATA_ARG=--data !DATASET!
)

REM --- 5. Rebuild the index when the mode/dataset changes ----
set SIG=sample
if "%USE_FULL%"=="1" set SIG=full
if "%USE_BERT%"=="1" set SIG=!SIG!+bert
set PREVSIG=
if exist artifacts\build.info set /p PREVSIG=<artifacts\build.info
if not "!SIG!"=="!PREVSIG!" (
  if not "!PREVSIG!"=="" echo [setup] Build config changed ^("!PREVSIG!" -^> "!SIG!"^); rebuilding...
  if exist artifacts\index.pkl del /q artifacts\index.pkl
  if exist artifacts\bert.pkl del /q artifacts\bert.pkl
)

REM --- 6. Build the search index (only if missing) ----------
if "%USE_BERT%"=="1" (
  if not exist artifacts\bert.pkl (
    echo [setup] Building index + BERT embeddings...
    python scripts\build_index.py !DATA_ARG! --bert || goto :error
  )
  if not exist artifacts\bert.pkl (
    echo [error] BERT embeddings were not created. Check the build output above.
    goto :error
  )
) else (
  if not exist artifacts\index.pkl (
    echo [setup] Building search index...
    python scripts\build_index.py !DATA_ARG! || goto :error
  )
)
> artifacts\build.info echo !SIG!

REM --- 7. Build the frontend (install once, always rebuild) --
pushd frontend
if not exist node_modules (
  echo [setup] Installing frontend packages...
  call npm install || (popd & goto :error)
)
echo [setup] Building frontend...
call npm run build || (popd & goto :error)
popd
if not exist frontend\dist\index.html (
  echo [error] Frontend build did not produce frontend\dist\index.html
  goto :error
)

REM --- 8. Open the browser once the server is actually up ----
start "" /b powershell -NoProfile -Command ^
  "for($i=0;$i -lt 90;$i++){try{Invoke-WebRequest 'http://localhost:8000/api/health' -UseBasicParsing -TimeoutSec 1 ^| Out-Null; Start-Process 'http://localhost:8000'; break}catch{Start-Sleep -Milliseconds 700}}"

REM --- 9. Launch --------------------------------------------
echo.
echo [run] Starting server at http://localhost:8000  (mode: !SIG!)  (Ctrl+C to stop)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir .
goto :eof

:error
echo.
echo [FAILED] Setup did not complete. See the message above.
pause
exit /b 1
