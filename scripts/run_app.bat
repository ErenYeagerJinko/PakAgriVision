@echo off
setlocal
cd /d "%~dp0.."
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" "scripts\run_app.py" %*
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "scripts\run_app.py" %*
) else (
  python "scripts\run_app.py" %*
)
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
  echo.
  echo Startup failed. Read the message above to resolve it.
  pause
)
exit /b %APP_EXIT%
