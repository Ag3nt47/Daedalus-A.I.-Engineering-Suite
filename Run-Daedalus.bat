@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if exist "%~dp0Daedalus.exe" (
  start "" "%~dp0Daedalus.exe"
  exit /b 0
)
set "PYTHON=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYTHON%" (
  echo Daedalus is not installed yet. Starting the installer...
  call "%~dp0Install-Daedalus.bat"
  if errorlevel 1 exit /b %ERRORLEVEL%
)
start "Daedalus AI Engineering Suite" "%PYTHON%" -m daedalus
exit /b 0
