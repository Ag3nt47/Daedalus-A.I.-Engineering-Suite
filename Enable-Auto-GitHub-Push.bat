@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Daedalus - Enable Guarded Auto Push
cd /d "%~dp0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\configure-auto-push.ps1"
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%

