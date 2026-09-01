@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Daedalus - Release Guard and GitHub Push
cd /d "%~dp0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\safe-push.ps1"
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%

