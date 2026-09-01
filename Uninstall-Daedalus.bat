@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Daedalus AI Engineering Suite - Uninstall
cd /d "%~dp0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\uninstall.ps1"
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%

