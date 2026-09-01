@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Daedalus AI Engineering Suite - Installer
cd /d "%~dp0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\install.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [ERROR] Installation did not complete. Exit code %RC%.
if /I not "%~1"=="-Unattended" pause
exit /b %RC%

