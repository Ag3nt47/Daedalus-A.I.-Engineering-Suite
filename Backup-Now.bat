@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Daedalus - Safe Backup
cd /d "%~dp0"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\backup.ps1" %*
set "RC=%ERRORLEVEL%"
if /I not "%~1"=="-Scheduled" pause
exit /b %RC%

