[CmdletBinding(SupportsShouldProcess = $true)]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$BackupScript = Join-Path $RepoRoot 'tools\backup.ps1'
$PowerShellExe = Join-Path `
    $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$TaskName = 'Daedalus AI Suite - Hourly Backup'

if (-not (Test-Path -LiteralPath $BackupScript -PathType Leaf)) {
    throw 'The guarded backup launcher is missing.'
}
if (-not (Test-Path -LiteralPath $PowerShellExe -PathType Leaf)) {
    throw 'Windows PowerShell is unavailable.'
}

# Validate the real configured volume without copying or writing health state.
& $PowerShellExe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
    -File $BackupScript -PreflightOnly
if ($LASTEXITCODE -ne 0) {
    throw 'Backup preflight failed. Repair the configured volume before scheduling backups.'
}

$Action = New-ScheduledTaskAction `
    -Execute $PowerShellExe `
    -Argument (
        '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden ' +
        '-ExecutionPolicy Bypass -File "' + $BackupScript + '" -Scheduled'
    )
$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(5) `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 45)

if ($PSCmdlet.ShouldProcess($TaskName, 'Register guarded hourly backup task')) {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description 'Verified, non-destructive backup after a fail-closed volume preflight.' `
        -Force | Out-Null
    Write-Host '[DONE] Guarded backup is scheduled every hour.' -ForegroundColor Green
}
