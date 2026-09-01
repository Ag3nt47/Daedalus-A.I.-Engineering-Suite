[CmdletBinding()]
param([string]$Confirm)

$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot '.git'))) {
    throw 'Initialize and publish the repository before enabling automatic push.'
}
$Origin = (& git.exe -C $RepoRoot remote get-url origin 2>$null)
if (-not $Origin -or $Origin -notmatch '^https://github\.com/[^/]+/[^/]+(?:\.git)?$') {
    throw 'A credential-free HTTPS GitHub origin is required.'
}
$Branch = (& git.exe -C $RepoRoot branch --show-current).Trim()
if ($Branch -cne 'main') { throw 'Automatic push can be enabled only from the main branch.' }
$Pending = @(& git.exe -C $RepoRoot status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0 -or $Pending.Count -ne 0) {
    throw 'Review and publish or discard every pending change before enabling automatic push.'
}
$ToolBin = Join-Path $RepoRoot '.dev-tools\bin'
if (-not (Test-Path -LiteralPath (Join-Path $ToolBin 'gh.exe') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $ToolBin 'gitleaks.exe') -PathType Leaf)) {
    & (Join-Path $PSScriptRoot 'bootstrap-security-tools.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Verified GitHub security tools could not be prepared.' }
}
& (Join-Path $ToolBin 'gh.exe') auth status --hostname github.com | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Authenticate GitHub CLI before enabling automatic push.' }
& git.exe -C $RepoRoot fetch --no-tags origin main
if ($LASTEXITCODE -ne 0) { throw 'Origin main could not be fetched.' }
$LocalHead = (& git.exe -C $RepoRoot rev-parse HEAD).Trim()
$RemoteHead = (& git.exe -C $RepoRoot rev-parse refs/remotes/origin/main).Trim()
if ($LocalHead -cne $RemoteHead) {
    throw 'Local main must exactly match origin/main before automatic push is enabled.'
}

$ConfirmationPhrase = 'ENABLE DAEDALUS AUTO PUSH'
if ([string]::IsNullOrWhiteSpace($Confirm)) {
    Write-Warning 'This opt-in task can stage and publish every non-ignored source change on main.'
    $Confirm = Read-Host "Type $ConfirmationPhrase to continue"
}
if ($Confirm -cne $ConfirmationPhrase) { throw 'Automatic push was not enabled.' }

$PowerShellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$Script = Join-Path $RepoRoot 'tools\auto-push.ps1'
$ActionArguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden ' +
    '-ExecutionPolicy Bypass -File "' + $Script + '" ' +
    '-ExpectedBranch main -ExpectedOrigin "' + $Origin + '"'
$Action = New-ScheduledTaskAction -Execute $PowerShellExe -Argument $ActionArguments
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(10) -RepetitionInterval (New-TimeSpan -Hours 4) -RepetitionDuration (New-TimeSpan -Days 3650)
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10) -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName 'Daedalus AI Suite - Guarded Auto Push' -Action $Action -Trigger $Trigger -Settings $Settings -Description 'Opt-in main-only push after Gitleaks, privacy, test, and advisory gates pass.' -Force | Out-Null
Write-Host '[DONE] Guarded automatic push is scheduled every four hours.' -ForegroundColor Green
