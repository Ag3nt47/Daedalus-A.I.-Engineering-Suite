[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Daedalus is not installed. Run Install-Daedalus.bat first.'
}
$Bootstrap = Join-Path $PSScriptRoot 'bootstrap-security-tools.ps1'
$ToolBin = Join-Path $RepoRoot '.dev-tools\bin'
$Gitleaks = Join-Path $ToolBin 'gitleaks.exe'
$GitHubCli = Join-Path $ToolBin 'gh.exe'
if (-not (Test-Path -LiteralPath $Gitleaks -PathType Leaf) -or
    -not (Test-Path -LiteralPath $GitHubCli -PathType Leaf)) {
    & $Bootstrap
    if ($LASTEXITCODE -ne 0) { throw 'Verified GitHub security tools could not be prepared.' }
}
$env:Path = "$ToolBin;$env:Path"

Write-Host 'Daedalus will scan exact staged blobs with Gitleaks, then run Release Guard'
Write-Host 'privacy, size, syntax, Ruff, tests, PyPI, and GitHub advisory gates.'
Write-Host ''
& git.exe -C $RepoRoot status --short
$Message = Read-Host 'Commit message (blank cancels)'
if ([string]::IsNullOrWhiteSpace($Message)) {
    Write-Host 'Cancelled. Nothing was staged, committed, or pushed.'
    exit 0
}
& git.exe -C $RepoRoot add -A
if ($LASTEXITCODE -ne 0) { throw 'Git could not stage the publish candidate.' }

$StagedPaths = @(& git.exe -C $RepoRoot diff --cached --name-only --diff-filter=ACMR)
if ($LASTEXITCODE -ne 0) { throw 'Git could not enumerate staged files.' }
if ($StagedPaths.Count -gt 0) {
    Write-Host "Scanning $($StagedPaths.Count) exact staged blobs with Gitleaks..."
    & $Gitleaks git --staged --redact --no-banner --timeout 120 $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Gitleaks blocked the staged candidate. Remove and rotate any exposed credential.'
    }
}
& $Python -m daedalus.services.release_guard --repo $RepoRoot push --message $Message
exit $LASTEXITCODE
