[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run Install-Daedalus.bat first.' }
$ToolBin = Join-Path $RepoRoot '.dev-tools\bin'
$GhPath = Join-Path $ToolBin 'gh.exe'
if (-not (Test-Path -LiteralPath $GhPath -PathType Leaf)) {
    $SystemGh = Get-Command gh.exe -ErrorAction SilentlyContinue
    if ($SystemGh) { $GhPath = $SystemGh.Source }
    else {
        & (Join-Path $PSScriptRoot 'bootstrap-security-tools.ps1')
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $GhPath -PathType Leaf)) {
            throw 'Verified GitHub CLI could not be prepared.'
        }
    }
}
$env:Path = "$(Split-Path -Parent $GhPath);$env:Path"

& $Python -m daedalus.services.release_guard --repo $RepoRoot init
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $GhPath auth status --hostname github.com
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Sign in to GitHub. Daedalus does not read or store your token.'
    & $GhPath auth login --hostname github.com --git-protocol https --web
    if ($LASTEXITCODE -ne 0) { throw 'GitHub authentication did not complete.' }
}

$Existing = (& git.exe -C $RepoRoot remote get-url origin 2>$null)
if (-not $Existing) {
    $DefaultName = 'daedalus-ai-engineering-suite'
    $Name = Read-Host "Public repository name [$DefaultName]"
    if ([string]::IsNullOrWhiteSpace($Name)) { $Name = $DefaultName }
    if ($Name -notmatch '^[A-Za-z0-9._-]+$') { throw 'Repository name contains unsupported characters.' }
    & $GhPath repo create $Name --public --source $RepoRoot --remote origin --disable-wiki --description 'Build and learn neural networks from NumPy primitives in a local-first desktop suite.'
    if ($LASTEXITCODE -ne 0) { throw 'GitHub repository creation failed.' }
}
$Origin = (& git.exe -C $RepoRoot remote get-url origin 2>$null)
if ($Origin -notmatch '^https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?$') {
    throw 'Origin must be credential-free HTTPS on github.com.'
}
$Repository = $Matches[1]
$RemoteHeads = @(& git.exe -C $RepoRoot ls-remote --heads origin 2>$null)
if ($LASTEXITCODE -ne 0) { throw 'The GitHub origin could not be inspected.' }
$InitialPublish = $RemoteHeads.Count -eq 0

$Message = Read-Host 'Initial commit message [feat: launch Daedalus AI Engineering Suite]'
if ([string]::IsNullOrWhiteSpace($Message)) { $Message = 'feat: launch Daedalus AI Engineering Suite' }
$GuardArguments = @('-m', 'daedalus.services.release_guard', '--repo', $RepoRoot, 'push', '--message', $Message)
if ($InitialPublish) { $GuardArguments += '--initial-publish' }
& $Python @GuardArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& (Join-Path $PSScriptRoot 'configure-github-security.ps1') -Repository $Repository -GitHubCli $GhPath
if ($LASTEXITCODE -ne 0) { throw 'The source was pushed, but GitHub security settings were not fully configured.' }
$Url = & $GhPath repo view $Repository --json url --jq '.url'
Write-Host "[DONE] Published and verified: $Url" -ForegroundColor Green

$Enable = Read-Host 'Enable unattended guarded push every four hours? Manual Safe Push is recommended. [y/N]'
if ($Enable -match '^(?i)y(?:es)?$') {
    & (Join-Path $RepoRoot 'tools\configure-auto-push.ps1')
}
