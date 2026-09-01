[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ExpectedBranch,
    [Parameter(Mandatory = $true)][string]$ExpectedOrigin
)

$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$HealthPath = Join-Path $RepoRoot '.git\daedalus-auto-push-health.json'
$Mutex = New-Object Threading.Mutex($false, 'Local\DaedalusGuardedAutoPush')
$HasMutex = $false

function Write-AutoPushHealth([string]$State, [string]$Code, [int]$ExitCode) {
    try {
        $Now = [datetime]::UtcNow.ToString('o')
        $Payload = [ordered]@{
            kind = 'daedalus-auto-push-health'
            schema = 1
            state = $State
            updated_utc = $Now
            code = $Code
            exit_code = $ExitCode
            branch = $ExpectedBranch
            origin = $ExpectedOrigin
        }
        $Temporary = "$HealthPath.$PID.tmp"
        $Utf8 = New-Object Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($Temporary, ($Payload | ConvertTo-Json -Depth 4), $Utf8)
        Move-Item -LiteralPath $Temporary -Destination $HealthPath -Force
    } catch {
        # Retain the actual publication exit code if local health evidence fails.
    }
}

try {
    try { $HasMutex = $Mutex.WaitOne(0) }
    catch [Threading.AbandonedMutexException] { $HasMutex = $true }
    if (-not $HasMutex) {
        Write-AutoPushHealth 'failed' 'already-running' 10
        exit 10
    }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'python-runtime-missing' }
    if ($ExpectedBranch -cne 'main' -or
        $ExpectedOrigin -notmatch '^https://github\.com/[^/]+/[^/]+(?:\.git)?$') {
        throw 'scheduled-identity-invalid'
    }
    $ActualOrigin = (& git.exe -C $RepoRoot remote get-url origin 2>$null)
    $ActualBranch = (& git.exe -C $RepoRoot branch --show-current).Trim()
    if ($ActualOrigin -cne $ExpectedOrigin -or $ActualBranch -cne $ExpectedBranch) {
        throw 'scheduled-identity-changed'
    }
    $ToolBin = Join-Path $RepoRoot '.dev-tools\bin'
    $Gitleaks = Join-Path $ToolBin 'gitleaks.exe'
    $GitHubCli = Join-Path $ToolBin 'gh.exe'
    if (-not (Test-Path -LiteralPath $Gitleaks -PathType Leaf) -or
        -not (Test-Path -LiteralPath $GitHubCli -PathType Leaf)) {
        throw 'verified-security-tools-missing'
    }
    $env:Path = "$ToolBin;$env:Path"
    & $GitHubCli auth status --hostname github.com | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'github-authentication-unavailable' }
    & git.exe -C $RepoRoot fetch --no-tags origin main
    if ($LASTEXITCODE -ne 0) { throw 'origin-fetch-failed' }
    $Counts = (& git.exe -C $RepoRoot rev-list --left-right --count 'origin/main...HEAD').Trim().Split()
    if ($LASTEXITCODE -ne 0 -or $Counts.Count -ne 2) { throw 'origin-divergence-unknown' }
    if ([int]$Counts[0] -ne 0) { throw 'origin-is-ahead-or-diverged' }

    & git.exe -C $RepoRoot add -A
    if ($LASTEXITCODE -ne 0) { throw 'staging-failed' }
    $StagedPaths = @(& git.exe -C $RepoRoot diff --cached --name-only --diff-filter=ACMR)
    if ($StagedPaths.Count -gt 0) {
        & $Gitleaks git --staged --redact --no-banner --timeout 120 $RepoRoot
        if ($LASTEXITCODE -ne 0) { throw 'gitleaks-blocked' }
    }
    $Stamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm UTC')
    & $Python -m daedalus.services.release_guard --repo $RepoRoot push --message "chore: guarded automatic sync $Stamp"
    $Result = $LASTEXITCODE
    if ($Result -ne 0) {
        Write-AutoPushHealth 'failed' 'release-guard-blocked' $Result
        exit $Result
    }
    Write-AutoPushHealth 'healthy' 'push-complete-or-current' 0
    exit 0
} catch {
    Write-AutoPushHealth 'failed' ([string]$_.Exception.Message) 11
    exit 11
} finally {
    if ($HasMutex) { $Mutex.ReleaseMutex() }
    $Mutex.Dispose()
}
