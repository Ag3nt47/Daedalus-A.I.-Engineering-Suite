[CmdletBinding()]
param(
    [string]$Repository,
    [string]$GitHubCli
)

$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ([string]::IsNullOrWhiteSpace($GitHubCli)) {
    $Command = Get-Command gh.exe -ErrorAction SilentlyContinue
    if ($Command) { $GitHubCli = $Command.Source }
    else { $GitHubCli = Join-Path $RepoRoot '.dev-tools\bin\gh.exe' }
}
if (-not (Test-Path -LiteralPath $GitHubCli -PathType Leaf)) {
    throw 'Verified GitHub CLI is unavailable. Run tools\bootstrap-security-tools.ps1.'
}
if ([string]::IsNullOrWhiteSpace($Repository)) {
    $Origin = (& git.exe -C $RepoRoot remote get-url origin 2>$null)
    if ($Origin -notmatch '^https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?$') {
        throw 'Origin must be credential-free HTTPS on github.com.'
    }
    $Repository = $Matches[1]
}
if ($Repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
    throw 'Repository must be OWNER/NAME.'
}

function Invoke-GitHubApi([string]$Method, [string]$Endpoint, [string]$Json = '') {
    $Arguments = @('api', '--method', $Method, '-H', 'Accept: application/vnd.github+json', '-H', 'X-GitHub-Api-Version: 2026-03-10', $Endpoint)
    if ($Json) {
        $Json | & $GitHubCli @Arguments --input - | Out-Null
    } else {
        & $GitHubCli @Arguments | Out-Null
    }
    if ($LASTEXITCODE -ne 0) { throw "GitHub security configuration failed for $Endpoint" }
}

& $GitHubCli auth status --hostname github.com | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'GitHub CLI is not authenticated.' }

Invoke-GitHubApi 'PUT' "repos/$Repository/vulnerability-alerts"
Invoke-GitHubApi 'PUT' "repos/$Repository/automated-security-fixes"
Invoke-GitHubApi 'PUT' "repos/$Repository/private-vulnerability-reporting"
$SecurityPayload = @{
    security_and_analysis = @{
        secret_scanning = @{status = 'enabled'}
        secret_scanning_push_protection = @{status = 'enabled'}
    }
} | ConvertTo-Json -Depth 5 -Compress
Invoke-GitHubApi 'PATCH' "repos/$Repository" $SecurityPayload

$State = & $GitHubCli repo view $Repository --json url,visibility,defaultBranchRef --jq '{url: .url, visibility: .visibility, default_branch: .defaultBranchRef.name}'
if ($LASTEXITCODE -ne 0) { throw 'GitHub repository state could not be verified.' }
Write-Host '[DONE] Dependabot alerts/updates, private vulnerability reporting, secret scanning, and push protection are enabled.' -ForegroundColor Green
Write-Host $State
