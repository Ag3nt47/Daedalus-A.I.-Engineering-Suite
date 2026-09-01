[CmdletBinding()]
param([switch]$ForceDownload)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$ToolRoot = Join-Path $RepoRoot '.dev-tools'
$DownloadRoot = Join-Path $ToolRoot 'downloads'
$SourceRoot = Join-Path $ToolRoot 'sources'
$BinRoot = Join-Path $ToolRoot 'bin'
$LockPath = Join-Path $PSScriptRoot 'security-tools.lock.json'

function Assert-ToolPath([string]$Path) {
    $Root = [IO.Path]::GetFullPath($ToolRoot).TrimEnd('\')
    $Candidate = [IO.Path]::GetFullPath($Path)
    if (-not ($Candidate.Equals($Root, [StringComparison]::OrdinalIgnoreCase) -or
        $Candidate.StartsWith($Root + '\', [StringComparison]::OrdinalIgnoreCase))) {
        throw "Refusing a development-tool write outside $ToolRoot"
    }
}

function Get-VerifiedAsset($Asset) {
    $Archive = Join-Path $DownloadRoot ([string]$Asset.asset)
    $Partial = "$Archive.partial"
    $ExtractRoot = Join-Path $SourceRoot ("{0}-{1}" -f $Asset.id, $Asset.version)
    foreach ($Path in @($Archive, $Partial, $ExtractRoot)) { Assert-ToolPath $Path }

    $DownloadNeeded = $ForceDownload -or -not (Test-Path -LiteralPath $Archive -PathType Leaf)
    if (-not $DownloadNeeded) {
        $Actual = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
        $DownloadNeeded = $Actual -cne ([string]$Asset.sha256).ToLowerInvariant()
    }
    if ($DownloadNeeded) {
        if (Test-Path -LiteralPath $Partial) { Remove-Item -LiteralPath $Partial -Force }
        Write-Host "Downloading pinned $($Asset.id) $($Asset.version)..."
        Invoke-WebRequest -UseBasicParsing -Uri ([string]$Asset.url) -OutFile $Partial
        $Actual = (Get-FileHash -LiteralPath $Partial -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -cne ([string]$Asset.sha256).ToLowerInvariant()) {
            Remove-Item -LiteralPath $Partial -Force
            throw "SHA-256 mismatch for $($Asset.asset)."
        }
        Move-Item -LiteralPath $Partial -Destination $Archive -Force
    }
    $Verified = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Verified -cne ([string]$Asset.sha256).ToLowerInvariant()) {
        throw "Cached $($Asset.asset) failed its pinned SHA-256 check."
    }

    New-Item -ItemType Directory -Path $ExtractRoot -Force | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $ExtractRoot -Force
    $Matches = @(Get-ChildItem -LiteralPath $ExtractRoot -File -Recurse -Filter ([string]$Asset.executable))
    if ($Matches.Count -ne 1) {
        throw "Expected one $($Asset.executable) in $($Asset.asset); found $($Matches.Count)."
    }
    if ($Asset.executable_sha256) {
        $ExecutableHash = (Get-FileHash -LiteralPath $Matches[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ExecutableHash -cne ([string]$Asset.executable_sha256).ToLowerInvariant()) {
            throw "Extracted $($Asset.executable) failed its pinned SHA-256 check."
        }
    }
    $Installed = Join-Path $BinRoot ([string]$Asset.executable)
    Assert-ToolPath $Installed
    Copy-Item -LiteralPath $Matches[0].FullName -Destination $Installed -Force
    if ($Asset.executable_sha256) {
        $InstalledHash = (Get-FileHash -LiteralPath $Installed -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($InstalledHash -cne ([string]$Asset.executable_sha256).ToLowerInvariant()) {
            throw "Installed $($Asset.executable) failed its pinned SHA-256 check."
        }
    }
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'The pinned Daedalus security toolkit currently supports 64-bit Windows only.'
}
$Manifest = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
if ([int]$Manifest.schema_version -ne 1) { throw 'Unsupported security-tool lock schema.' }
foreach ($Directory in @($ToolRoot, $DownloadRoot, $SourceRoot, $BinRoot)) {
    Assert-ToolPath $Directory
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}
foreach ($Asset in $Manifest.assets) { Get-VerifiedAsset $Asset }

Write-Host 'Verified Daedalus GitHub security tools:' -ForegroundColor Green
& (Join-Path $BinRoot 'gh.exe') --version | Select-Object -First 1
& (Join-Path $BinRoot 'gitleaks.exe') version
Write-Host "Location: $ToolRoot (development-only and ignored by Git)"
