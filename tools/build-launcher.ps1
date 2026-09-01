[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$SourcePath = [IO.Path]::GetFullPath((Join-Path $RepoRoot 'tools\launcher\DaedalusLauncher.cs'))
$IconPath = [IO.Path]::GetFullPath((Join-Path $RepoRoot 'src\daedalus\assets\daedalus.ico'))
$OutputPath = [IO.Path]::GetFullPath((Join-Path $RepoRoot 'Daedalus.exe'))

foreach ($Required in @($SourcePath, $IconPath)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required launcher input is missing: $Required"
    }
}
if ([IO.Directory]::GetParent($OutputPath).FullName.TrimEnd('\') -ne $RepoRoot.TrimEnd('\')) {
    throw "Launcher output escaped the repository root: $OutputPath"
}

$CompilerCandidates = @(
    (Join-Path $env:SystemRoot 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'),
    (Join-Path $env:SystemRoot 'Microsoft.NET\Framework\v4.0.30319\csc.exe')
)
$Compiler = $CompilerCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $Compiler) {
    $CompilerCommand = Get-Command csc.exe -ErrorAction SilentlyContinue
    if ($CompilerCommand) { $Compiler = $CompilerCommand.Source }
}
if (-not $Compiler) {
    throw 'The Windows C# compiler was not found. Enable .NET Framework 4.x and retry.'
}

$TemporaryOutput = Join-Path $RepoRoot ('.Daedalus-launcher-' + [Guid]::NewGuid().ToString('N') + '.exe')
try {
    $CompilerArguments = @(
        '/nologo',
        '/target:winexe',
        '/optimize+',
        '/platform:anycpu',
        "/reference:System.Windows.Forms.dll",
        "/win32icon:$IconPath",
        "/out:$TemporaryOutput",
        $SourcePath
    )
    & $Compiler @CompilerArguments
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $TemporaryOutput -PathType Leaf)) {
        throw "C# launcher compilation failed with exit code $LASTEXITCODE."
    }
    Move-Item -LiteralPath $TemporaryOutput -Destination $OutputPath -Force
} finally {
    if (Test-Path -LiteralPath $TemporaryOutput -PathType Leaf) {
        Remove-Item -LiteralPath $TemporaryOutput -Force
    }
}

$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutputPath).Hash
Write-Host "Built $OutputPath" -ForegroundColor Green
Write-Host "SHA-256 $Hash"
