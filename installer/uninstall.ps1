[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Desktop = [Environment]::GetFolderPath('Desktop')

Write-Host 'This removes Daedalus dependencies, shortcuts, and scheduled tasks.' -ForegroundColor Yellow
Write-Host 'Your source files, private workspace, Git history, and F: backups are preserved.'
$Answer = Read-Host 'Continue? [y/N]'
if ($Answer -notmatch '^(?i)y(?:es)?$') { exit 0 }

foreach ($Task in @('Daedalus AI Suite - Hourly Backup', 'Daedalus AI Suite - Guarded Auto Push')) {
    Unregister-ScheduledTask -TaskName $Task -Confirm:$false -ErrorAction SilentlyContinue
}
foreach ($Name in @('Daedalus AI Engineering Suite.lnk', 'Daedalus - Backup Now.lnk', 'Daedalus - Safe Push.lnk')) {
    $Path = Join-Path $Desktop $Name
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Force }
}
$Venv = [IO.Path]::GetFullPath((Join-Path $RepoRoot '.venv'))
if ($Venv.StartsWith($RepoRoot, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $Venv)) {
    Remove-Item -LiteralPath $Venv -Recurse -Force
}
Write-Host '[DONE] Runtime components removed. Data and source were preserved.' -ForegroundColor Green

