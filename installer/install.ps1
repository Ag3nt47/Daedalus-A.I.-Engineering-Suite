[CmdletBinding()]
param(
    [switch]$Unattended,
    [switch]$NoLaunch,
    [switch]$SkipInitialBackup,
    [switch]$SkipDependencyInstall,
    [switch]$SkipToolInstall,
    [switch]$InstalledCopy
)

$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Desktop = [Environment]::GetFolderPath('Desktop')
$PreferredRoot = [IO.Path]::GetFullPath((Join-Path $Desktop 'Daedalus A.I. Engineering Suite'))

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

if (-not $InstalledCopy -and $RepoRoot.TrimEnd('\') -ne $PreferredRoot.TrimEnd('\')) {
    # Publish a complete staged tree with a recoverable directory swap.  Copying
    # directly over the live install can mix versions, retain obsolete files, or
    # leave an unusable half-update after a crash.
    $InstallId = [Guid]::NewGuid().ToString('N')
    $StagingRoot = [IO.Path]::GetFullPath((Join-Path $Desktop ".Daedalus-install-$InstallId"))
    $RollbackRoot = [IO.Path]::GetFullPath((Join-Path $Desktop ("Daedalus A.I. Engineering Suite.rollback-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))))
    $FailedRoot = [IO.Path]::GetFullPath((Join-Path $Desktop ("Daedalus A.I. Engineering Suite.failed-$InstallId")))
    foreach ($Candidate in @($StagingRoot, $RollbackRoot, $FailedRoot, $PreferredRoot)) {
        if ([IO.Directory]::GetParent($Candidate).FullName.TrimEnd('\') -ne $Desktop.TrimEnd('\')) {
            throw "Installer path escaped the Desktop boundary: $Candidate"
        }
    }
    if ((Test-Path -LiteralPath $StagingRoot) -or (Test-Path -LiteralPath $RollbackRoot) -or (Test-Path -LiteralPath $FailedRoot)) {
        throw 'A unique staging, rollback, or failed-install path unexpectedly already exists.'
    }

    Write-Step "Staging the complete source tree at $StagingRoot"
    New-Item -ItemType Directory -Path $StagingRoot | Out-Null
    $Robocopy = Join-Path $env:SystemRoot 'System32\robocopy.exe'
    & $Robocopy $RepoRoot $StagingRoot /E /XJ /R:1 /W:1 /XD '.git' '.venv' 'venv' '__pycache__' 'build' 'dist' /XF '*.pyc' '*.tmp' | Out-Host
    $CopyExitCode = $LASTEXITCODE
    if ($CopyExitCode -ge 8) {
        throw "Could not stage Daedalus on the Desktop (Robocopy $CopyExitCode). The staging directory was preserved for diagnosis."
    }
    foreach ($RequiredRelative in @('pyproject.toml', 'installer\install.ps1', 'src\daedalus\__main__.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $StagingRoot $RequiredRelative) -PathType Leaf)) {
            throw "Staged install is incomplete: $RequiredRelative is missing."
        }
    }

    $PreviousRoot = $null
    if (Test-Path -LiteralPath $PreferredRoot) {
        Write-Step "Moving the previous installation to $RollbackRoot"
        Move-Item -LiteralPath $PreferredRoot -Destination $RollbackRoot
        $PreviousRoot = $RollbackRoot
    }
    try {
        Move-Item -LiteralPath $StagingRoot -Destination $PreferredRoot
    } catch {
        if ($PreviousRoot -and -not (Test-Path -LiteralPath $PreferredRoot)) {
            Move-Item -LiteralPath $PreviousRoot -Destination $PreferredRoot
        }
        throw
    }

    $TargetInstaller = Join-Path $PreferredRoot 'installer\install.ps1'
    $Arguments = @('-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $TargetInstaller, '-InstalledCopy')
    if ($Unattended) { $Arguments += '-Unattended' }
    if ($NoLaunch) { $Arguments += '-NoLaunch' }
    if ($SkipInitialBackup) { $Arguments += '-SkipInitialBackup' }
    if ($SkipDependencyInstall) { $Arguments += '-SkipDependencyInstall' }
    if ($SkipToolInstall) { $Arguments += '-SkipToolInstall' }
    & (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') @Arguments
    $InstallExitCode = $LASTEXITCODE
    if ($InstallExitCode -ne 0) {
        Write-Warning 'Installation failed; isolating the failed tree.'
        if (Test-Path -LiteralPath $PreferredRoot) {
            Move-Item -LiteralPath $PreferredRoot -Destination $FailedRoot
        }
        if ($PreviousRoot) {
            Write-Warning "Restoring the prior complete tree from $PreviousRoot"
            Move-Item -LiteralPath $PreviousRoot -Destination $PreferredRoot
        }
    }
    if ($InstallExitCode -eq 0 -and $PreviousRoot) {
        Write-Host "Rollback copy preserved at $PreviousRoot" -ForegroundColor Yellow
    }
    exit $InstallExitCode
}

function Find-CompatiblePython {
    $Launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($Launcher) {
        # Missing launcher versions write to the native error stream. Probe
        # every supported version without allowing that expected miss to abort
        # an installer running under ErrorActionPreference=Stop.
        $OriginalPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            foreach ($Version in @('3.13', '3.12', '3.11')) {
                $Candidate = (& $Launcher.Source "-$Version" -c 'import sys; print(sys.executable)' 2>$null)
                $ProbeExitCode = $LASTEXITCODE
                if ($ProbeExitCode -eq 0 -and $Candidate) { return $Candidate.Trim() }
            }
        } finally {
            $ErrorActionPreference = $OriginalPreference
        }
    }
    $PathPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($PathPython) {
        $OriginalPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $Candidate = (& $PathPython.Source -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 15); print(sys.executable)' 2>$null)
            $ProbeExitCode = $LASTEXITCODE
            if ($ProbeExitCode -eq 0 -and $Candidate) { return $Candidate.Trim() }
        } finally {
            $ErrorActionPreference = $OriginalPreference
        }
    }
    return $null
}

Write-Host @'
  ____                 _       _
 |  _ \  __ _  ___  __| | __ _| |_   _ ___
 | | | |/ _` |/ _ \/ _` |/ _` | | | | / __|
 | |_| | (_| |  __/ (_| | (_| | | |_| \__ \
 |____/ \__,_|\___|\__,_|\__,_|_|\__,_|___/
      A.I. ENGINEERING SUITE
'@ -ForegroundColor Cyan

$Python = Find-CompatiblePython
if (-not $Python) {
    Write-Step 'Installing Python 3.12 through Windows Package Manager'
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $Winget) { throw 'Python 3.11+ is required and winget is unavailable.' }
    & $Winget.Source install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
    $Python = Find-CompatiblePython
    if (-not $Python) { throw 'Python installed, but this shell cannot locate it. Restart Windows and rerun the installer.' }
}
Write-Host "Python: $Python"

if (-not (Get-Command git.exe -ErrorAction SilentlyContinue) -and -not $SkipToolInstall) {
    Write-Step 'Installing Git for Windows'
    & winget.exe install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw 'Git installation failed.' }
    $env:Path += ';C:\Program Files\Git\cmd'
}

if (-not (Get-Command gh.exe -ErrorAction SilentlyContinue) -and -not $SkipToolInstall) {
    Write-Step 'Installing GitHub CLI for optional Dependabot alert access'
    & winget.exe install --id GitHub.cli -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -eq 0) { $env:Path += ';C:\Program Files\GitHub CLI' }
    else { Write-Warning 'GitHub CLI was not installed. Local advisory scanning still works.' }
}

$Venv = Join-Path $RepoRoot '.venv'
$VenvPython = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Step 'Creating the isolated Daedalus environment'
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}

if (-not $SkipDependencyInstall) {
    Write-Step 'Installing Daedalus and all dependencies'
    & $VenvPython -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'Python packaging tools could not be installed.' }
    & $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $RepoRoot 'requirements-dev.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Daedalus dependencies could not be installed.' }
    & $VenvPython -m pip install --disable-pip-version-check --no-deps -e $RepoRoot
    if ($LASTEXITCODE -ne 0) { throw 'Daedalus dependencies could not be installed.' }
} else {
    Write-Step 'Using the pre-provisioned local dependency environment'
    & $VenvPython -c 'import daedalus, numpy, PySide6, psutil; print(daedalus.__version__)'
    if ($LASTEXITCODE -ne 0) { throw 'The pre-provisioned dependency environment is incomplete.' }
}

Write-Step 'Bootstrapping the private user workspace'
& $VenvPython -c 'from daedalus.workspace.manager import WorkspaceManager; m=WorkspaceManager.from_environment(); m.bootstrap(); print(m.workspace_root)'
if ($LASTEXITCODE -ne 0) { throw 'Private workspace initialization failed.' }

Write-Step 'Running installation self-tests'
& $VenvPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Daedalus tests failed; shortcuts and automation were not installed.' }

Write-Step 'Initializing guarded source control'
& $VenvPython -m daedalus.services.release_guard --repo $RepoRoot init
if ($LASTEXITCODE -ne 0) { throw 'Git initialization failed.' }
& git.exe -C $RepoRoot config core.hooksPath .githooks

Write-Step 'Preparing the branded Windows launcher'
$Launcher = Join-Path $RepoRoot 'Daedalus.exe'
if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
    $BuildLauncher = Join-Path $RepoRoot 'tools\build-launcher.ps1'
    & (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') -NoLogo -NoProfile -ExecutionPolicy Bypass -File $BuildLauncher
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
        throw 'The branded Daedalus launcher could not be built.'
    }
}

Write-Step 'Creating Desktop shortcuts'
$Shell = New-Object -ComObject WScript.Shell
$Shortcuts = @(
    @{Name='Daedalus AI Engineering Suite.lnk'; Target='Daedalus.exe'; Description='Build and learn AI from NumPy primitives'; Icon="$Launcher,0"},
    @{Name='Daedalus - Backup Now.lnk'; Target='Backup-Now.bat'; Description='Back up Daedalus source and private workspaces'; Icon="$env:SystemRoot\System32\imageres.dll,68"},
    @{Name='Daedalus - Safe Push.lnk'; Target='Safe-Push.bat'; Description='Scan, test, commit, and push safely'; Icon="$env:SystemRoot\System32\imageres.dll,68"}
)
foreach ($Definition in $Shortcuts) {
    $Shortcut = $Shell.CreateShortcut((Join-Path $Desktop $Definition.Name))
    $Shortcut.TargetPath = Join-Path $RepoRoot $Definition.Target
    $Shortcut.WorkingDirectory = $RepoRoot
    $Shortcut.Description = $Definition.Description
    $Shortcut.IconLocation = $Definition.Icon
    $Shortcut.Save()
}

Write-Step 'Registering hourly non-destructive backup'
try {
    $PowerShellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $BackupScript = Join-Path $RepoRoot 'tools\backup.ps1'
    $TaskAction = New-ScheduledTaskAction -Execute $PowerShellExe -Argument "-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$BackupScript`" -Scheduled"
    $TaskTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 3650)
    $TaskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Minutes 45)
    Register-ScheduledTask -TaskName 'Daedalus AI Suite - Hourly Backup' -Action $TaskAction -Trigger $TaskTrigger -Settings $TaskSettings -Description 'Non-destructive backup of Daedalus source and private workspaces.' -Force | Out-Null
} catch {
    Write-Warning "Automatic backup task could not be registered: $($_.Exception.Message)"
}

if (-not $SkipInitialBackup) {
    Write-Step 'Creating the initial guarded backup'
    $InitialBackupLauncher = Join-Path $RepoRoot 'tools\backup.ps1'
    $InitialPowerShell = Join-Path `
        $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    & $InitialPowerShell -NoLogo -NoProfile -NonInteractive `
        -ExecutionPolicy Bypass -File $InitialBackupLauncher -Scheduled
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Initial backup preflight or copy failed safely; the application will show details.'
    }
}

Write-Host "`n[DONE] Daedalus is installed at $RepoRoot" -ForegroundColor Green
Write-Host 'Use Publish-To-GitHub.bat once to create/connect the public repository.'
if (-not $NoLaunch) {
    & (Join-Path $RepoRoot 'Run-Daedalus.bat')
}
