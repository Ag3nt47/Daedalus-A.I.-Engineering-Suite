[CmdletBinding()]
param(
    [switch]$Scheduled,
    [Alias('ValidateOnly')]
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$DefaultBackupRoot = 'F:\Daedalus-Backups\DaedalusAI'
$MinimumFreeBytes = [uint64](5GB)

function Get-BackupHealthPath {
    $ConfiguredRoot = [Environment]::GetEnvironmentVariable('DAEDALUS_WORKSPACE_ROOT')
    if ([string]::IsNullOrWhiteSpace($ConfiguredRoot)) {
        $UserProfilePath = [Environment]::GetFolderPath('UserProfile')
        $WorkspaceRoot = Join-Path $UserProfilePath 'Daedalus Workspaces'
    } else {
        $WorkspaceRoot = [Environment]::ExpandEnvironmentVariables($ConfiguredRoot)
    }
    $WorkspaceRoot = [IO.Path]::GetFullPath($WorkspaceRoot)
    return Join-Path (Join-Path $WorkspaceRoot '.daedalus') 'backup-health.json'
}

function Write-BackupFailure(
    [string]$FailureCode,
    [int]$ExitCode,
    [datetime]$AttemptStartedUtc
) {
    try {
        $HealthPath = Get-BackupHealthPath
        $Existing = $null
        if (Test-Path -LiteralPath $HealthPath -PathType Leaf) {
            try {
                $Existing = Get-Content -LiteralPath $HealthPath -Raw | ConvertFrom-Json
                $ExistingUpdated = [datetime]::Parse($Existing.updated_utc).ToUniversalTime()
                if ($Existing.state -eq 'failed' -and $ExistingUpdated -ge $AttemptStartedUtc) {
                    return
                }
            } catch {
                $Existing = $null
            }
        }

        $Now = [datetime]::UtcNow.ToString('o')
        $Payload = [ordered]@{
            kind = 'daedalus-backup-health'
            schema = 1
            state = 'failed'
            updated_utc = $Now
            last_attempt_utc = $Now
            last_success_utc = if ($Existing) { $Existing.last_success_utc } else { $null }
            last_failure_utc = $Now
            failure_code = $FailureCode
            last_manifest = if ($Existing) { $Existing.last_manifest } else { $null }
            scheduled = [bool]$Scheduled
            last_exit_code = $ExitCode
        }
        $HealthDirectory = Split-Path -Parent $HealthPath
        New-Item -ItemType Directory -Path $HealthDirectory -Force | Out-Null
        $Temporary = Join-Path $HealthDirectory ('.backup-health.' + $PID + '.tmp')
        $Utf8 = New-Object System.Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($Temporary, ($Payload | ConvertTo-Json -Depth 5), $Utf8)
        Move-Item -LiteralPath $Temporary -Destination $HealthPath -Force
    } catch {
        # The scheduled command must retain its original exit code even if the
        # local health record cannot be written.
    }
}

function Get-NormalizedBackupDestination {
    $ConfiguredRoot = [Environment]::GetEnvironmentVariable('DAEDALUS_BACKUP_ROOT')
    if ([string]::IsNullOrWhiteSpace($ConfiguredRoot)) {
        $ConfiguredRoot = $DefaultBackupRoot
    } else {
        $ConfiguredRoot = [Environment]::ExpandEnvironmentVariables($ConfiguredRoot)
    }

    # The health and dirty-bit probes below are reliable only for an absolute
    # local Windows volume. Never reinterpret relative, UNC, or device syntax.
    if ($ConfiguredRoot -notmatch '^[A-Za-z]:[\\/]') {
        throw [System.ArgumentException]::new('Backup destination is not an absolute local path.')
    }
    $RawPath = $ConfiguredRoot.TrimEnd('\', '/')
    if ($RawPath.Length -le 3) {
        throw [System.ArgumentException]::new('A volume root cannot be a backup destination.')
    }
    $RawTail = $RawPath.Substring(3).Replace('/', '\')
    $ReservedNames = '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$'
    foreach ($Component in $RawTail.Split('\')) {
        if (
            [string]::IsNullOrWhiteSpace($Component) -or
            $Component -in @('.', '..') -or
            $Component.EndsWith(' ') -or
            $Component.EndsWith('.') -or
            $Component -match $ReservedNames -or
            $Component.IndexOfAny([char[]]'<>:"/|?*') -ge 0
        ) {
            throw [System.ArgumentException]::new('Backup destination contains an unsafe component.')
        }
    }

    try {
        $Destination = [IO.Path]::GetFullPath($RawPath)
        $DriveRoot = [IO.Path]::GetPathRoot($Destination)
    } catch {
        throw [System.ArgumentException]::new('Backup destination could not be normalized.')
    }
    if (
        [string]::IsNullOrWhiteSpace($DriveRoot) -or
        $DriveRoot -notmatch '^[A-Za-z]:\\$' -or
        [string]::Equals($Destination, $DriveRoot, [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw [System.ArgumentException]::new('Backup destination does not identify a safe child path.')
    }
    if ((Test-Path -LiteralPath $Destination) -and
        -not (Test-Path -LiteralPath $Destination -PathType Container)) {
        throw [System.ArgumentException]::new('Backup destination is not a directory.')
    }

    # Reject junctions and symbolic links in every existing ancestor so an F:
    # path cannot silently redirect writes to a different volume. A missing
    # drive is classified by the volume probe instead of being mistaken for a
    # malformed destination.
    if (Test-Path -LiteralPath $DriveRoot -PathType Container) {
        $Cursor = $Destination
        while (-not (Test-Path -LiteralPath $Cursor)) {
            $Parent = [IO.Directory]::GetParent($Cursor)
            if ($null -eq $Parent) {
                throw [System.ArgumentException]::new(
                    'Backup destination has no existing ancestor.'
                )
            }
            $Cursor = $Parent.FullName
        }
        while (-not [string]::IsNullOrWhiteSpace($Cursor)) {
            $Item = Get-Item -Force -LiteralPath $Cursor
            if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw [System.ArgumentException]::new(
                    'Backup destination has a reparse ancestor.'
                )
            }
            $Parent = [IO.Directory]::GetParent($Cursor)
            if ($null -eq $Parent) {
                break
            }
            $Cursor = $Parent.FullName
        }
    }

    return [pscustomobject]@{
        Destination = $Destination
        DriveRoot = $DriveRoot
        DriveLetter = $DriveRoot.Substring(0, 1).ToUpperInvariant()
    }
}

function Get-BackupVolumeProbe {
    param([Parameter(Mandatory = $true)][psobject]$Boundary)

    $DrivePresent = Test-Path -LiteralPath $Boundary.DriveRoot -PathType Container
    if (-not $DrivePresent) {
        return [pscustomobject]@{
            DrivePresent = $false
            MetadataResolved = $false
            DirtyKnown = $false
            Dirty = $false
            StorageResolved = $false
            HealthStatus = ''
            OperationalStatuses = @()
            FreeBytesKnown = $false
            FreeBytes = [uint64]0
        }
    }

    $DriveDesignator = $Boundary.DriveLetter + ':'
    try {
        $Volumes = @(
            Get-CimInstance -ClassName Win32_Volume `
                -Filter "DriveLetter = '$DriveDesignator'" -ErrorAction Stop
        )
    } catch {
        $Volumes = @()
    }
    if ($Volumes.Count -ne 1) {
        return [pscustomobject]@{
            DrivePresent = $true
            MetadataResolved = $false
            DirtyKnown = $false
            Dirty = $false
            StorageResolved = $false
            HealthStatus = ''
            OperationalStatuses = @()
            FreeBytesKnown = $false
            FreeBytes = [uint64]0
        }
    }
    $Volume = $Volumes[0]
    $DirtyKnown = $null -ne $Volume.DirtyBitSet
    $FreeBytesKnown = $null -ne $Volume.FreeSpace
    $FreeBytes = if ($FreeBytesKnown) { [uint64]$Volume.FreeSpace } else { [uint64]0 }

    try {
        $StorageVolumes = @(Get-Volume -DriveLetter $Boundary.DriveLetter -ErrorAction Stop)
    } catch {
        $StorageVolumes = @()
    }
    $StorageResolved = $StorageVolumes.Count -eq 1
    $HealthStatus = ''
    $OperationalStatuses = @()
    if ($StorageResolved) {
        $StorageVolume = $StorageVolumes[0]
        $HealthStatus = [string]$StorageVolume.HealthStatus
        $OperationalStatuses = @(
            $StorageVolume.OperationalStatus | ForEach-Object { [string]$_ }
        )
        if ($null -ne $StorageVolume.SizeRemaining) {
            $StorageFreeBytes = [uint64]$StorageVolume.SizeRemaining
            if (-not $FreeBytesKnown -or $StorageFreeBytes -lt $FreeBytes) {
                $FreeBytes = $StorageFreeBytes
            }
            $FreeBytesKnown = $true
        }
    }

    return [pscustomobject]@{
        DrivePresent = $true
        MetadataResolved = $true
        DirtyKnown = $DirtyKnown
        Dirty = if ($DirtyKnown) { [bool]$Volume.DirtyBitSet } else { $false }
        StorageResolved = $StorageResolved
        HealthStatus = $HealthStatus
        OperationalStatuses = $OperationalStatuses
        FreeBytesKnown = $FreeBytesKnown
        FreeBytes = $FreeBytes
    }
}

function Get-BackupPreflightFailureCode {
    param(
        [Parameter(Mandatory = $true)][psobject]$Probe,
        [Parameter(Mandatory = $true)][uint64]$RequiredFreeBytes
    )

    if (-not [bool]$Probe.DrivePresent) {
        return 'backup-volume-missing'
    }
    if (-not [bool]$Probe.MetadataResolved -or -not [bool]$Probe.DirtyKnown) {
        return 'backup-volume-probe-failed'
    }
    if ([bool]$Probe.Dirty) {
        return 'backup-volume-dirty'
    }
    if (-not [bool]$Probe.StorageResolved) {
        return 'backup-volume-probe-failed'
    }
    if (-not [string]::Equals(
            [string]$Probe.HealthStatus,
            'Healthy',
            [StringComparison]::OrdinalIgnoreCase
        )) {
        return 'backup-volume-unhealthy'
    }
    $Statuses = @($Probe.OperationalStatuses)
    if ($Statuses.Count -ne 1 -or
        -not [string]::Equals(
            [string]$Statuses[0],
            'OK',
            [StringComparison]::OrdinalIgnoreCase
        )) {
        return 'backup-volume-not-operational'
    }
    if (-not [bool]$Probe.FreeBytesKnown) {
        return 'backup-volume-probe-failed'
    }
    if ([uint64]$Probe.FreeBytes -lt $RequiredFreeBytes) {
        return 'backup-free-space-low'
    }
    return $null
}

function Get-PreflightExitCode {
    param([Parameter(Mandatory = $true)][string]$FailureCode)

    switch ($FailureCode) {
        'backup-destination-invalid' { return 20 }
        'backup-volume-missing' { return 21 }
        'backup-volume-dirty' { return 22 }
        'backup-volume-unhealthy' { return 23 }
        'backup-volume-not-operational' { return 24 }
        'backup-free-space-low' { return 25 }
        default { return 26 }
    }
}

function Invoke-BackupPreflight {
    try {
        $Boundary = Get-NormalizedBackupDestination
        $Probe = Get-BackupVolumeProbe -Boundary $Boundary
        $FailureCode = Get-BackupPreflightFailureCode `
            -Probe $Probe -RequiredFreeBytes $MinimumFreeBytes
    } catch {
        $Boundary = $null
        $Probe = $null
        $FailureCode = 'backup-destination-invalid'
    }

    if ($null -ne $FailureCode) {
        return [pscustomobject]@{
            Ok = $false
            FailureCode = $FailureCode
            ExitCode = Get-PreflightExitCode -FailureCode $FailureCode
            DriveLetter = if ($null -ne $Boundary) { $Boundary.DriveLetter } else { $null }
            FreeBytes = if ($null -ne $Probe -and $Probe.FreeBytesKnown) {
                [uint64]$Probe.FreeBytes
            } else {
                $null
            }
        }
    }
    return [pscustomobject]@{
        Ok = $true
        FailureCode = $null
        ExitCode = 0
        DriveLetter = $Boundary.DriveLetter
        FreeBytes = [uint64]$Probe.FreeBytes
    }
}

# Dot-sourcing exposes only the pure preflight helpers for focused tests. No
# probe, filesystem write, or backup is performed in that mode.
if ($MyInvocation.InvocationName -eq '.') {
    return
}

$AttemptStartedUtc = [datetime]::UtcNow
$Preflight = Invoke-BackupPreflight
if (-not $Preflight.Ok) {
    $Message = "Backup preflight failed ($($Preflight.FailureCode)). Nothing was copied."
    if ($PreflightOnly) {
        [Console]::Error.WriteLine($Message)
        exit $Preflight.ExitCode
    }
    Write-BackupFailure `
        -FailureCode $Preflight.FailureCode `
        -ExitCode $Preflight.ExitCode `
        -AttemptStartedUtc $AttemptStartedUtc
    if (-not $Scheduled) {
        [Console]::Error.WriteLine($Message)
    }
    exit $Preflight.ExitCode
}

if ($PreflightOnly) {
    [ordered]@{
        kind = 'daedalus-backup-preflight'
        schema = 1
        state = 'ready'
        drive = $Preflight.DriveLetter
        free_bytes = $Preflight.FreeBytes
        required_free_bytes = $MinimumFreeBytes
    } | ConvertTo-Json
    exit 0
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-BackupFailure `
        -FailureCode 'python-runtime-missing' `
        -ExitCode 4 `
        -AttemptStartedUtc $AttemptStartedUtc
    if ($Scheduled) { exit 4 }
    throw 'Daedalus is not installed. Run Install-Daedalus.bat first.'
}
try {
    & $Python -m daedalus.services.backup
    $BackupExitCode = $LASTEXITCODE
    if ($BackupExitCode -ne 0) {
        Write-BackupFailure `
            -FailureCode 'backup-command-failed' `
            -ExitCode $BackupExitCode `
            -AttemptStartedUtc $AttemptStartedUtc
        if (-not $Scheduled) {
            [Console]::Error.WriteLine(
                'Backup failed safely. Run BackupService status for details.'
            )
        }
    }
    exit $BackupExitCode
} catch {
    Write-BackupFailure `
        -FailureCode 'backup-script-exception' `
        -ExitCode 2 `
        -AttemptStartedUtc $AttemptStartedUtc
    if (-not $Scheduled) {
        [Console]::Error.WriteLine('Backup launcher failed safely.')
    }
    exit 2
}
