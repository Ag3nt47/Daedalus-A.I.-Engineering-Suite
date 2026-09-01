# Backup and restore

Daedalus backup is independent from Git and GitHub. It protects both the public
suite source and the external private workspace without making private projects
eligible for publication.

## Defaults and layout

The default destination is:

```text
F:\Daedalus-Backups\DaedalusAI
```

Override it explicitly before launch when required:

```powershell
$env:DAEDALUS_BACKUP_ROOT = "F:\Daedalus-Backups\DaedalusAI"
```

A valid root contains:

```text
.daedalus-backup-root.json
source-current\
workspace-current\
objects\sha256\<first-two-hex>\<sha256>
manifests\backup-<UTC timestamp>.json
latest.json
```

The marker identifies a directory Daedalus is allowed to manage. A nonempty
destination without that marker is refused. Source, workspace, and backup roots
must be separate; none may contain another.

## What a backup does

- Bootstraps and validates the private workspace.
- Refuses a missing configured Windows drive instead of redirecting elsewhere.
- Acquires a lock so two backup jobs do not run concurrently.
- Reclaims a structured dead-process lock. A malformed lock is preserved unless
  it is at least one day old and predates the current OS boot; proven crash
  debris is atomically renamed to a quarantine file, never silently deleted.
- Copies source into `source-current` and private workspace into
  `workspace-current` as convenient current mirrors.
- Skips environment/build/cache directories only at the public source root;
  identically named directories inside a project are private data and are kept.
  Symbolic links and path escapes are skipped.
- Copies through a temporary filename before replacement.
- Compares every regular file by SHA-256, regardless of timestamps or size.
- Snapshots live SQLite databases with SQLite's online-backup API and omits
  a WAL/SHM/journal file only when it belongs to a verified SQLite database;
  ordinary names such as `notes-journal` remain normal backup files.
- Publishes every captured file once beneath its immutable SHA-256 object path.
- Writes a schema-3 manifest containing each logical path, size, kind, SHA-256,
  and canonical object path. `latest.json` is atomically updated only after a
  complete error-free run; a failed run keeps its timestamped diagnostic
  manifest without replacing the last restorable pointer.
- Records sanitized local and destination health evidence for failed jobs.

The copy is intentionally non-destructive: if a local file is deleted, the next
run does not immediately delete the destination-only recovery copy. This is not
unlimited versioning. Modified files at the same relative path can replace the
current mirror, and storage grows until an operator reviews retention. Exact
restore does not copy the mirror wholesale: it reconstructs only the files named
by the selected verified manifest, so stale destination-only files are not
resurrected.

The content-addressed object store, not either mutable `*-current` tree, is the
recovery authority. A crash while a newer mirror is being refreshed cannot
invalidate objects referenced by the previously committed `latest.json`.

Backup currently does not encrypt data, upload to a cloud service, or follow
links. SQLite databases receive consistent snapshots; arbitrary non-database
files can still change while being read, so stable-hash checks refuse an unstable
copy. Use trusted encrypted storage or drive-level encryption when confidentiality
requires it.

## Fail-closed volume preflight

`Backup-Now.bat`, the hourly scheduled task, and the installer's initial backup
all enter through `tools\backup.ps1`. Before Python can bootstrap or copy data,
the launcher:

- normalizes the configured destination as a child of an absolute local Windows
  volume and rejects relative, UNC, device, volume-root, unsafe-component, and
  reparse-ancestor paths;
- confirms the volume is attached and that Windows can resolve both its
  `Win32_Volume` and Storage metadata;
- refuses a set or unknown filesystem dirty bit;
- requires `HealthStatus=Healthy` and the single operational status `OK`; and
- requires at least 5 GiB free as a minimum safety reserve.

Production managers created from the environment enforce that same read-only
launcher gate at the beginning of `BackupService.run()`. This closes direct
Python, `backup_once()`, main-window, and **Vault & Backup** write paths as well;
the gate runs before workspace bootstrap, lock creation, ownership markers, or
destination health records. A failed direct/GUI gate records only sanitized
health evidence in the local workspace and never updates the unsafe volume.

There is no scheduled or interactive unsafe override. A normal failed attempt
writes a sanitized local health code such as `backup-volume-missing`,
`backup-volume-dirty`, `backup-volume-unhealthy`,
`backup-volume-not-operational`, `backup-free-space-low`, or
`backup-volume-probe-failed`, but it does not write to the backup volume.

Run the read-only preflight without changing backup or health state:

```powershell
.\tools\backup.ps1 -PreflightOnly
```

The 5 GiB reserve is a floor, not a prediction of a large first backup. Confirm
that free capacity also covers the private datasets and checkpoints you expect
to retain.

If Windows reports that `F:` is dirty or needs repair, do not force a backup.
Close applications using the drive and run this from an elevated terminal:

```powershell
chkdsk F: /f
fsutil dirty query F:
.\tools\backup.ps1 -PreflightOnly
```

Honor any dismount or restart instructions from `chkdsk`. Resume or re-register
automation only after the final preflight succeeds.

## Run a manual backup

Preferred: open **Vault & Backup**, inspect all three roots, select **Validate**,
then **Run backup**. Review copied counts, skipped links, errors, and the manifest
path.

From an installed development environment, use the same guarded launcher as the
scheduled task:

```powershell
.\tools\backup.ps1
```

Treat any listed error as an incomplete backup. Do not mark the run successful
merely because some files copied.

Verify the complete latest inventory at any time:

```powershell
.\.venv\Scripts\python.exe -m daedalus.services.backup --verify
```

Verification reads and hashes the marked recovery set but never writes to the
backup volume. Its success or failure evidence is recorded only in the private
local workspace, so verification remains safe when the destination is mounted
read-only or Windows has flagged it for repair.

The command returns failure for an empty/malformed inventory, invalid ownership
marker, missing or changed object, unsafe/noncanonical path, count mismatch, or
any error recorded by the manifest. A zero-file verification is never success.

## First restore drill

Run this before depending on scheduled backup:

1. Create a small private project containing representative source, metadata,
   and a non-sensitive checkpoint fixture.
2. Record expected file paths and SHA-256 hashes.
3. Run backup and review `latest.json`.
4. Restore to a **new** directory; never target the active workspace.
5. Compare expected paths, sizes, and hashes.
6. Open the restored project read-only first, then run its tests in an isolated
   environment.
7. Record the drill date, backup manifest, result, and any exclusions.

The service re-verifies the selected manifest and its objects before creating the
destination, then restores only its `workspace-current/*` logical entries into a
timestamped sibling directory by default:

```powershell
.\.venv\Scripts\python.exe -c "from daedalus.services.backup import BackupService; from daedalus.workspace.manager import WorkspaceManager; m=WorkspaceManager.from_environment(); print(BackupService(m).restore_workspace())"
```

An existing destination is refused. After verification, change
`DAEDALUS_WORKSPACE_ROOT` to the restored path deliberately; do not rename or
delete the damaged workspace until recovery is accepted.

## Verification guidance

For a single file in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "D:\Private\example.npz"
```

For a full drill, first run the built-in `--verify` command, then compare the
restored tree against the schema-3 inventory by logical relative path, size, and
SHA-256.
The restore is not accepted when expected files are missing, hashes differ,
metadata is invalid, or the project cannot pass its offline smoke tests.

## Scheduled backup

A release installer may offer a Windows scheduled task. Before enabling it:

- confirm the task runs as the intended user without stored broad credentials;
- confirm `F:` uses a stable drive assignment and is normally attached at that
  time;
- run the exact command manually;
- ensure task history and Daedalus manifests reveal failures;
- choose a schedule that does not overlap large active writes;
- keep restore drills on a separate calendar.

An absent drive should produce a visible failure. Repeated failures must not
silently fall back to the system drive.

After the volume passes preflight, create or refresh the guarded hourly task with
the root `Enable-Auto-Backup.bat` launcher, or non-interactively with:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File .\tools\configure-backup.ps1
```

The registration helper itself runs `-PreflightOnly` first. It replaces the task
with a one-hour trigger, ignores overlapping instances, retries transient task
failures up to three times, starts missed runs when available, and limits one run
to 45 minutes.

## Recovery scenarios

### Accidental local deletion

Stop automatic jobs until the destination-only copy is located. Restore to a new
path and copy back only after hash and content review.

### Corrupted active workspace

Stop training and editor writes. Preserve the damaged tree for diagnosis. Restore
the last verified backup beside it, compare, test, then switch the configured
workspace root.

### Lost or replaced backup drive

Do not create a marker inside an unfamiliar nonempty directory. Configure a new
empty trusted root, run a full backup, and complete a restore drill.

### Source release recovery

Use the backup copy only as one source of evidence. Also verify tagged source and
published artifact hashes. Never restore private workspace content into the
public repository.

## Retention and deletion

Automatic deletion is intentionally absent from the initial service. Review disk
use and define a retention policy before introducing pruning. A future pruning
operation must resolve and display exact targets, preserve at least one verified
restore point, avoid active/current trees, and require explicit confirmation.
Backup is complete only when recovery has been tested.
