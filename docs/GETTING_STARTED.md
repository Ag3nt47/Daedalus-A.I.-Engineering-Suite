# Getting started

Daedalus is a local-first Windows workbench for learning and implementing neural
networks from Python and NumPy primitives. The public suite source and your
private projects are deliberately separate. Read the path summary on first
launch before creating or importing data.

## Choose an installation path

### Release bundle

For a packaged release, extract the bundle to a normal user-owned directory and
run `Install-Daedalus.bat`. The installer should create an isolated environment,
install the declared dependencies, initialize the private workspace, and create
shortcuts. Review every proposed path before accepting optional scheduled
backup. An installation is not proof that backup or GitHub authentication is
configured.

### Source checkout

Python 3.11 through 3.14 is supported by the project metadata. From PowerShell
in the source root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m daedalus
```

Do not reuse an environment from another AI project. If the test command fails,
resolve that failure before launching against valuable data.

## Know where data goes

| Content | Default location | Publication policy |
|---|---|---|
| Public suite source | Installation/source directory | Eligible only after Release Guard |
| Learner projects | `%USERPROFILE%\Daedalus Workspaces\projects` | Private; never part of a suite push |
| Datasets | `%USERPROFILE%\Daedalus Workspaces\datasets` | Private by default |
| Checkpoints | `%USERPROFILE%\Daedalus Workspaces\checkpoints` | Private; NPZ arrays plus JSON metadata |
| Run logs | `%USERPROFILE%\Daedalus Workspaces\training-runs` and `logs` | Private by default |
| Backup | `F:\Daedalus-Backups\DaedalusAI` | Recovery copy, not a Git remote |

Override the private roots before launch when needed:

```powershell
$env:DAEDALUS_WORKSPACE_ROOT = "D:\Private\Daedalus Workspaces"
$env:DAEDALUS_BACKUP_ROOT = "F:\Daedalus-Backups\DaedalusAI"
```

The workspace must be outside the public source tree. Daedalus marks owned
workspace and backup roots and refuses unsafe root relationships or an unmarked,
nonempty backup destination.

## First 30 minutes

The left rail is the build order. Each numbered stage has **Tools** for the work
and **Info** for explanations, evidence requirements, and boundaries. **Overview**
and **Settings** support the workflow without being numbered build steps.

1. Open **Overview** and confirm Source, Workspace, and Backup are three
   intentional paths. Then open **1 · Define**, describe a tiny first goal, and use
   its **Setup** tab to audit the generated project baseline.
2. Open **2 · Learn**, select **Launch Pad**, and read glossary cards in
   Plain mode. Math, Code, and Diagnostic modes expose progressively deeper
   detail without changing the underlying definition.
3. Open **4 · Plan** and complete **Shape Detective**. Predict `(4, 3) +
   (3,)`, then try an incompatible pair and read the axis-by-axis explanation.
   Open **Calculator Lab → Weight Lab** to try one bounded generated-weight,
   logic, recurrent, constraint, random-feature, or next-sample example. Its
   **More Info** tab links the mathematical scope to primary research and a live
   YouTube search.
4. Open **5 · Data & Train**, analyze the built-in XOR data, and run a short
   held-out experiment. Inspect its validation/final-test metrics and checkpoint.
5. Create an **XOR** project in **6 · Build**. Projects are created under the
   private workspace, never beneath the suite repository.
6. Run the project's test or starter script. The constrained runner checks paths,
   selected imports, execution time, and output size. It is not a hostile-code
   sandbox; use a disposable virtual machine for unknown code.
7. Open **8 · Protect**, validate the destination, run a backup, and inspect
   its manifest. Follow [BACKUP_RESTORE.md](BACKUP_RESTORE.md) for a restore
   drill before depending on automation.

## Guided AI build workflow

Create a private project from **Overview**, then open **1 · Define**.
Choose Beginner for one explained decision at a time, Builder for a compact
engineering checklist, or Expert for the full evidence matrix. The same canonical
session is used in every mode.

The bot walks through ten gates: discovery, recovery inventory, data readiness,
simple baseline, architecture, experiment, evaluation, deployment, security, and
release. It routes work into Learning Atlas, Architecture Builder, calculators,
Training Lab, Code Workshop, Model Evaluator, Vault, and Release Guard. It does
not call an external AI service or run project code.

The application header keeps the selected project and its evidence-based completion
visible on every step. The percentage changes only when a canonical gate passes or is
explicitly waived; visiting a tab does not manufacture progress. Use **Scan project &
logs** for an immediate read-only syntax/log/run/data/checkpoint report. Leave **Live
watch** enabled to rescan after bounded file-metadata changes. Each project receives a
private `logs` folder automatically, including older projects when they are opened.
Sandbox events record status and timing but never copy stdout or stderr values into the
automatic event log.

The Define stage's **Setup** tab is the provider-neutral bridge to a professional
toolchain. **Audit setup** checks the project manifest, Python requirement, dependency
evidence, smoke tests, cards, deployment/observability notes, source fingerprint, and
detected optional tools without importing project code or using the network. **Initialize
missing standards** creates only absent files and never replaces user content. **Capture
environment** writes a reviewable snapshot. New projects receive this baseline atomically;
completed project training runs add an immutable, path-redacted manifest beneath `runs/`.

In **7 · Evaluate**, choose a validation or final-test split, optionally select a metric
threshold and baseline, then replay the exact recorded split and preprocessing contract.
Daedalus verifies the run and checkpoint checksums before reconstruction and writes each
evaluation as a new immutable report beneath the private project's `reports/` directory.

In **3 · Design**, the **3D Model** tab renders the validated layer chain with sampled
neurons and connections. Drag or use the arrow keys to rotate, use the wheel or `+`/`-`
to zoom, and select a layer for exact dimensions. Detail and animation adapt to the
available display/resources; static software rendering remains available when graphics
acceleration is unavailable. Every step's **Info** tab also offers an explicit YouTube
search pre-filled for that stage.

Use **Generate missing draft plans** to create the project specification,
dataset/model cards, experiment/evaluation plans, threat model, deployment
runbook, and reproducibility record. Existing files are never overwritten, and
draft placeholders do not satisfy evidence gates until you replace them with real
results. Session export is versioned JSON; the local SQLite history can recover
the newest valid committed revision after a crash.

## Publish the public source safely

1. Run `Backup-Now.bat` only after Windows reports the destination volume clean
   and healthy; then run the built-in backup verification.
2. Double-click `Publish-To-GitHub.bat` and complete GitHub's browser sign-in.
3. Accept or change the default public repository name, inspect the candidate,
   and enter a meaningful initial commit message.
4. Confirm the resulting GitHub URL and wait for both **CI** and **Security
   assurance** to complete.
5. In GitHub settings, make successful CI/security checks required on `main` and
   review the enabled Dependabot, secret-scanning, push-protection, and private
   vulnerability-reporting controls.

For later reviewed changes, use `Safe-Push.bat`. Its pinned Gitleaks pass examines
the exact staged blobs before Release Guard commits, and the pre-push hook scans
the immutable outgoing history. Never use force push as a scanner workaround.

## Learning completion

A module is complete only when its artifact, automated checks, written
explanation, and safety gates pass. Reading alone does not unlock a completion
badge. Numeric checkpoints record dtype, tolerances, seed, Python/NumPy versions,
and input identity so “works on my machine” is not mistaken for reproducibility.

The packaged curriculum is in `src/daedalus/resources/`:

- `learning_paths.json` - tracks, modules, labs, and checkpoints.
- `project_recipes.json` - reusable build recipes.
- `glossary.json` - Plain, Math, Code, and Diagnostic explanations.
- `error_cards.json` - deterministic troubleshooting and safe actions.
- `sources.json` - official documentation and primary papers with original
  offline summaries.

## Safety defaults

- Never paste tokens, passwords, private keys, or proprietary data into code,
  logs, screenshots, or issue reports.
- `.gitignore` is defense in depth. It does not remove files already tracked by
  Git and it does not replace inspection of outgoing commits.
- Daedalus checkpoints use non-executable NPZ arrays and validated JSON.
  Untrusted pickle files are refused because loading one can execute code.
- Backup and GitHub publication are independent. A private project can be backed
  up without ever becoming eligible for a suite push.
- Automatic dependency changes belong on a reviewable branch and must pass the
  full test suite. An advisory with no fixed release cannot be “auto-fixed.”

## Fast troubleshooting

| Symptom | First check |
|---|---|
| Broadcast error | Right-align both shapes; each axis pair must match or contain a `1`. |
| Matrix multiplication error | Verify `(..., m, k) @ (..., k, n)`. |
| NaN/Inf loss | Find the first non-finite operation; do not replace NaN with zero. |
| Loss diverges | Pass gradient checks, overfit one fixed batch, then adjust one factor at a time. |
| Path boundary rejection | Confirm the resolved path is beneath the marked private workspace. |
| Ignored file still appears | Check whether Git already tracks it; ignore rules are not retroactive. |
| Backup drive missing | Reconnect the configured drive or change the destination explicitly. Nothing is copied silently elsewhere. |

See [SECURITY_MODEL.md](SECURITY_MODEL.md) for boundaries and
[LEARNING_PATHS.md](LEARNING_PATHS.md) for the full progression. See
[PROFESSIONAL_TOOLCHAIN.md](PROFESSIONAL_TOOLCHAIN.md) for the mapping from Daedalus
stages to common engineering tools.
