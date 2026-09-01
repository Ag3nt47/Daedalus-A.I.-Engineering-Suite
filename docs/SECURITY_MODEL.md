# Security model

Daedalus is an educational engineering suite with defensive controls for local
projects and public releases. It is not a malware-analysis platform, container,
virtual machine, or kernel security boundary. Unknown or hostile code and model
files belong in a disposable VM with no valuable credentials or data.

## Security objectives

- Keep learner projects, datasets, checkpoints, logs, and credentials out of the
  public source repository and outgoing pushes.
- Prevent ordinary path mistakes from reading or writing outside the configured
  private workspace.
- Refuse executable checkpoint formats and detect tampering before array load.
- Limit damage from accidental code execution with process, path, import,
  environment, output, and time controls.
- Report release and dependency risks with redacted evidence and fail closed at
  defined thresholds.
- Maintain a separate, restorable backup without treating GitHub as private
  backup storage.

## Assets and trust levels

| Asset | Default trust | Notes |
|---|---|---|
| Installed Daedalus source and signed/verified release artifacts | Trusted after verification | Still subject to dependency and update risk |
| User-authored local learning code | Cooperative but fallible | Eligible for constrained runner only beneath private projects |
| User datasets and logs | Private | May contain personal, proprietary, or regulated information |
| Daedalus NPZ + JSON checkpoint created locally | Trusted after checksum/schema validation | Trust does not transfer automatically between people or machines |
| External code, pickle, model, dataset, plugin, or archive | Untrusted | Inspect in a disposable VM or convert through a trusted producer |
| GitHub, package indexes, and remote APIs | External service | Network results require authentication, permissions, and freshness checks |

## Trust boundaries and controls

### Public source vs private workspace

The source root and workspace root may not equal or contain one another. Paths
are resolved before containment checks. The workspace has an ownership marker;
user content lives under dedicated projects, datasets, checkpoints, runs, and
logs directories.

`.gitignore` blocks common private artifacts as defense in depth, but ignore
rules affect untracked files only. Release Guard must inspect tracked and outgoing
content. If a secret was committed, adding an ignore rule or deleting the newest
copy does not remove historical exposure.

### Code Workshop runner

Current controls are intended for ordinary learner mistakes:

- only Python files beneath the private `projects` directory are eligible;
- source is parsed and selected imports/dynamic-code calls are flagged;
- Python starts in isolated mode with a controlled working directory;
- a minimal environment is passed and output is captured and bounded;
- execution has a configurable timeout.

Limitations are explicit: a Python process shares the host kernel and user
account; static checks can be bypassed; native extensions may escape Python-level
policy; Windows process spawning does not guarantee network denial; filesystem
permissions remain those of the current user. Approval to run restricted imports
does not make code safe. Use a disposable VM for hostile or unknown content.

Weight Lab uses the same runner for its per-tool project drafts. Opening a tool
only creates an in-memory starter preview. The first filesystem mutation is the
explicit exclusive **Create private draft** action under
`projects/<project>/experiments/weight_lab`; an existing draft is never silently
overwritten. Built-in numerical tools cannot be replaced from that directory.

### Checkpoints and serialization

Daedalus saves numeric arrays to NPZ and metadata to JSON. Object arrays are
forbidden. Load uses `allow_pickle=False` and validates the metadata schema,
archive checksum, ordered keys, parameter count, shapes, and dtypes before
copying values into an existing model.

Pickle is refused for untrusted checkpoints because deserialization can execute
code. Do not load a pickle merely to inspect it. Ask a trusted producer to export
named numeric arrays plus non-executable metadata.

### Backup destination

The backup root must be separate from source and workspace. A nonempty directory
without a Daedalus marker is refused to prevent copying into or later operating
on someone else's files. Symlinks are skipped and restore targets a new path.
Backup does not encrypt private data; use trusted encrypted storage or drive-level
encryption when confidentiality requires it.

### Release Guard and GitHub

The guard must identify the exact outgoing commit range and retain provenance for
each finding. Local lockfile audit, tests, secret/privacy/size checks, and GitHub
Dependabot alerts are different evidence sources. Dependabot alert APIs require
the right repository, enabled features, authentication, and permissions; no
network result must be fabricated when unavailable.

Blocking behavior includes secrets, private workspace material, failed required
tests, disallowed large/generated model artifacts, path-policy violations, and
advisories above policy threshold. A finding with no fixed version requires
remove/replace/isolate/mitigate or explicit time-bounded risk handling; it cannot
be automatically declared fixed.

The default flow never force pushes, dismisses alerts, rewrites dependency files,
or publishes a user project. Dependency remediation belongs on a reviewable
branch or pull request and must pass tests.

## Secret handling

1. Store credentials outside source and inject the minimum scoped value at use
   time.
2. Never print tokens or include them in diagnostic snapshots.
3. Redact finding evidence to type, path category, and line location.
4. If a real credential is exposed, stop publication, rotate or revoke it, then
   assess logs, artifacts, Git history, forks, and backups.
5. History rewriting is destructive and coordinated work, never an automatic
   scanner action.

Private repository visibility is not a reason to bypass secret protection.

## Threat scenarios

| Scenario | Expected response |
|---|---|
| `../../` or junction escapes project root | Resolve and reject before access |
| Infinite learner loop | Stop on timeout and preserve a bounded diagnostic |
| Learner imports `subprocess` | Block by default; explain VM requirement for untrusted/system code |
| NPZ archive modified after save | Checksum mismatch; fail without loading arrays |
| Pickle checkpoint supplied | Refuse and show conversion guidance |
| `.env` was staged before ignore rule | Block as secret/private content; remove from index, rotate if real, assess history |
| Remote branch advanced | Fetch and review; never default to force push |
| Dependabot advisory has no patch | Block per policy; remove, replace, isolate, or document owned temporary mitigation |
| `F:` is absent | Fail visibly; never redirect a backup to an unapproved local folder |
| Unmarked nonempty backup destination | Refuse without modifying it |

## Privacy and diagnostic data

Default numeric diagnostics collect operation name, shapes, dtypes, axis,
finite-value counts, min/max/mean/std, gradient norms, learning rate, seed, and
software versions. Raw samples, prompts, labels, tokens, full private paths, and
credential values are excluded unless the user explicitly exports them.

Reports intended for issues or GitHub require a preview. The preview must state
which files and fields leave the machine and provide a cancel path.

## Secure development and release

Daedalus uses the [NIST Secure Software Development Framework](https://csrc.nist.gov/pubs/sp/800/218/final)
to organize preparation, software protection, secure production, and vulnerability
response. AI-specific risk review uses the voluntary [NIST AI Risk Management
Framework](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)
to govern, map, measure, and manage risk. These frameworks guide evidence and
questions; they are not a certification claim.

Before release, run tests and static checks, scan exact outgoing content, review
dependency/secret findings, build from a clean state, install the final artifact
in a fresh environment, verify hashes and inventory, and rehearse rollback. See
[RELEASE_GUIDE.md](RELEASE_GUIDE.md).

## Reporting and response

Report vulnerabilities privately using the process in the repository's
`SECURITY.md`. Do not attach real credentials, private datasets, proprietary
projects, or weaponized payloads to a public issue.

For a suspected exposure:

1. stop the affected push, release, scheduled task, or runner;
2. preserve redacted timestamps, versions, hashes, and finding IDs;
3. rotate credentials and isolate affected artifacts;
4. determine scope across local files, Git history, remotes, releases, logs, and
   backups;
5. fix and test the root cause;
6. document recovery, notification, and prevention decisions.
