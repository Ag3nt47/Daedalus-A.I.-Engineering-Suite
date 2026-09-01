# Release guide

This guide is for maintainers preparing a public Daedalus release. Release Guard
is a blocking preflight, not permission to publish automatically. A human reviews
the exact outgoing content, dependency evidence, artifacts, and rollback plan.

## Release invariants

- Public source and the private user workspace are separate roots.
- Projects, datasets, checkpoints, logs, credentials, local settings, and backup
  content are not release inputs.
- Required numerical, security, backup, and resource-schema tests pass.
- Checkpoints use NPZ plus validated JSON; no pickle fixture is shipped except a
  harmless filename/policy fixture that is never deserialized.
- Findings identify their evidence source. Local package audit is not presented
  as a GitHub Dependabot alert, and unavailable network data is not invented.
- Dependency fixes are reviewable changes with tests. No alert is silently
  dismissed and no force push occurs.
- Final artifacts are built from the reviewed source state and installed in a
  clean environment before publication.

## 1. Prepare the candidate

1. Work on a focused release branch.
2. Confirm the intended version, supported Python/Windows versions, license,
   release notes, known limitations, and rollback owner.
3. Inspect configured roots and prove the private workspace is outside the
   repository.
4. Review `git status`, staged content, untracked content, and the exact diff from
   the target branch. Do not rely on `.gitignore` alone.
5. Update versioned schemas and migration notes when persisted or packaged data
   changes.
6. Re-run the backup restore drill for a release that changes workspace,
   checkpoints, installers, backup, or migration behavior.

## 2. Run local quality gates

In the release environment:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m bandit -r src -ll
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip_audit -r requirements.txt --progress-spinner=off
.\.venv\Scripts\python.exe -m pytest tests\test_resources.py
.\.venv\Scripts\python.exe -m daedalus.services.release_guard --repo . scan
```

The guard report should cover exact outgoing files, syntax/static checks, tests,
privacy/path policy, secret patterns, disallowed size/type patterns, dependency
manifests/lockfiles, and available advisory evidence. Preview the redacted report
before attaching it anywhere.

Hard blockers include:

- any real or suspected secret in outgoing content;
- private workspace/data/model/log material;
- failed required tests or schema checks;
- unreviewed generated/binary artifacts;
- a dependency advisory above the configured threshold without accepted
  mitigation;
- unknown publication scope or remote-ahead conflict.

An advisory with no patched version requires remove, replace, isolate, disable,
or explicit owner-approved time-bounded risk treatment. It cannot be marked fixed
by the scanner.

## 3. Review GitHub security evidence

For the public repository, configure and review as available:

- dependency graph and supported lockfiles;
- `.github/dependabot.yml` for each package ecosystem and manifest directory;
- Dependabot alerts and security-update pull requests;
- secret scanning and push protection;
- branch protection and required CI checks;
- least-privilege workflow permissions;
- artifact provenance/attestation and SBOM support.

Daedalus's hosted security workflow supplies Gitleaks, Bandit, and Zizmor workflow
lint at every repository visibility. Public repositories additionally run CodeQL,
pull-request dependency review, and OpenSSF Scorecard publication because those
GitHub-hosted result channels require public visibility or an eligible private-repository
security plan. Actions are pinned to full
commit SHAs. The first-publication helper enables the available repository-side
security features, but the owner must still review branch/ruleset policy and
required check names after the first successful run.

GitHub's [supply-chain documentation](https://docs.github.com/en/code-security/concepts/supply-chain-security/supply-chain-security)
describes feature behavior and availability. Dependabot alerts are repository-side
data; the [REST API](https://docs.github.com/en/rest/dependabot/alerts) requires
authentication and suitable permissions. Record when the query ran and which
repository it covered.

## 4. Build artifacts

Use a clean release environment. Install the PyPA build frontend there if it is
not already present, then build:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade build
.\.venv\Scripts\python.exe -m build
```

Expected Python artifacts are a source distribution and wheel. Build the Windows
one-click installer or bundle only through its reviewed release script. Do not
include `.venv`, caches, user workspaces, local reports, credentials, datasets,
checkpoints, or developer shortcuts.

Generate SHA-256 hashes after every artifact is final. Generate the dependency
inventory/SBOM from the same resolved build, not an earlier development
environment. If signing infrastructure is configured, sign final artifacts and
record certificate identity and timestamp; otherwise describe artifacts as
hashed, not signed.

## 5. Clean-install verification

Use a disposable Windows VM or clean test account:

1. Verify the downloaded artifact hash.
2. Install without an existing Daedalus environment.
3. Confirm the source/install and private workspace paths are separate.
4. Launch offline with `F:` absent; local learning and calculators must work and
   backup must report the missing drive without redirecting.
5. Run a Shape Detective fixture, create a private XOR project, and execute a
   constrained-runner smoke test.
6. Save and reload an NPZ+JSON checkpoint.
7. Configure a temporary marked backup destination, back up, and restore into a
   new path.
8. Run Release Guard on its clean fixture without authorizing a push.
9. Uninstall and confirm private user data is preserved unless the user explicitly
   chose a separately confirmed data-removal action.

Record OS build, Python/runtime versions, installer hash, test results, and
screenshots that contain no private paths or secrets.

## 6. Publish deliberately

1. Recheck the outgoing commit range after final fixes.
2. Require reviewed CI results from the same commit.
3. Merge according to repository policy; create an annotated version tag.
4. Create release notes covering features, security fixes, breaking/schema
   changes, known limitations, install/upgrade steps, artifact hashes, and
   rollback.
5. Upload only the verified artifacts and matching SBOM/attestation evidence.
6. Verify the public download hashes and clean install once more.
7. Enable immutable release controls when supported and appropriate.

Safe Push must not make steps 3-7 implicit. Publication remains an explicit
operator decision with a visible destination and artifact list.

## 7. Rollback and incident handling

Before publication, identify the last known-good version and how to obtain its
verified artifacts. A rollback must preserve private workspace data and avoid
loading newer incompatible metadata into an older release without a documented
migration path.

If a release contains a secret, vulnerable dependency, corrupted artifact, or
unsafe migration:

1. stop further distribution and automation;
2. revoke/rotate exposed credentials immediately;
3. preserve redacted hashes, timestamps, versions, and finding IDs;
4. withdraw or mark affected artifacts according to repository policy;
5. publish a corrected release from a reviewed branch;
6. notify users of exact affected versions, recovery steps, and data implications;
7. update regression fixtures and the security/release documentation.

Follow [SECURITY_MODEL.md](SECURITY_MODEL.md) for vulnerability reporting and
[BACKUP_RESTORE.md](BACKUP_RESTORE.md) before any destructive migration or
recovery operation.

## Authoritative references

- [PyPA Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Git hooks](https://git-scm.com/docs/githooks)
- [Git ignore rules](https://git-scm.com/docs/gitignore)
- [GitHub Actions](https://docs.github.com/en/actions/get-started/understand-github-actions)
- [GitHub Dependabot configuration](https://docs.github.com/en/code-security/concepts/supply-chain-security/about-the-dependabot-yml-file)
- [GitHub command-line push protection](https://docs.github.com/en/code-security/concepts/secret-security/command-line-push-protection)
- [NIST SSDF SP 800-218](https://csrc.nist.gov/pubs/sp/800/218/final)
- [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)
