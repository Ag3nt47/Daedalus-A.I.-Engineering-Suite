# Security policy

## Supported versions

The latest tagged release and the default branch receive security fixes.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature when the public repository
is available. Until then, contact the repository owner privately. Do not include
credentials, private datasets, proprietary projects, or exploit payloads in a
public issue.

## Product boundaries

Daedalus is an educational engineering suite, not a hardened malware-analysis
sandbox. Its code workshop constrains paths and process time but cannot promise
kernel-level isolation or network denial on every platform. Run hostile or
unknown code in a disposable virtual machine.

Checkpoints use non-executable NPZ and JSON. The suite refuses pickle loading.
Release Guard is defense in depth and does not replace repository rules,
protected branches, code review, or GitHub secret scanning.

## Automated assurance

Every public update runs cross-platform tests, Ruff, dependency auditing, CodeQL,
Gitleaks history scanning, Bandit, pull-request dependency review, Zizmor, and
OpenSSF Scorecard. Workflow actions are pinned to immutable full commit SHAs. Tagged
source releases include SHA-256 and CycloneDX SBOM evidence plus GitHub-hosted
provenance/SBOM attestations.

The first-publication helper enables Dependabot alerts and security updates,
private vulnerability reporting, secret scanning, and push protection. Repository
rules and required checks still need owner review in GitHub after the first CI run.
