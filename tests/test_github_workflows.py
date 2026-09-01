from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_every_external_action_is_pinned_to_a_full_commit_sha() -> None:
    action_pattern = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)", re.MULTILINE)
    pin_pattern = re.compile(r"^[^@]+@[0-9a-f]{40}$")
    found: list[str] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        source = workflow.read_text(encoding="utf-8")
        assert "pull_request_target:" not in source
        for action in action_pattern.findall(source):
            found.append(action)
            assert pin_pattern.fullmatch(action), f"{workflow.name}: mutable action pin {action}"
    assert found


def test_security_workflow_has_layered_scanners_and_bounded_jobs() -> None:
    source = (WORKFLOWS / "security.yml").read_text(encoding="utf-8")
    for scanner in (
        "github/codeql-action/init@",
        "gitleaks/gitleaks-action@",
        "python -m bandit",
        "actions/dependency-review-action@",
        "zizmor --collect=workflows",
        "ossf/scorecard-action@",
    ):
        assert scanner in source
    assert source.count("timeout-minutes:") >= 6
    assert "persist-credentials: false" in source


def test_release_publication_is_split_from_candidate_execution() -> None:
    source = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    package = source.split("  publish:\n", maxsplit=1)[0]
    publish = source.split("  publish:\n", maxsplit=1)[1]
    assert "contents: read" in package
    assert "python -m pytest" in package
    assert "contents: write" in publish
    assert "actions/checkout@" not in publish
    assert "attest-build-provenance@" in publish
    assert "attest-sbom@" in publish


def test_security_tool_manifest_is_hash_pinned_and_development_only() -> None:
    manifest = json.loads((ROOT / "tools" / "security-tools.lock.json").read_text("utf-8"))
    assert manifest["schema_version"] == 1
    assert {asset["id"] for asset in manifest["assets"]} == {"github-cli", "gitleaks"}
    for asset in manifest["assets"]:
        assert re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])
        assert asset["url"].startswith("https://github.com/")
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["assets"][1]["executable_sha256"])
    assert "/.dev-tools/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_local_publish_path_uses_exact_staged_scan_and_explicit_first_push() -> None:
    safe_push = (ROOT / "tools" / "safe-push.ps1").read_text(encoding="utf-8")
    publish = (ROOT / "tools" / "publish-github.ps1").read_text(encoding="utf-8")
    security = (ROOT / "tools" / "configure-github-security.ps1").read_text(
        encoding="utf-8"
    )
    assert "gitleaks git --staged" in safe_push.lower()
    assert "bootstrap-security-tools.ps1" in safe_push
    assert "--initial-publish" in publish
    assert "configure-github-security.ps1" in publish
    assert "[y/N]" in publish
    for endpoint in (
        "vulnerability-alerts",
        "automated-security-fixes",
        "private-vulnerability-reporting",
        "secret_scanning_push_protection",
    ):
        assert endpoint in security


def test_unattended_push_is_main_only_pinned_and_explicitly_enabled() -> None:
    configure = (ROOT / "tools" / "configure-auto-push.ps1").read_text(encoding="utf-8")
    automatic = (ROOT / "tools" / "auto-push.ps1").read_text(encoding="utf-8")
    assert "ENABLE DAEDALUS AUTO PUSH" in configure
    assert "Local main must exactly match origin/main" in configure
    assert "-ExpectedBranch main -ExpectedOrigin" in configure
    assert "scheduled-identity-changed" in automatic
    assert "origin-is-ahead-or-diverged" in automatic
    assert "gitleaks git --staged" in automatic.lower()
    assert "daedalus-auto-push-health.json" in automatic
    assert "--fix-dependencies" not in automatic
