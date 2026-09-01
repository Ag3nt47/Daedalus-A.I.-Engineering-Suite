import json
import subprocess
from pathlib import Path

import pytest

import daedalus.services.release_guard as release_guard_module
from daedalus.services.release_guard import BLOCK, ReleaseGuard

WORKSPACE_SOURCE_FILES = (
    "__init__.py",
    "checkpoints.py",
    "datasets.py",
    "manager.py",
    "run_registry.py",
)
ZERO_OID = "0" * 40


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text(
        "\n".join(
            [
                "/.env",
                "/.env.*",
                "!/.env.example",
                "/.venv/",
                "/workspace/",
                "/workspaces/",
                "/user-workspaces/",
                "/projects/",
                "/datasets/",
                "/weights/",
                "/checkpoints/",
                "/training-runs/",
                "/logs/",
                "/models/",
                "*.npz",
                "reports/local/",
            ]
        ),
        encoding="utf-8",
    )
    for name in ("LICENSE", "SECURITY.md", "pyproject.toml"):
        (repo / name).write_text("safe", encoding="utf-8")
    workspace_package = repo / "src" / "daedalus" / "workspace"
    workspace_package.mkdir(parents=True)
    for name in WORKSPACE_SOURCE_FILES:
        (workspace_package / name).write_text("# safe\n", encoding="utf-8")
    return repo


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def initialize_with_baseline(repo: Path) -> str:
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "release-guard@example.invalid")
    git(repo, "config", "user.name", "Release Guard Test")
    git(repo, "add", "-A")
    git(repo, "commit", "-m", "safe baseline")
    return git(repo, "rev-parse", "HEAD")


def outgoing_update(local_oid: str, remote_oid: str = ZERO_OID) -> str:
    return f"refs/heads/main {local_oid} refs/heads/main {remote_oid}\n"


def test_clean_tree_passes_privacy_checks(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "app.py").write_text("print('safe')\n", encoding="utf-8")
    report = ReleaseGuard(repo).scan()
    assert report.ok


def test_secret_fixture_is_blocked_and_redacted(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    token = "ghp_" + "A" * 36
    (repo / "bad.txt").write_text(token, encoding="utf-8")
    report = ReleaseGuard(repo).scan()
    secret_findings = [item for item in report.findings if item.check == "secret-scan"]
    assert secret_findings
    assert token not in report.format_text()


@pytest.mark.parametrize("suffix", [".cs", ".xml", ".properties"])
def test_normal_text_source_and_config_suffixes_are_secret_scanned(
    tmp_path: Path, suffix: str
) -> None:
    repo = make_repo(tmp_path)
    token = "github_" + "pat_" + "A" * 48
    (repo / f"config{suffix}").write_text(token, encoding="utf-8")

    report = ReleaseGuard(repo).scan()

    assert any(item.check == "secret-scan" for item in report.blocking)
    assert token not in report.format_text()


def test_forced_tracked_environment_file_is_blocked_but_example_is_allowed(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    initialize_with_baseline(repo)
    (repo / ".env.production").write_text("safe-looking-placeholder=true\n", encoding="utf-8")
    (repo / ".env.example").write_text("SETTING=example\n", encoding="utf-8")
    git(repo, "add", "-f", ".env.production")
    git(repo, "add", ".env.example")

    report = ReleaseGuard(repo).scan(staged=True)

    assert any(item.path == ".env.production" for item in report.blocking)
    assert not any(item.path == ".env.example" for item in report.blocking)


@pytest.mark.parametrize(
    "marker_name", [".daedalus-workspace.json", ".daedalus-backup-root.json"]
)
def test_private_ownership_markers_are_blocked_at_any_depth(
    tmp_path: Path, marker_name: str
) -> None:
    repo = make_repo(tmp_path)
    nested = repo / "innocent-looking" / "nested"
    nested.mkdir(parents=True)
    (nested / marker_name).write_text("{}\n", encoding="utf-8")

    report = ReleaseGuard(repo).scan()

    assert any(
        item.check == "private-marker" and item.path == f"innocent-looking/nested/{marker_name}"
        for item in report.blocking
    )


def test_large_text_receives_streaming_secret_scan(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    token = b"ghp_" + b"A" * 36
    # Put the token across a 1 MiB stream boundary after the former 3 MiB cutoff.
    prefix = b"x" * (4 * 1024 * 1024 - 11) + b"\n"
    (repo / "large.txt").write_bytes(prefix + token + b"\n")
    report = ReleaseGuard(repo).scan()
    matches = [item for item in report.blocking if item.check == "secret-scan"]
    assert matches
    assert token.decode() not in report.format_text()


def test_private_root_candidate_is_blocked(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    private = repo / "projects"
    private.mkdir()
    (private / "client.py").write_text("print('private')", encoding="utf-8")
    guard = ReleaseGuard(repo)
    # Non-git fallback intentionally sees every file, proving the second guard.
    report = guard.scan()
    assert any(item.level == BLOCK and item.check == "private-path" for item in report.findings)


def test_python_syntax_error_blocks_release(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "broken.py").write_text("def nope(:\n", encoding="utf-8")
    report = ReleaseGuard(repo).scan()
    assert any(item.check == "python-syntax" for item in report.blocking)


def test_unanchored_workspace_ignore_rule_is_blocked(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    with (repo / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("\nworkspace/\n")
    report = ReleaseGuard(repo).scan()
    assert any(
        item.check == "privacy-policy" and "workspace/" in item.message
        for item in report.blocking
    )


def test_workspace_package_is_a_publish_candidate(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    candidates = {path.as_posix() for path in ReleaseGuard(repo).candidate_paths()}
    assert "src/daedalus/workspace/manager.py" in candidates
    report = ReleaseGuard(repo).scan()
    assert any(item.check == "publish-scope" and item.level == "pass" for item in report.findings)


def test_outgoing_history_blocks_secret_removed_from_clean_head(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    baseline = initialize_with_baseline(repo)
    secret = "ghp_" + "B" * 36
    (repo / "temporary-secret.txt").write_text(secret, encoding="utf-8")
    git(repo, "add", "temporary-secret.txt")
    git(repo, "commit", "-m", "accidentally add secret")
    (repo / "temporary-secret.txt").unlink()
    git(repo, "add", "-u")
    git(repo, "commit", "-m", "remove secret")
    head = git(repo, "rev-parse", "HEAD")

    # The ordinary worktree is clean and no longer contains the credential.
    assert ReleaseGuard(repo).scan().ok
    report = ReleaseGuard(repo).scan_outgoing([outgoing_update(head, baseline)])
    assert any(item.check == "secret-scan" for item in report.blocking)
    assert secret not in report.format_text()


def test_outgoing_commit_blocks_model_weight_artifact(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    baseline = initialize_with_baseline(repo)
    (repo / "weights.pt").write_bytes(b"not actually a model")
    git(repo, "add", "-f", "weights.pt")
    git(repo, "commit", "-m", "add model weights")
    head = git(repo, "rev-parse", "HEAD")

    report = ReleaseGuard(repo).scan_outgoing([outgoing_update(head, baseline)])
    assert any(
        item.check == "private-artifact" and item.path == "weights.pt"
        for item in report.blocking
    )


def test_outgoing_non_fast_forward_branch_update_is_blocked(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    baseline = initialize_with_baseline(repo)
    (repo / "remote-change.py").write_text("print('remote')\n", encoding="utf-8")
    git(repo, "add", "remote-change.py")
    git(repo, "commit", "-m", "remote descendant")
    remote_head = git(repo, "rev-parse", "HEAD")

    report = ReleaseGuard(repo).scan_outgoing([outgoing_update(baseline, remote_head)])

    assert any(item.check == "non-fast-forward" for item in report.blocking)


def test_outgoing_protocol_fails_closed_when_scope_is_missing(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    initialize_with_baseline(repo)
    report = ReleaseGuard(repo).scan_outgoing([])
    assert any(item.check == "outgoing-history" for item in report.blocking)


def test_pre_push_hook_invokes_immutable_history_scan() -> None:
    hook = Path(__file__).parents[1] / ".githooks" / "pre-push"
    source = hook.read_text(encoding="utf-8")
    assert "pre-push --remote-name" in source
    assert " release_guard --repo \"$ROOT\" scan" not in source


def test_push_without_origin_blocks_before_staging_or_committing(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    baseline = initialize_with_baseline(repo)
    (repo / "pending.py").write_text("print('pending')\n", encoding="utf-8")

    report = ReleaseGuard(repo).safe_push("should not commit")

    assert any(item.check == "push" for item in report.blocking)
    assert git(repo, "rev-parse", "HEAD") == baseline
    assert "?? pending.py" in git(repo, "status", "--short")


def test_no_change_no_ahead_push_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    head = initialize_with_baseline(repo)
    git(repo, "remote", "add", "origin", "https://github.com/example/daedalus.git")
    git(repo, "update-ref", "refs/remotes/origin/main", head)
    git(repo, "branch", "--set-upstream-to=origin/main", "main")
    guard = ReleaseGuard(repo)
    monkeypatch.setattr(guard, "_scan_ruff", lambda _findings: None)
    monkeypatch.setattr(guard, "_scan_tests", lambda _findings: None)
    monkeypatch.setattr(guard, "_scan_dependencies", lambda _findings: None)
    monkeypatch.setattr(guard, "_scan_github_dependabot", lambda _findings: None)
    original_git = guard._git
    push_calls: list[tuple[str, ...]] = []

    def recording_git(*arguments: str, **kwargs):
        if arguments and arguments[0] == "push":
            push_calls.append(arguments)
            return subprocess.CompletedProcess(arguments, 1, "", "unexpected push")
        return original_git(*arguments, **kwargs)

    monkeypatch.setattr(guard, "_git", recording_git)

    report = guard.safe_push("nothing changed")

    assert report.ok
    assert not push_calls
    assert any(
        item.check == "push" and "up to date" in item.message for item in report.findings
    )


def test_dependency_remediation_requires_explicit_push_flag() -> None:
    parser = release_guard_module._parser()

    default_args = parser.parse_args(["push", "--message", "safe change"])
    opted_in_args = parser.parse_args(
        ["push", "--message", "safe change", "--fix-dependencies"]
    )

    assert default_args.fix_dependencies is False
    assert opted_in_args.fix_dependencies is True


def test_local_quality_gate_runs_ruff_before_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = ReleaseGuard(make_repo(tmp_path))
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        output = "All checks passed!" if "ruff" in command else "1 passed"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(guard, "_run", fake_run)

    report = guard.scan(include_tests=True)

    assert report.ok
    assert commands[0][2:] == ["ruff", "check", "."]
    assert commands[1][2:] == ["pytest", "-q"]


def test_test_gate_uses_clean_bounded_environment_and_blocks_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = ReleaseGuard(make_repo(tmp_path))
    captured: dict[str, object] = {}

    def timeout_run(command, **kwargs):
        captured["command"] = list(command)
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(guard, "_run", timeout_run)
    findings = []

    guard._scan_tests(findings)

    assert captured["command"][-3:] == ["-m", "pytest", "-q"]
    assert captured["timeout"] == 1200
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["QT_QPA_PLATFORM"] == "offscreen"
    assert len(findings) == 1
    assert findings[0].level == BLOCK
    assert findings[0].check == "tests"
    assert "20-minute timeout" in findings[0].message


def test_initial_publish_skips_only_github_query_for_proven_empty_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    head = initialize_with_baseline(repo)
    guard = ReleaseGuard(repo)
    checks: list[str] = []
    monkeypatch.setattr(guard, "_initial_publish_error", lambda _remote="origin": None)
    monkeypatch.setattr(guard, "_scan_ruff", lambda _findings: checks.append("ruff"))
    monkeypatch.setattr(guard, "_scan_tests", lambda _findings: checks.append("tests"))
    monkeypatch.setattr(
        guard, "_scan_dependencies", lambda _findings: checks.append("dependencies")
    )

    def unexpected_github_query(_findings):
        pytest.fail("initial publication must not query unavailable repository alerts")

    monkeypatch.setattr(guard, "_scan_github_dependabot", unexpected_github_query)

    report = guard.scan_outgoing(
        [outgoing_update(head)],
        include_tests=True,
        include_dependencies=True,
        include_github=True,
        initial_publish=True,
    )

    assert report.ok
    assert checks == ["ruff", "tests", "dependencies"]


def test_initial_publish_is_rejected_when_remote_has_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = ReleaseGuard(make_repo(tmp_path))
    remote_output = f"{'a' * 40}\trefs/heads/main\n"
    monkeypatch.setattr(
        guard,
        "_git",
        lambda *arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, remote_output, ""
        ),
    )

    error = guard._initial_publish_error()

    assert error is not None
    assert "no refs" in error


def test_initial_publish_with_populated_remote_blocks_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    baseline = initialize_with_baseline(repo)
    git(repo, "remote", "add", "origin", "https://github.com/example/daedalus.git")
    (repo / "pending.py").write_text("print('pending')\n", encoding="utf-8")
    guard = ReleaseGuard(repo)
    monkeypatch.setattr(
        guard,
        "_initial_publish_error",
        lambda _remote="origin": "Initial publication requires a remote with no refs.",
    )

    report = guard.safe_push("must not commit", initial_publish=True)

    assert any(item.check == "initial-publish" for item in report.blocking)
    assert git(repo, "rev-parse", "HEAD") == baseline
    assert "?? pending.py" in git(repo, "status", "--short")


def test_initial_publish_cli_flag_is_explicit() -> None:
    args = release_guard_module._parser().parse_args(
        ["push", "--message", "first publication", "--initial-publish"]
    )

    assert args.initial_publish is True


def _github_repo(repo: Path) -> ReleaseGuard:
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/example/daedalus.git"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return ReleaseGuard(repo)


def test_dependabot_unavailable_blocks_when_github_origin_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _github_repo(make_repo(tmp_path))
    monkeypatch.setattr(release_guard_module.shutil, "which", lambda _name: None)
    report = guard.scan(include_github=True)
    assert any(item.check == "dependabot" and item.level == BLOCK for item in report.findings)


def test_dependabot_api_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = ReleaseGuard(make_repo(tmp_path))
    monkeypatch.setattr(
        guard, "_origin_url", lambda: "https://github.com/example/daedalus.git"
    )
    monkeypatch.setattr(release_guard_module.shutil, "which", lambda _name: "gh")
    monkeypatch.setattr(
        guard,
        "_run",
        lambda _command, **_kwargs: subprocess.CompletedProcess([], 1, "", "denied"),
    )
    findings = []
    guard._scan_github_dependabot(findings)
    assert any(item.check == "dependabot" and item.level == BLOCK for item in findings)


def test_dependabot_paginates_and_flattens_all_alerts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = ReleaseGuard(make_repo(tmp_path))
    monkeypatch.setattr(
        guard, "_origin_url", lambda: "https://github.com/example/daedalus.git"
    )
    monkeypatch.setattr(release_guard_module.shutil, "which", lambda _name: "gh")
    commands: list[list[str]] = []
    pages = [
        [
            {
                "security_advisory": {"severity": "high", "ghsa_id": "GHSA-high"},
                "dependency": {"package": {"name": "first"}},
            }
        ],
        [
            {
                "security_advisory": {"severity": "low", "ghsa_id": "GHSA-low"},
                "dependency": {"package": {"name": "second"}},
            }
        ],
    ]

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, json.dumps(pages), "")

    monkeypatch.setattr(guard, "_run", fake_run)
    findings = []
    guard._scan_github_dependabot(findings)
    assert "--paginate" in commands[0]
    assert "--slurp" in commands[0]
    assert any("GHSA-high" in item.message and item.level == BLOCK for item in findings)
    assert any("GHSA-low" in item.message for item in findings)
