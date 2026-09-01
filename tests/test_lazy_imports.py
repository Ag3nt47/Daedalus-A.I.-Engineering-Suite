"""Regression coverage for package-level on-demand import boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_targeted_public_exports_do_not_import_unrelated_toolchains() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(source_root), environment.get("PYTHONPATH", ""))
        if value
    )
    script = r'''
import sys

import daedalus.gui as gui
assert "daedalus.gui.main_window" not in sys.modules
assert "daedalus.gui.widgets" not in sys.modules
_ = gui.THEMES
assert "daedalus.gui.theme" in sys.modules
assert "daedalus.gui.main_window" not in sys.modules

import daedalus.services as services
assert "daedalus.services.project_standards" not in sys.modules
assert "daedalus.services.weight_sandbox" not in sys.modules
_ = services.ProjectStandardsService
assert "daedalus.services.project_standards" in sys.modules
assert "daedalus.services.weight_sandbox" not in sys.modules
assert "numpy" not in sys.modules

import daedalus.engine as engine
assert "daedalus.engine.advanced_calculators" not in sys.modules
assert "daedalus.engine.weight_tools" not in sys.modules
_ = engine.convolution_output_shape
assert "daedalus.engine.advanced_calculators" in sys.modules
assert "daedalus.engine.weight_tools" not in sys.modules

import daedalus.developer as developer
assert "daedalus.developer.store" not in sys.modules
assert "daedalus.developer.diagnostics" not in sys.modules
_ = developer.DeveloperSessionStore
assert "daedalus.developer.store" in sys.modules
assert "daedalus.developer.diagnostics" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_root.parent,
        env=environment,
        capture_output=True,
        text=True,
        # This test verifies module boundaries, not cold-start performance. Keep
        # the subprocess bounded while allowing endpoint security and heavily
        # loaded Windows hosts to finish importing from a fresh interpreter.
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
