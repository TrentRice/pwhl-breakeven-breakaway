"""The calculator must reproduce the Python decision engine (runs tests/app_check.js with Node)."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_calculator_matches_python_engine():
    if not (ROOT / "app" / "data" / "model.json").exists():
        pytest.skip("run `python -m src.export_app` first")
    out = subprocess.run(["node", str(ROOT / "tests" / "app_check.js")], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
