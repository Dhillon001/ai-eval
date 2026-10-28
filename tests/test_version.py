from __future__ import annotations

import re
from pathlib import Path

import ai_eval

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_pyproject_and_changelog():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert ai_eval.__version__ == version
    assert f"## [{version}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
