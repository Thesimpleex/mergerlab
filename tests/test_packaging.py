"""Version metadata stays in one place and the typed-package marker ships."""

import re
import tomllib
from pathlib import Path

import mergerlab

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_defined_once_and_matches_the_citation_file():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" not in pyproject["project"]
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/mergerlab/__init__.py"
    cff = (ROOT / "CITATION.cff").read_text()
    match = re.search(r"^version:\s*(\S+)", cff, flags=re.M)
    assert match is not None and match.group(1) == mergerlab.__version__


def test_py_typed_marker_ships_with_the_package():
    assert (Path(mergerlab.__file__).parent / "py.typed").exists()
