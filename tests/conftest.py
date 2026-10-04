"""Shared fixtures."""

from __future__ import annotations

from typing import Any

import pytest

from helpers import load_json


@pytest.fixture(scope="session")
def pyblp_fixtures() -> dict[str, Any]:
    return load_json("pyblp_merger.json")


@pytest.fixture(scope="session")
def r_fixtures() -> dict[str, Any]:
    return load_json("r_antitrust.json")
