"""Shared fixtures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from travel_agent import TravelToolkit, build_database  # noqa: E402


@pytest.fixture(scope="session")
def conn():
    return build_database()


@pytest.fixture()
def toolkit(conn):
    """Toolkit with live weather disabled so tests never touch the network."""
    return TravelToolkit(conn, allow_live_weather=False)
