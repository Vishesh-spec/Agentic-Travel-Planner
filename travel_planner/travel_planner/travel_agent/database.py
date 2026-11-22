"""SQLite persistence layer.

The three JSON datasets are loaded into a *normalised* relational schema
(3NF) with indexes on every column used for filtering/sorting.  The agent
tools then query this database instead of scanning raw JSON in Python.

Schema overview::

    cities(city_id, name)
    airlines(airline_id, name)
    flights(flight_id, airline_id, origin_city_id, destination_city_id,
            departure_time, arrival_time, duration_minutes, price)
    hotels(hotel_id, name, city_id, stars, price_per_night)
    amenities(amenity_id, name)
    hotel_amenities(hotel_id, amenity_id)
    place_types(type_id, name)
    places(place_id, name, city_id, type_id, rating)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cities (
    city_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS airlines (
    airline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS flights (
    flight_id           TEXT PRIMARY KEY,
    airline_id          INTEGER NOT NULL REFERENCES airlines(airline_id),
    origin_city_id      INTEGER NOT NULL REFERENCES cities(city_id),
    destination_city_id INTEGER NOT NULL REFERENCES cities(city_id),
    departure_time      TEXT NOT NULL,
    arrival_time        TEXT NOT NULL,
    duration_minutes    INTEGER NOT NULL CHECK (duration_minutes > 0),
    price               INTEGER NOT NULL CHECK (price >= 0)
);

CREATE TABLE IF NOT EXISTS hotels (
    hotel_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    city_id         INTEGER NOT NULL REFERENCES cities(city_id),
    stars           INTEGER NOT NULL CHECK (stars BETWEEN 1 AND 5),
    price_per_night INTEGER NOT NULL CHECK (price_per_night >= 0)
);

CREATE TABLE IF NOT EXISTS amenities (
    amenity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS hotel_amenities (
    hotel_id   TEXT    NOT NULL REFERENCES hotels(hotel_id),
    amenity_id INTEGER NOT NULL REFERENCES amenities(amenity_id),
    PRIMARY KEY (hotel_id, amenity_id)
);

CREATE TABLE IF NOT EXISTS place_types (
    type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS places (
    place_id TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    city_id  INTEGER NOT NULL REFERENCES cities(city_id),
    type_id  INTEGER NOT NULL REFERENCES place_types(type_id),
    rating   REAL NOT NULL CHECK (rating BETWEEN 0 AND 5)
);

CREATE INDEX IF NOT EXISTS idx_flights_route
    ON flights (origin_city_id, destination_city_id, price);
CREATE INDEX IF NOT EXISTS idx_hotels_search
    ON hotels (city_id, stars, price_per_night);
CREATE INDEX IF NOT EXISTS idx_places_city_rating
    ON places (city_id, rating DESC);
CREATE INDEX IF NOT EXISTS idx_places_type
    ON places (type_id);
CREATE INDEX IF NOT EXISTS idx_hotel_amenities_amenity
    ON hotel_amenities (amenity_id);
"""


class DataLoadError(RuntimeError):
    """Raised when a dataset file is missing or is not valid JSON."""


def _read_json(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array from ``path`` with friendly error messages."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise DataLoadError(f"Dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Dataset is not valid JSON: {path} ({exc})") from exc
    if not isinstance(data, list):
        raise DataLoadError(f"Dataset must be a JSON array: {path}")
    return data


def _get_or_create(conn: sqlite3.Connection, table: str, id_col: str, name: str) -> int:
    """Return the id of ``name`` in a lookup table, inserting it if needed."""
    row = conn.execute(f"SELECT {id_col} FROM {table} WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    cursor = conn.execute(f"INSERT INTO {table} (name) VALUES (?)", (name,))
    return cursor.lastrowid


def _load_flights(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
    """Insert flight records, skipping (and logging) invalid ones."""
    loaded = 0
    for rec in records:
        try:
            departure = datetime.fromisoformat(rec["departure_time"])
            arrival = datetime.fromisoformat(rec["arrival_time"])
            duration = int((arrival - departure).total_seconds() // 60)
            if duration <= 0:
                raise ValueError("arrival is not after departure")
            conn.execute(
                "INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rec["flight_id"],
                    _get_or_create(conn, "airlines", "airline_id", rec["airline"]),
                    _get_or_create(conn, "cities", "city_id", rec["from"]),
                    _get_or_create(conn, "cities", "city_id", rec["to"]),
                    rec["departure_time"],
                    rec["arrival_time"],
                    duration,
                    int(rec["price"]),
                ),
            )
            loaded += 1
        except (KeyError, ValueError, sqlite3.Error) as exc:
            logger.warning("Skipping invalid flight %s: %s", rec.get("flight_id"), exc)
    return loaded


def _load_hotels(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
    """Insert hotel records plus their amenity links."""
    loaded = 0
    for rec in records:
        try:
            conn.execute(
                "INSERT INTO hotels VALUES (?, ?, ?, ?, ?)",
                (
                    rec["hotel_id"],
                    rec["name"],
                    _get_or_create(conn, "cities", "city_id", rec["city"]),
                    int(rec["stars"]),
                    int(rec["price_per_night"]),
                ),
            )
            for amenity in set(rec.get("amenities", [])):
                amenity_id = _get_or_create(conn, "amenities", "amenity_id", amenity)
                conn.execute(
                    "INSERT OR IGNORE INTO hotel_amenities VALUES (?, ?)",
                    (rec["hotel_id"], amenity_id),
                )
            loaded += 1
        except (KeyError, ValueError, sqlite3.Error) as exc:
            logger.warning("Skipping invalid hotel %s: %s", rec.get("hotel_id"), exc)
    return loaded


def _load_places(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
    """Insert place (POI) records."""
    loaded = 0
    for rec in records:
        try:
            conn.execute(
                "INSERT INTO places VALUES (?, ?, ?, ?, ?)",
                (
                    rec["place_id"],
                    rec["name"],
                    _get_or_create(conn, "cities", "city_id", rec["city"]),
                    _get_or_create(conn, "place_types", "type_id", rec["type"]),
                    float(rec["rating"]),
                ),
            )
            loaded += 1
        except (KeyError, ValueError, sqlite3.Error) as exc:
            logger.warning("Skipping invalid place %s: %s", rec.get("place_id"), exc)
    return loaded


def build_database(
    flights_path: Path = config.FLIGHTS_JSON,
    hotels_path: Path = config.HOTELS_JSON,
    places_path: Path = config.PLACES_JSON,
    db_path: str | Path = ":memory:",
) -> sqlite3.Connection:
    """Create the schema and load all three JSON datasets.

    Args:
        flights_path: Path to ``flights.json``.
        hotels_path: Path to ``hotels.json``.
        places_path: Path to ``places.json``.
        db_path: SQLite file path, or ``":memory:"`` (default).

    Returns:
        An open :class:`sqlite3.Connection` with ``row_factory`` set to
        :class:`sqlite3.Row`.

    Raises:
        DataLoadError: If a dataset is missing or malformed.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    counts = {
        "flights": _load_flights(conn, _read_json(Path(flights_path))),
        "hotels": _load_hotels(conn, _read_json(Path(hotels_path))),
        "places": _load_places(conn, _read_json(Path(places_path))),
    }
    conn.commit()
    logger.info("Database ready: %s", counts)
    return conn


def list_cities(conn: sqlite3.Connection) -> list[str]:
    """Return every city name known to the database, sorted."""
    return [r["name"] for r in conn.execute("SELECT name FROM cities ORDER BY name")]


def list_amenities(conn: sqlite3.Connection) -> list[str]:
    """Return every hotel amenity name, sorted."""
    return [r["name"] for r in conn.execute("SELECT name FROM amenities ORDER BY name")]


def list_place_types(conn: sqlite3.Connection) -> list[str]:
    """Return every place category (park, beach, ...), sorted."""
    return [r["name"] for r in conn.execute("SELECT name FROM place_types ORDER BY name")]


def dataset_stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts of the main tables (used by the UI and the report)."""
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("cities", "airlines", "flights", "hotels", "places")
    }
