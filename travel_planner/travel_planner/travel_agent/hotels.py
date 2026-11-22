"""Hotel filtering and ranking."""

from __future__ import annotations

import sqlite3

from . import config
from .models import Hotel

HOTEL_PREFERENCES = ("best_value", "best_rated", "cheapest")


def query_hotels(
    conn: sqlite3.Connection,
    city: str,
    min_stars: int = 1,
    max_price: int | None = None,
    required_amenities: list[str] | None = None,
) -> list[Hotel]:
    """Return every hotel in ``city`` that satisfies the filters.

    Args:
        conn: Open database connection.
        city: City name.
        min_stars: Minimum star rating (1-5).
        max_price: Maximum price per night, or ``None`` for no cap.
        required_amenities: Amenities that must ALL be present.
    """
    required = sorted({a.lower() for a in (required_amenities or [])})
    sql = """
        SELECT h.hotel_id, h.name, c.name AS city, h.stars, h.price_per_night
        FROM hotels h JOIN cities c ON c.city_id = h.city_id
        WHERE c.name = ? AND h.stars >= ?
    """
    params: list = [city, min_stars]
    if max_price is not None:
        sql += " AND h.price_per_night <= ?"
        params.append(max_price)
    for amenity in required:
        sql += """ AND EXISTS (
            SELECT 1 FROM hotel_amenities ha JOIN amenities a ON a.amenity_id = ha.amenity_id
            WHERE ha.hotel_id = h.hotel_id AND a.name = ?)"""
        params.append(amenity)

    hotels = []
    for row in conn.execute(sql, params).fetchall():
        names = [
            r["name"]
            for r in conn.execute(
                "SELECT a.name FROM hotel_amenities ha JOIN amenities a "
                "ON a.amenity_id = ha.amenity_id WHERE ha.hotel_id = ?",
                (row["hotel_id"],),
            )
        ]
        hotels.append(Hotel(row["hotel_id"], row["name"], row["city"], row["stars"],
                            row["price_per_night"], names))
    return hotels


def rank_hotels(hotels: list[Hotel], preference: str = "best_value") -> list[Hotel]:
    """Sort hotels by ``best_value``, ``best_rated`` or ``cheapest``."""
    if preference not in HOTEL_PREFERENCES:
        raise ValueError(f"preference must be one of {HOTEL_PREFERENCES}")
    if preference == "cheapest":
        return sorted(hotels, key=lambda h: (h.price_per_night, -h.stars, -len(h.amenities)))
    if preference == "best_rated":
        return sorted(hotels, key=lambda h: (-h.stars, -len(h.amenities), h.price_per_night))
    return sorted(hotels, key=lambda h: (-h.value_score, h.price_per_night))


def recommend_hotels(
    conn: sqlite3.Connection,
    city: str,
    min_stars: int = 1,
    max_price: int | None = None,
    required_amenities: list[str] | None = None,
    preference: str = "best_value",
    top_n: int = 3,
) -> dict:
    """Filter, rank and justify hotel choices for a city.

    Returns:
        ``{"found": bool, "selected": dict, "alternatives": [dict], "why": str,
        "candidates": int}``
    """
    hotels = query_hotels(conn, city, min_stars, max_price, required_amenities)
    if not hotels:
        return {
            "found": False,
            "candidates": 0,
            "selected": None,
            "alternatives": [],
            "why": (
                f"No hotel in {city} matches stars>={min_stars}, max price "
                f"{max_price or 'any'}, amenities {required_amenities or 'any'}."
            ),
        }
    ranked = rank_hotels(hotels, preference)
    best = ranked[0]
    cheapest = min(hotels, key=lambda h: h.price_per_night)
    reasons = {
        "best_value": f"highest value score ({best.value_score:.2f}) = quality per {config.CURRENCY}1,000",
        "best_rated": f"highest star rating ({best.stars}-star) with {len(best.amenities)} amenities",
        "cheapest": "lowest nightly price among matching hotels",
    }
    why = (
        f"{best.name} chosen because it has the {reasons[preference]}. It costs "
        f"{config.CURRENCY}{best.price_per_night:,}/night, {best.stars}-star, amenities: "
        f"{', '.join(sorted(best.amenities)) or 'none listed'}. "
        f"{len(hotels)} hotel(s) matched; cheapest match is {cheapest.name} at "
        f"{config.CURRENCY}{cheapest.price_per_night:,}."
    )
    return {
        "found": True,
        "candidates": len(hotels),
        "preference": preference,
        "selected": best.to_dict(),
        "alternatives": [h.to_dict() for h in ranked[1 : top_n + 1]],
        "why": why,
    }
