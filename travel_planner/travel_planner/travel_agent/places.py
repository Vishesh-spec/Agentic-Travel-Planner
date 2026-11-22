"""Attraction (POI) discovery and ranking."""

from __future__ import annotations

import sqlite3

from .models import Place


def query_places(
    conn: sqlite3.Connection,
    city: str,
    categories: list[str] | None = None,
    min_rating: float = 0.0,
) -> list[Place]:
    """Return places in ``city`` filtered by category and rating.

    Args:
        conn: Open database connection.
        city: City name.
        categories: Allowed categories (``None``/empty = all).
        min_rating: Minimum rating (0-5).
    """
    sql = """
        SELECT p.place_id, p.name, c.name AS city, t.name AS category, p.rating
        FROM places p
        JOIN cities c      ON c.city_id = p.city_id
        JOIN place_types t ON t.type_id = p.type_id
        WHERE c.name = ? AND p.rating >= ?
    """
    params: list = [city, min_rating]
    wanted = [c.lower() for c in (categories or [])]
    if wanted:
        sql += f" AND t.name IN ({','.join('?' * len(wanted))})"
        params.extend(wanted)
    sql += " ORDER BY p.rating DESC, p.place_id"
    return [Place(r["place_id"], r["name"], r["city"], r["category"], r["rating"])
            for r in conn.execute(sql, params)]


def discover_places(
    conn: sqlite3.Connection,
    city: str,
    interests: list[str] | None = None,
    min_rating: float = 0.0,
    limit: int = 12,
) -> dict:
    """Recommend attractions, boosting the categories the traveller likes.

    Ranking: places whose category is in ``interests`` come first (sorted by
    rating); the remaining places follow (sorted by rating) so that the
    itinerary can still be filled on longer trips.

    Returns:
        ``{"found": bool, "places": [dict], "why": str}``
    """
    all_places = query_places(conn, city, None, min_rating)
    if not all_places:
        return {"found": False, "places": [],
                "why": f"No places in {city} with rating >= {min_rating}."}
    wanted = {i.lower() for i in (interests or [])}
    matched = [p for p in all_places if p.category in wanted]
    others = [p for p in all_places if p.category not in wanted]
    ranked = (matched + others)[:limit]
    if wanted and not matched:
        why = (f"No {', '.join(sorted(wanted))} attractions in {city}; "
               f"showing the highest-rated places instead.")
    elif wanted:
        why = (f"{len(matched)} place(s) match your interests ({', '.join(sorted(wanted))}) "
               f"and are ranked first by rating; the rest follow by rating.")
    else:
        why = "No specific interests given - places ranked purely by rating."
    return {
        "found": True,
        "why": why,
        "places": [p.to_dict() for p in ranked],
        "_places": ranked,
    }
