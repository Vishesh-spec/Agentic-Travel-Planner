"""Flight search, routing and ranking.

``flights.json`` contains only 30 *directed* flights on fixed 2025 dates, so
many city pairs have no direct service (e.g. Delhi -> Goa).  We therefore
treat every record as a recurring *daily schedule template*:

* the airline, route, price and duration come from the dataset;
* the clock time of departure is re-used on the requested travel date;
* connecting itineraries (up to 3 stops) are assembled with a depth-first
  search that enforces a minimum layover.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, time, timedelta

from . import config
from .models import FlightLeg, FlightOption

logger = logging.getLogger(__name__)

_PREFERENCES = ("cheapest", "fastest", "balanced")


def _load_templates(conn: sqlite3.Connection) -> dict[str, list[sqlite3.Row]]:
    """Group all scheduled flights by origin city name."""
    rows = conn.execute(
        """
        SELECT f.flight_id, a.name AS airline, o.name AS origin, d.name AS destination,
               f.departure_time, f.duration_minutes, f.price
        FROM flights f
        JOIN airlines a ON a.airline_id = f.airline_id
        JOIN cities   o ON o.city_id    = f.origin_city_id
        JOIN cities   d ON d.city_id    = f.destination_city_id
        ORDER BY f.price
        """
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["origin"], []).append(row)
    return grouped


def _next_departure(clock: time, not_before: datetime) -> datetime:
    """First datetime with the given clock time that is >= ``not_before``."""
    candidate = datetime.combine(not_before.date(), clock)
    return candidate if candidate >= not_before else candidate + timedelta(days=1)


def find_flight_options(
    conn: sqlite3.Connection,
    origin: str,
    destination: str,
    travel_date: date,
    max_stops: int = config.MAX_STOPS_HARD_LIMIT,
    max_layover_hours: float = config.PREFERRED_MAX_LAYOVER_HOURS,
) -> list[FlightOption]:
    """Enumerate every valid journey from ``origin`` to ``destination``.

    Args:
        conn: Open database connection.
        origin: Departure city.
        destination: Arrival city.
        travel_date: Date of the first departure.
        max_stops: Maximum number of intermediate stops (0 = direct only).
        max_layover_hours: Longest acceptable single layover.

    Returns:
        All journeys found (unsorted).  Empty if the cities are not
        connected within ``max_stops`` stops.
    """
    templates = _load_templates(conn)
    results: list[FlightOption] = []
    min_layover = timedelta(minutes=config.MIN_LAYOVER_MINUTES)
    max_layover = timedelta(hours=max_layover_hours)

    def extend(city: str, ready_at: datetime | None, legs: list[FlightLeg], seen: set[str]) -> None:
        for row in templates.get(city, []):
            if row["destination"] in seen:
                continue
            clock = datetime.fromisoformat(row["departure_time"]).time()
            if ready_at is None:
                departure = datetime.combine(travel_date, clock)
            else:
                departure = _next_departure(clock, ready_at + min_layover)
                if departure - ready_at > max_layover:
                    continue
            leg = FlightLeg(
                flight_id=row["flight_id"],
                airline=row["airline"],
                origin=row["origin"],
                destination=row["destination"],
                departure=departure,
                arrival=departure + timedelta(minutes=row["duration_minutes"]),
                price=row["price"],
            )
            new_legs = legs + [leg]
            if leg.destination == destination:
                results.append(FlightOption(new_legs))
            elif len(new_legs) <= max_stops:
                extend(leg.destination, leg.arrival, new_legs, seen | {leg.destination})

    extend(origin, None, [], {origin})
    return results


def rank_options(options: list[FlightOption], preference: str = "cheapest") -> list[FlightOption]:
    """Sort journeys by ``cheapest``, ``fastest`` or ``balanced``.

    ``balanced`` min-max normalises price and duration (50/50) and adds a
    small penalty per stop.
    """
    if preference not in _PREFERENCES:
        raise ValueError(f"preference must be one of {_PREFERENCES}")
    if not options:
        return []
    if preference == "cheapest":
        return sorted(options, key=lambda o: (o.total_price, o.total_duration_minutes, o.stops))
    if preference == "fastest":
        return sorted(options, key=lambda o: (o.total_duration_minutes, o.total_price, o.stops))

    prices = [o.total_price for o in options]
    times = [o.total_duration_minutes for o in options]

    def norm(value: float, low: float, high: float) -> float:
        return 0.0 if high == low else (value - low) / (high - low)

    def score(option: FlightOption) -> float:
        return (
            0.5 * norm(option.total_price, min(prices), max(prices))
            + 0.5 * norm(option.total_duration_minutes, min(times), max(times))
            + 0.05 * option.stops
        )

    return sorted(options, key=score)


def search_flights(
    conn: sqlite3.Connection,
    origin: str,
    destination: str,
    travel_date: date,
    preference: str = "cheapest",
    max_stops: int = config.MAX_STOPS_HARD_LIMIT,
    top_n: int = 3,
) -> dict:
    """Find, rank and explain the best journeys between two cities.

    Strategy: prefer journeys without overnight layovers; if none exist,
    relax the layover limit.  Direct flights are always listed first in the
    explanation when available.

    Returns:
        ``{"found": bool, "selected": dict, "alternatives": [dict], "why": str,
        "notes": [str], "options_considered": int}``
    """
    notes: list[str] = []
    options = find_flight_options(conn, origin, destination, travel_date, max_stops)
    if not options:
        options = find_flight_options(
            conn, origin, destination, travel_date, max_stops,
            max_layover_hours=config.RELAXED_MAX_LAYOVER_HOURS,
        )
        if options:
            notes.append("Only itineraries with a long (overnight) layover exist for this route.")
    if not options:
        return {
            "found": False,
            "options_considered": 0,
            "why": f"No flight route from {origin} to {destination} within {max_stops} stops.",
            "selected": None,
            "alternatives": [],
            "notes": notes,
        }

    ranked = rank_options(options, preference)
    best = ranked[0]
    direct = [o for o in options if o.stops == 0]
    if not direct:
        notes.append(
            f"No direct {origin}->{destination} flight exists in the dataset; "
            f"a connection with {best.stops} stop(s) is used."
        )
    cheapest = min(options, key=lambda o: o.total_price)
    fastest = min(options, key=lambda o: o.total_duration_minutes)
    why = (
        f"Selected by '{preference}' ranking out of {len(options)} possible journeys: "
        f"{config.CURRENCY}{best.total_price:,}, {best.total_duration_minutes // 60}h "
        f"{best.total_duration_minutes % 60:02d}m, {best.stops} stop(s). "
        f"(Cheapest overall: {config.CURRENCY}{cheapest.total_price:,}; fastest overall: "
        f"{fastest.total_duration_minutes // 60}h {fastest.total_duration_minutes % 60:02d}m.)"
    )
    return {
        "found": True,
        "options_considered": len(options),
        "preference": preference,
        "selected": best.to_dict(),
        "alternatives": [o.to_dict() for o in ranked[1 : top_n + 1]],
        "why": why,
        "notes": notes,
        "_option": best,  # in-process object for the planner; stripped before JSON output
    }
