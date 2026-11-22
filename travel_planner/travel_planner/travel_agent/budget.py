"""Budget estimation: flights + hotel + per-day local expenses."""

from __future__ import annotations

import math

from . import config


def rooms_needed(travellers: int) -> int:
    """Rooms required, assuming two people share one room."""
    return math.ceil(travellers / config.PEOPLE_PER_ROOM)


def estimate_budget(
    flight_cost_per_person: int,
    hotel_price_per_night: int,
    nights: int,
    days: int,
    travellers: int = 1,
    spending_level: str = config.DEFAULT_SPENDING_LEVEL,
    budget_total: int | None = None,
) -> dict:
    """Sum all trip costs and compare against an optional budget.

    Args:
        flight_cost_per_person: Total airfare for ONE traveller (all legs,
            outbound plus return if applicable).
        hotel_price_per_night: Price of one room for one night.
        nights: Number of nights.
        days: Number of days (for local expenses).
        travellers: Number of travellers.
        spending_level: ``budget`` | ``standard`` | ``premium``.
        budget_total: Optional maximum the user wants to spend.

    Returns:
        A dict with a line-item breakdown, ``total`` and budget status.
    """
    if spending_level not in config.SPENDING_LEVELS:
        raise ValueError(f"spending_level must be one of {list(config.SPENDING_LEVELS)}")
    if min(travellers, days) < 1 or nights < 0:
        raise ValueError("travellers/days must be >= 1 and nights >= 0")

    rooms = rooms_needed(travellers)
    per_day = config.SPENDING_LEVELS[spending_level]
    flights = flight_cost_per_person * travellers
    hotel = hotel_price_per_night * nights * rooms
    local = per_day * days * travellers
    total = flights + hotel + local

    result = {
        "currency": config.CURRENCY,
        "travellers": travellers,
        "rooms": rooms,
        "breakdown": {
            "flights": flights,
            "hotel": hotel,
            "food_and_local_travel": local,
        },
        "assumptions": {
            "flight_cost_per_person": flight_cost_per_person,
            "hotel_price_per_night": hotel_price_per_night,
            "nights": nights,
            "local_expense_per_person_per_day": per_day,
            "spending_level": spending_level,
        },
        "total": total,
        "per_person": round(total / travellers),
        "budget_total": budget_total,
        "within_budget": None,
        "difference": None,
    }
    if budget_total is not None:
        result["within_budget"] = total <= budget_total
        result["difference"] = budget_total - total
    return result
