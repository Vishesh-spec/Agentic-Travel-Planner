"""Natural-language query parser (used by the rule-based agent).

When an LLM is available it understands the request itself; this lightweight
regex parser gives the offline agent the same ability for typical phrasings
such as::

    "Plan a 3 day trip from Delhi to Goa starting 12 Dec under 25000 for 2 people,
     I like beaches and want a pool"
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from . import config
from .models import InvalidTripRequest, TripRequest

_CITY_ALIASES = {
    "bengaluru": "Bangalore", "bangalore": "Bangalore", "bombay": "Mumbai",
    "mumbai": "Mumbai", "delhi": "Delhi", "new delhi": "Delhi", "goa": "Goa",
    "chennai": "Chennai", "madras": "Chennai", "hyderabad": "Hyderabad",
    "kolkata": "Kolkata", "calcutta": "Kolkata", "jaipur": "Jaipur",
}
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_INTEREST_WORDS = {
    "beach": "beach", "beaches": "beach", "temple": "temple", "temples": "temple",
    "museum": "museum", "museums": "museum", "fort": "fort", "forts": "fort",
    "market": "market", "markets": "market", "shopping": "market", "park": "park",
    "parks": "park", "lake": "lake", "lakes": "lake", "monument": "monument",
    "monuments": "monument", "heritage": "monument", "history": "museum",
}
_AMENITY_WORDS = ["wifi", "pool", "gym", "spa", "parking", "breakfast"]


def _find_cities(text: str) -> tuple[str | None, str | None]:
    """Extract origin/destination from 'from X to Y' or 'X to Y' or 'trip to Y'."""
    names = "|".join(sorted(_CITY_ALIASES, key=len, reverse=True))
    match = re.search(rf"from\s+({names})\s+to\s+({names})", text) or \
        re.search(rf"\b({names})\s+(?:to|->|→)\s+({names})", text)
    if match:
        return _CITY_ALIASES[match.group(1)], _CITY_ALIASES[match.group(2)]
    dest = re.search(rf"\bto\s+({names})", text)
    orig = re.search(rf"\bfrom\s+({names})", text)
    return (_CITY_ALIASES[orig.group(1)] if orig else None,
            _CITY_ALIASES[dest.group(1)] if dest else None)


def _find_date(text: str, today: date) -> date | None:
    """Understand ISO dates, '12 Dec', 'Dec 12', 'tomorrow', 'next week'."""
    iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    months = "|".join(_MONTHS)
    day_first = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})[a-z]*(?:\s+(\d{{4}}))?", text)
    month_first = re.search(rf"\b({months})[a-z]*\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?", text)
    found = day_first and (int(day_first.group(1)), day_first.group(2), day_first.group(3)) or \
        month_first and (int(month_first.group(2)), month_first.group(1), month_first.group(3))
    if found:
        day, month, year = found
        candidate = date(int(year) if year else today.year, _MONTHS[month[:3]], day)
        if not year and candidate < today:
            candidate = candidate.replace(year=today.year + 1)
        return candidate
    if "tomorrow" in text:
        return today + timedelta(days=1)
    if "next week" in text:
        return today + timedelta(days=7)
    return None


def parse_query(text: str, origin_default: str = "Delhi", today: date | None = None) -> TripRequest:
    """Turn a free-text request into a :class:`TripRequest`.

    Missing details fall back to sensible defaults (3 days, starting a week
    from today, cheapest flight, best-value hotel).

    Raises:
        InvalidTripRequest: if no destination can be found.
    """
    today = today or date.today()
    lowered = text.lower()
    origin, destination = _find_cities(lowered)
    if not destination:
        raise InvalidTripRequest(
            "I could not find a destination. Try: 'Plan a 4 day trip from Delhi to Goa'.")
    origin = origin or origin_default

    days = None
    if m := re.search(r"(\d+)\s*[- ]?\s*day", lowered):
        days = int(m.group(1))
    elif m := re.search(r"(\d+)\s*[- ]?\s*night", lowered):
        days = int(m.group(1)) + 1
    days = days or config.MIN_TRIP_DAYS

    budget = None
    if m := re.search(r"(?:under|below|within|budget(?:\s+of)?|upto|up to|max(?:imum)?)\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(k|lakh|lac)?", lowered):
        budget = float(m.group(1).replace(",", ""))
        budget *= {"k": 1_000, "lakh": 100_000, "lac": 100_000}.get(m.group(2) or "", 1)
        budget = int(budget)

    travellers = 1
    if m := re.search(r"for\s+(\d+)\s*(?:people|persons|travell?ers|adults|pax)", lowered) or \
            re.search(r"(\d+)\s*(?:people|persons|travell?ers|adults|pax)", lowered):
        travellers = int(m.group(1))
    elif re.search(r"\b(couple|honeymoon)\b", lowered):
        travellers = 2

    stars = 1
    if m := re.search(r"(\d)\s*[- ]?\s*star", lowered):
        stars = max(1, min(5, int(m.group(1))))

    flight_pref = "fastest" if re.search(r"fastest|quickest|shortest", lowered) else \
        "balanced" if "balanced" in lowered else "cheapest"
    hotel_pref = "best_rated" if re.search(r"best[- ]rated|luxury|top[- ]rated", lowered) else \
        "cheapest" if re.search(r"cheapest hotel|budget hotel", lowered) else "best_value"
    level = "premium" if re.search(r"luxury|premium", lowered) else \
        "budget" if re.search(r"backpack|budget trip|low cost|cheap", lowered) else "standard"

    interests = sorted({canon for word, canon in _INTEREST_WORDS.items()
                        if re.search(rf"\b{word}\b", lowered)})
    amenities = [a for a in _AMENITY_WORDS if re.search(rf"\b{a}\b", lowered)]

    return TripRequest(
        origin=origin, destination=destination,
        start_date=_find_date(lowered, today) or today + timedelta(days=7),
        num_days=days, travellers=travellers, budget_total=budget,
        flight_preference=flight_pref, hotel_preference=hotel_pref,
        min_hotel_stars=stars, required_amenities=amenities, interests=interests,
        spending_level=level,
    )
