"""Day-wise itinerary construction.

Rules implemented (deterministic, easy to test):

* The number of attractions on the first day depends on the arrival time of
  the outbound journey, and on the last day on the departure time of the
  return journey (late arrival / early departure => fewer activities).
* Top-rated places are spread across days (round-robin) so every active day
  gets a highlight.
* Within a day, sightseeing (museum/fort/temple/monument) is scheduled for
  the morning and relaxed places (market/park/lake/beach) later.
* Rainy days (rain probability >= 60 %) are re-ordered so that indoor
  categories (museum, temple, market) are preferred.
* If there are fewer attractions than days, remaining days become flexible
  "leisure days" and the best-rated place is suggested for a revisit.
"""

from __future__ import annotations

from datetime import date, datetime

from .models import DayWeather, Place

MAX_PLACES_PER_DAY = 3
_TIME_SLOTS = {1: ["10:30"], 2: ["10:00", "16:00"], 3: ["09:30", "13:30", "17:00"]}
_MORNING_FIRST = {"museum": 0, "fort": 0, "temple": 0, "monument": 0,
                  "market": 1, "park": 2, "lake": 2, "beach": 3}
_INDOOR = {"museum", "temple", "market"}
RAIN_THRESHOLD = 60


def arrival_capacity(arrival: datetime | None) -> int:
    """Sightseeing slots available on arrival day."""
    if arrival is None:
        return MAX_PLACES_PER_DAY
    hour = arrival.hour + arrival.minute / 60
    if hour <= 11:
        return 3
    if hour <= 15:
        return 2
    if hour <= 19:
        return 1
    return 0


def departure_capacity(departure: datetime | None) -> int:
    """Sightseeing slots available on departure day."""
    if departure is None:
        return MAX_PLACES_PER_DAY
    hour = departure.hour + departure.minute / 60
    if hour >= 18:
        return 3
    if hour >= 14:
        return 2
    if hour >= 10:
        return 1
    return 0


def arrival_day_offset(arrival: datetime | None, start_date: date | None) -> int:
    """Whole days spent travelling before reaching the destination (0 = same day)."""
    if arrival is None or start_date is None:
        return 0
    return max(0, (arrival.date() - start_date).days)


def _day_capacities(num_days: int, arrival: datetime | None, departure: datetime | None,
                    offset: int = 0) -> list[int]:
    caps = [MAX_PLACES_PER_DAY] * num_days
    for day in range(min(offset, num_days)):
        caps[day] = 0                       # still in transit
    if offset < num_days:
        caps[offset] = min(caps[offset], arrival_capacity(arrival))
    caps[-1] = min(caps[-1], departure_capacity(departure))
    return caps


def _order_for_day(places: list[Place], rainy: bool) -> list[Place]:
    """Order a day's places morning -> evening; indoor first when raining."""
    def key(place: Place) -> tuple:
        outdoor_penalty = 0 if (not rainy or place.category in _INDOOR) else 1
        return (outdoor_penalty, _MORNING_FIRST.get(place.category, 2), -place.rating)
    return sorted(places, key=key)


def build_itinerary(
    places: list[Place],
    num_days: int,
    weather: list[DayWeather] | None = None,
    arrival: datetime | None = None,
    departure: datetime | None = None,
    start_date: date | None = None,
) -> list[dict]:
    """Assign ranked ``places`` to ``num_days`` days.

    Args:
        places: Attractions ordered best-first (as returned by discovery).
        num_days: Trip length in days.
        weather: Optional per-day weather aligned with the days.
        arrival: Arrival time at the destination on day 1.
        departure: Departure time of the return journey on the last day.
        start_date: Date of the first departure; used to detect journeys that
            arrive on a later day (those days become "In transit").

    Returns:
        One dict per day: ``{"day", "theme", "activities": [{"time", "place_id",
        "name", "category", "rating"}], "notes": [str]}``.
    """
    offset = arrival_day_offset(arrival, start_date)
    caps = _day_capacities(num_days, arrival, departure, offset)
    buckets: list[list[Place]] = [[] for _ in range(num_days)]
    pending = list(places)

    # Round-robin: best places go to different days first.
    while pending and any(len(buckets[d]) < caps[d] for d in range(num_days)):
        progressed = False
        for day in range(num_days):
            if pending and len(buckets[day]) < caps[day]:
                buckets[day].append(pending.pop(0))
                progressed = True
        if not progressed:
            break

    top_place = places[0] if places else None
    itinerary = []
    for day in range(num_days):
        day_weather = weather[day] if weather and day < len(weather) else None
        rainy = bool(day_weather and day_weather.rain_probability >= RAIN_THRESHOLD)
        ordered = _order_for_day(buckets[day], rainy)
        slots = _TIME_SLOTS.get(len(ordered), [])
        activities = [
            {"time": slot, "place_id": p.place_id, "name": p.name,
             "category": p.category, "rating": p.rating, "label": p.label}
            for slot, p in zip(slots, ordered)
        ]
        notes: list[str] = []
        if day < offset:
            notes.append("In transit - the journey continues today.")
        if day == offset:
            if arrival:
                notes.append(f"Arrive {arrival.strftime('%H:%M')} and check in to the hotel.")
            if caps[day] == 0:
                notes.append("Late arrival - rest and enjoy a local dinner.")
        if day == num_days - 1:
            if departure:
                notes.append(f"Check out and head to the airport (flight at {departure.strftime('%H:%M')}).")
            if caps[day] == 0:
                notes.append("Early departure - no sightseeing planned.")
        if rainy and day >= offset:
            notes.append("High chance of rain - indoor places scheduled first; carry an umbrella.")

        if activities:
            theme = " & ".join(sorted({a["category"].title() for a in activities}))
        elif day < offset:
            theme = "In transit"
        elif caps[day] == 0:
            theme = "Travel day"
        else:
            theme = "Leisure day"
            if top_place:
                notes.append(f"Flexible day: relax, shop locally, or revisit {top_place.label}.")
        itinerary.append({"day": day + 1, "theme": theme, "activities": activities, "notes": notes})
    return itinerary
