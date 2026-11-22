"""Unit tests for the database, flight/hotel/place tools, budget and weather."""

import json
from datetime import date, datetime

import pytest
import requests

from travel_agent import config, database, flights, hotels, places, weather
from travel_agent.budget import estimate_budget
from travel_agent.itinerary import arrival_capacity, build_itinerary, departure_capacity
from travel_agent.models import Place


# ------------------------------------------------------------------ database
def test_all_records_loaded(conn):
    stats = database.dataset_stats(conn)
    assert stats["flights"] == 30 and stats["hotels"] == 40 and stats["places"] == 40
    assert stats["cities"] == 8


def test_indexes_exist(conn):
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_flights_route", "idx_hotels_search", "idx_places_city_rating"} <= names


def test_missing_file_raises(tmp_path):
    with pytest.raises(database.DataLoadError):
        database.build_database(flights_path=tmp_path / "nope.json")


def test_invalid_record_is_skipped(tmp_path):
    bad = tmp_path / "flights.json"
    bad.write_text(json.dumps([{"flight_id": "X", "airline": "A", "from": "A", "to": "B",
                                "departure_time": "2025-01-01T10:00:00",
                                "arrival_time": "2025-01-01T09:00:00", "price": 1}]))
    conn = database.build_database(flights_path=bad)
    assert conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0] == 0


# ------------------------------------------------------------------ flights
def test_direct_flight_found(conn):
    result = flights.search_flights(conn, "Hyderabad", "Delhi", date(2026, 12, 1))
    assert result["found"]
    assert result["selected"]["stops"] == 0 and result["selected"]["total_price"] == 2907


def test_connection_used_when_no_direct(conn):
    # Delhi only flies to Kolkata in the dataset, so Delhi->Goa needs 3 stops.
    result = flights.search_flights(conn, "Delhi", "Goa", date(2026, 12, 1))
    assert result["found"] and result["selected"]["stops"] == 3
    assert any("No direct" in n for n in result["notes"])


def test_every_city_pair_is_reachable(conn):
    cities = database.list_cities(conn)
    for origin in cities:
        for destination in cities:
            if origin != destination:
                assert flights.search_flights(conn, origin, destination, date(2026, 12, 1))["found"], \
                    (origin, destination)


def test_connection_layover_is_respected(conn):
    result = flights.search_flights(conn, "Delhi", "Goa", date(2026, 12, 1))
    legs = result["_option"].legs
    for prev, nxt in zip(legs, legs[1:]):
        gap = (nxt.departure - prev.arrival).total_seconds() / 60
        assert gap >= config.MIN_LAYOVER_MINUTES


def test_cheapest_vs_fastest(conn):
    cheap = flights.search_flights(conn, "Bangalore", "Kolkata", date(2026, 12, 1), "cheapest")
    fast = flights.search_flights(conn, "Bangalore", "Kolkata", date(2026, 12, 1), "fastest")
    assert cheap["selected"]["total_price"] <= fast["selected"]["total_price"]
    assert fast["selected"]["total_duration_minutes"] <= cheap["selected"]["total_duration_minutes"]


def test_invalid_preference(conn):
    with pytest.raises(ValueError):
        flights.rank_options(
            flights.find_flight_options(conn, "Hyderabad", "Delhi", date(2026, 1, 1)), "weird")


def test_no_route_for_unknown_city(conn):
    result = flights.search_flights(conn, "Atlantis", "Goa", date(2026, 12, 1))
    assert not result["found"]


# ------------------------------------------------------------------ hotels
def test_hotel_filters(conn):
    result = hotels.recommend_hotels(conn, "Goa", min_stars=4, required_amenities=["pool"])
    assert result["found"]
    for hotel in [result["selected"], *result["alternatives"]]:
        assert hotel["stars"] >= 4 and "pool" in hotel["amenities"]


def test_hotel_price_cap_and_cheapest(conn):
    result = hotels.recommend_hotels(conn, "Delhi", max_price=3700, preference="cheapest")
    assert result["selected"]["price_per_night"] == 2743


def test_hotel_no_match(conn):
    result = hotels.recommend_hotels(conn, "Delhi", min_stars=5, max_price=1000)
    assert not result["found"]


def test_best_rated_prefers_stars(conn):
    result = hotels.recommend_hotels(conn, "Mumbai", preference="best_rated")
    assert result["selected"]["stars"] == 5


# ------------------------------------------------------------------ places
def test_places_interest_boost(conn):
    result = places.discover_places(conn, "Delhi", interests=["temple"])
    assert result["places"][0]["category"] == "temple"


def test_places_sorted_by_rating(conn):
    result = places.discover_places(conn, "Kolkata")
    ratings = [p["rating"] for p in result["places"]]
    assert ratings == sorted(ratings, reverse=True)


def test_places_unknown_interest_falls_back(conn):
    result = places.discover_places(conn, "Goa", interests=["beach"])
    assert result["found"] and "No beach" in result["why"]


# ------------------------------------------------------------------ budget
def test_budget_math():
    b = estimate_budget(4800, 3200, nights=2, days=3, travellers=1, spending_level="standard")
    assert b["breakdown"] == {"flights": 4800, "hotel": 6400, "food_and_local_travel": 4500}
    assert b["total"] == 15700


def test_budget_rooms_and_budget_check():
    b = estimate_budget(1000, 1000, 2, 3, travellers=3, spending_level="budget", budget_total=5000)
    assert b["rooms"] == 2 and b["breakdown"]["hotel"] == 4000
    assert b["within_budget"] is False and b["difference"] < 0


def test_budget_validation():
    with pytest.raises(ValueError):
        estimate_budget(1, 1, 1, 1, spending_level="lavish")


# ------------------------------------------------------------------ weather
class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(str(self.status))

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.params = response, error, None

    def get(self, url, params=None, timeout=None):
        self.params = params
        if self.error:
            raise self.error
        return self.response


def test_live_weather_parsed():
    payload = {"daily": {"time": ["2026-11-20", "2026-11-21"],
                         "temperature_2m_max": [31.2, 30.0], "temperature_2m_min": [22.0, 21.5],
                         "precipitation_probability_max": [10, None], "weathercode": [0, 61]}}
    session = FakeSession(FakeResponse(payload))
    result = weather.get_weather("Goa", date(2026, 11, 20), date(2026, 11, 21),
                                 today=date(2026, 11, 19), session=session)
    assert result["source"] == "open-meteo-live"
    assert result["days"][0]["condition"] == "Clear sky"
    assert result["days"][1]["condition"] == "Slight rain" and result["days"][1]["rain_probability_pct"] == 0
    assert session.params["latitude"] == 15.2993


def test_weather_falls_back_on_network_error():
    session = FakeSession(error=requests.ConnectionError("down"))
    result = weather.get_weather("Goa", date(2026, 11, 20), date(2026, 11, 22),
                                 today=date(2026, 11, 19), session=session)
    assert result["source"] == "climate-normals" and "unavailable" in result["note"]
    assert len(result["days"]) == 3


def test_weather_far_future_skips_network():
    session = FakeSession(error=AssertionError("must not be called"))
    result = weather.get_weather("Jaipur", date(2027, 3, 1), date(2027, 3, 3),
                                 today=date(2026, 11, 19), session=session)
    assert result["source"] == "climate-normals"


# ------------------------------------------------------------------ itinerary
def _places(n):
    return [Place(f"P{i}", f"Place {i}", "X", "museum" if i % 2 else "beach", 5 - i / 10)
            for i in range(n)]


def test_capacity_rules():
    assert arrival_capacity(datetime(2026, 1, 1, 9, 0)) == 3
    assert arrival_capacity(datetime(2026, 1, 1, 22, 0)) == 0
    assert departure_capacity(datetime(2026, 1, 1, 7, 0)) == 0
    assert departure_capacity(datetime(2026, 1, 1, 20, 0)) == 3


def test_itinerary_has_every_day_and_no_duplicates():
    plan = build_itinerary(_places(9), 4)
    assert [d["day"] for d in plan] == [1, 2, 3, 4]
    ids = [a["place_id"] for d in plan for a in d["activities"]]
    assert len(ids) == len(set(ids)) == 9


def test_itinerary_leisure_days_when_few_places():
    plan = build_itinerary(_places(2), 5)
    assert sum(1 for d in plan if d["theme"] == "Leisure day") >= 2


def test_late_arrival_day_has_no_sightseeing():
    plan = build_itinerary(_places(6), 3, arrival=datetime(2026, 1, 1, 22, 30))
    assert plan[0]["activities"] == [] and plan[0]["theme"] == "Travel day"


def test_multi_day_journey_marks_transit_days():
    plan = build_itinerary(_places(6), 4, arrival=datetime(2026, 1, 2, 7, 0), start_date=date(2026, 1, 1))
    assert plan[0]["theme"] == "In transit" and plan[0]["activities"] == []
    assert plan[1]["activities"]  # arrival day (07:00) is usable
