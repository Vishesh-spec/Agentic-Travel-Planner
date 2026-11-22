"""Tool layer: the six tools the agent can call.

:class:`TravelToolkit` owns the database connection and a mutable
:class:`TripState`.  Every public method is a *tool* that:

1. takes simple, JSON-friendly arguments (so an LLM can call it),
2. returns a JSON-serialisable ``dict``,
3. records its result in the shared state, so that the final
   :class:`~travel_agent.planner.TripPlan` can be assembled no matter whether
   a real LLM or the rule-based agent drove the calls.

``TravelToolkit.as_langchain_tools()`` wraps the methods as LangChain
``StructuredTool`` objects with validated argument schemas.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from . import config, flights, hotels, places, weather
from .budget import estimate_budget as compute_budget
from .itinerary import build_itinerary as compute_itinerary
from .models import DayWeather, Place

logger = logging.getLogger(__name__)


@dataclass
class TripState:
    """Everything the tools have discovered so far in one planning session."""

    outbound: dict | None = None
    inbound: dict | None = None
    hotel: dict | None = None
    places: dict | None = None
    weather: dict | None = None
    itinerary: list[dict] | None = None
    budget: dict | None = None
    decisions: list[str] = field(default_factory=list)


def _clean(payload: dict) -> dict:
    """Drop private in-process objects (keys starting with ``_``)."""
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Date must be in YYYY-MM-DD format, got '{value}'") from exc


class TravelToolkit:
    """Collection of agent tools sharing one database and one trip state."""

    def __init__(self, conn: sqlite3.Connection, allow_live_weather: bool = True,
                 today: date | None = None) -> None:
        self.conn = conn
        self.allow_live_weather = allow_live_weather
        self.today = today
        self.state = TripState()

    def reset(self) -> None:
        """Forget previous results before planning a new trip."""
        self.state = TripState()

    # ------------------------------------------------------------------ flights
    def search_flights(self, origin: str, destination: str, travel_date: str,
                       preference: str = "cheapest", leg: str = "outbound",
                       max_stops: int = config.MAX_STOPS_HARD_LIMIT) -> dict:
        """Search flights (direct or connecting) and pick the best one."""
        if leg not in {"outbound", "return"}:
            return {"error": "leg must be 'outbound' or 'return'"}
        try:
            result = flights.search_flights(self.conn, origin, destination,
                                            _parse_date(travel_date), preference, max_stops)
        except ValueError as exc:
            return {"error": str(exc)}
        if result["found"]:
            setattr(self.state, "outbound" if leg == "outbound" else "inbound", result)
            self.state.decisions.append(f"[{leg} flight] {result['why']}")
        return _clean(result)

    # ------------------------------------------------------------------ hotels
    def recommend_hotels(self, city: str, min_stars: int = 1,
                         max_price_per_night: int | None = None,
                         amenities: list[str] | None = None,
                         preference: str = "best_value") -> dict:
        """Filter hotels by city/stars/price/amenities and rank them."""
        try:
            result = hotels.recommend_hotels(self.conn, city, min_stars, max_price_per_night,
                                             amenities, preference)
        except ValueError as exc:
            return {"error": str(exc)}
        if result["found"]:
            self.state.hotel = result
            self.state.decisions.append(f"[hotel] {result['why']}")
        return _clean(result)

    # ------------------------------------------------------------------ places
    def discover_places(self, city: str, interests: list[str] | None = None,
                        min_rating: float = 0.0, limit: int = 12) -> dict:
        """Recommend attractions by category and rating."""
        result = places.discover_places(self.conn, city, interests, min_rating, limit)
        if result["found"]:
            self.state.places = result
            self.state.decisions.append(f"[places] {result['why']}")
        return _clean(result)

    # ------------------------------------------------------------------ weather
    def get_weather(self, city: str, start_date: str, end_date: str) -> dict:
        """Weather for each trip day (Open-Meteo live, climate-normal fallback)."""
        try:
            result = weather.get_weather(city, _parse_date(start_date), _parse_date(end_date),
                                         today=self.today, allow_live=self.allow_live_weather)
        except (ValueError, weather.WeatherUnavailable) as exc:
            return {"error": str(exc)}
        self.state.weather = result
        self.state.decisions.append(f"[weather] source={result['source']}. {result['note']}".strip())
        return _clean(result)

    # ------------------------------------------------------------------ itinerary
    def build_itinerary(self, num_days: int, place_ids: list[str] | None = None) -> dict:
        """Create the day-wise plan from discovered places, weather and flight times.

        Call after ``discover_places`` (and ideally after flights + weather so
        that arrival/departure times and rain are taken into account).
        """
        if not self.state.places:
            return {"error": "Call discover_places first."}
        pool: list[Place] = list(self.state.places["_places"])
        if place_ids:
            by_id = {p.place_id: p for p in pool}
            pool = [by_id[i] for i in place_ids if i in by_id] or pool
        arrival = departure = start = None
        if self.state.outbound:
            option = self.state.outbound["_option"]
            arrival, start = option.arrival, option.departure.date()
        if self.state.inbound:
            departure = self.state.inbound["_option"].departure
        days: list[DayWeather] | None = self.state.weather["_days"] if self.state.weather else None
        try:
            itinerary = compute_itinerary(pool, int(num_days), days, arrival, departure, start)
        except (ValueError, TypeError) as exc:
            return {"error": str(exc)}
        self.state.itinerary = itinerary
        return {"itinerary": itinerary}

    # ------------------------------------------------------------------ budget
    def estimate_budget(self, flight_cost_per_person: int, hotel_price_per_night: int,
                        nights: int, days: int, travellers: int = 1,
                        spending_level: str = config.DEFAULT_SPENDING_LEVEL,
                        budget_total: int | None = None) -> dict:
        """Sum flights + hotel + per-day local expenses."""
        try:
            result = compute_budget(flight_cost_per_person, hotel_price_per_night, nights,
                                    days, travellers, spending_level, budget_total)
        except ValueError as exc:
            return {"error": str(exc)}
        self.state.budget = result
        return result

    # ------------------------------------------------------------------ LangChain
    def as_langchain_tools(self) -> list[StructuredTool]:
        """Expose the six tools to LangChain with pydantic-validated schemas."""

        def as_json(func):
            def wrapper(**kwargs: Any) -> str:
                try:
                    return json.dumps(func(**kwargs), ensure_ascii=False, default=str)
                except Exception as exc:  # tool errors must reach the LLM, not crash the loop
                    logger.exception("Tool failed")
                    return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            return wrapper

        class FlightArgs(BaseModel):
            origin: str = Field(description="Departure city, e.g. 'Delhi'")
            destination: str = Field(description="Arrival city, e.g. 'Goa'")
            travel_date: str = Field(description="Travel date YYYY-MM-DD")
            preference: str = Field("cheapest", description="cheapest | fastest | balanced")
            leg: str = Field("outbound", description="'outbound' or 'return' (stores the result)")
            max_stops: int = Field(3, description="Max intermediate stops (0-3)")

        class HotelArgs(BaseModel):
            city: str = Field(description="City to stay in")
            min_stars: int = Field(1, description="Minimum star rating 1-5")
            max_price_per_night: int | None = Field(None, description="Max price per night in INR")
            amenities: list[str] = Field(default_factory=list,
                                         description="Required amenities: wifi, pool, gym, spa, parking, breakfast")
            preference: str = Field("best_value", description="best_value | best_rated | cheapest")

        class PlaceArgs(BaseModel):
            city: str = Field(description="City to explore")
            interests: list[str] = Field(default_factory=list,
                                         description="Categories: park, market, museum, temple, lake, fort, beach, monument")
            min_rating: float = Field(0.0, description="Minimum rating 0-5")
            limit: int = Field(12, description="Max places to return")

        class WeatherArgs(BaseModel):
            city: str = Field(description="City name")
            start_date: str = Field(description="First day YYYY-MM-DD")
            end_date: str = Field(description="Last day YYYY-MM-DD")

        class ItineraryArgs(BaseModel):
            num_days: int = Field(description="Trip length in days (3-7)")
            place_ids: list[str] = Field(default_factory=list,
                                         description="Optional ordered place_ids to use (default: all discovered)")

        class BudgetArgs(BaseModel):
            flight_cost_per_person: int = Field(description="Total airfare per person (outbound + return)")
            hotel_price_per_night: int = Field(description="Price of one room per night")
            nights: int = Field(description="Number of nights (= days - 1)")
            days: int = Field(description="Number of days")
            travellers: int = Field(1, description="Number of travellers")
            spending_level: str = Field("standard", description="budget | standard | premium")
            budget_total: int | None = Field(None, description="User's maximum total budget in INR")

        specs = [
            ("flight_search", self.search_flights, FlightArgs,
             "Search flights between two cities (direct or with connections) from the flights "
             "dataset, rank by cheapest/fastest/balanced and return the best option plus alternatives."),
            ("hotel_recommendation", self.recommend_hotels, HotelArgs,
             "Filter hotels in a city by stars, price and amenities and rank them (best_value, best_rated, cheapest)."),
            ("places_discovery", self.discover_places, PlaceArgs,
             "Recommend attractions in a city by category and rating."),
            ("weather_lookup", self.get_weather, WeatherArgs,
             "Get the day-by-day weather forecast for the trip dates (Open-Meteo API)."),
            ("itinerary_builder", self.build_itinerary, ItineraryArgs,
             "Build the day-wise itinerary from discovered places, arrival/departure times and weather. "
             "Call after places_discovery."),
            ("budget_estimation", self.estimate_budget, BudgetArgs,
             "Estimate the total cost: flights + hotel + per-day local expenses; compares against the user's budget."),
        ]
        return [
            StructuredTool.from_function(func=as_json(fn), name=name, description=desc,
                                         args_schema=schema)
            for name, fn, schema, desc in specs
        ]
