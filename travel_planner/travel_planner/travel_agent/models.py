"""Typed data models shared by every module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from . import config


class InvalidTripRequest(ValueError):
    """Raised when a trip request violates a business rule."""


@dataclass
class TripRequest:
    """Everything the user can ask for when planning a trip."""

    origin: str
    destination: str
    start_date: date
    num_days: int = 3
    travellers: int = 1
    round_trip: bool = True
    budget_total: int | None = None
    flight_preference: str = "cheapest"      # cheapest | fastest | balanced
    hotel_preference: str = "best_value"     # best_value | best_rated | cheapest
    min_hotel_stars: int = 1
    max_hotel_price: int | None = None
    required_amenities: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    spending_level: str = config.DEFAULT_SPENDING_LEVEL
    max_stops: int = config.MAX_STOPS_HARD_LIMIT

    @property
    def end_date(self) -> date:
        """Last day of the trip (inclusive)."""
        return self.start_date + timedelta(days=self.num_days - 1)

    @property
    def nights(self) -> int:
        """Number of hotel nights (a 3-day trip has 2 nights)."""
        return self.num_days - 1

    def validate(self, known_cities: list[str] | None = None) -> None:
        """Check business rules and raise :class:`InvalidTripRequest`."""
        if known_cities is not None:
            for label, city in (("origin", self.origin), ("destination", self.destination)):
                if city not in known_cities:
                    raise InvalidTripRequest(
                        f"Unknown {label} city '{city}'. Available: {', '.join(known_cities)}"
                    )
        if self.origin == self.destination:
            raise InvalidTripRequest("Origin and destination must be different cities.")
        if not config.MIN_TRIP_DAYS <= self.num_days <= config.MAX_TRIP_DAYS:
            raise InvalidTripRequest(
                f"Trip length must be {config.MIN_TRIP_DAYS}-{config.MAX_TRIP_DAYS} days."
            )
        if self.travellers < 1:
            raise InvalidTripRequest("At least one traveller is required.")
        if self.flight_preference not in {"cheapest", "fastest", "balanced"}:
            raise InvalidTripRequest("flight_preference must be cheapest, fastest or balanced.")
        if self.hotel_preference not in {"best_value", "best_rated", "cheapest"}:
            raise InvalidTripRequest("hotel_preference must be best_value, best_rated or cheapest.")
        if self.spending_level not in config.SPENDING_LEVELS:
            raise InvalidTripRequest(
                f"spending_level must be one of {', '.join(config.SPENDING_LEVELS)}."
            )
        if not 1 <= self.min_hotel_stars <= 5:
            raise InvalidTripRequest("min_hotel_stars must be between 1 and 5.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        data["end_date"] = self.end_date.isoformat()
        return data


@dataclass
class FlightLeg:
    """A single scheduled flight (one row of ``flights.json``)."""

    flight_id: str
    airline: str
    origin: str
    destination: str
    departure: datetime
    arrival: datetime
    price: int

    @property
    def duration_minutes(self) -> int:
        return int((self.arrival - self.departure).total_seconds() // 60)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flight_id": self.flight_id,
            "airline": self.airline,
            "from": self.origin,
            "to": self.destination,
            "departure": self.departure.strftime("%Y-%m-%d %H:%M"),
            "arrival": self.arrival.strftime("%Y-%m-%d %H:%M"),
            "duration_minutes": self.duration_minutes,
            "price": self.price,
        }


@dataclass
class FlightOption:
    """A complete one-way journey made of one or more legs."""

    legs: list[FlightLeg]

    @property
    def origin(self) -> str:
        return self.legs[0].origin

    @property
    def destination(self) -> str:
        return self.legs[-1].destination

    @property
    def departure(self) -> datetime:
        return self.legs[0].departure

    @property
    def arrival(self) -> datetime:
        return self.legs[-1].arrival

    @property
    def stops(self) -> int:
        return len(self.legs) - 1

    @property
    def total_price(self) -> int:
        return sum(leg.price for leg in self.legs)

    @property
    def total_duration_minutes(self) -> int:
        """Door-to-door time including layovers."""
        return int((self.arrival - self.departure).total_seconds() // 60)

    @property
    def max_layover_minutes(self) -> int:
        gaps = [
            int((nxt.departure - prev.arrival).total_seconds() // 60)
            for prev, nxt in zip(self.legs, self.legs[1:])
        ]
        return max(gaps, default=0)

    @property
    def is_overnight(self) -> bool:
        """True if any layover is longer than 12 hours."""
        return self.max_layover_minutes > 12 * 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": " -> ".join([self.legs[0].origin] + [leg.destination for leg in self.legs]),
            "airlines": sorted({leg.airline for leg in self.legs}),
            "stops": self.stops,
            "departure": self.departure.strftime("%Y-%m-%d %H:%M"),
            "arrival": self.arrival.strftime("%Y-%m-%d %H:%M"),
            "total_duration_minutes": self.total_duration_minutes,
            "max_layover_minutes": self.max_layover_minutes,
            "total_price": self.total_price,
            "legs": [leg.to_dict() for leg in self.legs],
        }


@dataclass
class Hotel:
    """A hotel row with its amenities."""

    hotel_id: str
    name: str
    city: str
    stars: int
    price_per_night: int
    amenities: list[str]

    @property
    def value_score(self) -> float:
        """Quality per rupee: (stars + 0.25 * amenities) per Rs. 1,000 / night."""
        return (self.stars + 0.25 * len(self.amenities)) / max(self.price_per_night / 1000, 0.1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotel_id": self.hotel_id,
            "name": self.name,
            "city": self.city,
            "stars": self.stars,
            "price_per_night": self.price_per_night,
            "amenities": sorted(self.amenities),
            "value_score": round(self.value_score, 2),
        }


@dataclass
class Place:
    """A point of interest."""

    place_id: str
    name: str
    city: str
    category: str
    rating: float

    @property
    def label(self) -> str:
        """Display label.

        The dataset's ``name`` and ``type`` are independent random fields
        (e.g. "Famous Fort" of type ``lake``), so we show both.
        """
        return f"{self.name} ({self.category})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "place_id": self.place_id,
            "name": self.name,
            "category": self.category,
            "rating": self.rating,
            "label": self.label,
        }


@dataclass
class DayWeather:
    """Weather expectation for one calendar day."""

    day: date
    condition: str
    temp_max_c: float
    temp_min_c: float
    rain_probability: int
    source: str  # "open-meteo-live" | "climate-normals"

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.day.isoformat(),
            "condition": self.condition,
            "temp_max_c": round(self.temp_max_c, 1),
            "temp_min_c": round(self.temp_min_c, 1),
            "rain_probability_pct": self.rain_probability,
            "source": self.source,
        }
