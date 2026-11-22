"""Central configuration for the Agentic AI Travel Planning Assistant.

All tunable constants (paths, city coordinates, spending levels, climate
normals, ranking weights) live here so that the rest of the code base stays
free of magic numbers.
"""

from __future__ import annotations

import os
from pathlib import Path



def _load_dotenv(path: Path) -> None:
    """Tiny .env reader (KEY=VALUE lines) so no extra dependency is needed."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TRAVEL_DATA_DIR", PROJECT_ROOT / "data"))
FLIGHTS_JSON = DATA_DIR / "flights.json"
HOTELS_JSON = DATA_DIR / "hotels.json"
PLACES_JSON = DATA_DIR / "places.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------
# Trip constraints (from the project brief: 3-7 day itineraries)
# --------------------------------------------------------------------------
MIN_TRIP_DAYS = 3
MAX_TRIP_DAYS = 7
CURRENCY = "₹"

# --------------------------------------------------------------------------
# Flight routing
# --------------------------------------------------------------------------
MIN_LAYOVER_MINUTES = 90          # minimum safe connection time
PREFERRED_MAX_LAYOVER_HOURS = 12  # avoid overnight layovers when possible
RELAXED_MAX_LAYOVER_HOURS = 36    # used only if nothing else exists
MAX_STOPS_HARD_LIMIT = 3          # the dataset needs up to 3 stops (e.g. Delhi->Goa)

# --------------------------------------------------------------------------
# Local spending (food + local transport + entry tickets) per person per day
# --------------------------------------------------------------------------
SPENDING_LEVELS = {
    "budget": 800,
    "standard": 1500,
    "premium": 3000,
}
DEFAULT_SPENDING_LEVEL = "standard"
PEOPLE_PER_ROOM = 2

# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT_SECONDS = 8
OPEN_METEO_MAX_FORECAST_DAYS = 15  # Open-Meteo serves ~16 days ahead

# Latitude / longitude of every city present in the datasets.
CITY_COORDINATES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Goa": (15.2993, 74.1240),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Jaipur": (26.9124, 75.7873),
}

# Approximate monthly climate normals (Jan..Dec) used ONLY as an offline
# fallback when the live Open-Meteo call is unavailable or the trip is too
# far in the future for a real forecast.
#   tmax / tmin : average daily max / min temperature in deg C
#   rain        : approximate chance (%) that a given day is rainy
CLIMATE_NORMALS = {
    "Delhi": {
        "tmax": [21, 24, 30, 37, 41, 40, 35, 34, 34, 33, 28, 23],
        "tmin": [8, 11, 16, 22, 26, 28, 27, 26, 24, 18, 12, 8],
        "rain": [10, 10, 8, 8, 10, 25, 60, 65, 40, 8, 3, 6],
    },
    "Mumbai": {
        "tmax": [31, 32, 33, 33, 33, 31, 29, 29, 30, 33, 34, 32],
        "tmin": [19, 20, 23, 26, 28, 27, 26, 25, 25, 25, 23, 21],
        "rain": [2, 2, 2, 3, 10, 70, 90, 85, 65, 25, 8, 2],
    },
    "Goa": {
        "tmax": [32, 32, 33, 33, 33, 30, 29, 29, 30, 32, 33, 33],
        "tmin": [21, 22, 25, 27, 28, 26, 25, 25, 25, 25, 24, 22],
        "rain": [1, 1, 2, 5, 20, 80, 95, 90, 70, 30, 10, 3],
    },
    "Bangalore": {
        "tmax": [28, 31, 34, 35, 34, 30, 28, 28, 28, 28, 27, 27],
        "tmin": [15, 16, 19, 21, 21, 20, 19, 19, 19, 19, 17, 15],
        "rain": [3, 4, 6, 20, 35, 40, 45, 50, 55, 50, 30, 10],
    },
    "Chennai": {
        "tmax": [29, 31, 33, 35, 38, 37, 35, 34, 34, 32, 29, 28],
        "tmin": [21, 22, 24, 27, 28, 27, 26, 26, 25, 24, 23, 21],
        "rain": [10, 5, 4, 5, 10, 15, 25, 30, 30, 55, 65, 40],
    },
    "Hyderabad": {
        "tmax": [29, 32, 35, 38, 39, 34, 31, 30, 31, 31, 29, 28],
        "tmin": [15, 17, 21, 24, 26, 23, 22, 21, 22, 21, 17, 14],
        "rain": [3, 4, 5, 8, 15, 40, 55, 60, 50, 25, 8, 3],
    },
    "Kolkata": {
        "tmax": [27, 30, 34, 36, 36, 34, 32, 32, 32, 32, 29, 26],
        "tmin": [13, 16, 21, 25, 26, 27, 26, 26, 26, 24, 18, 13],
        "rain": [5, 10, 15, 25, 40, 70, 85, 85, 75, 35, 8, 3],
    },
    "Jaipur": {
        "tmax": [22, 25, 31, 37, 41, 40, 35, 33, 34, 33, 28, 23],
        "tmin": [8, 11, 16, 22, 27, 28, 26, 25, 24, 19, 13, 9],
        "rain": [5, 6, 5, 5, 8, 25, 60, 65, 35, 5, 2, 4],
    },
}

# WMO weather interpretation codes -> human readable text (Open-Meteo docs).
WMO_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AGENT_RECURSION_LIMIT = 40
