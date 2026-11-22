"""Weather lookup via the free Open-Meteo API with an offline fallback."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import requests

from . import config
from .models import DayWeather

logger = logging.getLogger(__name__)


class WeatherUnavailable(RuntimeError):
    """Raised internally when the live API cannot serve a request."""


def _condition_from_rain(rain_probability: int) -> str:
    """Translate a rain chance into a short condition label."""
    if rain_probability >= 70:
        return "Rainy"
    if rain_probability >= 40:
        return "Showers likely"
    if rain_probability >= 20:
        return "Partly cloudy, chance of rain"
    return "Mostly sunny"


def climate_normal_forecast(city: str, start: date, end: date) -> list[DayWeather]:
    """Estimate weather from monthly climate normals (offline fallback)."""
    normals = config.CLIMATE_NORMALS.get(city)
    if normals is None:
        raise WeatherUnavailable(f"No climate data for {city}")
    days = []
    for offset in range((end - start).days + 1):
        day = start + timedelta(days=offset)
        month = day.month - 1
        rain = normals["rain"][month]
        days.append(DayWeather(day, _condition_from_rain(rain), normals["tmax"][month],
                               normals["tmin"][month], rain, "climate-normals"))
    return days


def fetch_open_meteo(city: str, start: date, end: date, session: requests.Session | None = None) -> list[DayWeather]:
    """Call the Open-Meteo forecast API (no API key needed).

    Raises:
        WeatherUnavailable: on network failure, bad status or malformed data.
    """
    if city not in config.CITY_COORDINATES:
        raise WeatherUnavailable(f"Unknown coordinates for {city}")
    latitude, longitude = config.CITY_COORDINATES[city]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    http = session or requests
    try:
        response = http.get(config.OPEN_METEO_URL, params=params,
                            timeout=config.OPEN_METEO_TIMEOUT_SECONDS)
        response.raise_for_status()
        daily = response.json()["daily"]
        result = []
        for i, iso_day in enumerate(daily["time"]):
            code = daily["weathercode"][i]
            rain = daily["precipitation_probability_max"][i]
            result.append(DayWeather(
                day=date.fromisoformat(iso_day),
                condition=config.WMO_WEATHER_CODES.get(code, f"Weather code {code}"),
                temp_max_c=float(daily["temperature_2m_max"][i]),
                temp_min_c=float(daily["temperature_2m_min"][i]),
                rain_probability=int(rain) if rain is not None else 0,
                source="open-meteo-live",
            ))
        if not result:
            raise WeatherUnavailable("Empty response from Open-Meteo")
        return result
    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        raise WeatherUnavailable(f"Open-Meteo request failed: {exc}") from exc


def get_weather(
    city: str,
    start: date,
    end: date,
    today: date | None = None,
    allow_live: bool = True,
    session: requests.Session | None = None,
) -> dict:
    """Weather for every day of the trip.

    Live data is requested only when the trip falls inside Open-Meteo's
    forecast window; otherwise (or on any error) climate normals are used
    and the result says so explicitly in ``source`` / ``note``.

    Returns:
        ``{"city", "source", "note", "days": [dict], "_days": [DayWeather]}``
    """
    today = today or date.today()
    in_window = today <= start and end <= today + timedelta(days=config.OPEN_METEO_MAX_FORECAST_DAYS)
    note = ""
    days: list[DayWeather] | None = None

    if allow_live and in_window:
        try:
            days = fetch_open_meteo(city, start, end, session)
        except WeatherUnavailable as exc:
            logger.warning("%s - falling back to climate normals", exc)
            note = "Live forecast unavailable (network/API error); showing seasonal averages."
    elif allow_live:
        note = ("Trip dates are outside the 15-day live forecast window; "
                "showing typical seasonal weather for that month.")

    if days is None:
        days = climate_normal_forecast(city, start, end)
        note = note or "Live lookup disabled; showing typical seasonal weather."

    return {
        "city": city,
        "source": days[0].source,
        "note": note,
        "days": [d.to_dict() for d in days],
        "_days": days,
    }
