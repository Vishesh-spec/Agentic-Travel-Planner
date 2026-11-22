"""Render a plan as human-readable text (matches the brief's example) or JSON."""

from __future__ import annotations

import json
from datetime import datetime

from .config import CURRENCY


def _money(value: int | float) -> str:
    return f"{CURRENCY}{int(value):,}"


def _hm(minutes: int) -> str:
    return f"{minutes // 60}h {minutes % 60:02d}m"


def _flight_line(flight: dict) -> str:
    dep = datetime.strptime(flight["departure"], "%Y-%m-%d %H:%M")
    stops = "Direct" if flight["stops"] == 0 else f"{flight['stops']} stop(s)"
    return (f"{' + '.join(flight['airlines'])} ({_money(flight['total_price'])}) – "
            f"{flight['route']} – departs {dep.strftime('%d %b %H:%M')}, "
            f"{_hm(flight['total_duration_minutes'])}, {stops}")


def to_json(plan: dict, indent: int = 2) -> str:
    """Serialise the plan as pretty JSON."""
    return json.dumps(plan, indent=indent, ensure_ascii=False, default=str)


def to_text(plan: dict, include_reasoning: bool = True) -> str:
    """Format the plan like the example in the project brief."""
    lines = [plan["title"], "=" * len(plan["title"]), ""]
    req = plan["request"]
    lines += [f"{req['origin']} → {req['destination']} | {req['travellers']} traveller(s) | "
              f"{'round trip' if req['round_trip'] else 'one-way'} | planner: {plan['agent_mode']}", ""]

    flights = plan["flights"]
    lines.append("Flight Selected:")
    if flights["outbound"]:
        lines.append(f"- Outbound: {_flight_line(flights['outbound'])}")
    if flights["return"]:
        lines.append(f"- Return:   {_flight_line(flights['return'])}")
    lines.append("")

    hotel = plan["hotel"]["selected"]
    lines.append("Hotel Recommended:")
    if hotel:
        lines.append(f"- {hotel['name']} ({_money(hotel['price_per_night'])}/night, "
                     f"{hotel['stars']}-star) – {', '.join(hotel['amenities'])}")
    lines.append("")

    lines.append("Weather:")
    if plan["weather"]:
        for index, day in enumerate(plan["weather"]["days"], 1):
            lines.append(f"- Day {index}: {day['condition']} ({day['temp_max_c']:.0f}°C / "
                         f"{day['temp_min_c']:.0f}°C, rain {day['rain_probability_pct']}%)")
        lines.append(f"  (source: {plan['weather']['source']})")
    lines.append("")

    lines.append("Itinerary:")
    for day in plan["itinerary"]:
        names = ", ".join(a["label"] for a in day["activities"]) or day["theme"]
        lines.append(f"Day {day['day']}: {names}")
        for activity in day["activities"]:
            lines.append(f"    {activity['time']}  {activity['label']}  ★{activity['rating']}")
        for note in day["notes"]:
            lines.append(f"    • {note}")
    lines.append("")

    budget = plan["budget"]
    if budget:
        b = budget["breakdown"]
        lines.append("Estimated Total Budget:")
        lines += [f"- Flight: {_money(b['flights'])}",
                  f"- Hotel: {_money(b['hotel'])} ({budget['rooms']} room(s))",
                  f"- Food & Travel: {_money(b['food_and_local_travel'])}",
                  "-------------------------------------",
                  f"Total Cost: {_money(budget['total'])}  ({_money(budget['per_person'])} per person)"]
        if budget["budget_total"]:
            verdict = ("within" if budget["within_budget"] else "OVER")
            lines.append(f"Budget check: {verdict} your {_money(budget['budget_total'])} budget "
                         f"({_money(abs(budget['difference']))} "
                         f"{'to spare' if budget['within_budget'] else 'over'})")
        lines.append("")

    if plan["warnings"]:
        lines.append("Notes:")
        lines += [f"- {w}" for w in dict.fromkeys(plan["warnings"])]
        lines.append("")

    if include_reasoning and plan["reasoning"]:
        lines.append("Why we selected this:")
        lines += [f"- {r}" for r in plan["reasoning"]]
    return "\n".join(lines).rstrip() + "\n"
