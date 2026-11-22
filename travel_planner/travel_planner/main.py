#!/usr/bin/env python3
"""Command-line interface for the Travel Planning Assistant.

Examples::

    python main.py "Plan a 4 day trip from Delhi to Goa starting 2026-12-12 under 40000"
    python main.py --origin Mumbai --destination Jaipur --days 5 --travellers 2 --json
    python main.py            # interactive prompt
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from travel_agent import (InvalidTripRequest, TravelToolkit, TripRequest, build_database,
                          create_trip_plan, parse_query, to_json, to_text)
from travel_agent import config
from travel_agent.database import DataLoadError, list_cities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic AI travel planner (LangChain)")
    parser.add_argument("query", nargs="?", help="Free-text request, e.g. 'Plan 3 days in Goa from Delhi'")
    parser.add_argument("--origin", default=None)
    parser.add_argument("--destination", default=None)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD (default: 7 days from today)")
    parser.add_argument("--days", type=int, default=None, help="3-7")
    parser.add_argument("--travellers", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None, help="Total budget in INR")
    parser.add_argument("--flight-pref", choices=["cheapest", "fastest", "balanced"], default=None)
    parser.add_argument("--hotel-pref", choices=["best_value", "best_rated", "cheapest"], default=None)
    parser.add_argument("--min-stars", type=int, default=None)
    parser.add_argument("--amenities", nargs="*", default=None)
    parser.add_argument("--interests", nargs="*", default=None)
    parser.add_argument("--spending", choices=list(config.SPENDING_LEVELS), default=None)
    parser.add_argument("--one-way", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Force the rule-based agent (no LLM)")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    parser.add_argument("--save", metavar="FILE", help="Also write the JSON plan to FILE")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def request_from_args(args: argparse.Namespace) -> TripRequest:
    """Merge free text (if any) with explicit flags; flags win."""
    if args.query:
        request = parse_query(args.query)
    elif args.origin and args.destination:
        request = TripRequest(args.origin, args.destination, date.today() + timedelta(days=7))
    else:
        text = input("Where do you want to go? (e.g. 'Plan a 4 day trip from Delhi to Goa'): ")
        request = parse_query(text)
    overrides = {
        "origin": args.origin, "destination": args.destination,
        "start_date": date.fromisoformat(args.start_date) if args.start_date else None,
        "num_days": args.days, "travellers": args.travellers, "budget_total": args.budget,
        "flight_preference": args.flight_pref, "hotel_preference": args.hotel_pref,
        "min_hotel_stars": args.min_stars, "required_amenities": args.amenities,
        "interests": args.interests, "spending_level": args.spending,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(request, key, value)
    if args.one_way:
        request.round_trip = False
    return request


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        conn = build_database()
        request = request_from_args(args)
        request.validate(list_cities(conn))
        plan = create_trip_plan(request, TravelToolkit(conn), use_llm=not args.offline)
    except (InvalidTripRequest, DataLoadError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(to_json(plan) if args.json else to_text(plan))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(to_json(plan), encoding="utf-8")
        print(f"[saved JSON plan to {args.save}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
