"""Plan assembly and the rule-based (offline) ReAct agent.

* :func:`assemble_plan` converts the shared :class:`TripState` into the final
  structured plan (identical for the LLM and the rule-based agent).
* :class:`RuleBasedAgent` follows the same Thought -> Action -> Observation
  loop as a ReAct agent, but the "thoughts" come from deterministic rules.
  It is used when no LLM API key is configured, in tests and for the CI
  demo, and it also *completes* a plan if an LLM agent skipped a step.
"""

from __future__ import annotations

import logging
from datetime import date

from . import config
from .models import InvalidTripRequest, TripRequest
from .toolkit import TravelToolkit

logger = logging.getLogger(__name__)


def _title(request: TripRequest) -> str:
    start, end = request.start_date, request.end_date
    same_month = (start.month, start.year) == (end.month, end.year)
    span = (f"{start.strftime('%b')} {start.day}–{end.day}" if same_month
            else f"{start.strftime('%b')} {start.day}–{end.strftime('%b')} {end.day}")
    return f"Your {request.num_days}-Day Trip to {request.destination} ({span})"


def assemble_plan(request: TripRequest, toolkit: TravelToolkit, agent_mode: str,
                  trace: list[dict] | None = None, agent_answer: str | None = None) -> dict:
    """Build the final structured plan from whatever the tools produced.

    Missing pieces are reported in ``warnings`` rather than raising, so the
    UI can still show a partial plan.
    """
    state = toolkit.state
    warnings: list[str] = []

    outbound = state.outbound["selected"] if state.outbound else None
    inbound = state.inbound["selected"] if state.inbound else None
    if not outbound:
        warnings.append("No outbound flight could be found.")
    if request.round_trip and not inbound:
        warnings.append("No return flight found; return fare assumed equal to the outbound fare.")
    for source in (state.outbound, state.inbound):
        if source:
            warnings.extend(source.get("notes", []))
    offset = _arrival_offset(request, toolkit)
    if offset:
        selected = state.outbound["selected"]
        lost = _hotel_nights_lost(request, toolkit)
        message = (f"The outbound journey arrives on {selected['arrival']}, {offset} day(s) after "
                   f"departure, so sightseeing starts later")
        message += (f" and {lost} hotel night(s) are not charged." if lost
                    else "; the hotel is still needed for the first night (arrival after midnight).")
        if selected["stops"]:
            message += " Only slow multi-stop connections exist for this route in the dataset."
        warnings.append(message)
    if state.weather and state.weather.get("note"):
        warnings.append(state.weather["note"])

    plan = {
        "title": _title(request),
        "agent_mode": agent_mode,
        "request": request.to_dict(),
        "flights": {
            "outbound": outbound,
            "outbound_alternatives": state.outbound["alternatives"] if state.outbound else [],
            "return": inbound,
            "return_alternatives": state.inbound["alternatives"] if state.inbound else [],
        },
        "hotel": {
            "selected": state.hotel["selected"] if state.hotel else None,
            "alternatives": state.hotel["alternatives"] if state.hotel else [],
        },
        "weather": ({"source": state.weather["source"], "days": state.weather["days"]}
                    if state.weather else None),
        "attractions": state.places["places"] if state.places else [],
        "itinerary": state.itinerary or [],
        "budget": state.budget,
        "reasoning": list(state.decisions),
        "warnings": warnings,
        "agent_trace": trace or [],
    }
    if agent_answer:
        plan["agent_answer"] = agent_answer
    return plan


def _arrival_offset(request: TripRequest, toolkit: TravelToolkit) -> int:
    """Calendar days between the first departure and the arrival at the destination."""
    if not toolkit.state.outbound:
        return 0
    arrival = toolkit.state.outbound["_option"].arrival
    return max(0, (arrival.date() - request.start_date).days)


def _hotel_nights_lost(request: TripRequest, toolkit: TravelToolkit) -> int:
    """Hotel nights that are not needed because the traveller is still in the air.

    An arrival shortly after midnight (before 06:00) still needs that night's
    room, so it does not reduce the number of nights.
    """
    offset = _arrival_offset(request, toolkit)
    if offset and toolkit.state.outbound["_option"].arrival.hour < 6:
        offset -= 1
    return offset


class RuleBasedAgent:
    """Deterministic ReAct-style agent that drives the six tools."""

    mode = "rule-based agent (no LLM key found)"

    def __init__(self, toolkit: TravelToolkit) -> None:
        self.toolkit = toolkit
        self.trace: list[dict] = []

    # -- helpers -----------------------------------------------------------
    def _act(self, thought: str, action: str, **kwargs) -> dict:
        """Execute one tool call and log the ReAct step."""
        observation = getattr(self.toolkit, action)(**kwargs)
        summary = observation.get("error") or observation.get("why") or "ok"
        self.trace.append({"step": len(self.trace) + 1, "thought": thought,
                           "action": action, "action_input": kwargs,
                           "observation": str(summary)[:300]})
        return observation

    def run(self, request: TripRequest) -> dict:
        """Plan a complete trip for ``request`` and return the plan dict."""
        tk = self.toolkit
        tk.reset()
        self.trace = []
        request.validate()
        dest = request.destination

        out = self._act(
            f"I need a way to get from {request.origin} to {dest} on {request.start_date}; "
            f"preference is '{request.flight_preference}'.",
            "search_flights", origin=request.origin, destination=dest,
            travel_date=request.start_date.isoformat(),
            preference=request.flight_preference, leg="outbound", max_stops=request.max_stops)
        if not out.get("selected"):
            raise InvalidTripRequest(out.get("why") or out.get("error") or "No outbound flight.")

        if request.round_trip:
            self._act("The traveller must get home, so I search the return journey on the last day.",
                      "search_flights", origin=dest, destination=request.origin,
                      travel_date=request.end_date.isoformat(),
                      preference=request.flight_preference, leg="return",
                      max_stops=request.max_stops)

        hotel = self._act(
            f"Next I need accommodation in {dest} for {request.nights} night(s) matching the filters.",
            "recommend_hotels", city=dest, min_stars=request.min_hotel_stars,
            max_price_per_night=request.max_hotel_price, amenities=request.required_amenities,
            preference=request.hotel_preference)
        if not hotel.get("selected"):
            hotel = self._relax_hotel_filters(request)
        if not hotel.get("selected"):
            raise InvalidTripRequest(f"No hotel available in {dest}.")

        self._act("Now I look for attractions that fit the traveller's interests.",
                  "discover_places", city=dest, interests=request.interests, limit=12)
        self._act("Weather affects which attractions go on which day, so I fetch it before planning.",
                  "get_weather", city=dest, start_date=request.start_date.isoformat(),
                  end_date=request.end_date.isoformat())
        self._act("I have flights, weather and places; I can build the day-wise plan now.",
                  "build_itinerary", num_days=request.num_days)

        budget = self._estimate(request)
        if request.budget_total and not budget["within_budget"]:
            budget = self._optimise_for_budget(request, budget)

        plan = assemble_plan(request, tk, self.mode, self.trace)
        return plan

    # -- budget ------------------------------------------------------------
    def _flight_cost_per_person(self, request: TripRequest) -> int:
        state = self.toolkit.state
        out = state.outbound["selected"]["total_price"]
        if not request.round_trip:
            return out
        back = state.inbound["selected"]["total_price"] if state.inbound else out
        return out + back

    def _hotel_nights(self, request: TripRequest) -> int:
        """Nights actually spent in the hotel (shorter if the journey takes days)."""
        return max(0, request.nights - _hotel_nights_lost(request, self.toolkit))

    def _estimate(self, request: TripRequest) -> dict:
        state = self.toolkit.state
        return self._act(
            "Finally I add up flights, hotel and local expenses and compare with the budget.",
            "estimate_budget",
            flight_cost_per_person=self._flight_cost_per_person(request),
            hotel_price_per_night=state.hotel["selected"]["price_per_night"],
            nights=self._hotel_nights(request), days=request.num_days, travellers=request.travellers,
            spending_level=request.spending_level, budget_total=request.budget_total)

    def _optimise_for_budget(self, request: TripRequest, budget: dict) -> dict:
        """Try cheaper flights, then cheaper hotels, until the budget fits."""
        tk = self.toolkit
        tk.state.decisions.append(
            f"[budget] Total {config.CURRENCY}{budget['total']:,} exceeds the budget of "
            f"{config.CURRENCY}{request.budget_total:,}; trying cheaper options.")
        dest = request.destination
        if request.flight_preference != "cheapest":
            self._act("Over budget: switch to the cheapest flights.", "search_flights",
                      origin=request.origin, destination=dest,
                      travel_date=request.start_date.isoformat(), preference="cheapest",
                      leg="outbound", max_stops=request.max_stops)
            if request.round_trip:
                self._act("Same for the return journey.", "search_flights", origin=dest,
                          destination=request.origin, travel_date=request.end_date.isoformat(),
                          preference="cheapest", leg="return", max_stops=request.max_stops)
            budget = self._estimate(request)
        if not budget["within_budget"] and request.hotel_preference != "cheapest":
            self._act("Still over budget: pick the cheapest hotel that meets the filters.",
                      "recommend_hotels", city=dest, min_stars=request.min_hotel_stars,
                      max_price_per_night=request.max_hotel_price,
                      amenities=request.required_amenities, preference="cheapest")
            budget = self._estimate(request)
        if budget["within_budget"]:
            tk.state.decisions.append(
                f"[budget] Optimised plan fits: {config.CURRENCY}{budget['total']:,} "
                f"({config.CURRENCY}{budget['difference']:,} to spare).")
        else:
            tk.state.decisions.append(
                f"[budget] Even the cheapest options cost {config.CURRENCY}{budget['total']:,}, "
                f"{config.CURRENCY}{-budget['difference']:,} over budget. Consider fewer days, "
                f"a lower spending level or a different destination.")
        return budget

    def _relax_hotel_filters(self, request: TripRequest) -> dict:
        """No hotel matched: drop amenity requirement, then the price cap, then stars."""
        for label, kwargs in (
            ("amenity requirements", dict(amenities=[], max_price_per_night=request.max_hotel_price,
                                          min_stars=request.min_hotel_stars)),
            ("price cap", dict(amenities=[], max_price_per_night=None,
                               min_stars=request.min_hotel_stars)),
            ("minimum stars", dict(amenities=[], max_price_per_night=None, min_stars=1)),
        ):
            result = self._act(f"No hotel matched; relaxing the {label}.", "recommend_hotels",
                               city=request.destination, preference=request.hotel_preference, **kwargs)
            if result.get("selected"):
                self.toolkit.state.decisions.append(
                    f"[hotel] Original filters had no match, so the {label} was relaxed.")
                return result
        return {}

    def complete_missing(self, request: TripRequest) -> None:
        """Fill any step an LLM agent skipped so the plan is always complete."""
        s = self.toolkit.state
        dest = request.destination
        if not s.outbound:
            self._act("Outbound flight missing - searching.", "search_flights",
                      origin=request.origin, destination=dest,
                      travel_date=request.start_date.isoformat(),
                      preference=request.flight_preference, leg="outbound")
        if request.round_trip and not s.inbound:
            self._act("Return flight missing - searching.", "search_flights", origin=dest,
                      destination=request.origin, travel_date=request.end_date.isoformat(),
                      preference=request.flight_preference, leg="return")
        if not s.hotel:
            self._act("Hotel missing - searching.", "recommend_hotels", city=dest,
                      min_stars=request.min_hotel_stars, amenities=request.required_amenities,
                      preference=request.hotel_preference)
            if not self.toolkit.state.hotel:
                self._relax_hotel_filters(request)
        if not s.places:
            self._act("Places missing - searching.", "discover_places", city=dest,
                      interests=request.interests)
        if not s.weather:
            self._act("Weather missing - fetching.", "get_weather", city=dest,
                      start_date=request.start_date.isoformat(), end_date=request.end_date.isoformat())
        if not s.itinerary and s.places:
            self._act("Itinerary missing - building.", "build_itinerary", num_days=request.num_days)
        if s.outbound and s.hotel:
            self._estimate(request)
