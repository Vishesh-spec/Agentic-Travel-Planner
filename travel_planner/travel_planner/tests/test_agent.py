"""Tests for the agent layer: parser, rule-based agent, LangChain agent loop, CLI, UI."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import main as cli
from travel_agent import (InvalidTripRequest, TravelToolkit, TripRequest, create_trip_plan,
                          parse_query, to_json, to_text)

TODAY = date(2026, 11, 19)


# ------------------------------------------------------------------ parser
def test_parse_full_sentence():
    req = parse_query("Plan a 4 day trip from Bengaluru to Goa on 12 Dec under 25k for 3 people, "
                      "I like beaches and markets, need pool and wifi, 4 star, fastest flight", today=TODAY)
    assert (req.origin, req.destination, req.num_days) == ("Bangalore", "Goa", 4)
    assert req.start_date == date(2026, 12, 12) and req.budget_total == 25000
    assert req.travellers == 3 and req.min_hotel_stars == 4
    assert req.interests == ["beach", "market"] and set(req.required_amenities) == {"pool", "wifi"}
    assert req.flight_preference == "fastest"


def test_parse_nights_and_defaults():
    req = parse_query("trip to Jaipur for 3 nights", today=TODAY)
    assert req.num_days == 4 and req.destination == "Jaipur" and req.origin == "Delhi"
    assert req.start_date == date(2026, 11, 26)


def test_parse_without_destination():
    with pytest.raises(InvalidTripRequest):
        parse_query("I want a holiday", today=TODAY)


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("kwargs", [
    dict(origin="Goa", destination="Goa"),
    dict(origin="Delhi", destination="Goa", num_days=2),
    dict(origin="Delhi", destination="Goa", num_days=8),
    dict(origin="Delhi", destination="Goa", travellers=0),
    dict(origin="Delhi", destination="Paris"),
])
def test_invalid_requests(kwargs, conn):
    from travel_agent.database import list_cities
    req = TripRequest(start_date=date(2026, 12, 1), **kwargs)
    with pytest.raises(InvalidTripRequest):
        req.validate(list_cities(conn))


# ------------------------------------------------------------------ rule-based agent
def make_request(**overrides):
    base = dict(origin="Mumbai", destination="Jaipur", start_date=date(2026, 12, 12), num_days=5,
                travellers=2)
    base.update(overrides)
    return TripRequest(**base)


def test_plan_is_complete(toolkit):
    plan = create_trip_plan(make_request(), toolkit, use_llm=False)
    assert plan["flights"]["outbound"] and plan["flights"]["return"]
    assert plan["hotel"]["selected"]
    assert len(plan["itinerary"]) == 5 and len(plan["weather"]["days"]) == 5
    assert plan["title"] == "Your 5-Day Trip to Jaipur (Dec 12–16)"
    assert [s["action"] for s in plan["agent_trace"]][:3] == \
        ["search_flights", "search_flights", "recommend_hotels"]


def test_budget_total_matches_components(toolkit):
    plan = create_trip_plan(make_request(), toolkit, use_llm=False)
    b = plan["budget"]
    assert b["total"] == sum(b["breakdown"].values())
    expected_flights = (plan["flights"]["outbound"]["total_price"]
                        + plan["flights"]["return"]["total_price"]) * 2
    assert b["breakdown"]["flights"] == expected_flights
    nights = 4
    assert b["breakdown"]["hotel"] == plan["hotel"]["selected"]["price_per_night"] * nights * 1


def test_one_way_has_no_return(toolkit):
    plan = create_trip_plan(make_request(round_trip=False), toolkit, use_llm=False)
    assert plan["flights"]["return"] is None


def test_budget_optimisation_switches_to_cheaper(toolkit):
    loose = create_trip_plan(make_request(flight_preference="fastest", hotel_preference="best_rated"),
                             toolkit, use_llm=False)
    tight = create_trip_plan(make_request(flight_preference="fastest", hotel_preference="best_rated",
                                          budget_total=loose["budget"]["total"] - 1),
                             toolkit, use_llm=False)
    assert tight["budget"]["total"] < loose["budget"]["total"]
    assert any("[budget]" in r for r in tight["reasoning"])


def test_hotel_filters_relaxed_when_nothing_matches(toolkit):
    plan = create_trip_plan(make_request(destination="Delhi", min_hotel_stars=5, max_hotel_price=1000),
                            toolkit, use_llm=False)
    assert plan["hotel"]["selected"] is not None
    assert any("relaxed" in r for r in plan["reasoning"])


def test_text_and_json_output(toolkit):
    plan = create_trip_plan(make_request(), toolkit, use_llm=False)
    text = to_text(plan)
    for heading in ("Flight Selected:", "Hotel Recommended:", "Weather:", "Itinerary:",
                    "Estimated Total Budget:", "Total Cost:", "Why we selected this:"):
        assert heading in text
    assert json.loads(to_json(plan))["title"] == plan["title"]


# ------------------------------------------------------------------ LangChain agent loop
class ScriptedToolModel(BaseChatModel):
    """Minimal fake chat model that emits scripted tool calls, then a final answer.

    Used to prove the real ``create_agent`` loop executes our LangChain tools.
    """

    script: list[Any] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.script[min(self.cursor, len(self.script) - 1)]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_langchain_agent_runs_tools(toolkit):
    request = make_request(origin="Hyderabad", destination="Kolkata", num_days=3, travellers=1)
    script = [
        _call("flight_search", {"origin": "Hyderabad", "destination": "Kolkata",
                                "travel_date": "2026-12-12", "leg": "outbound"}, "1"),
        _call("hotel_recommendation", {"city": "Kolkata"}, "2"),
        _call("places_discovery", {"city": "Kolkata"}, "3"),
        AIMessage(content="Here is your Kolkata trip."),
    ]
    plan = create_trip_plan(request, toolkit, llm=ScriptedToolModel(script=script))
    assert plan["agent_answer"] == "Here is your Kolkata trip."
    actions = [s["action"] for s in plan["agent_trace"]]
    assert actions[:3] == ["flight_search", "hotel_recommendation", "places_discovery"]
    # steps the "LLM" skipped (weather, itinerary, return flight, budget) were completed
    assert plan["weather"] and plan["itinerary"] and plan["budget"] and plan["flights"]["return"]
    assert "LangChain tool-calling agent" in plan["agent_mode"]


def test_llm_failure_degrades_to_offline(toolkit):
    class Broken(ScriptedToolModel):
        def _generate(self, *a, **k):
            raise RuntimeError("API down")

    plan = create_trip_plan(make_request(), toolkit, llm=Broken(script=[AIMessage(content="x")]))
    assert plan["itinerary"] and any("LLM agent failed" in w for w in plan["warnings"])


def test_langchain_tools_return_json_errors(toolkit):
    tools = {t.name: t for t in toolkit.as_langchain_tools()}
    assert set(tools) == {"flight_search", "hotel_recommendation", "places_discovery",
                          "weather_lookup", "itinerary_builder", "budget_estimation"}
    out = json.loads(tools["flight_search"].invoke({"origin": "Delhi", "destination": "Goa",
                                                    "travel_date": "not-a-date"}))
    assert "error" in out


# ------------------------------------------------------------------ CLI / UI
def test_cli_json_output(capsys):
    code = cli.main(["Plan 3 day trip from Chennai to Hyderabad starting 2026-12-05", "--offline", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["request"]["destination"] == "Hyderabad"


def test_cli_rejects_bad_input(capsys):
    assert cli.main(["--origin", "Goa", "--destination", "Goa", "--offline"]) == 2
    assert "different" in capsys.readouterr().err


def test_streamlit_app_smoke():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60).run()
    assert not app.exception
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert any("Trip to" in t.value for t in app.title)


def test_slow_outbound_reduces_hotel_nights(toolkit):
    # Bangalore -> Chennai has no direct flight; the cheapest route arrives the next day.
    plan = create_trip_plan(make_request(origin="Bangalore", destination="Chennai", num_days=3,
                                         start_date=date(2026, 12, 12), travellers=1),
                            toolkit, use_llm=False)
    assert plan["itinerary"][0]["theme"] == "In transit"
    assert plan["budget"]["assumptions"]["nights"] == 1
    assert any("sightseeing starts later" in w and "1 hotel night" in w for w in plan["warnings"])
