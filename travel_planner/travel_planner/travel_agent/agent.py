"""LangChain agent wiring.

``create_trip_plan`` is the single entry point used by the CLI and the
Streamlit app:

* If an ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` is present (or an LLM is
  injected) a real **LangChain tool-calling ReAct agent**
  (``langchain.agents.create_agent``) decides which tools to call, in which
  order and with which arguments.
* Otherwise the deterministic :class:`~travel_agent.planner.RuleBasedAgent`
  performs the same tool sequence.

In both cases the final structured plan is assembled from the tool results,
and any step the LLM skipped is completed automatically.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.agents import create_agent

from . import config
from .models import TripRequest
from .planner import RuleBasedAgent, assemble_plan
from .toolkit import TravelToolkit

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI travel agent for Indian domestic trips.
Plan the trip by CALLING TOOLS - never invent flights, hotels, places or prices.

Workflow (ReAct): think about what information is missing, call a tool, read the
observation, then decide the next step.
1. flight_search for the outbound journey (leg='outbound') and, for round trips, the
   return journey on the last day (leg='return').
2. hotel_recommendation for the destination city.
3. places_discovery for the destination (use the traveller's interests).
4. weather_lookup for the trip dates.
5. itinerary_builder with the trip length (after places and weather).
6. budget_estimation: flight_cost_per_person = outbound + return fare; if the plan is
   over the user's budget, retry cheaper flights/hotels and justify the change.

Dates are YYYY-MM-DD. A 3-day trip has 2 nights. Prices are in Indian rupees.
Final answer format (concise, human readable):
  Trip title, Flight Selected (outbound/return), Hotel, Weather per day, Day-wise
  Itinerary, Estimated Total Budget (flight, hotel, food & travel, total) and a short
  'Why we selected this' section that justifies flight, hotel and activity choices.
If a tool returns an error, fix the arguments and retry once."""


def build_llm() -> Any | None:
    """Return a chat model if credentials are configured, else ``None``."""
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=config.ANTHROPIC_MODEL, temperature=0, timeout=60)
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0, timeout=60)
    except Exception as exc:  # missing optional dependency, bad config, ...
        logger.warning("Could not initialise LLM: %s", exc)
    return None


def _request_to_prompt(request: TripRequest) -> str:
    """Describe the structured request to the LLM agent."""
    return (
        f"Plan a {request.num_days}-day trip from {request.origin} to {request.destination}, "
        f"{request.start_date.isoformat()} to {request.end_date.isoformat()} "
        f"({request.nights} nights), {request.travellers} traveller(s), "
        f"{'round trip' if request.round_trip else 'one-way'}. "
        f"Flight preference: {request.flight_preference}. Hotel preference: {request.hotel_preference}, "
        f"min {request.min_hotel_stars} stars, max price/night {request.max_hotel_price or 'none'}, "
        f"required amenities: {request.required_amenities or 'none'}. "
        f"Interests: {request.interests or 'none'}. Spending level: {request.spending_level}. "
        f"Total budget: {request.budget_total or 'not specified'}."
    )


def run_llm_agent(request: TripRequest, toolkit: TravelToolkit, llm: Any) -> dict:
    """Let a LangChain tool-calling agent plan the trip."""
    toolkit.reset()
    agent = create_agent(model=llm, tools=toolkit.as_langchain_tools(), system_prompt=SYSTEM_PROMPT)
    result = agent.invoke({"messages": [{"role": "user", "content": _request_to_prompt(request)}]},
                          config={"recursion_limit": config.AGENT_RECURSION_LIMIT})

    trace, answer = [], ""
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            trace.append({"step": len(trace) + 1, "thought": "(chosen by LLM)",
                          "action": call["name"], "action_input": call["args"], "observation": ""})
        if message.type == "tool" and trace:
            observation = message.content if isinstance(message.content, str) else str(message.content)
            for entry in reversed(trace):
                if not entry["observation"]:
                    entry["observation"] = observation[:300]
                    break
        if message.type == "ai":
            text = message.content if isinstance(message.content, str) else " ".join(
                block.get("text", "") for block in message.content if isinstance(block, dict))
            if text.strip():
                answer = text

    helper = RuleBasedAgent(toolkit)
    helper.trace = trace
    helper.complete_missing(request)  # safety net: guarantees a complete plan
    return assemble_plan(request, toolkit, f"LangChain tool-calling agent ({type(llm).__name__})",
                         helper.trace, answer)


def create_trip_plan(request: TripRequest, toolkit: TravelToolkit, llm: Any | None = None,
                     use_llm: bool = True) -> dict:
    """Plan a trip with the best available agent.

    Args:
        request: Validated trip request.
        toolkit: Toolkit bound to the database.
        llm: Optional pre-built chat model (for tests); defaults to ``build_llm()``.
        use_llm: Set ``False`` to force the offline rule-based agent.

    Returns:
        The structured plan dictionary (see :func:`assemble_plan`).
    """
    request.validate()
    model = (llm or build_llm()) if use_llm else None
    if model is not None:
        try:
            return run_llm_agent(request, toolkit, model)
        except Exception as exc:  # API/network/limit errors -> degrade gracefully
            logger.error("LLM agent failed (%s); using rule-based agent.", exc)
            plan = RuleBasedAgent(toolkit).run(request)
            plan["warnings"].append(f"LLM agent failed ({type(exc).__name__}); used offline agent.")
            return plan
    return RuleBasedAgent(toolkit).run(request)
