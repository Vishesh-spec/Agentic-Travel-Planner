"""Streamlit interface for the Agentic AI Travel Planning Assistant.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from travel_agent import (InvalidTripRequest, TravelToolkit, TripRequest, build_database,
                          build_llm, create_trip_plan, parse_query, to_json, to_text)
from travel_agent import config
from travel_agent.database import (dataset_stats, list_amenities, list_cities,
                                   list_place_types)

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_connection():
    """Load the three JSON datasets into SQLite once per server process."""
    return build_database()


def money(value: int | float) -> str:
    return f"{config.CURRENCY}{int(value):,}"


def flight_rows(option: dict | None) -> list[dict]:
    """Flatten one journey into table rows (one per leg)."""
    if not option:
        return []
    return [{"Flight": leg["flight_id"], "Airline": leg["airline"],
             "From": leg["from"], "To": leg["to"], "Departs": leg["departure"],
             "Arrives": leg["arrival"], "Price": leg["price"]} for leg in option["legs"]]


def sidebar_request(conn) -> tuple[TripRequest, bool, str, bool]:
    """Render the sidebar form; return (request, submitted, free_text)."""
    cities = list_cities(conn)
    st.sidebar.header("Plan your trip")
    free_text = st.sidebar.text_area(
        "Describe your trip (optional)",
        placeholder="Plan a 4 day trip from Delhi to Goa starting 2026-12-12 under 40000 for 2 people",
        help="If filled, this sentence is parsed and the form below is ignored.")
    origin = st.sidebar.selectbox("From", cities, index=cities.index("Delhi"))
    destination = st.sidebar.selectbox("To", [c for c in cities if c != origin],
                                       index=0 if origin != "Goa" else 1)
    start = st.sidebar.date_input("Start date", date.today() + timedelta(days=7),
                                  min_value=date.today())
    days = st.sidebar.slider("Trip length (days)", config.MIN_TRIP_DAYS, config.MAX_TRIP_DAYS, 4)
    travellers = st.sidebar.number_input("Travellers", 1, 10, 2)
    round_trip = st.sidebar.checkbox("Round trip", True)
    budget_on = st.sidebar.checkbox("Set a total budget")
    budget = st.sidebar.number_input("Total budget (₹)", 5000, 500000, 40000, 1000) if budget_on else None

    with st.sidebar.expander("Flight & hotel preferences", expanded=False):
        flight_pref = st.selectbox("Flight priority", ["cheapest", "fastest", "balanced"])
        hotel_pref = st.selectbox("Hotel priority", ["best_value", "best_rated", "cheapest"])
        min_stars = st.slider("Minimum hotel stars", 1, 5, 1)
        cap_on = st.checkbox("Cap hotel price / night")
        max_price = st.slider("Max price per night (₹)", 1000, 7000, 4000, 100) if cap_on else None
        amenities = st.multiselect("Required amenities", list_amenities(conn))
        spending = st.select_slider("Spending level (food + local travel)",
                                    list(config.SPENDING_LEVELS), "standard")
    interests = st.sidebar.multiselect("Interests", list_place_types(conn))
    use_llm = st.sidebar.checkbox("Use LLM agent when an API key is set", True)
    submitted = st.sidebar.button("Plan my trip", type="primary", use_container_width=True)

    request = TripRequest(origin, destination, start, days, int(travellers), round_trip,
                          budget, flight_pref, hotel_pref, min_stars, max_price,
                          amenities, interests, spending)
    return request, submitted, free_text, use_llm


def render_plan(plan: dict) -> None:
    """Show a finished plan in tabs."""
    st.title(plan["title"])
    st.caption(f"Planned by: {plan['agent_mode']}")
    budget = plan["budget"]
    if budget:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total cost", money(budget["total"]))
        c2.metric("Per person", money(budget["per_person"]))
        c3.metric("Flights", money(budget["breakdown"]["flights"]))
        c4.metric("Hotel", money(budget["breakdown"]["hotel"]))
        if budget["budget_total"]:
            (st.success if budget["within_budget"] else st.error)(
                f"{'Within' if budget['within_budget'] else 'Over'} your budget of "
                f"{money(budget['budget_total'])} by {money(abs(budget['difference']))}.")
    for warning in dict.fromkeys(plan["warnings"]):
        st.warning(warning)

    tabs = st.tabs(["Itinerary", "Flights", "Hotel", "Weather", "Budget", "Why this plan?", "JSON"])

    with tabs[0]:
        weather_days = plan["weather"]["days"] if plan["weather"] else []
        for day in plan["itinerary"]:
            weather = weather_days[day["day"] - 1] if day["day"] - 1 < len(weather_days) else None
            with st.expander(f"Day {day['day']} – {day['theme']}", expanded=True):
                if weather:
                    st.caption(f"{weather['date']} · {weather['condition']} · "
                               f"{weather['temp_max_c']:.0f}°C / {weather['temp_min_c']:.0f}°C")
                for activity in day["activities"]:
                    st.markdown(f"**{activity['time']}** – {activity['label']} · ★ {activity['rating']}")
                for note in day["notes"]:
                    st.write(f"• {note}")

    with tabs[1]:
        for label, key in (("Outbound", "outbound"), ("Return", "return")):
            option = plan["flights"][key]
            if option:
                st.subheader(f"{label}: {option['route']}")
                st.write(f"{money(option['total_price'])} · {option['total_duration_minutes'] // 60}h "
                         f"{option['total_duration_minutes'] % 60:02d}m · {option['stops']} stop(s)")
                st.dataframe(pd.DataFrame(flight_rows(option)), hide_index=True, use_container_width=True)
                alts = plan["flights"][f"{key}_alternatives"]
                if alts:
                    st.caption("Alternatives")
                    st.dataframe(pd.DataFrame([{"Route": a["route"], "Price": a["total_price"],
                                                "Minutes": a["total_duration_minutes"],
                                                "Stops": a["stops"]} for a in alts]),
                                 hide_index=True, use_container_width=True)

    with tabs[2]:
        hotel = plan["hotel"]["selected"]
        if hotel:
            st.subheader(f"{hotel['name']} – {hotel['stars']}★")
            st.write(f"{money(hotel['price_per_night'])}/night · {', '.join(hotel['amenities'])}")
            if plan["hotel"]["alternatives"]:
                st.caption("Alternatives")
                st.dataframe(pd.DataFrame(plan["hotel"]["alternatives"]).drop(columns=["hotel_id"]),
                             hide_index=True, use_container_width=True)

    with tabs[3]:
        if plan["weather"]:
            frame = pd.DataFrame(plan["weather"]["days"]).set_index("date")
            st.line_chart(frame[["temp_max_c", "temp_min_c"]])
            st.dataframe(frame, use_container_width=True)
            st.caption(f"Source: {plan['weather']['source']}")

    with tabs[4]:
        if budget:
            frame = pd.Series(budget["breakdown"], name="₹")
            st.bar_chart(frame)
            st.json(budget["assumptions"])

    with tabs[5]:
        for reason in plan["reasoning"]:
            st.write(f"• {reason}")
        if plan.get("agent_answer"):
            st.markdown("**LLM final answer**")
            st.write(plan["agent_answer"])
        with st.expander("Agent trace (Thought → Action → Observation)"):
            for step in plan["agent_trace"]:
                st.markdown(f"**{step['step']}. {step['action']}**")
                st.caption(step["thought"])
                st.code(f"input: {step['action_input']}\nobservation: {step['observation']}")

    with tabs[6]:
        st.download_button("Download JSON", to_json(plan), "trip_plan.json", "application/json")
        st.download_button("Download text", to_text(plan), "trip_plan.txt")
        st.json(plan, expanded=False)


def main() -> None:
    conn = get_connection()
    request, submitted, free_text, use_llm = sidebar_request(conn)
    if submitted:
        try:
            if free_text.strip():
                request = parse_query(free_text)
            request.validate(list_cities(conn))
            with st.spinner("Agent is searching flights, hotels, places and weather..."):
                st.session_state["plan"] = create_trip_plan(
                    request, TravelToolkit(conn), use_llm=use_llm)
        except InvalidTripRequest as exc:
            st.error(str(exc))
        except Exception as exc:  # last-resort guard so the UI never shows a stack trace
            st.error(f"Something went wrong while planning: {exc}")
    if "plan" in st.session_state:
        render_plan(st.session_state["plan"])
    else:
        st.title("✈️ Agentic AI Travel Planner")
        st.write("Pick your trip in the sidebar (or type it in a sentence) and press **Plan my trip**.")
        stats = dataset_stats(conn)
        st.caption(f"Loaded datasets: {stats['flights']} flights, {stats['hotels']} hotels, "
                   f"{stats['places']} places across {stats['cities']} cities. "
                   f"LLM available: {'yes' if build_llm() else 'no (offline rule-based agent)'}.")


main()
