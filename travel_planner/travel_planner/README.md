# ✈️ Agentic AI-Based Travel Planning Assistant (LangChain)

An agentic travel planner that understands a trip request, **decides which tools to call**
(flights, hotels, places, weather, itinerary, budget), reasons over the results and returns a
day-wise itinerary with costs and a "why we selected this" justification – as readable text
**and** structured JSON.

* **Domain:** Travel / Tourism  
* **Skills:** Python · LangChain agents · LLM tool-calling · Prompt engineering · API integration (Open-Meteo) · SQLite · Streamlit  
* Full write-up: [`docs/PROJECT_REPORT.pdf`](docs/PROJECT_REPORT.pdf) (source: `docs/PROJECT_REPORT.md`)

## Features

| Brief requirement | Where it is implemented |
|---|---|
| Flight search tool (JSON) – filter source→destination, cheapest / fastest | `travel_agent/flights.py` (direct **and** connecting routes, up to 3 stops) |
| Hotel recommendation tool (JSON) – city, rating, price | `travel_agent/hotels.py` (stars, price cap, amenities, 3 ranking modes) |
| Places discovery tool (JSON) – type & rating | `travel_agent/places.py` |
| Weather tool – free Open-Meteo API | `travel_agent/weather.py` (live + seasonal fallback) |
| Budget estimation tool | `travel_agent/budget.py` (+ automatic cost optimisation) |
| ReAct / tool-calling agent | `travel_agent/agent.py` (`langchain.agents.create_agent`) + offline `RuleBasedAgent` |
| 3–7 day itinerary, weather per day, accommodation, budget | `travel_agent/itinerary.py`, `planner.py` |
| Justification ("Why we selected this?") | every tool returns a `why`; collected in `plan["reasoning"]` |
| Clean JSON + human-readable output | `travel_agent/formatter.py` |
| CLI and Streamlit interface | `main.py`, `app.py` |
| SQL practices (normalised tables, indexes) | `travel_agent/database.py` (3NF SQLite schema, 5 indexes) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (optional) enable the real LLM agent – otherwise the offline rule-based agent is used
cp .env.example .env        # then put ANTHROPIC_API_KEY or OPENAI_API_KEY in it

# Streamlit app
streamlit run app.py

# Command line
python main.py "Plan a 4 day trip from Mumbai to Jaipur starting 2026-12-20 for 2 people under 60000, I like forts and markets"
python main.py --origin Hyderabad --destination Kolkata --days 7 --flight-pref fastest --hotel-pref best_rated --json
python main.py "3 day trip to Goa" --offline          # force the rule-based agent

# Tests
pytest -q
```

## Project structure

```
travel_planner/
├── app.py                  Streamlit UI (filters, tabs, charts, JSON download)
├── main.py                 CLI
├── data/                   flights.json, hotels.json, places.json (provided datasets)
├── travel_agent/
│   ├── config.py           constants, city coordinates, climate normals, spending levels
│   ├── database.py         JSON → normalised SQLite (+ indexes, validation)
│   ├── models.py           dataclasses: TripRequest, FlightOption, Hotel, Place, DayWeather
│   ├── flights.py          route search (DFS with layover rules), ranking
│   ├── hotels.py           filtering + ranking (value score)
│   ├── places.py           interest-aware attraction ranking
│   ├── weather.py          Open-Meteo client + seasonal fallback
│   ├── itinerary.py        day-wise scheduling rules
│   ├── budget.py           cost estimation
│   ├── toolkit.py          the 6 tools + LangChain StructuredTool wrappers + shared state
│   ├── agent.py            LLM factory, system prompt, LangChain agent, graceful fallback
│   ├── planner.py          RuleBasedAgent (ReAct trace) + final plan assembly
│   ├── query_parser.py     free-text → TripRequest (offline mode)
│   └── formatter.py        text / JSON rendering
├── tests/                  50 pytest tests (data, tools, weather mocks, agent loop, CLI, Streamlit)
├── outputs/                sample plans (.txt + .json)
└── docs/                   PROJECT_REPORT.md / .pdf and UI screenshots
```

## How the agent works

```
User request ──► Agent (LLM tool-calling, or rule-based ReAct)
                    │  Thought → Action → Observation (repeat)
                    ├─ flight_search            (outbound + return)
                    ├─ hotel_recommendation
                    ├─ places_discovery
                    ├─ weather_lookup           (Open-Meteo)
                    ├─ itinerary_builder
                    └─ budget_estimation  ─► over budget? retry cheaper flights / hotel
                    ▼
        Structured plan (JSON) + readable text + reasoning + agent trace
```

* **With an API key** the LLM chooses tools and arguments itself. Any step it skips is completed
  automatically, and if the API fails the app falls back to the offline agent.
* **Without a key** the same six tools run in a fixed ReAct-style sequence; the Thought/Action/Observation trace is shown in the UI.

## Important notes about the data

* `flights.json` has 25 **directed** direct routes out of 56 possible city pairs and only fixed 2025 dates. Each flight is
  used as a recurring daily schedule (airline, price, duration, clock time); missing pairs are served with connections
  (e.g. Delhi → Goa = 3 stops). This is reported in the plan's notes.
* In `places.json` the `name` ("Famous Fort") and `type` (`lake`) are independent, so both are displayed: *Famous Fort (lake)*.
  Goa has no `beach` places in the data.
* Weather: live Open-Meteo data is used for trips within the next 15 days. Beyond that window (or if the network fails)
  approximate monthly climate normals are used and clearly labelled `climate-normals`.

## Configuration

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | enables the LLM agent (Anthropic is preferred if both are set) |
| `ANTHROPIC_MODEL`, `OPENAI_MODEL` | model overrides |
| `TRAVEL_DATA_DIR` | folder with custom `flights/hotels/places.json` |

Spending assumptions (₹ per person per day): budget 800 · standard 1,500 · premium 3,000; two travellers share a room
(`travel_agent/config.py`).
