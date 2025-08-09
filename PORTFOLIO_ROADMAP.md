# Multi-Agent Transit Simulation – Portfolio Upgrade Roadmap

## Objectives
- Make the simulation realistic on real street networks with accurate travel times.
- Showcase AI-driven decision-making (optimization, forecasting, LLM-assisted operations).
- Improve architecture, reliability, and observability so it scales and is demo-ready.

## High‑impact upgrades (summary)
- Real maps and routing: build the road graph from OpenStreetMap and compute routes and ETAs with a local engine (OSRM/Valhalla/GraphHopper). Use GTFS for stops, lines, and schedules.
- Optimization and AI: OR‑Tools for dynamic dispatch and re‑routing, ML for demand/ETA prediction, RL for control policies, LLM for natural‑language control center.
- Simulation realism: discrete‑event engine, stochastic demand, traffic incidents, headway control strategies.
- Architecture: typed message schemas, persistent state store, FastAPI control API, containerized services.
- Dashboard: live map with buses and incidents, KPIs, scenario runner, chat copilot.

---

## Real‑world maps and routing
- OpenStreetMap ingestion and graph building
  - OSMnx (build and analyze city networks; exports to NetworkX) – local, open source.
  - NetworkX (graph ops and simple routing) – good for topology; travel times best from a routing engine.
- Routing engines (pick one; all open source and free to self‑host)
  - OSRM: very fast, Docker images available, profiles for car/bus; supports table (distance matrix), route, match, nearest.
  - Valhalla: supports multimodal routing (driving, walking), time‑dependent costing, map‑matching.
  - GraphHopper: flexible profiles, good docs, also offers commercial cloud; open source self‑host possible.
  - OpenTripPlanner: focuses on public transit; consumes GTFS + OSM to produce timetable‑aware routes and transfers.
- Data sources
  - GTFS static: stops, routes, trips, stop times. Use to place stations and lines realistically.
  - OSM extracts: Geofabrik or BBBike for city‑level maps.
- Map visualization
  - pydeck/Deck.gl in Streamlit for real‑time animated layers (buses, routes, incidents, heatmaps).
  - Folium + streamlit‑folium for simpler overlays.
  - Kepler.gl (via pydeck) for interactive layers and playback.
- Geocoding and metadata
  - Nominatim (OSM) for address ↔ coordinates; respect usage policy or self‑host.

Recommendation: start with OSRM in Docker + OSMnx to build station network from OSM/GTFS; query OSRM for route polyline and travel time matrices.

## AI and decision‑making
- Optimization (deterministic, explainable)
  - Google OR‑Tools: Vehicle Routing Problem with Time Windows (VRPTW) for dispatching, assignment, and re‑routing. Use OSRM/Valhalla matrices as costs.
  - PuLP or Pyomo for custom LP/MIP (e.g., driver shifts, depot constraints).
- Prediction
  - Demand forecasting at stations: Prophet, statsmodels, sktime, LightGBM; optionally online learning with river.
  - ETA prediction: gradient boosting (XGBoost/LightGBM/CatBoost) with features from routing engine + congestion state.
  - Incident risk models: anomaly detection with PyOD over edge speeds; or Bayesian updates.
- Reinforcement Learning (advanced)
  - Ray RLlib or Stable‑Baselines3 for headway control, holding strategies, and re‑routing under uncertainty.
  - PettingZoo for multi‑agent RL environment wrappers around your SPADE sim.
- LLM‑assisted control
  - Local LLM runtime (Ollama or llama.cpp) to avoid API keys; use function/tool calling to request: “recompute routes for line X due to incident Y”, “summarize system status”, “explain why Bus 12 is delayed”.
  - Retrieval‑augmented generation: index current system state (Redis/SQLite/Parquet) for natural language Q&A.
  - Natural language scenario builder: “simulate heavy rain from 7–9am and reduce speeds by 20%”.

Recommendation: combine OR‑Tools for reliability with a small LLM (via Ollama) for operator UX and explainability.

## Simulation realism
- Discrete‑event simulation
  - Replace sleep‑based clock with SimPy; model buses, dwell times, signals, and incidents as events. Much faster and more scalable.
- Traffic and incidents
  - Import historical speeds (if available) or synthesize with time‑of‑day profiles; dynamically adjust edge costs.
  - Incident generator with severities and durations; integrate with routing engine by updating weights/costing.
- Passenger demand
  - Stochastic arrival processes (Poisson with time‑varying rate) per station and line; couple to forecast models.
- Operations strategies
  - Headway‑based control, stop‑skipping, holding, short‑turns; evaluate KPIs (on‑time performance, average wait, load factor).

## Architecture and data
- Messaging and schemas
  - Keep SPADE/XMPP; introduce strict typed message schemas (Pydantic models) and explicit performatives.
  - Heartbeats, retries, and idempotent commands for reliability; correlation IDs for traceability.
- Persistence and state
  - Replace CSV with SQLite or Postgres (or TimescaleDB) for states and events; or Redis for fast pub/sub and caches.
  - Parquet for historical logs to enable offline analytics.
- Services and APIs
  - FastAPI microservice exposing control endpoints (assign routes, inject incidents, snapshot state); dashboard consumes it.
  - Background workers for routing queries and optimization jobs.
- Packaging and deployment
  - Docker Compose stack: openfire, simulator, routing (OSRM/Valhalla), API, dashboard, (optional) DB.
  - Configuration via Hydra or Pydantic Settings; environment variables in .env (no secrets in repo).
- Observability
  - Structured logging (JSON) and metrics (prometheus_client) with Grafana dashboards.
  - Tracing with OpenTelemetry for end‑to‑end message flows (optional but impressive).

## Dashboard (portfolio‑friendly)
- Live map
  - pydeck layers for buses (color by delay), routes, stops, and incident icons; tooltips with ETAs and loads.
  - Time‑scrubber playback of a full day scenario.
- KPIs and operations
  - Network‑level: on‑time performance, average wait, max load, missed connections, re‑route count.
  - Per‑line and per‑station drill‑downs; compare scenarios A/B.
- Scenario runner
  - UI to load GTFS city, pick date/time window, add incidents, toggle control strategies.
- Chat copilot
  - Panel in Streamlit to ask: “Why is Line 4 off schedule?”, “What if we add one bus at 8am?”, “Show bottlenecks now”.

## Engineering polish
- Testing and quality
  - Unit/integration tests with pytest; contract tests for message schemas; simulation determinism via fixed seeds.
  - Static analysis and formatting: mypy, ruff, black; pre‑commit hooks.
- CI/CD
  - GitHub Actions: run tests, lint, build Docker images, publish docs.
- Documentation
  - System architecture and sequence diagrams; scenario playbook; how to run locally with Docker Compose.
  - Dataset lineage and licenses (OSM ODbL, GTFS licenses vary).

## Recommended rebuild path (phased)
1) Foundations
   - Migrate to SimPy time; define Pydantic message schemas; move state to SQLite; containerize Openfire and simulator.
2) Maps and routing
   - Add OSRM container with a small city extract; integrate for routes and matrices; render live map with pydeck.
3) Operations logic
   - Add OR‑Tools dispatcher for route assignment/re‑routing; implement headway control; incident engine tied to routing.
4) Prediction
   - Add demand and ETA models; feed forecasts into dispatcher; record metrics for evaluation.
5) LLM copilot
   - Run a local LLM via Ollama; add natural‑language controls and explanations with function calls into the API.
6) Polish and demo
   - KPIs, scenario runner, A/B compare, documentation, one‑click Docker Compose demo.

## Stretch ideas
- Multimodal trips (walk + bus) with OpenTripPlanner or Valhalla multimodal costing.
- Real GTFS‑Realtime ingestion to react to live feed (if a city provides it), or simulate GTFS‑RT from your engine.
- Offline what‑if optimizer that proposes timetable changes for a given fleet size and demand profile.
- Federated multi‑city benchmark: switch between cities by swapping OSM/GTFS archives.

## Tooling shortlist (all open source or free to self‑host)
- Maps/routing: OSMnx, OSRM, Valhalla, GraphHopper, OpenTripPlanner, NetworkX.
- Optimization: OR‑Tools, PuLP, Pyomo.
- ML: LightGBM/XGBoost/CatBoost, Prophet/statsmodels/sktime, river (online), PyOD (anomaly).
- RL: Stable‑Baselines3, Ray RLlib, PettingZoo.
- LLM: Ollama, llama.cpp; LangChain/LlamaIndex for tool/RAG orchestration (optional).
- Sim: SimPy; SUMO/CityFlow for traffic‑level realism (advanced alt path).
- Backend/infra: FastAPI, Redis, SQLite/Postgres, Prometheus + Grafana, OpenTelemetry.
- UI: Streamlit with pydeck/folium; or Dash/Panel if preferred.
