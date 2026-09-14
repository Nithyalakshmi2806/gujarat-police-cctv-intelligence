# Cham's Module — Database + Watchlist + Alert Engine

Working FastAPI + SQLAlchemy backend covering every task under your
name in the work-division chart. Tested end-to-end (seed → 3 detections
across cameras → auto-alert → journey reconstruction → stats) — all green.

## Run it

```bash
pip install fastapi uvicorn sqlalchemy pydantic
python seed.py              # loads 4 cameras + the 3 representative watchlist plates
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` — interactive Swagger UI, useful both
for your own testing and for Naveen/Nithya-2/Monika to see exactly what
to call.

## Files

| File | Purpose |
|---|---|
| `database.py` | SQLite for now (swap `DATABASE_URL` for Postgres later — zero other changes needed) |
| `models.py` | All tables + relationships (Stage 1) |
| `schemas.py` | Request/response validation |
| `crud.py` | All business logic: camera state, detection ingest, watchlist matching, alert generation, movement history, vehicle search |
| `main.py` | API routes |
| `seed.py` | Loads the demo data from the brief (CAM-07/14/21/32, GJ01AB1234 etc.) |

## Stage → endpoint map

**Stage 1 — Architecture**: `models.py` (5 tables: `cameras`, `camera_logs`, `detections`, `watchlist`, `alerts`, `movement_history`)

**Stage 2 — Camera metadata/status/logs**
- `POST /cameras` — register a camera (code, location, lat/lon, RTSP, etc.)
- `PATCH /cameras/{id}/status` — update online/offline/degraded, auto-logs the event
- `GET /cameras`, `GET /cameras/{id}/logs`

**Stage 3 — Detections**
- `POST /detections` — this is what Nithya-2's AI pipeline calls per plate read. It stores the detection **and** immediately runs the watchlist check + writes a movement-history point, in one call.

**Stage 4 — Watchlist tables + alert logic + CRUD**
- `POST /watchlist`, `PATCH /watchlist/{plate}`, `DELETE /watchlist/{plate}`, `GET /watchlist`
- Matching logic lives in `crud.check_watchlist_and_alert()` — this is your "Detected → Search Watchlist → Match Found → 🚨 Alert" flow from the architecture diagram, called automatically on every detection.

**Stage 5 — Movement history / timeline**
- `GET /vehicles/{plate}/journey` — ordered sightings with timestamps; this is the data behind Monika's "Vehicle Journey" timeline bar.

**Stage 6 — Watchlist categories + alert severity/history/status**
- Categories: `STOLEN`, `WANTED`, `BLACKLISTED` (enum in `models.py`, edit freely)
- Severity auto-assigned by category (`_CATEGORY_SEVERITY` in `crud.py`)
- `PATCH /alerts/{id}/status` — move an alert through `NEW → ACKNOWLEDGED → RESOLVED / FALSE_POSITIVE`
- `GET /alerts?status=&severity=` — history/filtering

**Stage 7 — Location data / route history / GIS feed**
- Same `movement_history` table doubles as the GIS feed: each row has `latitude`/`longitude`, so `GET /vehicles/{plate}/journey` is what you hand Monika to draw the route line on the map.

**Stage 8 — Investigation + search + stats**
- `GET /vehicles/{plate}/search` — the full investigation payload: watchlist status, full journey, every alert, every raw detection. This is Scene 7 of your demo ("Vehicle Number → Camera → Timestamp → Route → Last Seen → Evidence") in one call.
- `GET /alerts/stats` — counts by status/category/severity, for dashboard cards.

## Integration notes for your teammates

- **Nithya-2 (AI/ANPR)** calls `POST /detections` with `camera_id`, `plate_number`, `vehicle_type`, `vehicle_color`, `confidence`. You return whether an alert fired — no extra round trip needed.
- **Naveen (Streaming)** calls `POST /cameras` once per onboarded camera and `PATCH /cameras/{id}/status` on connect/disconnect.
- **Monika (Dashboard/GIS)** consumes `GET /alerts`, `GET /alerts/stats`, `GET /vehicles/{plate}/journey`, `GET /vehicles/{plate}/search` — she shouldn't need anything you haven't already exposed.
- **Nithya (Architecture)** — this service is stateless behind the DB, so it's fine wherever it sits in the final container/deployment layout.

## Before the real demo

1. Point `DATABASE_URL` in `database.py` at Postgres if you want persistence beyond a single run.
2. Add auth (even a shared API key header) before this is reachable from other machines on the LAN — nothing here does auth yet.
3. Load your real representative watchlist via `seed.py` or the `POST /watchlist` endpoint.
