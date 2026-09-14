"""
Cham's backend service — Database + Watchlist + Alert Engine.

Run:
    pip install fastapi uvicorn sqlalchemy pydantic
    uvicorn main:app --reload --port 8000

Interactive API docs (great for the demo / for teammates integrating
against you): http://localhost:8000/docs
"""

from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
import crud
from database import engine, get_db

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Unified CCTV Platform - Database/Watchlist/Alert Service")


# =============================== Stage 2: Cameras ===============================

@app.post("/cameras", response_model=schemas.CameraOut)
def register_camera(camera: schemas.CameraCreate, db: Session = Depends(get_db)):
    return crud.create_camera(db, camera)


@app.patch("/cameras/{camera_id}/status", response_model=schemas.CameraOut)
def set_camera_status(
    camera_id: int, update: schemas.CameraStatusUpdate, db: Session = Depends(get_db)
):
    camera = crud.update_camera_status(db, camera_id, update)
    if not camera:
        raise HTTPException(404, "Camera not found")
    return camera


@app.get("/cameras", response_model=List[schemas.CameraOut])
def get_cameras(db: Session = Depends(get_db)):
    return crud.list_cameras(db)


@app.get("/cameras/{camera_id}/logs")
def camera_logs(camera_id: int, db: Session = Depends(get_db)):
    return crud.get_camera_logs(db, camera_id)


# ============================= Stage 3: Detections =============================

@app.post("/detections")
def ingest_detection(detection: schemas.DetectionCreate, db: Session = Depends(get_db)):
    """
    Called by Nithya-2's AI/ANPR pipeline for every vehicle+plate it reads.
    `camera_id` in the request body is Sentinel's camera_code (e.g. "CAM-07"),
    resolved to our internal camera automatically — register the camera via
    POST /cameras first (Naveen's side) or this returns a 404.

    Every detection is stored regardless of confidence (full audit trail).
    Watchlist matching only runs when plate_format_valid is true and
    plate_confidence >= 0.5, and returns zero or more alerts: an exact
    match fires a full-severity alert, a one-character-off match fires a
    LOW-severity "needs review" alert instead of auto-escalating.
    """
    try:
        db_detection, alerts = crud.create_detection(db, detection)
    except ValueError as e:
        raise HTTPException(404, str(e))

    return {
        "detection": schemas.DetectionOut.model_validate(db_detection),
        "alerts_triggered": [schemas.AlertOut.model_validate(a) for a in alerts],
    }


# ============================ Stage 4/6: Watchlist ============================

@app.post("/watchlist", response_model=schemas.WatchlistOut)
def add_to_watchlist(entry: schemas.WatchlistCreate, db: Session = Depends(get_db)):
    return crud.create_watchlist_entry(db, entry)


@app.patch("/watchlist/{plate_number}", response_model=schemas.WatchlistOut)
def edit_watchlist(
    plate_number: str, update: schemas.WatchlistUpdate, db: Session = Depends(get_db)
):
    entry = crud.update_watchlist_entry(db, plate_number, update)
    if not entry:
        raise HTTPException(404, "Watchlist entry not found")
    return entry


@app.delete("/watchlist/{plate_number}")
def remove_from_watchlist(plate_number: str, db: Session = Depends(get_db)):
    ok = crud.delete_watchlist_entry(db, plate_number)
    if not ok:
        raise HTTPException(404, "Watchlist entry not found")
    return {"deleted": plate_number}


@app.get("/watchlist", response_model=List[schemas.WatchlistOut])
def get_watchlist(category: Optional[str] = None, db: Session = Depends(get_db)):
    return crud.list_watchlist(db, category)


# =============================== Stage 4/6: Alerts ===============================

@app.get("/alerts", response_model=List[schemas.AlertOut])
def get_alerts(
    status: Optional[str] = None, severity: Optional[str] = None, db: Session = Depends(get_db)
):
    return crud.list_alerts(db, status, severity)


@app.patch("/alerts/{alert_id}/status", response_model=schemas.AlertOut)
def set_alert_status(
    alert_id: int, update: schemas.AlertStatusUpdate, db: Session = Depends(get_db)
):
    alert = crud.update_alert_status(db, alert_id, update)
    if not alert:
        raise HTTPException(404, "Alert not found")
    return alert


@app.get("/alerts/stats", response_model=schemas.AlertStats)
def get_alert_stats(db: Session = Depends(get_db)):
    return crud.alert_stats(db)


# ======================= Stage 5/7: Movement / GIS feed =======================

@app.get("/vehicles/{plate_number}/journey", response_model=schemas.VehicleJourney)
def vehicle_journey(plate_number: str, db: Session = Depends(get_db)):
    """Powers Monika's 'Vehicle Journey' timeline strip and the GIS route line."""
    return crud.get_vehicle_journey(db, plate_number)


# ============================ Stage 8: Vehicle search ============================

@app.get("/vehicles/{plate_number}/search")
def search_vehicle(plate_number: str, db: Session = Depends(get_db)):
    """
    The investigation-view endpoint: Vehicle Number -> Camera -> Timestamp
    -> Route -> Last Seen -> Evidence, all in one response.
    """
    return crud.vehicle_search(db, plate_number)


@app.get("/")
def health():
    return {"status": "ok", "service": "cham-database-watchlist-alerts"}
