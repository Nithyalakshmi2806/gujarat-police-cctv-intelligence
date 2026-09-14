"""
Business logic layer.
Splits cleanly along your stages so it's obvious what to demo when:

  Stage 2 -> camera_*
  Stage 3 -> create_detection (also runs the Stage 4 watchlist check inline,
             exactly like the flow diagram: Detected -> Search Watchlist ->
             Match Found -> Alert)
  Stage 4/6 -> watchlist_*, alert_*
  Stage 5/7 -> movement / journey / GIS feed
  Stage 8 -> vehicle_search, alert_stats
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy.orm import Session
from sqlalchemy import func

import models
import schemas


# ============================= Stage 2: Cameras =============================

def create_camera(db: Session, camera: schemas.CameraCreate) -> models.Camera:
    """
    Upsert by camera_code. Sentinel's catalogue (/api/ingest) can be
    re-pulled at any time — same camera_code, possibly updated details
    (new RTSP URL, resolution, etc.). We treat camera_code as the lookup
    key (per the integration reference), not the row's identity, so a
    repeat registration updates the existing row instead of crashing on
    the unique constraint or creating a duplicate.
    """
    existing = (
        db.query(models.Camera)
        .filter(models.Camera.camera_code == camera.camera_code)
        .first()
    )
    if existing:
        for field, value in camera.model_dump(exclude_unset=True).items():
            setattr(existing, field, value)
        db.commit()
        db.refresh(existing)
        _log_camera_event(db, existing.id, "RE-REGISTERED", "Camera catalogue re-synced")
        return existing

    db_camera = models.Camera(**camera.model_dump())
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    _log_camera_event(db, db_camera.id, "REGISTERED", "Camera onboarded")
    return db_camera


def update_camera_status(
    db: Session, camera_id: int, update: schemas.CameraStatusUpdate
) -> Optional[models.Camera]:
    camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not camera:
        return None
    camera.status = update.status
    camera.last_heartbeat = datetime.utcnow()
    db.commit()
    db.refresh(camera)
    _log_camera_event(db, camera_id, update.status.value, update.message)
    return camera


def _log_camera_event(db: Session, camera_id: int, event: str, message: Optional[str]):
    log = models.CameraLog(camera_id=camera_id, event=event, message=message)
    db.add(log)
    db.commit()


def list_cameras(db: Session) -> List[models.Camera]:
    return db.query(models.Camera).all()


def get_camera_logs(db: Session, camera_id: int) -> List[models.CameraLog]:
    return (
        db.query(models.CameraLog)
        .filter(models.CameraLog.camera_id == camera_id)
        .order_by(models.CameraLog.timestamp.desc())
        .all()
    )


# ============================ Stage 3: Detections ============================

def create_detection(
    db: Session, detection: schemas.DetectionCreate
) -> tuple[models.Detection, list[models.Alert]]:
    """
    Persist a detection coming from the AI/ANPR pipeline, then immediately
    run the watchlist check + movement history write. This mirrors the
    architecture diagram:  Detected -> Search Watchlist -> Match -> Alert.

    detection.camera_id is Sentinel's camera_code (e.g. "CAM-07"), not our
    internal integer PK — resolved via camera_code below so the AI pipeline
    never needs to know our internal IDs.
    """
    camera = (
        db.query(models.Camera)
        .filter(models.Camera.camera_code == detection.camera_id)
        .first()
    )
    if not camera:
        raise ValueError(
            f"Unknown camera_code '{detection.camera_id}' — camera must be "
            f"registered via POST /cameras before it can send detections."
        )

    db_detection = models.Detection(
        camera_id=camera.id,
        plate_number=detection.plate_number,
        vehicle_type=detection.vehicle_type,
        vehicle_confidence=detection.vehicle_confidence,
        plate_confidence=detection.plate_confidence,
        plate_format_valid=detection.plate_format_valid,
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)

    # Stage 5/7: every detection is also a movement/timeline point
    _record_movement(db, detection.plate_number, camera, db_detection.timestamp)

    # Stage 4: automatic watchlist match -> alert(s).
    # Only feed matching off detections we trust — noisy/invalid-format
    # reads are still stored above (full audit trail) but don't trigger
    # alerts on their own.
    alerts = []
    if db_detection.plate_format_valid and db_detection.plate_confidence >= 0.5:
        alerts = check_watchlist_and_alert(db, db_detection, camera)

    return db_detection, alerts


# ============================ Stage 4/6: Watchlist ============================

def create_watchlist_entry(
    db: Session, entry: schemas.WatchlistCreate
) -> models.WatchlistEntry:
    db_entry = models.WatchlistEntry(**entry.model_dump())
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry


def update_watchlist_entry(
    db: Session, plate_number: str, update: schemas.WatchlistUpdate
) -> Optional[models.WatchlistEntry]:
    entry = (
        db.query(models.WatchlistEntry)
        .filter(models.WatchlistEntry.plate_number == plate_number)
        .first()
    )
    if not entry:
        return None
    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(entry, field, value)
    entry.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(entry)
    return entry


def delete_watchlist_entry(db: Session, plate_number: str) -> bool:
    entry = (
        db.query(models.WatchlistEntry)
        .filter(models.WatchlistEntry.plate_number == plate_number)
        .first()
    )
    if not entry:
        return False
    db.delete(entry)
    db.commit()
    return True


def list_watchlist(db: Session, category: Optional[str] = None) -> List[models.WatchlistEntry]:
    q = db.query(models.WatchlistEntry).filter(models.WatchlistEntry.is_active == True)  # noqa: E712
    if category:
        q = q.filter(models.WatchlistEntry.category == category)
    return q.all()


# Severity mapping per category (Stage 6: alert severity)
_CATEGORY_SEVERITY = {
    models.WatchlistCategory.STOLEN: models.AlertSeverity.HIGH,
    models.WatchlistCategory.WANTED: models.AlertSeverity.HIGH,
    models.WatchlistCategory.BLACKLISTED: models.AlertSeverity.MEDIUM,
}


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    """
    True if `a` and `b` differ by at most one character (insert, delete,
    or substitute) — i.e. plausibly the same plate with one OCR misread.
    Cheap early-exit version, not full Levenshtein: fine at our data size.
    """
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        # same length -> must be exactly one substitution
        diffs = sum(1 for x, y in zip(a, b) if x != y)
        return diffs == 1
    # lengths differ by 1 -> check one insertion/deletion aligns the rest
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    i = j = 0
    skipped = False
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        elif not skipped:
            skipped = True
            j += 1
        else:
            return False
    return True


def check_watchlist_and_alert(
    db: Session, detection: models.Detection, camera: Optional[models.Camera]
) -> list[models.Alert]:
    """
    Stage 4: 'Search Watchlist -> Match Found -> Alert'.

    Exact plate match -> full-severity alert, fires immediately (this is
    the auto-alert police act on). A near-match (one character off, likely
    OCR noise) never auto-escalates to a HIGH alert on its own — it's
    logged as a LOW-severity "needs review" alert instead, so a human
    checks it rather than police getting paged on a misread character.
    """
    active_entries = (
        db.query(models.WatchlistEntry)
        .filter(models.WatchlistEntry.is_active == True)  # noqa: E712
        .all()
    )

    alerts = []
    for entry in active_entries:
        if entry.plate_number == detection.plate_number:
            match_type = models.AlertMatchType.EXACT
            severity = _CATEGORY_SEVERITY.get(entry.category, models.AlertSeverity.MEDIUM)
            notes = None
        elif _edit_distance_at_most_one(entry.plate_number, detection.plate_number):
            match_type = models.AlertMatchType.NEAR
            severity = models.AlertSeverity.LOW
            notes = (
                f"Near-match: detected '{detection.plate_number}' is one "
                f"character off from watchlist plate '{entry.plate_number}' "
                f"— possible OCR misread, needs human review."
            )
        else:
            continue

        alert = models.Alert(
            detection_id=detection.id,
            watchlist_id=entry.id,
            plate_number=detection.plate_number,
            category=entry.category,
            severity=severity,
            status=models.AlertStatus.NEW,
            match_type=match_type,
            camera_id=detection.camera_id,
            location_name=camera.location_name if camera else None,
            notes=notes,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
        alerts.append(alert)

    return alerts


def update_alert_status(
    db: Session, alert_id: int, update: schemas.AlertStatusUpdate
) -> Optional[models.Alert]:
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        return None
    alert.status = update.status
    if update.notes:
        alert.notes = update.notes
    if update.status in (models.AlertStatus.RESOLVED, models.AlertStatus.FALSE_POSITIVE):
        alert.resolved_at = datetime.utcnow()
    db.commit()
    db.refresh(alert)
    return alert


def list_alerts(
    db: Session, status: Optional[str] = None, severity: Optional[str] = None
) -> List[models.Alert]:
    q = db.query(models.Alert)
    if status:
        q = q.filter(models.Alert.status == status)
    if severity:
        q = q.filter(models.Alert.severity == severity)
    return q.order_by(models.Alert.created_at.desc()).all()


def alert_stats(db: Session) -> dict:
    """Stage 8: alert statistics for the dashboard."""
    total = db.query(models.Alert).count()

    def count_status(s):
        return db.query(models.Alert).filter(models.Alert.status == s).count()

    by_category = dict(
        db.query(models.Alert.category, func.count(models.Alert.id))
        .group_by(models.Alert.category)
        .all()
    )
    by_severity = dict(
        db.query(models.Alert.severity, func.count(models.Alert.id))
        .group_by(models.Alert.severity)
        .all()
    )
    return {
        "total_alerts": total,
        "new": count_status(models.AlertStatus.NEW),
        "acknowledged": count_status(models.AlertStatus.ACKNOWLEDGED),
        "resolved": count_status(models.AlertStatus.RESOLVED),
        "false_positive": count_status(models.AlertStatus.FALSE_POSITIVE),
        "by_category": {k.value if hasattr(k, "value") else k: v for k, v in by_category.items()},
        "by_severity": {k.value if hasattr(k, "value") else k: v for k, v in by_severity.items()},
    }


# ==================== Stage 5/7: Movement history & GIS feed ====================

def _record_movement(
    db: Session, plate_number: str, camera: Optional[models.Camera], timestamp: datetime
):
    movement = models.MovementHistory(
        plate_number=plate_number,
        camera_id=camera.id if camera else None,
        location_name=camera.location_name if camera else None,
        latitude=camera.latitude if camera else None,
        longitude=camera.longitude if camera else None,
        timestamp=timestamp,
    )
    db.add(movement)
    db.commit()


def get_vehicle_journey(db: Session, plate_number: str) -> schemas.VehicleJourney:
    """
    Stage 5 (timeline) + Stage 7 (ordered lat/lon points for the GIS route
    line) + Stage 8 (feeds the investigation report / vehicle search UI).
    Query is indexed on (plate_number, timestamp) for speed at scale.
    """
    rows = (
        db.query(models.MovementHistory)
        .filter(models.MovementHistory.plate_number == plate_number)
        .order_by(models.MovementHistory.timestamp.asc())
        .all()
    )
    route = [
        schemas.MovementPoint(
            camera_id=r.camera_id,
            location_name=r.location_name,
            latitude=r.latitude,
            longitude=r.longitude,
            timestamp=r.timestamp,
        )
        for r in rows
    ]
    return schemas.VehicleJourney(
        plate_number=plate_number,
        total_sightings=len(route),
        first_seen=route[0].timestamp if route else None,
        last_seen=route[-1].timestamp if route else None,
        route=route,
    )


# ============================ Stage 8: Vehicle search ============================

def vehicle_search(db: Session, plate_number: str) -> dict:
    """
    One call that gives the dashboard everything for an investigation view:
    Vehicle Number -> Camera -> Timestamp -> Route -> Last Seen -> Evidence
    (matches Scene 7 of the demo script).
    """
    journey = get_vehicle_journey(db, plate_number)
    watchlist_entry = (
        db.query(models.WatchlistEntry)
        .filter(models.WatchlistEntry.plate_number == plate_number)
        .first()
    )
    alerts = (
        db.query(models.Alert)
        .filter(models.Alert.plate_number == plate_number)
        .order_by(models.Alert.created_at.desc())
        .all()
    )
    detections = (
        db.query(models.Detection)
        .filter(models.Detection.plate_number == plate_number)
        .order_by(models.Detection.timestamp.desc())
        .all()
    )
    return {
        "plate_number": plate_number,
        "watchlist_status": watchlist_entry.category if watchlist_entry else None,
        "journey": journey,
        "alerts": alerts,
        "detections": detections,
    }
