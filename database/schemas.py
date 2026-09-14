from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict

from models import WatchlistCategory, AlertSeverity, AlertStatus, AlertMatchType, CameraStatus


# ---------------- Camera (Stage 2) ----------------

class CameraCreate(BaseModel):
    camera_code: str
    department: Optional[str] = None
    location_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rtsp_url: Optional[str] = None
    resolution: Optional[str] = None
    codec: Optional[str] = None


class CameraStatusUpdate(BaseModel):
    status: CameraStatus
    message: Optional[str] = None


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_code: str
    location_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    status: CameraStatus
    last_heartbeat: Optional[datetime]


# ---------------- Detection (Stage 3) ----------------

class DetectionCreate(BaseModel):
    camera_id: str  # Sentinel's camera_code, e.g. "CAM-07" — resolved to our internal FK server-side
    plate_number: str
    vehicle_type: Optional[str] = None
    vehicle_confidence: Optional[float] = None
    plate_confidence: float
    plate_format_valid: bool = False


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    camera_id: int
    plate_number: str
    vehicle_type: Optional[str]
    vehicle_confidence: Optional[float]
    plate_confidence: float
    plate_format_valid: bool
    timestamp: datetime


# ---------------- Watchlist (Stage 4 / 6) ----------------

class WatchlistCreate(BaseModel):
    plate_number: str
    category: WatchlistCategory
    reason: Optional[str] = None
    added_by: Optional[str] = None


class WatchlistUpdate(BaseModel):
    category: Optional[WatchlistCategory] = None
    reason: Optional[str] = None
    is_active: Optional[bool] = None


class WatchlistOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    plate_number: str
    category: WatchlistCategory
    reason: Optional[str]
    is_active: bool
    created_at: datetime


# ---------------- Alerts (Stage 4 / 6) ----------------

class AlertStatusUpdate(BaseModel):
    status: AlertStatus
    notes: Optional[str] = None


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    plate_number: str
    category: WatchlistCategory
    severity: AlertSeverity
    status: AlertStatus
    match_type: AlertMatchType
    camera_id: int
    location_name: Optional[str]
    created_at: datetime
    resolved_at: Optional[datetime]
    notes: Optional[str]


# ---------------- Movement / Investigation (Stage 5 / 7 / 8) ----------------

class MovementPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    camera_id: int
    location_name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timestamp: datetime


class VehicleJourney(BaseModel):
    plate_number: str
    total_sightings: int
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    route: List[MovementPoint]


class AlertStats(BaseModel):
    total_alerts: int
    new: int
    acknowledged: int
    resolved: int
    false_positive: int
    by_category: dict
    by_severity: dict
