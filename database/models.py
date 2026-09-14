"""
Stage 1: Design database architecture / required tables / relationships
------------------------------------------------------------------------
Tables owned by Cham:

  Camera             -> Stage 2 (metadata, location, status, logs)
  Detection          -> Stage 3 (vehicle/plate detections from AI engine)
  WatchlistEntry      -> Stage 4 & 6 (watchlist + categories)
  Alert              -> Stage 4 & 6 (alert logic, severity, history, status)
  MovementHistory    -> Stage 5 & 7 (timeline, last-seen, route/GIS feed)
  CameraLog          -> Stage 2 (health/status logs)

Relationships:
  Camera 1---* Detection        (a camera produces many detections)
  Camera 1---* CameraLog        (a camera has many status log entries)
  WatchlistEntry 1---* Alert    (a watchlist entry can trigger many alerts)
  Detection 1---1 Alert         (an alert is raised from one detection)
  Detection --- MovementHistory (each detection is a point in a vehicle's
                                  movement timeline, keyed by plate number)
"""

from datetime import datetime
import enum

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Enum, Text, Boolean
)
from sqlalchemy.orm import relationship

from database import Base


# ---------- Enums (Stage 4 / 6: alert logic, severity, status) ----------

class WatchlistCategory(str, enum.Enum):
    STOLEN = "STOLEN"
    WANTED = "WANTED"
    BLACKLISTED = "BLACKLISTED"


class AlertSeverity(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AlertStatus(str, enum.Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class AlertMatchType(str, enum.Enum):
    EXACT = "EXACT"   # plate matched a watchlist entry exactly
    NEAR = "NEAR"      # matched within 1 character (possible OCR noise) — needs human review


class CameraStatus(str, enum.Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"


# ---------------------------- Stage 2: Cameras ---------------------------

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    camera_code = Column(String, unique=True, index=True, nullable=False)  # e.g. "CAM-07"
    department = Column(String, nullable=True)
    location_name = Column(String, nullable=False)   # e.g. "Ahmedabad - Ring Road Junction"
    latitude = Column(Float, nullable=True)           # Stage 7: feeds GIS map
    longitude = Column(Float, nullable=True)
    rtsp_url = Column(String, nullable=True)
    status = Column(Enum(CameraStatus), default=CameraStatus.OFFLINE)
    resolution = Column(String, nullable=True)
    codec = Column(String, nullable=True)
    last_heartbeat = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    detections = relationship("Detection", back_populates="camera")
    logs = relationship("CameraLog", back_populates="camera")


class CameraLog(Base):
    """Stage 2: Maintain logs (connect/disconnect/health events)."""
    __tablename__ = "camera_logs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    event = Column(String, nullable=False)   # "CONNECTED", "DISCONNECTED", "DEGRADED", ...
    message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    camera = relationship("Camera", back_populates="logs")


# -------------------------- Stage 3: Detections ---------------------------

class Detection(Base):
    """
    Stage 3: Store detections coming from Nithya-2's AI/ANPR pipeline.
    Cham owns persistence + integrity, not the AI itself.
    """
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)  # internal FK, resolved from camera_code
    plate_number = Column(String, index=True, nullable=False)   # e.g. "GJ01AB1234"
    vehicle_type = Column(String, nullable=True)                # car/bike/truck
    vehicle_color = Column(String, nullable=True)
    vehicle_confidence = Column(Float, nullable=True)           # AI pipeline's vehicle-detection confidence
    plate_confidence = Column(Float, nullable=False)            # OCR confidence — drives watchlist matching
    plate_format_valid = Column(Boolean, default=False)         # whether OCR text matched expected plate format
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    camera = relationship("Camera", back_populates="detections")
    alerts = relationship("Alert", back_populates="detection")


# -------------------------- Stage 4/6: Watchlist ---------------------------

class WatchlistEntry(Base):
    """Stage 4 & 6: watchlist tables + categories."""
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, unique=True, index=True, nullable=False)
    category = Column(Enum(WatchlistCategory), nullable=False)
    reason = Column(Text, nullable=True)
    added_by = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    alerts = relationship("Alert", back_populates="watchlist_entry")


# ----------------------------- Stage 4/6: Alerts ----------------------------

class Alert(Base):
    """Stage 4 & 6: alert logic, severity, history, status management."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False)
    watchlist_id = Column(Integer, ForeignKey("watchlist.id"), nullable=False)
    plate_number = Column(String, index=True, nullable=False)
    category = Column(Enum(WatchlistCategory), nullable=False)
    severity = Column(Enum(AlertSeverity), default=AlertSeverity.HIGH)
    status = Column(Enum(AlertStatus), default=AlertStatus.NEW)
    match_type = Column(Enum(AlertMatchType), default=AlertMatchType.EXACT)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    location_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    detection = relationship("Detection", back_populates="alerts")
    watchlist_entry = relationship("WatchlistEntry", back_populates="alerts")


# ------------------------- Stage 5/7: Movement history ------------------------

class MovementHistory(Base):
    """
    Stage 5 & 7: timeline / last-seen / route data feed for GIS.
    One row per (plate, camera, timestamp) sighting — this table IS
    the vehicle journey timeline Monika's dashboard renders, and the
    ordered lat/lon feed for the GIS route line.
    """
    __tablename__ = "movement_history"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, index=True, nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    location_name = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
