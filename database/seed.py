"""
Loads the representative demo data straight out of the brief:
  - Cameras CAM-07, CAM-14, CAM-21, CAM-32
  - Watchlist: GJ01AB1234 (STOLEN), GJ05XY7788 (WANTED), GJ18PQ4455 (BLACKLISTED)

Run once before your demo:
    python seed.py
"""

from database import SessionLocal, engine
import models

models.Base.metadata.create_all(bind=engine)
db = SessionLocal()

cameras = [
    {"camera_code": "CAM-07", "location_name": "Ahmedabad", "latitude": 23.0225, "longitude": 72.5714},
    {"camera_code": "CAM-14", "location_name": "Junction", "latitude": 23.0300, "longitude": 72.5800},
    {"camera_code": "CAM-21", "location_name": "Highway", "latitude": 23.0450, "longitude": 72.6000},
    {"camera_code": "CAM-32", "location_name": "City Road", "latitude": 23.0550, "longitude": 72.6100},
]

for c in cameras:
    if not db.query(models.Camera).filter_by(camera_code=c["camera_code"]).first():
        db.add(models.Camera(**c, status=models.CameraStatus.ONLINE))

watchlist = [
    {"plate_number": "GJ01AB1234", "category": models.WatchlistCategory.STOLEN, "reason": "Reported stolen"},
    {"plate_number": "GJ05XY7788", "category": models.WatchlistCategory.WANTED, "reason": "Wanted in FIR"},
    {"plate_number": "GJ18PQ4455", "category": models.WatchlistCategory.BLACKLISTED, "reason": "Blacklisted vehicle"},
]

for w in watchlist:
    if not db.query(models.WatchlistEntry).filter_by(plate_number=w["plate_number"]).first():
        db.add(models.WatchlistEntry(**w))

db.commit()
db.close()
print("Seed complete: 4 cameras + 3 watchlist entries loaded.")
