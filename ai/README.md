# AI + ANPR Module — Nithya 2

This is your slice of the architecture: **Camera → Vehicle Detection → Plate Detection → OCR → Vehicle Number + Timestamp**, exactly as laid out in STEP 3 of the master plan. Everything below is tested and runs.

## What's in this folder

```
anpr_pipeline/
├── models/
│   ├── haarcascade_car.xml      # vehicle detector weights
│   └── haarcascade_plate.xml    # plate detector weights
├── src/
│   ├── vehicle_detector.py      # Camera frame -> vehicle bounding boxes
│   ├── plate_detector.py        # vehicle crop -> plate bounding box
│   ├── ocr_reader.py            # plate crop -> plate text + confidence
│   └── ai_pipeline.py           # wires all three into one pipeline + JSON output
├── sample_data/test_car.jpg     # synthetic test image (for sanity-checking OCR)
├── output/                      # debug crops from the test run
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
# tesseract-ocr binary also required: sudo apt-get install tesseract-ocr

cd src
python3 ai_pipeline.py ../sample_data/test_car.jpg CAM-07 Ahmedabad
```

Output is exactly the JSON record your teammates need downstream:

```json
[
  {
    "detection_id": "...",
    "camera_id": "CAM-07",
    "camera_location": "Ahmedabad",
    "timestamp": "2026-09-11T...",
    "vehicle_type": "vehicle",
    "vehicle_confidence": 0.60,
    "plate_number": "GJ01AB1234",
    "plate_confidence": 0.82,
    "plate_format_valid": true,
    "bbox": {"x": 150, "y": 120, "w": 300, "h": 180}
  }
]
```

- `plate_number` → what Cham's watchlist engine searches against
- `camera_id`, `bbox`, `timestamp` → what Monika's dashboard overlays on Live CCTV
- the whole record → what feeds STEP 4 (cross-camera tracking)

## Why Haar cascades, not YOLOv8, right now

Your team's dev sandbox/laptops may not always have GPU + several GB free for
`torch`/`ultralytics`. Haar cascades are CPU-only, install in seconds, and get
you a **working end-to-end pipeline today** — which is exactly what the doc says
to prioritize ("first make one vehicle → one plate → one timestamp work
reliably. Then expand.").

**Swap-in path when ready:** `vehicle_detector.py` has a `detect_yolo()` stub
with the exact code to paste in once `ultralytics` is installed. The public
interface (`detect(frame) -> List[Detection]`) doesn't change, so
`plate_detector.py`, `ocr_reader.py`, and `ai_pipeline.py` need zero edits.
Similarly, `ocr_reader.py` can be swapped for `easyocr` for better real-world
accuracy — Tesseract is solid but easyocr generally does better on angled/low-res
plates from live CCTV.

## Known limitations to fix before demo day

1. **0 vs O, 1 vs I confusion** — very common OCR issue. Add a post-processing
   correction step in `ocr_reader.py` using known Indian plate structure
   (positions 3-4 are always digits, positions 5-7 are always letters, etc.)
   to disambiguate.
2. **Haar cascade car detector** is trained on Western vehicles/angles — test
   it against real Sentinel camera footage early (Step 2 output) and if
   recall is poor, prioritize the YOLOv8 swap before AI/ANPR demo rehearsal.
3. **Plate detector fallback (contour method)** works well when the plate has
   good contrast against the bumper — verify against night/low-light frames,
   since that's a realistic condition for police CCTV.
4. **Confidence scores** from Haar cascades are placeholders (cascades don't
   emit real confidence) — once on YOLOv8, replace `confidence=0.60` with the
   real `box.conf[0]` value so Cham's alert engine can threshold on it properly.

---

## Your full step-by-step plan (from the work-division table)

### Stage 1 — Architecture & Setup
- [ ] Design the AI pipeline (this repo is the answer: Vehicle → Plate → OCR)
- [ ] Choose models & approach — **decision made**: Haar cascades now, YOLOv8 swap-in later (see above)
- [ ] Define input/output format — **decision made**: see `VehicleRecord` dataclass in `ai_pipeline.py`
- [ ] Plan required AI resources — flag to the team: GPU access needed once you move to YOLOv8/easyocr

### Stage 2 — Sentinel / CCTV Integration
- [ ] Test `AIPipeline.process_video()` with a Sentinel RTSP URL once Naveen hands you one
      (it already accepts RTSP transparently via `cv2.VideoCapture` — see the docstring)
- [ ] Check video quality & FPS against real feeds
- [ ] Try different `sample_every_n_frames` values to balance speed vs. missed detections
- [ ] Validate the frame read loop doesn't crash on stream drops (wrap in try/reconnect)

### Stage 3 — AI/ANPR (core of this repo — done, needs real-data validation)
- [x] Build vehicle detection — `vehicle_detector.py`
- [x] Detect number plate — `plate_detector.py`
- [x] Implement OCR — `ocr_reader.py`
- [ ] Extract vehicle attributes (color, type) — **next task**: add a simple
      color-histogram classifier on the vehicle crop; type can come from
      YOLOv8 class names once swapped in

### Stage 4 — Database & Watchlist
- [ ] Hand `records_to_json()` output to Cham to store detections
- [ ] Agree on the exact JSON schema (this repo's `VehicleRecord` is the proposal — confirm with Cham)
- [ ] Verify plate numbers round-trip correctly into the DB (watch for the 0/O issue above)

### Stage 5 — Cross-Camera Tracking
- [ ] Use `plate_number` as vehicle identity key across camera detections
- [ ] Coordinate with Nithya (integration) on the matching/linking logic
- [ ] Help validate tracking accuracy end-to-end

### Stage 6 — Watchlist & Alerts
- [ ] Compare OCR'd plates against watchlist entries
- [ ] Generate match confidence (combine plate OCR confidence + string similarity for near-matches)
- [ ] Test stolen/wanted detection scenarios with your test images

### Stage 7 — GIS Mapping
- [ ] Send `{plate_number, camera_id, timestamp}` to the GIS layer for route plotting
- [ ] Sanity-check detection ordering is chronological

### Stage 8 — Full Dashboard & Command Centre
- [ ] Feed live `process_video()` output into Monika's dashboard
- [ ] Show plate number + vehicle type + confidence scores on the AI analytics cards
- [ ] Support demo Scene 3 (Camera 07 detects `GJ01AB1234`) end-to-end
