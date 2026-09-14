"""
ai_pipeline.py
--------------
Owner: Nithya 2 (AI + ANPR)
Stage: STEP 3 - Build AI pipeline (full block, matches the architecture doc)
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np

from vehicle_detector import VehicleDetector, YOLOVehicleDetector, Detection, draw_detections
from plate_detector import PlateDetector
from ocr_reader import PlateOCR, OCRResult
from vehicle_attributes import extract_color


@dataclass
class VehicleRecord:
    detection_id: str
    camera_code: str
    camera_location: str
    timestamp: str
    vehicle_type: str
    vehicle_confidence: float
    vehicle_color: str
    vehicle_color_confidence: float
    plate_number: Optional[str]
    plate_confidence: Optional[float]
    plate_format_valid: Optional[bool]
    bbox: dict
    is_plausible: bool


class AIPipeline:
    MIN_VEHICLE_CONFIDENCE = 0.5
    MIN_PLATE_LENGTH = 7

    def __init__(self, camera_id: str, camera_location: str, use_yolo: bool = False,
                 use_easyocr: bool = False,
                 min_vehicle_confidence: float = None, min_plate_length: int = None):
        self.camera_id = camera_id
        self.camera_location = camera_location
        self.min_vehicle_confidence = (min_vehicle_confidence
                                        if min_vehicle_confidence is not None
                                        else self.MIN_VEHICLE_CONFIDENCE)
        self.min_plate_length = (min_plate_length
                                  if min_plate_length is not None
                                  else self.MIN_PLATE_LENGTH)
        if use_yolo:
            self.vehicle_detector = YOLOVehicleDetector()
        else:
            self.vehicle_detector = VehicleDetector()
        self.plate_detector = PlateDetector()
        if use_easyocr:
            from ocr_reader_easyocr import EasyOCRPlateReader
            self.ocr = EasyOCRPlateReader()
        else:
            self.ocr = PlateOCR()

    def _is_plausible(self, vehicle_confidence: float, plate_number: Optional[str]) -> bool:
        if vehicle_confidence < self.min_vehicle_confidence:
            return False
        if plate_number is not None and len(plate_number) < self.min_plate_length:
            return False
        return True

    def process_frame(self, frame: np.ndarray) -> List[VehicleRecord]:
        records: List[VehicleRecord] = []
        vehicle_detections = self.vehicle_detector.detect(frame)

        for det in vehicle_detections:
            vehicle_crop = det.crop(frame)

            os.makedirs("../output", exist_ok=True)
            cv2.imwrite(f"../output/vehicle_{det.x}_{det.y}.jpg", vehicle_crop)

            plate_candidates = self.plate_detector.detect_candidates(vehicle_crop)

            plate_number, plate_conf, plate_valid = None, None, None
            best_result: Optional[OCRResult] = None

            print(f"Vehicle at ({det.x},{det.y}): {len(plate_candidates)} plate candidates")

            for i, plate_box in enumerate(plate_candidates):
                plate_crop = plate_box.crop(vehicle_crop)

                cv2.imwrite(f"../output/plate_{det.x}_{det.y}_candidate{i}.jpg", plate_crop)

                result = self.ocr.read(plate_crop)
                print(f"  candidate {i}: raw={result.cleaned_text if result else None}, normalized={result.normalized_text if result else None}, conf={result.confidence if result else None}, valid={result.is_valid_format if result else None}")

                if not result or not result.cleaned_text:
                    continue
                if result.is_valid_format:
                    best_result = result
                    break
                if best_result is None:
                    best_result = result
                    continue

                # A candidate that's roughly plate-length (7+ chars after
                # cleanup) is almost always the real plate, even without a
                # strict format match (e.g. one obscured character).
                # Prefer it over short junk text regardless of confidence --
                # Tesseract's confidence near 0 is too noisy to trust alone.
                result_plate_like = len(result.normalized_text) >= 7
                best_plate_like = len(best_result.normalized_text) >= 7

                if result_plate_like and not best_plate_like:
                    best_result = result
                    continue
                if best_plate_like and not result_plate_like:
                    continue

                better_confidence = result.confidence > best_result.confidence
                same_confidence_longer = (result.confidence == best_result.confidence
                                           and len(result.cleaned_text) > len(best_result.cleaned_text))
                if better_confidence or same_confidence_longer:
                    best_result = result

            if best_result is not None:
                plate_number = best_result.normalized_text
                plate_conf = best_result.confidence
                plate_valid = best_result.is_valid_format

            attrs = extract_color(vehicle_crop)

            records.append(VehicleRecord(
                detection_id=str(uuid.uuid4()),
                camera_code=self.camera_id,
                camera_location=self.camera_location,
                timestamp=datetime.now(timezone.utc).isoformat(),
                vehicle_type=det.label,
                vehicle_confidence=det.confidence,
                vehicle_color=attrs.color,
                vehicle_color_confidence=attrs.color_confidence,
                plate_number=plate_number,
                plate_confidence=plate_conf,
                plate_format_valid=plate_valid,
                bbox={"x": det.x, "y": det.y, "w": det.w, "h": det.h},
                is_plausible=self._is_plausible(det.confidence, plate_number),
            ))

        return records

    def process_video(self, video_path: str, sample_every_n_frames: int = 5,
                       save_annotated_to: Optional[str] = None,
                       max_reconnect_attempts: int = 10):
        is_live_stream = video_path.startswith(("rtsp://", "http://", "https://"))

        def _open_capture():
            c = cv2.VideoCapture(video_path)
            if not c.isOpened():
                raise RuntimeError(f"Could not open video source: {video_path}")
            return c

        cap = _open_capture()

        writer = None
        frame_idx = 0
        reconnect_attempts = 0
        backoff_seconds = 2
        all_records: List[VehicleRecord] = []

        while True:
            ok, frame = cap.read()

            if not ok:
                if not is_live_stream:
                    break

                if reconnect_attempts >= max_reconnect_attempts:
                    print(f"[ai_pipeline] Gave up after {max_reconnect_attempts} "
                          f"reconnect attempts on {video_path}")
                    break

                reconnect_attempts += 1
                print(f"[ai_pipeline] Stream read failed, reconnecting in "
                      f"{backoff_seconds}s (attempt {reconnect_attempts}/{max_reconnect_attempts})...")
                cap.release()
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 30)

                try:
                    cap = _open_capture()
                except RuntimeError:
                    continue

                continue

            reconnect_attempts = 0
            backoff_seconds = 2

            if frame_idx % sample_every_n_frames == 0:
                records = self.process_frame(frame)
                all_records.extend(records)

                if save_annotated_to:
                    annotated = draw_detections(frame, self.vehicle_detector.detect(frame))
                    if writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(save_annotated_to, fourcc, 10, (w, h))
                    writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        return all_records


def records_to_json(records: List[VehicleRecord]) -> str:
    return json.dumps([asdict(r) for r in records], indent=2)


def to_cham_detection_payload(record: VehicleRecord) -> dict:
    CAMERA_CODE_TO_ID = {
        "CAM-07": 1,
        "CAM-14": 2,
        "CAM-21": 3,
        "CAM-32": 4,
    }

    camera_id = CAMERA_CODE_TO_ID.get(record.camera_code)
    if camera_id is None:
        raise ValueError(
            f"Unknown camera_code '{record.camera_code}' -- not in "
            f"CAMERA_CODE_TO_ID. Check Cham's GET /cameras for the current "
            f"list and add it here."
        )

    combined_confidence = min(record.vehicle_confidence, record.plate_confidence)

    return {
        "camera_id": camera_id,
        "plate_number": record.plate_number,
        "vehicle_type": record.vehicle_type,
        "vehicle_color": record.vehicle_color,
        "confidence": combined_confidence,
    }


def send_to_cham(record: VehicleRecord, base_url: str = "http://localhost:8000"):
    import requests
    payload = to_cham_detection_payload(record)
    response = requests.post(f"{base_url}/detections", json=payload)
    if response.status_code == 404:
        raise RuntimeError(
            f"Camera '{record.camera_code}' isn't registered yet on Cham's side "
            f"-- ask Naveen to POST it to /cameras first."
        )
    if response.status_code == 422:
        raise RuntimeError(f"422 validation error: {response.json()}")
    response.raise_for_status()
    return response.json()


def filter_plausible(records: List[VehicleRecord]) -> List[VehicleRecord]:
    return [r for r in records if r.is_plausible]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 ai_pipeline.py <path_to_image> [camera_id] [camera_location]")
        sys.exit(1)

    path = sys.argv[1]
    cam_id = sys.argv[2] if len(sys.argv) > 2 else "CAM-07"
    cam_loc = sys.argv[3] if len(sys.argv) > 3 else "Ahmedabad"
    use_yolo = "--yolo" in sys.argv
    only_plausible = "--filter" in sys.argv
    use_easyocr = "--easyocr" in sys.argv

    pipeline = AIPipeline(camera_id=cam_id, camera_location=cam_loc, use_yolo=use_yolo,
                           use_easyocr=use_easyocr)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png"):
        frame = cv2.imread(path)
        records = pipeline.process_frame(frame)
    else:
        records = pipeline.process_video(path)

    if only_plausible:
        records = filter_plausible(records)

    print(records_to_json(records))

    if "--send-to-cham" in sys.argv:
        print("\n--- Sending to Cham's API ---")
        for r in records:
            try:
                result = send_to_cham(r)
                print(f"Sent {r.plate_number}: alerts_triggered = {result.get('alerts_triggered', [])}")
            except Exception as e:
                print(f"Failed to send {r.plate_number}: {e}")