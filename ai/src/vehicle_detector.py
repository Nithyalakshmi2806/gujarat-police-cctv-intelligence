"""
vehicle_detector.py
--------------------
Owner: Nithya 2 (AI + ANPR)
Stage: STEP 3 - Build AI pipeline (Vehicle Detection block)

Detects vehicles in a single video frame and returns bounding boxes.

NOTE ON MODEL CHOICE:
This starter uses an OpenCV Haar Cascade so it runs anywhere with zero GPU
and zero heavy dependencies (good for getting Camera -> Vehicle working today).

For the real hackathon build, swap `VehicleDetector` internals to YOLOv8
(ultralytics) once you're on a machine with more disk/GPU -- the public
interface (`detect(frame) -> List[Detection]`) stays identical, so nothing
downstream (plate detector, OCR, pipeline) needs to change. See the
`detect_yolo()` stub at the bottom for the drop-in replacement.
"""

from dataclasses import dataclass
from typing import List
import cv2
import numpy as np
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]


class VehicleDetector:
    def __init__(self, cascade_path: str = None, min_size=(60, 60)):
        cascade_path = cascade_path or os.path.join(MODELS_DIR, "haarcascade_car.xml")
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Vehicle cascade not found at {cascade_path}")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.min_size = min_size

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect vehicles in a BGR frame. Returns list of Detection boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=self.min_size,
        )

        detections = []
        for (x, y, w, h) in boxes:
            detections.append(Detection(
                x=int(x), y=int(y), w=int(w), h=int(h),
                label="vehicle",
                confidence=0.60,  # Haar cascades don't give real confidence; placeholder
            ))
        return detections

    # ------------------------------------------------------------------
    # DROP-IN UPGRADE PATH (use once you have disk/GPU headroom, e.g. on
    # your own laptop or Colab). Requires: pip install ultralytics
    # ------------------------------------------------------------------
    @staticmethod
    def detect_yolo(frame: np.ndarray, model=None) -> List["Detection"]:
        """
        Example YOLOv8 swap-in (not wired up by default):

            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            results = model(frame, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(Detection(
                        x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1),
                        label=model.names[int(box.cls[0])],
                        confidence=float(box.conf[0]),
                    ))
            return detections
        """
        raise NotImplementedError("Wire this up once ultralytics is installed in your env.")


class YOLOVehicleDetector:
    """
    Real vehicle detector using YOLOv8 (nano model, CPU-friendly).
    Same public interface as VehicleDetector (.detect(frame) -> List[Detection])
    so it's a true drop-in replacement everywhere in the pipeline.

    Requires: pip install ultralytics
    First run auto-downloads yolov8n.pt (~6MB) from Ultralytics, so you need
    internet the first time you use it.
    """

    # COCO class ids: 2=car, 3=motorcycle, 5=bus, 7=truck
    VEHICLE_CLASS_IDS = {2, 3, 5, 7}

    def __init__(self, conf_threshold: float = 0.35):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from e
        self.model = YOLO("yolov8n.pt")
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, classes=list(self.VEHICLE_CLASS_IDS),
                              conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    x=int(x1), y=int(y1),
                    w=int(x2 - x1), h=int(y2 - y1),
                    label=self.model.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                ))
        return detections


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), (0, 255, 0), 2)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (d.x, max(d.y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out
