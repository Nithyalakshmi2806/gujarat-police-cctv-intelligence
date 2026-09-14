"""
vehicle_attributes.py
----------------------
Owner: Nithya 2 (AI + ANPR)
Stage: STEP 3 - Build AI pipeline (vehicle attributes, next task after core ANPR)

Input:  a cropped vehicle image (from vehicle_detector.Detection.crop())
Output: dominant color name + vehicle type (type comes straight from YOLO's
        class label -- this module only adds color, which YOLO doesn't give you)

Why this matters for the demo: "white Suzuki Ciaz, plate 25BH2229" is far
more useful to a police officer scanning an alert list than just "car" --
color is often the first thing a witness or officer remembers, and it lets
Cham's watchlist do a secondary check (e.g. "stolen car was silver, this
detection is red -- probably not a match" even before the plate is 100% sure).
"""

from dataclasses import dataclass
import cv2
import numpy as np

# Reference colors in BGR (OpenCV's default channel order), tuned for
# typical vehicle paint tones rather than generic web colors.
COLOR_REFERENCE = {
    "white":  (240, 240, 240),
    "black":  (25, 25, 25),
    "silver": (190, 190, 190),
    "gray":   (128, 128, 128),
    "red":    (40, 40, 180),
    "blue":   (150, 60, 20),
    "green":  (60, 120, 40),
    "yellow": (40, 210, 230),
    "brown":  (40, 70, 100),
    "orange": (30, 100, 220),
}


@dataclass
class VehicleAttributes:
    color: str
    color_confidence: float  # 0-1, how close the dominant color was to the reference


def _closest_color_name(bgr: np.ndarray) -> tuple:
    best_name, best_dist = "unknown", float("inf")
    for name, ref in COLOR_REFERENCE.items():
        dist = np.linalg.norm(bgr.astype(float) - np.array(ref, dtype=float))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    # Convert distance to a rough 0-1 confidence (max possible distance ~441)
    confidence = max(0.0, 1.0 - (best_dist / 441.0))
    return best_name, round(confidence, 3)


def extract_color(vehicle_crop: np.ndarray) -> VehicleAttributes:
    """
    Estimate the vehicle's dominant body-panel color.

    Strategy: sample the center-most region of the crop rather than the
    whole box, since vehicle bounding boxes often include windows (dark,
    non-representative), wheels (black, not the body color), and background
    slivers around the edges. The center-band of a car is almost always
    actual bodywork.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return VehicleAttributes(color="unknown", color_confidence=0.0)

    h, w = vehicle_crop.shape[:2]

    # Sample a horizontal band through the middle-lower portion of the box
    # (avoids the roof/window area which is often glass or sky reflection)
    band_top = int(h * 0.45)
    band_bottom = int(h * 0.75)
    band_left = int(w * 0.15)
    band_right = int(w * 0.85)

    sample = vehicle_crop[band_top:band_bottom, band_left:band_right]
    if sample.size == 0:
        sample = vehicle_crop  # fall back to the whole crop if the box is tiny

    # Median is more robust than mean against small bright/dark outliers
    # (reflections, shadows, a stray taillight in the sample region)
    median_bgr = np.median(sample.reshape(-1, 3), axis=0)

    color_name, confidence = _closest_color_name(median_bgr)
    return VehicleAttributes(color=color_name, color_confidence=confidence)
