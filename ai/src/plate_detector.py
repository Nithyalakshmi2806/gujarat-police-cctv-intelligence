"""
plate_detector.py
------------------
Owner: Nithya 2 (AI + ANPR)
Stage: STEP 3 - Build AI pipeline (Plate Detection block)

Input:  a cropped vehicle image (from vehicle_detector.Detection.crop())
Output: a cropped plate image (or None if no plate found), ready for OCR
"""

from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@dataclass
class PlateBox:
    x: int
    y: int
    w: int
    h: int

    def crop(self, vehicle_img: np.ndarray) -> np.ndarray:
        return vehicle_img[self.y:self.y + self.h, self.x:self.x + self.w]


class PlateDetector:
    def __init__(self, cascade_path: str = None):
        cascade_path = cascade_path or os.path.join(MODELS_DIR, "haarcascade_plate.xml")
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Plate cascade not found at {cascade_path}")
        self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, vehicle_img: np.ndarray) -> Optional[PlateBox]:
        """
        Return a single best-guess plate box (for simple callers).
        For robust results, prefer detect_candidates() + OCR-based selection,
        which is what AIPipeline actually uses -- Haar cascade boxes all tend
        to have near-identical aspect ratios (~3:1) whether or not they're
        really on the plate, so geometry alone can't reliably pick the winner.
        """
        candidates = self.detect_candidates(vehicle_img)
        return candidates[0] if candidates else None

    def detect_candidates(self, vehicle_img: np.ndarray) -> List[PlateBox]:
        """Return ALL plausible plate boxes, most-likely first, for the
        caller (ai_pipeline) to disambiguate by actually trying OCR on each."""
        if vehicle_img is None or vehicle_img.size == 0:
            return []

        gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 12)
        )

        img_h = vehicle_img.shape[0]
        # Plates sit in the lower ~60% of the vehicle bbox -- discard cascade
        # hits clearly above that zone (rear windshield, roof trim, badges).
        lower_boxes = [b for b in boxes if b[1] > img_h * 0.35]

        candidates = [PlateBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in lower_boxes]

        fallback = self._fallback_contour_detect(vehicle_img)
        if fallback is not None:
            candidates.append(fallback)

        # Refine every candidate found so far by looking for the bright
        # (white/yellow plate background) sub-region within it. This
        # narrows an approximately-right box down to a tight, bumper-free
        # crop, which matters a lot for OCR accuracy -- it must run on an
        # already-narrowed candidate rather than the whole vehicle, since
        # on light-colored vehicles the entire body is "bright" too and a
        # whole-vehicle search can't tell the plate apart from the paint.
        refined = []
        for cand in candidates:
            crop = cand.crop(vehicle_img)
            bright_box = self._bright_region_within(crop)
            if bright_box is not None:
                refined.append(PlateBox(
                    x=cand.x + bright_box.x, y=cand.y + bright_box.y,
                    w=bright_box.w, h=bright_box.h,
                ))
        candidates = refined + candidates  # refined boxes tried first

        return candidates

    def _bright_region_within(self, sub_img: np.ndarray) -> Optional[PlateBox]:
        """Find the largest bright, plate-shaped region inside an already
        roughly-correct candidate crop (see detect_candidates)."""
        if sub_img is None or sub_img.size == 0:
            return None

        gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        img_h, img_w = sub_img.shape[:2]
        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            aspect = w / float(h)
            if 1.8 <= aspect <= 6.5 and w > img_w * 0.35:
                candidates.append((x, y, w, h))

        if not candidates:
            return None

        best = max(candidates, key=lambda b: b[2] * b[3])
        x, y, w, h = best
        return PlateBox(int(x), int(y), int(w), int(h))

    def _fallback_contour_detect(self, vehicle_img: np.ndarray) -> Optional[PlateBox]:
        """
        Backup method when the Haar cascade misses (common with Indian plates,
        which the cascade wasn't trained on). Looks for a rectangular,
        high-contrast region with a plate-like aspect ratio (~2:1 to 5:1),
        restricted to the lower portion of the vehicle crop since plates are
        essentially always mounted there -- this avoids false positives on
        trunk-lid reflections, badges, and window trim higher up the car.
        """
        img_h, img_w = vehicle_img.shape[:2]

        # Only search the bottom 45% of the vehicle crop
        search_top = int(img_h * 0.55)
        search_region = vehicle_img[search_top:, :]

        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            aspect = w / float(h)
            # Plates are wide+short and shouldn't span the full width of the bumper
            if 2.0 <= aspect <= 5.5 and img_w * 0.15 < w < img_w * 0.85:
                candidates.append((x, y, w, h))

        if not candidates:
            return None

        # Prefer the widest matching candidate (real plates are usually the
        # most prominent wide rectangle in the bumper region)
        best = max(candidates, key=lambda b: b[2])
        x, y, w, h = best
        return PlateBox(int(x), int(y + search_top), int(w), int(h))
