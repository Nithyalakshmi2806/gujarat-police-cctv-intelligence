"""
ocr_reader.py
-------------
Owner: Nithya 2 (AI + ANPR)
Stage: STEP 3 - Build AI pipeline (OCR block)

Input:  a cropped plate image
Output: cleaned plate text + confidence, validated against Indian plate format
"""

import re
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
import pytesseract

# Indian registration plate formats:
#   Standard state-code plates, e.g. GJ01AB1234
#   BH-series (Bharat series) plates, e.g. 25BH22294  -> YY BH #### L
PLATE_REGEX = re.compile(
    r"^([A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}|[0-9]{2}BH[0-9]{4}[A-Z])$"
)

# Characters we allow the OCR engine to output (cuts down on noisy junk)
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@dataclass
class OCRResult:
    raw_text: str
    cleaned_text: str        # exactly what OCR produced, untouched
    normalized_text: str     # best-guess corrected plate (stray chars trimmed)
    is_valid_format: bool    # True if normalized_text matches a known plate format
    confidence: float


# Loose variants -- same shape, but the single character whose class is
# genuinely ambiguous in noisy/obscured plates is allowed to be either a
# letter or digit. Used only to find the right *trim window*; the strict
# PLATE_REGEX above still decides is_valid_format.
LOOSE_BH_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z0-9]$")


def normalize_plate(cleaned: str) -> tuple:
    """
    OCR sometimes adds a stray extra character at the start or end (e.g.
    reflections, dirt, or plate-frame edges getting picked up as a digit).
    This looks for a plate-shaped window inside the noisy reading -- first
    strictly, then loosely (allowing the one ambiguous trailing character
    on BH-series plates to be either a letter or digit, which matters when
    that character is obscured/blacked out in the source image).

    Returns (normalized_text, is_valid). is_valid is only True on a strict
    format match -- a loose-window trim still shortens the noisy reading,
    but is correctly reported as not fully certain.
    """
    if PLATE_REGEX.match(cleaned):
        return cleaned, True

    # Try every contiguous window of plausible plate lengths (8-10 chars)
    # within the noisy text, strict match first.
    for length in (9, 8, 10):
        for start in range(0, max(len(cleaned) - length + 1, 0)):
            candidate = cleaned[start:start + length]
            if PLATE_REGEX.match(candidate):
                return candidate, True

    # No strict match anywhere -- try the loose BH window (ambiguous last
    # character allowed) so we at least trim stray characters even when
    # the final character can't be confirmed as a letter.
    for start in range(0, max(len(cleaned) - 9 + 1, 0)):
        candidate = cleaned[start:start + 9]
        if LOOSE_BH_REGEX.match(candidate):
            return candidate, False

    return cleaned, False


class PlateOCR:
    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def _preprocess(self, plate_img: np.ndarray) -> np.ndarray:
        """Upscale + threshold the plate crop so Tesseract has an easier time."""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if plate_img.ndim == 3 else plate_img
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def read(self, plate_img: np.ndarray) -> Optional[OCRResult]:
        if plate_img is None or plate_img.size == 0:
            return None

        processed = self._preprocess(plate_img)

        # Different plate crops respond better to different Tesseract page
        # segmentation modes -- psm 7 (single line) is usually best, but
        # psm 6/8 sometimes recover text that 7 misses (e.g. when the crop
        # still has a little background noise at the edges). Try all three
        # and keep whichever produces the best-looking result.
        best: Optional[OCRResult] = None
        for psm in (7, 6, 8):
            config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
            data = pytesseract.image_to_data(
                processed, lang=self.lang, config=config, output_type=pytesseract.Output.DICT
            )

            words = [w for w in data["text"] if w.strip()]
            confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]

            raw_text = "".join(words)
            cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
            avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
            normalized, is_valid = normalize_plate(cleaned)

            result = OCRResult(
                raw_text=raw_text,
                cleaned_text=cleaned,
                normalized_text=normalized,
                is_valid_format=is_valid,
                confidence=round(avg_conf, 3),
            )

            if is_valid:
                return result  # strict format match -- good enough, stop here
            if best is None or len(cleaned) > len(best.cleaned_text):
                best = result

        return best
