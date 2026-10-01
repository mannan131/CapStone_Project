"""Image quality analysis: blur, resolution, contrast, sharpness -> composite score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.common.config import load_config
from src.common.logging import get_logger

log = get_logger("image-quality")


@dataclass
class QualityReport:
    image_path: str
    blur_score: float
    resolution_score: float
    contrast_score: float
    sharpness_score: float
    quality_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _load_gray(path: str | Path) -> np.ndarray:
    p = str(path)
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # fallback via PIL
        with Image.open(p) as pil:
            img = np.array(pil.convert("L"))
    if img is None or img.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return img


def blur_score(gray: np.ndarray, threshold: float = 100.0) -> float:
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # normalize: score = var / (var + threshold) -> 0..1
    return float(var / (var + threshold)) if var >= 0 else 0.0


def resolution_score(image_path: str | Path, min_res=(32, 32), good_res=(512, 512)) -> float:
    with Image.open(image_path) as img:
        w, h = img.size
    pixels = w * h
    min_px = min_res[0] * min_res[1]
    good_px = good_res[0] * good_res[1]
    if pixels <= min_px:
        return 0.0
    if pixels >= good_px:
        return 1.0
    return float((pixels - min_px) / (good_px - min_px))


def contrast_score(gray: np.ndarray, ref_std: float = 64.0) -> float:
    # histogram spread: std / ref_std clipped to 0..1
    std = float(gray.std())
    return float(min(1.0, std / ref_std)) if ref_std > 0 else 0.0


def sharpness_score(gray: np.ndarray) -> float:
    # mean gradient magnitude via Sobel, normalized
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = float(np.mean(np.sqrt(gx**2 + gy**2)))
    return float(mag / (mag + 50.0))


class ImageQualityAnalyzer:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        iq = self.config["image_quality"]
        self.weights = iq["weights"]
        self.blur_threshold = float(iq.get("blur_threshold", 100.0))
        self.min_res = tuple(iq.get("min_resolution", [32, 32]))
        # New spec: target_resolution; legacy: good_resolution.
        self.good_res = tuple(iq.get("target_resolution", iq.get("good_resolution", [512, 512])))
        # Contrast reference std: new `contrast_min_std`, else legacy scale of 64.
        self.contrast_std = float(iq.get("contrast_min_std", 64.0))

    def analyze(self, image_path: str | Path) -> QualityReport:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        try:
            gray = _load_gray(p)
        except Exception as e:
            raise ValueError(f"Corrupt image {p}: {e}") from e
        b = blur_score(gray, self.blur_threshold)
        r = resolution_score(p, self.min_res, self.good_res)
        c = contrast_score(gray, self.contrast_std)
        s = sharpness_score(gray)
        w = self.weights
        total = w["blur"] + w["resolution"] + w["contrast"] + w["sharpness"]
        if total <= 0:
            raise ValueError("Image quality weights sum to zero; check config.yaml")
        composite = (
            w["blur"] * b + w["resolution"] * r + w["contrast"] * c + w["sharpness"] * s
        ) / total
        report = QualityReport(
            image_path=str(p),
            blur_score=round(float(b), 4),
            resolution_score=round(float(r), 4),
            contrast_score=round(float(c), 4),
            sharpness_score=round(float(s), 4),
            quality_score=round(float(max(0.0, min(1.0, composite))), 4),
        )
        log.info("quality_analyzed", **report.to_dict())
        return report

    def analyze_batch(self, paths: list[str | Path]) -> list[QualityReport]:
        reports = []
        for p in paths:
            try:
                reports.append(self.analyze(p))
            except (FileNotFoundError, ValueError) as e:
                log.info("quality_skipped", path=str(p), error=str(e))
        return reports
