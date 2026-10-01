import numpy as np
from PIL import Image

from src.image_quality.analyzer import ImageQualityAnalyzer


def _make_image(path, size=(256, 256), blur=False):
    img = Image.new("RGB", size, (200, 50, 50))
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    for i in range(0, size[0], 16):
        d.line([i, 0, i, size[1]], fill=(0, 0, 0), width=2)
    if blur:
        import cv2

        arr = np.array(img)
        arr = cv2.GaussianBlur(arr, (15, 15), 0)
        img = Image.fromarray(arr)
    img.save(path, "JPEG", quality=90)
    return path


def test_quality_report_structure(tmp_path):
    p = _make_image(tmp_path / "a.jpg")
    rep = ImageQualityAnalyzer().analyze(p)
    assert 0 <= rep.quality_score <= 1
    assert 0 <= rep.blur_score <= 1
    d = rep.to_dict()
    assert "contrast_score" in d and "sharpness_score" in d


def test_blurry_scores_lower(tmp_path):
    sharp = _make_image(tmp_path / "sharp.jpg", blur=False)
    blurry = _make_image(tmp_path / "blur.jpg", blur=True)
    az = ImageQualityAnalyzer()
    assert az.analyze(sharp).blur_score >= az.analyze(blurry).blur_score


def test_missing_image_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        ImageQualityAnalyzer().analyze("/nonexistent/x.jpg")
