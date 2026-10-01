"""Image upload handler: path/URL/base64 -> validated storage."""

from __future__ import annotations

import base64
import shutil
import urllib.request
from pathlib import Path

from PIL import Image

from src.common.config import artifact_paths, load_config
from src.common.logging import get_logger

log = get_logger("image-upload")


class ImageUploadHandler:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        iq = self.config["image_quality"]
        self.allowed = set(iq.get("allowed_extensions", [".jpg", ".jpeg", ".png"]))
        self.max_mb = float(iq.get("max_file_size_mb", 10))

    def _check_size(self, path: Path) -> None:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_mb:
            raise ValueError(f"Image {size_mb:.2f}MB exceeds limit of {self.max_mb}MB")

    def _validate_image(self, path: Path) -> None:
        if path.suffix.lower() not in self.allowed:
            raise ValueError(
                f"Unsupported image type: {path.suffix}. Allowed: {sorted(self.allowed)}"
            )
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable image file: {e}") from e

    def dest_path(self, customer_id: str, product_id: str = "default") -> Path:
        base = artifact_paths(self.config)["raw_images_dir"]
        d = base / customer_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{product_id}.jpg"

    def save_file(self, src: str | Path, customer_id: str, product_id: str = "default") -> Path:
        src_p = Path(src)
        if not src_p.exists():
            raise FileNotFoundError(f"Source image not found: {src}")
        self._check_size(src_p)
        self._validate_image(src_p)
        dest = self.dest_path(customer_id, product_id)
        # normalize to JPEG
        try:
            with Image.open(src_p) as img:
                img.convert("RGB").save(dest, "JPEG")
        except Exception:
            shutil.copy(src_p, dest)
        log.info("image_saved", dest=str(dest), customer_id=customer_id)
        return dest

    def save_bytes(self, data: bytes, customer_id: str, product_id: str = "default") -> Path:
        dest = self.dest_path(customer_id, product_id)
        dest.write_bytes(data)
        try:
            self._check_size(dest)
        except ValueError:
            dest.unlink(missing_ok=True)
            raise
        try:
            self._validate_image(dest)
            with Image.open(dest) as img:
                img.convert("RGB").save(dest, "JPEG")
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise ValueError(f"Invalid image bytes: {e}") from e
        return dest

    def save_base64(self, b64: str, customer_id: str, product_id: str = "default") -> Path:
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            data = base64.b64decode(b64)
        except Exception as e:
            raise ValueError(f"Invalid base64 payload: {e}") from e
        return self.save_bytes(data, customer_id, product_id)

    def save_url(self, url: str, customer_id: str, product_id: str = "default") -> Path:
        dest = self.dest_path(customer_id, product_id)
        try:
            with urllib.request.urlopen(url, timeout=20) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise ValueError(f"Could not download image from URL: {e}") from e
        try:
            self._check_size(dest)
            self._validate_image(dest)
        except ValueError:
            dest.unlink(missing_ok=True)
            raise
        return dest
