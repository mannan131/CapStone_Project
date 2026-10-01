"""API container entrypoint: bootstrap data + models on first boot, then serve.

Zero-intervention clean-room boot: if the mounted volumes lack the synthetic
dataset or trained artifacts, they are generated/trained in place before
uvicorn starts. Restarting with warm volumes skips straight to serving.
"""

from __future__ import annotations

import os
import sys

ROOT = "/app"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.common.config import artifact_paths  # noqa: E402


def _bootstrap() -> None:
    paths = artifact_paths()
    csv_ok = paths["raw_customers_csv"].exists()
    model_ok = paths["best_model"].exists() and paths["fusion_pipeline"].exists()
    if csv_ok and model_ok:
        print("Artifacts present, skipping bootstrap.", flush=True)
        return
    print("First boot: generating synthetic data...", flush=True)
    from src.common.config import load_config, project_seed, resolve_path
    from src.data_ingestion.generate_synthetic import (
        generate_customers,
        generate_sample_images,
    )

    cfg = load_config()
    df = generate_customers(n=600, seed=project_seed(cfg))
    csv_path = paths["raw_customers_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    img_dir = resolve_path(cfg["paths"]["raw_images_dir"])
    generate_sample_images(df, img_dir, n=20)
    print("First boot: training models (this takes several minutes)...", flush=True)
    from src.models.training import train_all

    report = train_all(cfg)
    print(f"Bootstrap complete. Best model: {report['best_model']}", flush=True)


if __name__ == "__main__":
    _bootstrap()
    os.execvp("uvicorn", ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"])
