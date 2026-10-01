"""Synthetic data + sample product image generation.

Column NAMES come from the config data contract at runtime
(``data_schema`` + rule-referenced extras); the statistical shapes attached to
each schema position encode the churn story. Rename a column in the config and
the emitted CSV follows it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.common.config import (
    artifact_paths,
    load_config,
    project_seed,
    resolve_path,
    schema,
)
from src.common.logging import get_logger
from src.recommendation_engine.rules import extra_rule_features

log = get_logger("synthetic-data")


def _column_plan(cfg: dict) -> dict:
    """Map semantic roles -> configured column names, positionally by schema order."""
    s = schema(cfg)
    fusion = cfg.get("feature_fusion", {})
    img_col = fusion.get("image_score_column", fusion.get("image_score_col", "image_quality_score"))
    cats = set(s.get("categorical_features", []))
    beh = list(s.get("behavioral_features", []))
    tx = list(s.get("transactional_features", []))
    tx_num = [c for c in tx if c not in cats]
    tx_cat = [c for c in tx if c in cats]
    if len(beh) != 4 or len(tx_num) != 3 or len(tx_cat) != 2:
        raise ValueError(
            "Synthetic generator expects 4 behavioral, 3 numeric transactional and "
            f"2 categorical columns; got {beh} / {tx}"
        )
    return {
        "id": s["id_column"],
        "target": s["target_column"],
        "login": beh[0],
        "session": beh[1],
        "usage": beh[2],
        "tickets": beh[3],
        "purchases": tx_num[0],
        "avg_invoice": tx_num[1],
        "late": tx_num[2],
        "pay_method": tx_cat[0],
        "tier": tx_cat[1],
        "clv": s.get("clv_column", "customer_lifetime_value"),
        "image": img_col,
        "extras": extra_rule_features(cfg),
    }


def generate_customers(n: int = 600, seed: int = 42, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    cols = _column_plan(cfg)
    rng = np.random.default_rng(seed)
    ids = [f"CUST-{i:05d}" for i in range(1, n + 1)]
    login_freq = rng.gamma(2.0, 2.5, n).clip(0, 60)
    avg_session = rng.normal(12, 7, n).clip(1, 120)
    usage_score = rng.normal(55, 18, n).clip(0, 100)
    support_tickets = rng.poisson(1.2, n).clip(0, 15)
    total_purchases = rng.poisson(4, n).clip(0, 40)
    avg_invoice = rng.normal(180, 60, n).clip(20, None)
    invoice_total = total_purchases * avg_invoice + rng.normal(50, 20, n).clip(0)
    late_payments = rng.poisson(0.6, n).clip(0, 10)
    days_since_login = rng.exponential(10, n).clip(0, 120)
    clv = (invoice_total * rng.uniform(0.8, 1.5, n)).clip(0)
    payment_method = rng.choice(
        ["credit_card", "paypal", "bank_transfer", "crypto"], n, p=[0.5, 0.25, 0.2, 0.05]
    )
    tier = rng.choice(["basic", "pro", "enterprise"], n, p=[0.55, 0.3, 0.15])
    image_quality = rng.beta(5, 2, n).clip(0.05, 1.0)
    # degrade quality for some customers to create signal
    degrade = rng.random(n) < 0.25
    image_quality[degrade] = rng.beta(2, 5, degrade.sum()).clip(0.05, 1.0)

    # Rule-referenced extras are emitted first: the first extra column carries
    # the recency signal in the churn story below (generic days-like shape).
    extras: dict[str, np.ndarray] = {}
    for extra in cols["extras"]:
        extras[extra] = np.round(rng.exponential(10, n).clip(0, 120), 2)
    recency = extras[cols["extras"][0]] if cols["extras"] else days_since_login

    # churn logit with realistic patterns
    logit = (
        -0.35 * (login_freq / 5)
        - 0.25 * (avg_session / 10)
        - 0.5 * ((usage_score - 50) / 20)
        + 0.55 * support_tickets
        - 0.3 * (total_purchases / 3)
        - 0.15 * (avg_invoice / 100)
        + 0.6 * late_payments
        + 0.5 * (recency / 15)
        - 1.2 * (image_quality - 0.6)
        + (payment_method == "crypto") * 0.4
        + (tier == "basic") * 0.3
        - (tier == "enterprise") * 0.5
        - 0.8
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = (rng.random(n) < prob).astype(int)

    data = {
        cols["id"]: ids,
        cols["login"]: np.round(login_freq, 2),
        cols["session"]: np.round(avg_session, 2),
        cols["usage"]: np.round(usage_score, 2),
        cols["tickets"]: support_tickets,
        cols["purchases"]: total_purchases,
        cols["avg_invoice"]: np.round(avg_invoice, 2),
        cols["late"]: late_payments,
        cols["clv"]: np.round(clv, 2),
        cols["pay_method"]: payment_method,
        cols["tier"]: tier,
        cols["image"]: np.round(image_quality, 3),
        cols["target"]: churned,
        **extras,
    }
    return pd.DataFrame(data)


def generate_sample_images(
    df: pd.DataFrame, out_dir: Path, n: int = 20, seed: int = 42, cfg: dict | None = None
) -> list[Path]:
    cfg = cfg or load_config()
    s = schema(cfg)
    fusion = cfg.get("feature_fusion", {})
    img_col = fusion.get("image_score_column", fusion.get("image_score_col", "image_quality_score"))
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    sample = df.sample(min(n, len(df)), random_state=seed)
    for _, row in sample.iterrows():
        cid = row[s["id_column"]]
        q = float(row[img_col])
        # Higher quality -> bigger, sharper synthetic image
        size = int(128 + q * 384)
        img = Image.new(
            "RGB", (size, size), (int(200 * q + 30), int(180 * q + 30), int(170 * q + 30))
        )
        d = ImageDraw.Draw(img)
        for _ in range(12):
            x0, y0 = int(rng.integers(0, size)), int(rng.integers(0, size))
            x1, y1 = int(rng.integers(0, size)), int(rng.integers(0, size))
            d.line(
                [x0, y0, x1, y1],
                fill=tuple(int(x) for x in rng.integers(0, 255, 3)),
                width=3,
            )
        d.text((10, 10), cid, fill=(0, 0, 0))
        cdir = out_dir / cid
        cdir.mkdir(parents=True, exist_ok=True)
        p = cdir / "default.jpg"
        # low quality -> heavy JPEG compression
        quality = int(20 + q * 75)
        img.save(p, "JPEG", quality=quality)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--images", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config()
    s = schema(cfg)
    df = generate_customers(n=args.n, seed=project_seed(cfg))
    csv_path = artifact_paths(cfg)["raw_customers_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info(
        "synthetic_csv_written",
        path=str(csv_path),
        rows=len(df),
        churn_rate=float(df[s["target_column"]].mean()),
    )
    img_dir = resolve_path(cfg["paths"]["raw_images_dir"])
    paths = generate_sample_images(df, img_dir, n=args.images)
    log.info("sample_images_written", count=len(paths), dir=str(img_dir))
    print(
        f"Wrote {len(df)} customers to {csv_path} (churn_rate={df[s['target_column']].mean():.3f})"
    )
    print(f"Generated {len(paths)} sample images under {img_dir}.")


if __name__ == "__main__":
    main()
