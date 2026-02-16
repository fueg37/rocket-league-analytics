from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from analytics.features import FEATURE_VERSION, POST_SHOT_FEATURE_COLUMNS, PRE_SHOT_FEATURE_COLUMNS

ARTIFACT_DIR = Path("analytics/artifacts")
MANIFEST_FILE = ARTIFACT_DIR / "xg_manifest.json"


@dataclass
class XGMetadata:
    model_version: str
    calibration_version: str
    feature_version: str
    trained_at_utc: str
    calibration_method: str
    pre_features: list[str]
    post_features: list[str]


class XGScorer:
    def __init__(self, pre_model, post_model, metadata: XGMetadata):
        self.pre_model = pre_model
        self.post_model = post_model
        self.metadata = metadata

    def predict(self, pre_features: Dict[str, float], post_features: Dict[str, float]) -> Tuple[float, float]:
        pre_x = pd.DataFrame([{k: pre_features.get(k, np.nan) for k in PRE_SHOT_FEATURE_COLUMNS}])
        post_x = pd.DataFrame([{k: post_features.get(k, np.nan) for k in POST_SHOT_FEATURE_COLUMNS}])
        xg_pre = float(self.pre_model.predict_proba(pre_x)[:, 1][0])
        xg_post = float(self.post_model.predict_proba(post_x)[:, 1][0])
        return float(np.clip(xg_pre, 0.001, 0.999)), float(np.clip(xg_post, 0.001, 0.999))



def _train_model(df: pd.DataFrame, feature_cols: list[str], calibration_method: str):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    work = df.copy()
    y = pd.to_numeric(work["is_goal"], errors="coerce").fillna(0).astype(int)
    X = work[feature_cols]
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=800, class_weight="balanced")),
    ])

    class_counts = y.value_counts(dropna=False)
    min_class = int(class_counts.min()) if not class_counts.empty else 0
    if min_class >= 3:
        model = CalibratedClassifierCV(base, method=calibration_method, cv=3)
    else:
        model = base
    model.fit(X, y)
    return model


def _bootstrap_training_data(n: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    pre = pd.DataFrame({
        "distance_to_goal": rng.uniform(200, 9000, n),
        "shot_angle": rng.uniform(0.01, 1.5, n),
        "ball_height": rng.uniform(0, 1000, n),
        "shooter_speed": rng.uniform(0, 2300, n),
        "pressure_score": rng.uniform(0, 1, n),
        "nearest_defender_distance": rng.uniform(120, 2600, n),
        "shooter_boost": rng.uniform(0, 100, n),
        "buildup_seconds": rng.uniform(0, 8, n),
        "touches_in_chain": rng.integers(1, 8, n),
        "chain_duration": rng.uniform(0.1, 12, n),
        "chain_avg_ball_speed": rng.uniform(300, 3000, n),
        "chain_final_third_entries": rng.integers(0, 4, n),
        "chain_turnovers_forced": rng.integers(0, 2, n),
    })
    post = pd.DataFrame({
        "shot_speed": rng.uniform(200, 3500, n),
        "shot_vx_norm": rng.uniform(-1, 1, n),
        "shot_vy_norm": rng.uniform(-1, 1, n),
        "shot_vz_norm": rng.uniform(-1, 1, n),
        "target_placement_x": rng.uniform(0, 1.7, n),
        "shot_height": rng.uniform(0, 900, n),
        "keeper_distance": rng.uniform(100, 3500, n),
        "keeper_line_offset": rng.uniform(0, 2000, n),
    })

    logit = (
        1.2 * pre["shot_angle"]
        - 0.00042 * pre["distance_to_goal"]
        - 1.1 * pre["pressure_score"]
        + 0.00028 * post["shot_speed"]
        + 0.65 * (post["target_placement_x"] > 0.65).astype(float)
        - 0.00025 * post["keeper_distance"]
        + rng.normal(0, 0.35, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)

    return pd.concat([pre, post, pd.DataFrame({"is_goal": y})], axis=1)


def train_and_persist(training_df: pd.DataFrame | None = None, model_version: str = "v1", calibration_method: str = "isotonic") -> XGScorer:
    from joblib import dump

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = training_df.copy() if training_df is not None and not training_df.empty else _bootstrap_training_data()
    for col in PRE_SHOT_FEATURE_COLUMNS + POST_SHOT_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if "is_goal" not in df.columns:
        raise ValueError("training data must include is_goal label")

    pre_model = _train_model(df, PRE_SHOT_FEATURE_COLUMNS, calibration_method)
    post_model = _train_model(df, POST_SHOT_FEATURE_COLUMNS, calibration_method)

    metadata = XGMetadata(
        model_version=model_version,
        calibration_version=f"{model_version}-{calibration_method}",
        feature_version=FEATURE_VERSION,
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        calibration_method=calibration_method,
        pre_features=PRE_SHOT_FEATURE_COLUMNS,
        post_features=POST_SHOT_FEATURE_COLUMNS,
    )

    artifact_name = f"xg_model_{metadata.model_version}_{metadata.calibration_method}.joblib"
    artifact_path = ARTIFACT_DIR / artifact_name
    dump({"pre_model": pre_model, "post_model": post_model, "metadata": asdict(metadata)}, artifact_path)

    manifest = {
        "latest": artifact_name,
        "artifacts": [{"artifact": artifact_name, **asdict(metadata)}],
    }
    if MANIFEST_FILE.exists():
        with MANIFEST_FILE.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_items = existing.get("artifacts", [])
        existing_items = [a for a in existing_items if a.get("artifact") != artifact_name]
        manifest["artifacts"] = existing_items + manifest["artifacts"]
    with MANIFEST_FILE.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return XGScorer(pre_model, post_model, metadata)


def load_or_train_latest() -> XGScorer:
    from joblib import load

    if MANIFEST_FILE.exists():
        with MANIFEST_FILE.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        latest = manifest.get("latest")
        if latest:
            path = ARTIFACT_DIR / latest
            if path.exists():
                payload = load(path)
                meta = XGMetadata(**payload["metadata"])
                return XGScorer(payload["pre_model"], payload["post_model"], meta)
    return train_and_persist()
