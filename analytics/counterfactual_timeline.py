"""Counterfactual overlays for timeline tracks."""

from __future__ import annotations

import pandas as pd


def apply_counterfactual(
    timeline_df: pd.DataFrame,
    *,
    event_id: str | None = None,
    intervention: str = "none",
    scale: float = 0.15,
) -> pd.DataFrame:
    """Return an alternative trajectory series after a simple intervention."""
    if timeline_df is None or timeline_df.empty:
        return pd.DataFrame()

    out = timeline_df.copy()
    if "time" not in out.columns:
        out["time"] = pd.to_numeric(out.get("Time", 0.0), errors="coerce").fillna(0.0)

    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and c not in {"time", "Time"}]
    if not numeric_cols:
        return out

    if intervention == "remove_event" and event_id and "event_id" in out.columns:
        out = out[out["event_id"] != event_id].copy()
    elif intervention == "swap_action_class":
        for col in numeric_cols:
            out[col] = out[col] * (1.0 - scale)
    elif intervention == "adjust_possession_outcome":
        for col in numeric_cols:
            out[col] = out[col] + scale

    return out.reset_index(drop=True)
