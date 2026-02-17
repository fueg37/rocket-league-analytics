"""Deterministic recommendation policy for partnership intelligence cards."""

from __future__ import annotations

import pandas as pd


EMPTY_RECOMMENDATION = {
    "label": "No qualifying pair",
    "detail": "Needs additional chemistry samples.",
}


def build_pair_recommendations(pair_df: pd.DataFrame, *, min_samples: int) -> dict[str, dict[str, str]]:
    """Build recommendation cards from partnership contract fields.

    Selection policy is deterministic and stable under ties.
    """
    if pair_df is None or pair_df.empty:
        return {
            "aggressive": EMPTY_RECOMMENDATION,
            "stabilizer": EMPTY_RECOMMENDATION,
            "high_confidence": EMPTY_RECOMMENDATION,
            "watchlist": EMPTY_RECOMMENDATION,
        }

    data = pair_df.copy()
    sample_col = "sample_count" if "sample_count" in data.columns else "Samples"
    index_col = "Partnership Index" if "Partnership Index" in data.columns else (
        "ChemistryScore_Shrunk" if "ChemistryScore_Shrunk" in data.columns else "ChemistryScore"
    )
    impact_col = "expected_xgd_lift_per_match" if "expected_xgd_lift_per_match" in data.columns else "ExpectedValueGain_Shrunk"
    conf_col = "confidence_level" if "confidence_level" in data.columns else "Reliability"

    data[sample_col] = pd.to_numeric(data.get(sample_col, 0), errors="coerce").fillna(0).astype(int)
    data[index_col] = pd.to_numeric(data.get(index_col, 0), errors="coerce").fillna(0.0)
    data[impact_col] = pd.to_numeric(data.get(impact_col, 0), errors="coerce").fillna(0.0)

    def _normalize_conf(value: str) -> str:
        bucket = str(value or "Low").strip().lower()
        if bucket == "high":
            return "High"
        if bucket == "medium":
            return "Medium"
        return "Low"

    data[conf_col] = data.get(conf_col, "Low").apply(_normalize_conf)
    data["pair_name"] = data["Player1"].astype(str) + " + " + data["Player2"].astype(str)

    qualifying = data[data[sample_col] >= int(min_samples)].copy()
    high_conf = qualifying[qualifying[conf_col] == "High"].copy()

    def _pick(df: pd.DataFrame, sort_cols: list[str], ascending: list[bool]):
        if df.empty:
            return None
        return df.sort_values(sort_cols, ascending=ascending, kind="mergesort").iloc[0]

    aggressive = _pick(
        qualifying,
        [impact_col, index_col, sample_col, "pair_name"],
        [False, False, False, True],
    )

    has_stabilizer_fields = "Pressure Escape" in qualifying.columns and "Rotation Fit" in qualifying.columns
    stabilizer = _pick(
        qualifying,
        ["Pressure Escape", "Rotation Fit", index_col, sample_col, "pair_name"],
        [False, False, False, False, True],
    ) if has_stabilizer_fields else _pick(
        qualifying,
        [index_col, sample_col, "pair_name"],
        [False, False, True],
    )

    best_high_conf = _pick(
        high_conf,
        [index_col, impact_col, sample_col, "pair_name"],
        [False, False, False, True],
    )

    watchlist = _pick(
        data[
            (data[sample_col] < int(min_samples))
            & (data[sample_col] >= 1)
            & (data[conf_col].isin(["Low", "Medium"]))
        ].copy(),
        [impact_col, index_col, sample_col, "pair_name"],
        [False, False, True, True],
    )

    def _format_card(row, fallback: str) -> dict[str, str]:
        if row is None:
            return {"label": fallback, "detail": "No qualifying pair under current sample/confidence filters."}
        detail = (
            f"Index {float(row[index_col]):.1f} • "
            f"xGD {float(row[impact_col]):+.3f} • "
            f"{str(row[conf_col]).title()} confidence ({int(row[sample_col])} samples)"
        )
        driver = str(row.get("primary_driver_label", "Balanced chemistry")).strip()
        return {"label": str(row["pair_name"]), "detail": detail + f" • Driver: {driver}"}

    return {
        "aggressive": _format_card(aggressive, "No recommended aggressive partner"),
        "stabilizer": _format_card(stabilizer, "No recommended stabilizer"),
        "high_confidence": _format_card(best_high_conf, "No high-confidence pair"),
        "watchlist": _format_card(watchlist, "No watchlist pair"),
    }
