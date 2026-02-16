from __future__ import annotations

import argparse

import pandas as pd

from analytics.xg_model import label_unique_count, train_and_persist


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline training for Rocket League xG models")
    parser.add_argument("--shots", required=False, help="CSV with engineered shot features and an is_goal column")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--calibration", default="isotonic", choices=["isotonic", "sigmoid"])
    args = parser.parse_args()

    df = pd.DataFrame()
    if args.shots:
        df = pd.read_csv(args.shots)
        if "is_goal" not in df.columns and "Result" in df.columns:
            df["is_goal"] = (df["Result"].astype(str).str.lower() == "goal").astype(int)

        if "is_goal" in df.columns and label_unique_count(df) < 2:
            raise SystemExit(
                "Training data has fewer than 2 distinct labels in is_goal. "
                "Refine your --shots input to include both goals and non-goals. "
                "If persisted as-is, training would fall back to a degenerate constant-probability scorer."
            )

    scorer = train_and_persist(df if not df.empty else None, model_version=args.model_version, calibration_method=args.calibration)
    print(
        f"trained model_version={scorer.metadata.model_version} "
        f"calibration={scorer.metadata.calibration_version} "
        f"pre_mode={scorer.metadata.pre_training_mode} "
        f"post_mode={scorer.metadata.post_training_mode}"
    )


if __name__ == "__main__":
    main()
