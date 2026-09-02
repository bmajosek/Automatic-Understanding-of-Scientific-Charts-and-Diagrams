"""Invalidate predictions whose reviewed claim content changed.

Only the 20 replaced unverifiable records are removed.  Every original
prediction file is copied to a timestamp-free audit snapshot before editing so
the operation is reversible and deterministic.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed_review")
    parser.add_argument(
        "--experiment-dir",
        default="results/experiments/review_test_stratified_1000",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    experiment_dir = Path(args.experiment_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not experiment_dir.is_absolute():
        experiment_dir = ROOT / experiment_dir

    reviewed = pd.read_csv(data_dir / "unverifiable_reviewed.csv")
    affected = set(
        reviewed.loc[
            reviewed["review_replaced"].astype(str).str.lower().eq("true"),
            "claim_id",
        ].astype(str)
    )
    if len(affected) != 20:
        raise ValueError(f"Expected 20 replaced claims, found {len(affected)}")

    predictions_dir = experiment_dir / "predictions"
    snapshot_dir = experiment_dir / "pre_review_prediction_snapshot"
    changed_files = 0
    removed_rows = 0
    for path in sorted(predictions_dir.glob("*/claims_pred.csv")):
        frame = pd.read_csv(path)
        if "claim_id" not in frame.columns:
            continue
        mask = frame["claim_id"].astype(str).isin(affected)
        count = int(mask.sum())
        if count == 0:
            continue
        backup = snapshot_dir / path.parent.name / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        retained = frame.loc[~mask].copy()
        temporary = path.with_suffix(path.suffix + ".tmp")
        retained.to_csv(temporary, index=False)
        temporary.replace(path)
        changed_files += 1
        removed_rows += count
        print(f"{path.parent.name}: removed {count}, retained {len(retained)}")

    print(f"Changed files: {changed_files}")
    print(f"Invalidated prediction rows: {removed_rows}")
    print(f"Original files retained below: {snapshot_dir}")


if __name__ == "__main__":
    main()
