"""Prepare CICIDS2017 data for the single-agent NIDS baseline.

Reads every CSV in data/raw/, cleans it, and writes:
    data/processed/train.csv   (80%)
    data/processed/test.csv    (20%)
    data/sample_flows.csv      (50 test rows used by the demo)

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --max-rows 300000   # smaller, faster subset
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
SAMPLE_PATH = ROOT / "data" / "sample_flows.csv"


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(
            f"No CSV files found in {RAW_DIR}. "
            "Download the CICIDS2017 MachineLearningCVE files and put them there."
        )
    frames = []
    for f in files:
        print(f"Reading {f.name} ...")
        frames.append(
            pd.read_csv(f, encoding="utf-8", encoding_errors="replace", low_memory=False)
        )
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Column names have leading/trailing spaces in CICIDS2017.
    df.columns = df.columns.str.strip()
    # A duplicated column name appears in some files; keep the first copy.
    df = df.loc[:, ~df.columns.duplicated()]

    if "Label" not in df.columns:
        raise SystemExit("Expected a 'Label' column but did not find one.")

    # 2. Infinity / NaN (mostly in 'Flow Bytes/s' and 'Flow Packets/s').
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 3. Exact duplicate rows.
    df = df.drop_duplicates()

    # 4. Labels: keep the attack name, add a binary target.
    df["Label"] = df["Label"].astype(str).str.strip()
    df = df.rename(columns={"Label": "attack_type"})
    df["label"] = (df["attack_type"].str.upper() != "BENIGN").astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Randomly keep at most this many rows after cleaning.",
    )
    args = parser.parse_args()

    df = clean(load_raw())
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=SEED).reset_index(drop=True)

    # Stratify on attack type so rare attacks appear in both splits.
    # Fall back to the binary label if some attack type has a single row.
    strat = df["attack_type"]
    if strat.value_counts().min() < 2:
        strat = df["label"]
    train, test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=strat
    )

    # Demo sample: about 25 benign and 25 attack flows from the test set.
    benign = test[test["label"] == 0]
    attack = test[test["label"] == 1]
    sample = pd.concat(
        [
            benign.sample(n=min(25, len(benign)), random_state=SEED),
            attack.sample(n=min(25, len(attack)), random_state=SEED),
        ]
    ).sample(frac=1, random_state=SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    sample.to_csv(SAMPLE_PATH, index=False)

    print(f"\nRows after cleaning: {len(df):,}")
    print(f"Train: {len(train):,}   Test: {len(test):,}   Demo sample: {len(sample)}")
    print("\nAttack types in the training set:")
    print(train["attack_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
