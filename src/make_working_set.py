"""Make the small working set used by the single-agent co-evolution task.

Reads data/processed/train.csv and test.csv (from prepare_data.py) and writes
a class-capped, stratified sample that is small enough to commit to Git and
to retrain on in a few seconds:
    data/working/train.csv.gz
    data/working/test.csv.gz

Usage:
    python src/make_working_set.py
"""

from pathlib import Path

import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "working"
CAPS = {"train": {"BENIGN": 30000, "attack": 5000},
        "test": {"BENIGN": 10000, "attack": 2000}}


def cap(df: pd.DataFrame, caps: dict) -> pd.DataFrame:
    parts = []
    for name, group in df.groupby("attack_type"):
        n = caps["BENIGN"] if name == "BENIGN" else caps["attack"]
        parts.append(group.sample(n=min(n, len(group)), random_state=SEED))
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        path = IN_DIR / f"{split}.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found. Run: python src/prepare_data.py")
        df = cap(pd.read_csv(path), CAPS[split])
        out = OUT_DIR / f"{split}.csv.gz"
        df.to_csv(out, index=False, compression="gzip")
        print(f"{split}: {len(df):,} rows -> {out.relative_to(ROOT)} "
              f"({out.stat().st_size / 1e6:.1f} MB)")
        print(df["attack_type"].value_counts().to_string(), "\n")


if __name__ == "__main__":
    main()
