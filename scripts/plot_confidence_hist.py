import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np


def _normalize_scores(df: pd.DataFrame, column: str, mode: str, group_cols) -> pd.Series:
    if mode == "none":
        return df[column]

    def _softmax(x):
        v = x - x.max()
        e = np.exp(v)
        return e / e.sum()

    def _percentile(x):
        return x.rank(pct=True)

    if mode == "softmax":
        return df.groupby(group_cols)[column].transform(_softmax)
    if mode == "zscore":
        return df.groupby(group_cols)[column].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    if mode == "minmax":
        return df.groupby(group_cols)[column].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
    if mode == "percentile":
        return df.groupby(group_cols)[column].transform(_percentile)
    raise ValueError(f"Unknown normalize mode: {mode}")


def plot_confidence_hist(
    csv_path: Path,
    out_path: Path,
    column: str = "confidence",
    normalize: str = "none",
    group_by: str = "head,rel",
) -> None:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")

    group_cols = [c.strip() for c in group_by.split(",") if c.strip()]
    for c in group_cols:
        if c and c not in df.columns:
            raise ValueError(f"Group-by column '{c}' not found in {csv_path}")

    df = df.copy()
    df["score_norm"] = _normalize_scores(df, column, normalize, group_cols if group_cols else [column])

    xlabel = {
        "none": "Ranking score (cos/dot)",
        "softmax": "Normalized score (per query, softmax)",
        "zscore": "Z-score (per query)",
        "minmax": "Min-max (per query)",
        "percentile": "Percentile (per query)",
    }.get(normalize, "Score")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.hist(df["score_norm"], bins=20, color="#7da0f7", alpha=0.9, edgecolor="white")
    ax.set_title("Confidence Distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot confidence histogram from a predictions CSV")
    parser.add_argument("csv", type=Path, help="CSV with a confidence column")
    parser.add_argument("--column", type=str, default="confidence", help="Column to plot (default: confidence)")
    parser.add_argument("--normalize", type=str, default="none", choices=["none", "softmax", "zscore", "minmax", "percentile"], help="Per-query normalization mode")
    parser.add_argument("--group-by", type=str, default="head,rel", help="Columns to group per query (comma-separated)")
    parser.add_argument("--out", type=Path, default=Path("confidence_hist.png"), help="Output image path")
    args = parser.parse_args()

    plot_confidence_hist(args.csv, args.out, column=args.column, normalize=args.normalize, group_by=args.group_by)
    print(f"Saved histogram to {args.out}")


if __name__ == "__main__":
    main()
