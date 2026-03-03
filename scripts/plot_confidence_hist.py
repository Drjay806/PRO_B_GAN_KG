import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_confidence_hist(csv_path: Path, out_path: Path, column: str = "confidence") -> None:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.hist(df[column], bins=20, color="#7da0f7", alpha=0.9, edgecolor="white")
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Ranking score (cos/dot)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot confidence histogram from a predictions CSV")
    parser.add_argument("csv", type=Path, help="CSV with a confidence column")
    parser.add_argument("--column", type=str, default="confidence", help="Column to plot (default: confidence)")
    parser.add_argument("--out", type=Path, default=Path("confidence_hist.png"), help="Output image path")
    args = parser.parse_args()

    plot_confidence_hist(args.csv, args.out, column=args.column)
    print(f"Saved histogram to {args.out}")


if __name__ == "__main__":
    main()
