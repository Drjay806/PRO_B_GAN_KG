import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


PHASE_ORDER: List[str] = ["pretrain", "warmup", "gan"]
PHASE_COLORS: Dict[str, str] = {
    "pretrain": "#d0e1ff",
    "warmup": "#dfffe0",
    "gan": "#ffe6d9",
}


def compute_global_epochs(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]]]:
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 1
    df = df.copy()
    global_epochs = []
    for phase in PHASE_ORDER:
        phase_df = df[df["phase"] == phase]
        count = len(phase_df)
        if count == 0:
            continue
        start = cursor
        end = cursor + count - 1
        spans[phase] = (start, end)
        global_epochs.extend(range(start, end + 1))
        cursor = end + 1
        df.loc[phase_df.index, "global_epoch"] = list(range(start, end + 1))
    return df.sort_values("global_epoch"), spans


def plot_mrr_hero(df: pd.DataFrame, spans: Dict[str, Tuple[int, int]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    # Phase shading
    for phase, (start, end) in spans.items():
        color = PHASE_COLORS.get(phase, "#f0f0f0")
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.3, label=phase.capitalize())
        ax.text((start + end) / 2, ax.get_ylim()[1] if ax.lines else 0.05, phase.capitalize(),
            ha="center", va="bottom", fontsize=11, color="#444")

    # MRR curve
    ax.plot(df["global_epoch"], df["mrr"], color="#1f77b4", linewidth=2.2, marker="o", markersize=4, label="MRR")

    # Titles and labels
    ax.set_title("MRR Convergence (PoC)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch (global)", fontsize=13)
    ax.set_ylabel("Filtered MRR", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(df["global_epoch"].min() - 0.5, df["global_epoch"].max() + 0.5)

    # Annotations (max two) if spans exist
    if "warmup" in spans:
        ws, we = spans["warmup"]
        mid_w = (ws + we) / 2
        y_w = df[df["phase"] == "warmup"]["mrr"].mean()
        ax.annotate("Warmup stabilizes",
                xy=(we, df[df["phase"] == "warmup"]["mrr"].iloc[-1]),
                xytext=(mid_w, y_w + 0.002),
                arrowprops=dict(arrowstyle="->", color="#555"),
                fontsize=12, color="#333")
    if "gan" in spans:
        gs, ge = spans["gan"]
        mid_g = (gs + ge) / 2
        y_g = df[df["phase"] == "gan"]["mrr"].mean()
        ax.annotate("GAN improves trend",
                    xy=(gs, df[df["phase"] == "gan"]["mrr"].iloc[0] if len(df[df["phase"] == "gan"]) > 0 else y_g),
                    xytext=(mid_g, y_g + 0.002),
                    arrowprops=dict(arrowstyle="->", color="#555"),
                    fontsize=12, color="#333")

    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MRR convergence hero chart")
    parser.add_argument("csv", type=Path, help="Path to metrics_log.csv")
    parser.add_argument("--out", type=Path, default=Path("hero_mrr.png"), help="Output image path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "mrr" not in df.columns or "phase" not in df.columns:
        raise ValueError("metrics_log.csv must have columns: phase, mrr")

    df, spans = compute_global_epochs(df)
    if df.empty:
        raise ValueError("metrics_log.csv is empty")

    plot_mrr_hero(df, spans, args.out)
    print(f"Saved hero chart to {args.out}")


if __name__ == "__main__":
    main()
