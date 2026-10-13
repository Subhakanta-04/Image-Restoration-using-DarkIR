
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results" / "research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Previously measured results from the existing DarkIR evaluation.
# These are aggregate values, not individual-image observations.
data = pd.DataFrame({
    "Condition": ["Low blur", "Low blur + noise", "Low-light sharp"],
    "PSNR (dB)": [23.792050, 23.121011, 28.016862],
    "SSIM": [0.770613, 0.714039, 0.941119],
    "LPIPS": [0.313093, 0.383748, 0.120857],
})

data.to_csv(OUT_DIR / "condition_metrics.csv", index=False)

metrics = [
    ("PSNR (dB)", "PSNR (dB) — higher is better"),
    ("SSIM", "SSIM — higher is better"),
    ("LPIPS", "LPIPS — lower is better"),
]

# Separate plots are easier to read in a conference paper.
for column, title in metrics:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(data["Condition"], data[column],
                  edgecolor="black", color="white", hatch="//")

    ax.set_title(title)
    ax.set_ylabel(column)
    ax.tick_params(axis="x", rotation=12)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, value in zip(bars, data[column]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center", va="bottom", fontsize=9
        )

    fig.tight_layout()
    filename = column.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Combined, normalized view for comparing metric patterns.
# Each metric is scaled independently for visualization only.
normalized = data[["PSNR (dB)", "SSIM", "LPIPS"]].copy()
normalized["LPIPS"] = 1 - normalized["LPIPS"]  # reverse direction for display
normalized = (normalized - normalized.min()) / (
    normalized.max() - normalized.min()
)

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(data))
width = 0.23

for index, metric in enumerate(["PSNR (dB)", "SSIM", "LPIPS"]):
    ax.bar(
        x + (index - 1) * width,
        normalized[metric],
        width,
        label=metric,
        edgecolor="black",
        facecolor="white",
        hatch=["//", "\\\\", "xx"][index],
    )

ax.set_xticks(x)
ax.set_xticklabels(data["Condition"], rotation=10)
ax.set_ylabel("Min–max normalized score")
ax.set_title("DarkIR Performance Across Test Conditions")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "metric_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(data.to_string(index=False))
print(f"\nCharts and CSV saved in: {OUT_DIR}")
