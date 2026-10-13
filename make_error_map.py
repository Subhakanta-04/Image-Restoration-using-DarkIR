
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

BASE = Path(__file__).resolve().parent

# Change these two paths to a matching restored image and ground truth.
RESTORED = BASE / "results" / "custom_evaluation" / "restored" / "restored1.png"
GROUND_TRUTH = BASE / "custom_images" / "gt1.jpeg"

OUT = BASE / "results" / "research" / "error_maps"
OUT.mkdir(parents=True, exist_ok=True)

restored = np.asarray(Image.open(RESTORED).convert("RGB"), dtype=np.float32) / 255.0
gt = np.asarray(Image.open(GROUND_TRUTH).convert("RGB"), dtype=np.float32) / 255.0

if restored.shape != gt.shape:
    raise ValueError(
        f"Image dimensions differ: restored={restored.shape}, GT={gt.shape}"
    )

error = np.abs(restored - gt)
mean_absolute_error = float(error.mean())

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
axes[0].imshow(restored)
axes[0].set_title("Restored")
axes[1].imshow(gt)
axes[1].set_title("Ground truth")
axes[2].imshow(error.mean(axis=2), cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Absolute error")

for ax in axes:
    ax.axis("off")

fig.suptitle(f"Absolute reconstruction error | MAE = {mean_absolute_error:.5f}")
fig.tight_layout()
fig.savefig(OUT / "error_map_pair1.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Mean absolute error: {mean_absolute_error:.6f}")
print(f"Saved: {OUT / 'error_map_pair1.png'}")
