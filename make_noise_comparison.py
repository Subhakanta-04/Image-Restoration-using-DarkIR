
from pathlib import Path
import csv
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from archs import create_model
from options.options import parse

PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DARKIR_LOLBLUR_TEST", PROJECT_ROOT / "data" / "datasets" / "LOLBlur" / "test"))
INPUT_DIR = ROOT / "low_blur_noise"
TARGET_DIR = ROOT / "high_sharp_scaled"
CONFIG = PROJECT_ROOT / "options" / "test" / "LOLBlur.yml"
CHECKPOINT = PROJECT_ROOT / "models" / "training_lolblur" / "best_official_format_fixed.pt"
OUT = PROJECT_ROOT / "results" / "noise_robustness" / "qualitative_v2"
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = [5, 10, 20, 30]
SEED = 2025
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def tensor_of(image):
    return torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    ).float().unsqueeze(0)

def restore(image, model):
    x = tensor_of(image).to(DEVICE)
    _, _, h, w = x.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    x = torch.nn.functional.pad(x, (0, pw, 0, ph))

    with torch.inference_mode():
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y = y[..., :h, :w]

    return y.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()

def metrics(pred, target, lpips_model):
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)

    psnr = peak_signal_noise_ratio(target, pred, data_range=1.0)
    ssim = structural_similarity(
        target, pred, channel_axis=2, data_range=1.0
    )

    a = (tensor_of(pred).to(DEVICE) * 2) - 1
    b = (tensor_of(target).to(DEVICE) * 2) - 1
    with torch.inference_mode():
        lp = lpips_model(a, b).item()

    return psnr, ssim, lp

# Find matching scene/frame pairs
inputs = {
    p.relative_to(INPUT_DIR).as_posix(): p
    for p in INPUT_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTS
}
targets = {
    p.relative_to(TARGET_DIR).as_posix(): p
    for p in TARGET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTS
}
keys = sorted(set(inputs) & set(targets))
if not keys:
    raise RuntimeError("No matching input/reference pairs found.")

# Deterministically select an image
key = keys[0]
print("Selected pair:", key)

# Load the existing trained checkpoint
opt = parse(str(CONFIG))
model, _, _ = create_model(opt["network"], rank=0)
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
weights = checkpoint.get(
    "params", checkpoint.get("model_state_dict", checkpoint)
)

model_keys = set(model.state_dict())
if weights and all(k.startswith("module.") for k in weights):
    weights = {k[7:]: v for k, v in weights.items()}
elif model_keys and all(k.startswith("module.") for k in model_keys):
    weights = {"module." + k: v for k, v in weights.items()}

model.load_state_dict(weights, strict=True)
model = model.to(DEVICE).eval()
lpips_model = lpips.LPIPS(net="alex").to(DEVICE).eval()

original = read_rgb(inputs[key])
reference = read_rgb(targets[key])
if original.shape != reference.shape:
    raise RuntimeError(
        f"Input/reference dimensions differ: {original.shape} vs {reference.shape}"
    )

# Save the exact noise inputs and outputs used for this figure.
# Each noise realization is reproducible from the fixed seed.
rng = np.random.default_rng(SEED)
panels = [("Reference", reference), ("Original degraded input", original)]
rows = []

for sigma in LEVELS:
    noise = rng.normal(
        0, sigma / 255.0, size=original.shape
    ).astype(np.float32)
    noisy = np.clip(original + noise, 0, 1)
    restored = restore(noisy, model)

    input_metrics = metrics(noisy, reference, lpips_model)
    restored_metrics = metrics(restored, reference, lpips_model)

    panels.append((f"Noisy input (sigma={sigma})", noisy))
    panels.append((f"DarkIR restored (sigma={sigma})", restored))

    rows.append({
        "image": key,
        "sigma": sigma,
        "noisy_psnr": input_metrics[0],
        "restored_psnr": restored_metrics[0],
        "noisy_ssim": input_metrics[1],
        "restored_ssim": restored_metrics[1],
        "noisy_lpips": input_metrics[2],
        "restored_lpips": restored_metrics[2],
    })

    Image.fromarray((noisy * 255).round().astype(np.uint8)).save(
        OUT / f"noisy_sigma_{sigma}.png"
    )
    Image.fromarray((restored * 255).round().astype(np.uint8)).save(
        OUT / f"restored_sigma_{sigma}.png"
    )

# Save per-image metrics for this exact figure
with (OUT / "qualitative_metrics.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Ten-panel figure, with shared display scaling
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for ax, (title, image) in zip(axes.flat, panels):
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

fig.suptitle(
    f"DarkIR Gaussian Noise Robustness — {key}",
    fontsize=16
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "qualitative_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSaved figure:", OUT / "qualitative_comparison.png")
print("Saved per-image metrics:", OUT / "qualitative_metrics.csv")
for row in rows:
    print(row)
