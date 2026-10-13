
import os
import csv
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from archs import create_model
from options.options import parse

# ---------------- CONFIGURATION ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DARKIR_LOLBLUR_TEST", PROJECT_ROOT / "data" / "datasets" / "LOLBlur" / "test"))
INPUT_DIR = ROOT / "low_blur_noise"
TARGET_DIR = ROOT / "high_sharp_scaled"

CONFIG = PROJECT_ROOT / "options" / "test" / "LOLBlur.yml"
CHECKPOINT = PROJECT_ROOT / "models" / "training_lolblur" / "best_official_format_fixed.pt"

OUTPUT_DIR = PROJECT_ROOT / "results" / "noise_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 100
NOISE_LEVELS = [5, 10, 20, 30]  # 8-bit pixel scale
SEED = 2025

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- CHECK FILES ----------------
for path in [INPUT_DIR, TARGET_DIR, CONFIG, CHECKPOINT]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

# Match by relative path, e.g. 0012/0010.png
extensions = {".png", ".jpg", ".jpeg", ".bmp"}
input_files = {
    p.relative_to(INPUT_DIR).as_posix(): p
    for p in INPUT_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
}
target_files = {
    p.relative_to(TARGET_DIR).as_posix(): p
    for p in TARGET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
}

keys = sorted(set(input_files) & set(target_files))
if not keys:
    raise RuntimeError("No matching input/reference image pairs found.")

random.shuffle(keys)
keys = keys[:min(NUM_IMAGES, len(keys))]
print(f"Matched pairs selected: {len(keys)}")

# ---------------- LOAD DARKIR ----------------
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
model = model.to(device).eval()
print("DarkIR checkpoint loaded.")

# LPIPS expects RGB tensors in [-1, 1].
lpips_model = lpips.LPIPS(net="alex").to(device).eval()

def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def to_tensor(image):
    return torch.from_numpy(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    ).float().unsqueeze(0)

def pad_to_multiple(tensor, multiple=8):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))

def restore(image):
    tensor = to_tensor(image).to(device)
    h, w = tensor.shape[-2:]
    padded = pad_to_multiple(tensor)

    with torch.inference_mode():
        prediction = model(padded)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        prediction = prediction[..., :h, :w]

    return prediction.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()

def calculate_metrics(pred, target):
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)

    psnr = peak_signal_noise_ratio(target, pred, data_range=1.0)
    ssim = structural_similarity(
        target, pred, channel_axis=2, data_range=1.0
    )

    pred_tensor = to_tensor(pred).to(device) * 2 - 1
    target_tensor = to_tensor(target).to(device) * 2 - 1

    with torch.inference_mode():
        lp = lpips_model(pred_tensor, target_tensor).item()

    return psnr, ssim, lp

# ---------------- RUN EXPERIMENT ----------------
results = []
preview_saved = set()

for sigma in NOISE_LEVELS:
    print(f"\nGaussian noise sigma={sigma}")

    for index, key in enumerate(keys, start=1):
        clean_input = read_rgb(input_files[key])
        target = read_rgb(target_files[key])

        if clean_input.shape != target.shape:
            print(f"Skipping mismatched dimensions: {key}")
            continue

        # Noise standard deviation is sigma/255.
        noise = np.random.normal(
            loc=0.0, scale=sigma / 255.0, size=clean_input.shape
        ).astype(np.float32)

        noisy_input = np.clip(clean_input + noise, 0, 1)
        restored = restore(noisy_input)

        input_metrics = calculate_metrics(noisy_input, target)
        output_metrics = calculate_metrics(restored, target)

        results.append({
            "image": key,
            "noise_sigma_8bit": sigma,
            "input_psnr": input_metrics[0],
            "input_ssim": input_metrics[1],
            "input_lpips": input_metrics[2],
            "restored_psnr": output_metrics[0],
            "restored_ssim": output_metrics[1],
            "restored_lpips": output_metrics[2],
        })

        if index == 1 and sigma in (5, 20, 30):
            preview_dir = OUTPUT_DIR / f"examples_sigma_{sigma}"
            preview_dir.mkdir(exist_ok=True)
            Image.fromarray((noisy_input * 255).round().astype(np.uint8)).save(
                preview_dir / "noisy.png"
            )
            Image.fromarray((restored * 255).round().astype(np.uint8)).save(
                preview_dir / "restored.png"
            )
            Image.fromarray((target * 255).round().astype(np.uint8)).save(
                preview_dir / "reference.png"
            )

        if index % 25 == 0:
            print(f"  Processed {index}/{len(keys)}")

# ---------------- SAVE CSV ----------------
csv_path = OUTPUT_DIR / "noise_robustness_metrics.csv"
fields = list(results[0].keys())

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

# Aggregate means by noise level
summary = []
for sigma in NOISE_LEVELS:
    rows = [r for r in results if r["noise_sigma_8bit"] == sigma]
    if not rows:
        continue

    summary.append({
        "sigma": sigma,
        **{
            metric: float(np.mean([r[metric] for r in rows]))
            for metric in [
                "input_psnr", "restored_psnr",
                "input_ssim", "restored_ssim",
                "input_lpips", "restored_lpips"
            ]
        }
    })

summary_path = OUTPUT_DIR / "noise_robustness_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

# ---------------- PLOT METRICS ----------------
plots = [
    ("PSNR (dB)", "input_psnr", "restored_psnr", "noise_psnr.png"),
    ("SSIM", "input_ssim", "restored_ssim", "noise_ssim.png"),
    ("LPIPS (lower is better)", "input_lpips", "restored_lpips", "noise_lpips.png"),
]

for title, input_key, output_key, filename in plots:
    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["sigma"] for r in summary],
        [r[input_key] for r in summary],
        marker="o", label="Noisy input"
    )
    plt.plot(
        [r["sigma"] for r in summary],
        [r[output_key] for r in summary],
        marker="s", label="DarkIR restored"
    )
    plt.xlabel("Gaussian noise standard deviation (8-bit scale)")
    plt.ylabel(title)
    plt.title(f"{title} under increasing Gaussian noise")
    plt.xticks(NOISE_LEVELS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

print("\nExperiment complete!")
print("Summary:", summary_path)
print("Per-image metrics:", csv_path)
print("Graphs:", OUTPUT_DIR)
print("\nMean metrics by noise level:")
for row in summary:
    print(row)
