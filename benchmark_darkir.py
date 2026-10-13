
from pathlib import Path
import os
import time

import torch
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import to_tensor
from archs.DarkIR import DarkIR

BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE / "models" / "training_lolblur" / "best_official_format_fixed.pt"
TEST_ROOT = Path(os.environ.get("DARKIR_LOLBLUR_TEST", BASE / "data" / "datasets" / "LOLBlur" / "test"))
TEST_DIR = TEST_ROOT / "low_blur_noise"
OUT_DIR = BASE / "results" / "research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WARMUP_RUNS = 10
MEASURED_RUNS = 50

model = DarkIR(
    img_channel=3,
    width=32,
    middle_blk_num_enc=2,
    middle_blk_num_dec=2,
    enc_blk_nums=[1, 2, 3],
    dec_blk_nums=[3, 1, 1],
    dilations=[1, 4, 9],
    extra_depth_wise=True,
)

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
weights = checkpoint.get("params", checkpoint.get("model_state_dict", checkpoint))
weights = {k.removeprefix("module."): v for k, v in weights.items()}
model.load_state_dict(weights, strict=True)
model.to(DEVICE).eval()

image_paths = sorted(
    p for p in TEST_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)
if not image_paths:
    raise FileNotFoundError(f"No test images found under {TEST_DIR}")

image = Image.open(image_paths[0]).convert("RGB")
tensor = to_tensor(image).unsqueeze(0).to(DEVICE)

def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

with torch.inference_mode():
    for _ in range(WARMUP_RUNS):
        model(tensor)
    synchronize()

    times_ms = []
    for _ in range(MEASURED_RUNS):
        synchronize()
        start = time.perf_counter()
        model(tensor)
        synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)

summary = {
    "Device": str(DEVICE),
    "Input image": image_paths[0].relative_to(TEST_ROOT).as_posix(),
    "Image width": image.width,
    "Image height": image.height,
    "Trainable parameters": params,
    "Warm-up runs": WARMUP_RUNS,
    "Measured runs": MEASURED_RUNS,
    "Mean latency (ms)": sum(times_ms) / len(times_ms),
    "Median latency (ms)": float(pd.Series(times_ms).median()),
    "Std latency (ms)": float(pd.Series(times_ms).std()),
    "Approx. throughput (images/s)": 1000 / (sum(times_ms) / len(times_ms)),
}

pd.DataFrame([summary]).to_csv(OUT_DIR / "runtime_benchmark.csv", index=False)
(OUT_DIR / "model_complexity.txt").write_text(
    "\n".join(f"{key}: {value}" for key, value in summary.items()),
    encoding="utf-8"
)

print("\n".join(f"{key}: {value}" for key, value in summary.items()))
print(f"\nSaved benchmark to {OUT_DIR}")
