
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms.functional import to_tensor, to_pil_image
from archs.DarkIR import DarkIR
import lpips

# Paths. The sample images are intentionally excluded from public Git history.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "custom_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "custom_evaluation")
CHECKPOINT = os.path.join(
    BASE_DIR, "models", "training_lolblur",
    "best_official_format_fixed.pt"
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_model():
    model = DarkIR(
        img_channel=3,
        width=32,
        middle_blk_num_enc=2,
        middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3],
        dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9],
        extra_depth_wise=True
    )

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    weights = checkpoint.get(
        "params", checkpoint.get("model_state_dict", checkpoint)
    )
    weights = {
        key.removeprefix("module."): value
        for key, value in weights.items()
    }

    model.load_state_dict(weights, strict=True)
    model.to(DEVICE).eval()
    print(f"Loaded DarkIR on {DEVICE}")
    return model


def find_image(folder, stem):
    for ext in EXTENSIONS:
        path = os.path.join(folder, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def main():
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    restored_dir = os.path.join(OUTPUT_DIR, "restored")
    comparison_dir = os.path.join(OUTPUT_DIR, "comparisons")
    os.makedirs(restored_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    model = load_model()

    # LPIPS uses the VGG network and expects RGB tensors in [-1, 1].
    lpips_model = lpips.LPIPS(net="vgg").to(DEVICE).eval()

    results = []

    for i in range(1, 13):
        low_path = find_image(INPUT_DIR, f"low{i}")
        gt_path = find_image(INPUT_DIR, f"gt{i}")

        if not low_path or not gt_path:
            print(f"Skipping pair {i}: missing low{i} or gt{i}")
            continue

        low_img = Image.open(low_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if low_img.size != gt_img.size:
            print(
                f"Skipping pair {i}: dimensions differ "
                f"(low={low_img.size}, GT={gt_img.size})."
            )
            continue

        low_tensor = to_tensor(low_img).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            restored_tensor = model(low_tensor).clamp(0, 1)
            gt_tensor = to_tensor(gt_img).unsqueeze(0).to(DEVICE)

            # PSNR and SSIM use RGB images with values in [0, 1].
            restored_np = restored_tensor[0].cpu().permute(1, 2, 0).numpy()
            gt_np = gt_tensor[0].cpu().permute(1, 2, 0).numpy()

            psnr = peak_signal_noise_ratio(gt_np, restored_np, data_range=1.0)
            ssim = structural_similarity(
                gt_np, restored_np, channel_axis=2, data_range=1.0
            )

            lpips_value = lpips_model(
                restored_tensor * 2 - 1,
                gt_tensor * 2 - 1
            ).item()

        restored_img = to_pil_image(restored_tensor[0].cpu())
        restored_img.save(
            os.path.join(restored_dir, f"restored{i}.png")
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        panels = [
            (low_img, f"Low-light input {i}"),
            (restored_img, "DarkIR restored"),
            (gt_img, "Ground truth")
        ]

        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(
            f"Pair {i} | PSNR: {psnr:.3f} dB | "
            f"SSIM: {ssim:.4f} | LPIPS: {lpips_value:.4f}"
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(comparison_dir, f"comparison{i}.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        results.append({
            "Pair": i,
            "PSNR (dB)": psnr,
            "SSIM": ssim,
            "LPIPS": lpips_value
        })

        print(
            f"Pair {i}: PSNR={psnr:.3f} dB | "
            f"SSIM={ssim:.4f} | LPIPS={lpips_value:.4f}"
        )

    if not results:
        print("No valid image pairs found. Check filenames and dimensions.")
        return

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "custom_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\nIndividual results:")
    print(df.to_string(index=False))

    print("\nAverage metrics:")
    print(df[["PSNR (dB)", "SSIM", "LPIPS"]].mean().to_string())

    print(f"\nMetrics saved to: {csv_path}")
    print(f"Restored images: {restored_dir}")
    print(f"Comparisons: {comparison_dir}")


if __name__ == "__main__":
    main()
