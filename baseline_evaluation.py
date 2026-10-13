
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lpips import LPIPS
from losses import SSIM

# Dataset paths. Override DARKIR_LOLBLUR_TEST when the dataset is elsewhere.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.environ.get(
    "DARKIR_LOLBLUR_TEST",
    os.path.join(PROJECT_ROOT, "data", "datasets", "LOLBlur", "test"),
)
INPUT_DIR = os.path.join(TEST_ROOT, "low_blur_noise")
TARGET_DIR = os.path.join(TEST_ROOT, "high_sharp_scaled")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PairedImages(Dataset):
    def __init__(self):
        self.pairs = []

        for root, _, files in os.walk(INPUT_DIR):
            for filename in files:
                if not filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                ):
                    continue

                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, INPUT_DIR)
                target_path = os.path.join(TARGET_DIR, relative_path)

                if os.path.isfile(target_path):
                    self.pairs.append((input_path, target_path))

        self.pairs.sort()

        if not self.pairs:
            raise RuntimeError("No matching input/target pairs found.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if input_image.size != target_image.size:
            raise ValueError(f"Image size mismatch: {input_path}")

        return to_tensor(input_image), to_tensor(target_image)


def main():
    dataset = PairedImages()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Matched pairs: {len(dataset)}")
    print(f"Device: {DEVICE}")

    ssim_fn = SSIM(data_range=1.).to(DEVICE)
    lpips_fn = LPIPS(net="vgg", verbose=False).to(DEVICE)

    psnr_values, ssim_values, lpips_values = [], [], []

    with torch.no_grad():
        for low, target in tqdm(loader, desc="Evaluating baseline"):
            low = low.to(DEVICE)
            target = target.to(DEVICE)

            mse = torch.mean((target - low) ** 2)

            # Match the PSNR formula used by testing.py
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            ssim = ssim_fn(low, target)
            lpips = lpips_fn(low, target)

            psnr_values.append(psnr.item())
            ssim_values.append(ssim.item())
            lpips_values.append(lpips.mean().item())

    print("\n===== ORIGINAL INPUT BASELINE =====")
    print(f"Images: {len(dataset)}")
    print(f"PSNR:   {np.mean(psnr_values):.6f} dB")
    print(f"SSIM:   {np.mean(ssim_values):.6f}")
    print(f"LPIPS:  {np.mean(lpips_values):.6f}")

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "baseline_results.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Original Input Baseline\n")
        f.write(f"Images: {len(dataset)}\n")
        f.write(f"PSNR: {np.mean(psnr_values):.6f} dB\n")
        f.write(f"SSIM: {np.mean(ssim_values):.6f}\n")
        f.write(f"LPIPS: {np.mean(lpips_values):.6f}\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
