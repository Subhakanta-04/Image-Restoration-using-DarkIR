
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lpips import LPIPS
from losses import SSIM

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.environ.get(
    "DARKIR_LOLBLUR_TEST",
    os.path.join(PROJECT_ROOT, "data", "datasets", "LOLBlur", "test"),
)
TARGET_DIR = os.path.join(TEST_ROOT, "high_sharp_scaled")

INPUT_DIRS = {
    "low_blur": os.path.join(TEST_ROOT, "low_blur"),
    "low_sharp": os.path.join(TEST_ROOT, "low_sharp"),
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PairedImages(Dataset):
    def __init__(self, input_dir):
        self.pairs = []

        for root, _, files in os.walk(input_dir):
            for filename in files:
                if not filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                ):
                    continue

                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_path, input_dir)
                target_path = os.path.join(TARGET_DIR, relative_path)

                if os.path.isfile(target_path):
                    self.pairs.append((input_path, target_path))

        self.pairs.sort()

        if not self.pairs:
            raise RuntimeError(
                f"No matching pairs found for {input_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if input_image.size != target_image.size:
            raise ValueError(f"Image size mismatch: {input_path}")

        return to_tensor(input_image), to_tensor(target_image)


def evaluate_condition(name, input_dir, ssim_fn, lpips_fn):
    dataset = PairedImages(input_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    psnr_values = []
    ssim_values = []
    lpips_values = []

    print(f"\nEvaluating condition: {name}")
    print(f"Matched image pairs: {len(dataset)}")

    with torch.no_grad():
        for low, target in tqdm(loader, desc=name):
            low = low.to(DEVICE)
            target = target.to(DEVICE)

            mse = torch.mean((target - low) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            ssim = ssim_fn(low, target)
            lpips = lpips_fn(low, target)

            psnr_values.append(psnr.item())
            ssim_values.append(ssim.item())
            lpips_values.append(lpips.mean().item())

    results = {
        "condition": name,
        "images": len(dataset),
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "LPIPS": float(np.mean(lpips_values)),
    }

    print(f"\n{name} baseline results:")
    print(f"PSNR:  {results['PSNR']:.6f} dB")
    print(f"SSIM:  {results['SSIM']:.6f}")
    print(f"LPIPS: {results['LPIPS']:.6f}")

    return results


def main():
    print(f"Device: {DEVICE}")

    ssim_fn = SSIM(data_range=1.).to(DEVICE)
    lpips_fn = LPIPS(net="vgg", verbose=False).to(DEVICE)

    all_results = []

    for name, input_dir in INPUT_DIRS.items():
        result = evaluate_condition(
            name, input_dir, ssim_fn, lpips_fn
        )
        all_results.append(result)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "remaining_baseline_results.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Original Input Baseline Results\n\n")

        for result in all_results:
            f.write(f"Condition: {result['condition']}\n")
            f.write(f"Images: {result['images']}\n")
            f.write(f"PSNR: {result['PSNR']:.6f} dB\n")
            f.write(f"SSIM: {result['SSIM']:.6f}\n")
            f.write(f"LPIPS: {result['LPIPS']:.6f}\n\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
