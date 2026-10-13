
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lpips import LPIPS
from losses import SSIM
from archs.DarkIR import DarkIR

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.environ.get(
    "DARKIR_LOLBLUR_TEST",
    os.path.join(PROJECT_ROOT, "data", "datasets", "LOLBlur", "test"),
)
TARGET_DIR = os.path.join(TEST_ROOT, "high_sharp_scaled")

CHECKPOINT = (
    os.path.join(PROJECT_ROOT, "models", "training_lolblur", "best_official_format_fixed.pt")
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "darkir_remaining_results.txt"
)

INPUT_DIRS = {
    "low_blur": os.path.join(TEST_ROOT, "low_blur"),
    "low_sharp": os.path.join(TEST_ROOT, "low_sharp"),
}

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


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
            raise RuntimeError(f"No matching pairs found: {input_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if input_image.size != target_image.size:
            raise ValueError(f"Image dimensions differ: {input_path}")

        return to_tensor(input_image), to_tensor(target_image)


def load_model():
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

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    weights = checkpoint.get(
        "params",
        checkpoint.get("model_state_dict", checkpoint),
    )

    weights = {
        key.removeprefix("module."): value
        for key, value in weights.items()
    }

    model.load_state_dict(weights, strict=True)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}")

    return model


def evaluate_condition(model, name, input_dir, ssim_fn, lpips_fn):
    dataset = PairedImages(input_dir)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    psnr_values, ssim_values, lpips_values = [], [], []

    print(f"\nCondition: {name}")
    print(f"Matched pairs: {len(dataset)}")

    with torch.no_grad():
        for low, target in tqdm(loader, desc=name):
            low = low.to(DEVICE)
            target = target.to(DEVICE)

            restored = model(low).clamp(0.0, 1.0)

            mse = torch.mean((target - restored) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            ssim = ssim_fn(restored, target)
            lpips = lpips_fn(restored, target)

            psnr_values.append(psnr.item())
            ssim_values.append(ssim.item())
            lpips_values.append(lpips.mean().item())

    result = {
        "condition": name,
        "images": len(dataset),
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "LPIPS": float(np.mean(lpips_values)),
    }

    print(f"PSNR:  {result['PSNR']:.6f} dB")
    print(f"SSIM:  {result['SSIM']:.6f}")
    print(f"LPIPS: {result['LPIPS']:.6f}")

    return result


def main():
    model = load_model()
    ssim_fn = SSIM(data_range=1.0).to(DEVICE)
    lpips_fn = LPIPS(net="vgg", verbose=False).to(DEVICE)

    results = []

    for name, input_dir in INPUT_DIRS.items():
        results.append(
            evaluate_condition(
                model, name, input_dir, ssim_fn, lpips_fn
            )
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("DarkIR Restored Results\n\n")

        for result in results:
            file.write(f"Condition: {result['condition']}\n")
            file.write(f"Images: {result['images']}\n")
            file.write(f"PSNR: {result['PSNR']:.6f} dB\n")
            file.write(f"SSIM: {result['SSIM']:.6f}\n")
            file.write(f"LPIPS: {result['LPIPS']:.6f}\n\n")

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
