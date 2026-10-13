
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from archs.DarkIR import DarkIR


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.environ.get(
    "DARKIR_LOLBLUR_TEST",
    os.path.join(PROJECT_ROOT, "data", "datasets", "LOLBlur", "test"),
)

INPUT_DIR = os.path.join(TEST_ROOT, "low_blur_noise")
TARGET_DIR = os.path.join(TEST_ROOT, "high_sharp_scaled")

CHECKPOINT = os.path.join(
    PROJECT_ROOT, "models", "training_lolblur", "best_official_format_fixed.pt"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "test_comparisons")

NUM_IMAGES = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Load the trained DarkIR model
# --------------------------------------------------

def load_darkir():
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

    # Remove DDP prefix if present.
    weights = {
        key.removeprefix("module."): value
        for key, value in weights.items()
    }

    model.load_state_dict(weights, strict=True)
    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}")

    return model


# --------------------------------------------------
# Find matching input and target images
# --------------------------------------------------


def find_pairs():
    pairs_by_scene = {}

    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ):
                continue

            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(input_path, INPUT_DIR)
            target_path = os.path.join(TARGET_DIR, relative_path)

            if not os.path.isfile(target_path):
                continue

            # The first relative folder is the scene ID.
            scene_id = os.path.dirname(relative_path)

            pairs_by_scene.setdefault(scene_id, []).append(
                (relative_path, input_path, target_path)
            )

    # Sort frames within each scene.
    for scene_id in pairs_by_scene:
        pairs_by_scene[scene_id].sort(key=lambda item: item[0])

    # Pick one frame from each scene first.
    selected = []
    scene_ids = sorted(pairs_by_scene)

    for scene_id in scene_ids:
        selected.append(pairs_by_scene[scene_id][0])

    # If more images are requested, add more frames from scenes.
    max_images = NUM_IMAGES if NUM_IMAGES is not None else sum(
        len(items) for items in pairs_by_scene.values()
    )

    if len(selected) < max_images:
        for frame_index in range(1, max(
            (len(items) for items in pairs_by_scene.values()),
            default=0
        )):
            for scene_id in scene_ids:
                scene_frames = pairs_by_scene[scene_id]

                if frame_index < len(scene_frames):
                    selected.append(scene_frames[frame_index])

                    if len(selected) >= max_images:
                        break

            if len(selected) >= max_images:
                break

    selected = selected[:max_images]

    print(f"Found {sum(len(v) for v in pairs_by_scene.values())} matching pairs")
    print(f"Selected {len(selected)} images from {len(scene_ids)} scenes")

    return selected



# --------------------------------------------------
# Save individual images and side-by-side comparisons
# --------------------------------------------------


def save_comparisons(model, pairs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = len(pairs) if NUM_IMAGES is None else min(NUM_IMAGES, len(pairs))
    saved = 0

    for index, (relative_path, input_path, target_path) in enumerate(
        pairs[:total], start=1
    ):
        relative_stem = os.path.splitext(relative_path)[0]
        save_folder = os.path.join(
            OUTPUT_DIR, os.path.dirname(relative_stem)
        )
        os.makedirs(save_folder, exist_ok=True)

        filename = os.path.basename(relative_stem)

        try:
            input_image = Image.open(input_path).convert("RGB")
            target_image = Image.open(target_path).convert("RGB")

            if input_image.size != target_image.size:
                print(f"Skipping size mismatch: {relative_path}")
                continue

            input_tensor = to_tensor(input_image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output_tensor = model(input_tensor).clamp(0.0, 1.0)[0].cpu()

            output_image = to_pil_image(output_tensor)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            panels = [
                (input_image, "Input: Low-light / Blur"),
                (output_image, "DarkIR Restored Output"),
                (target_image, "Ground Truth: Sharp Reference"),
            ]

            for ax, (image, title) in zip(axes, panels):
                ax.imshow(image)
                ax.set_title(title)
                ax.axis("off")

            fig.suptitle(relative_path, fontsize=9)
            fig.tight_layout()

            comparison_path = os.path.join(
                save_folder, f"{filename}_comparison.png"
            )
            fig.savefig(comparison_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

            saved += 1

            if saved % 25 == 0 or saved == total:
                print(f"Progress: {saved}/{total} comparisons saved")

        except Exception as exc:
            print(f"Error processing {relative_path}: {exc}")

    print(f"\nCompleted: {saved} comparisons saved.")
    print(f"Output directory: {OUTPUT_DIR}")



# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    model = load_darkir()
    image_pairs = find_pairs()
    save_comparisons(model, image_pairs)
