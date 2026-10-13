
import os
import torch
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

CHECKPOINT = os.path.join(
    PROJECT_ROOT, "models", "training_lolblur", "best_official_format_fixed.pt"
)

# Existing low_blur_noise comparisons are preserved.
CONDITIONS = {
    "low_blur": os.path.join(PROJECT_ROOT, "results", "comparisons_low_blur"),
    "low_blur_noise": os.path.join(PROJECT_ROOT, "results", "test_comparisons"),
    "low_sharp": os.path.join(PROJECT_ROOT, "results", "comparisons_low_sharp"),
}

TARGET_DIR = os.path.join(TEST_ROOT, "high_sharp_scaled")
NUM_IMAGES = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VALID_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
)


# --------------------------------------------------
# Load trained DarkIR
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

    weights = {
        key.removeprefix("module."): value
        for key, value in weights.items()
    }

    model.load_state_dict(weights, strict=True)
    model = model.to(DEVICE)
    model.eval()

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}")

    return model


# --------------------------------------------------
# Find matching pairs and select scenes
# --------------------------------------------------

def find_pairs(input_dir):
    pairs_by_scene = {}

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue

            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(input_path, input_dir)
            target_path = os.path.join(TARGET_DIR, relative_path)

            if not os.path.isfile(target_path):
                continue

            scene_id = os.path.dirname(relative_path)

            pairs_by_scene.setdefault(scene_id, []).append(
                (relative_path, input_path, target_path)
            )

    for scene_id in pairs_by_scene:
        pairs_by_scene[scene_id].sort(key=lambda item: item[0])

    scene_ids = sorted(pairs_by_scene)

    # Select one frame from every scene first.
    selected = [
        pairs_by_scene[scene_id][0]
        for scene_id in scene_ids
    ]

    max_images = (
        NUM_IMAGES
        if NUM_IMAGES is not None
        else sum(len(items) for items in pairs_by_scene.values())
    )

    # Add more frames, cycling across scenes.
    if len(selected) < max_images:
        max_frames = max(
            (len(items) for items in pairs_by_scene.values()),
            default=0,
        )

        for frame_index in range(1, max_frames):
            for scene_id in scene_ids:
                frames = pairs_by_scene[scene_id]

                if frame_index < len(frames):
                    selected.append(frames[frame_index])

                    if len(selected) >= max_images:
                        break

            if len(selected) >= max_images:
                break

    selected = selected[:max_images]

    print(
        f"Matched {sum(len(v) for v in pairs_by_scene.values())} "
        f"pairs across {len(scene_ids)} scenes."
    )
    print(f"Selected {len(selected)} images.")

    return selected


# --------------------------------------------------
# Generate comparisons
# --------------------------------------------------

def save_comparisons(model, pairs, output_dir, condition):
    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    for index, (relative_path, input_path, target_path) in enumerate(
        pairs, start=1
    ):
        try:
            input_image = Image.open(input_path).convert("RGB")
            target_image = Image.open(target_path).convert("RGB")

            if input_image.size != target_image.size:
                print(f"Skipping size mismatch: {relative_path}")
                continue

            input_tensor = to_tensor(input_image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                output_tensor = model(input_tensor).clamp(0.0, 1.0)[0].cpu()

            output_image = to_pil_image(output_tensor)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            panels = [
                (input_image, f"Input: {condition}"),
                (output_image, "DarkIR Restored Output"),
                (target_image, "Ground Truth: Sharp Reference"),
            ]

            for ax, (image, title) in zip(axes, panels):
                ax.imshow(image)
                ax.set_title(title)
                ax.axis("off")

            fig.suptitle(relative_path, fontsize=9)
            fig.tight_layout()

            # Keep the scene/frame path structure.
            relative_stem = os.path.splitext(relative_path)[0]
            save_folder = os.path.join(
                output_dir, os.path.dirname(relative_stem)
            )
            os.makedirs(save_folder, exist_ok=True)

            filename = os.path.basename(relative_stem)
            comparison_path = os.path.join(
                save_folder, f"{filename}_comparison.png"
            )

            fig.savefig(
                comparison_path,
                dpi=120,
                bbox_inches="tight",
            )
            plt.close(fig)
            saved += 1

            print(f"[{index}/{len(pairs)}] Saved: {comparison_path}")

        except Exception as exc:
            print(f"Error processing {relative_path}: {exc}")

    print(f"\n{condition}: saved {saved} comparisons.")
    print(f"Output: {output_dir}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not os.path.isdir(TEST_ROOT):
        raise FileNotFoundError(f"Test root not found: {TEST_ROOT}")

    model = load_darkir()

    for condition, output_dir in CONDITIONS.items():
        input_dir = os.path.join(TEST_ROOT, condition)

        if not os.path.isdir(input_dir):
            print(f"Skipping missing input folder: {input_dir}")
            continue

        print("\n" + "=" * 60)
        print(f"PROCESSING CONDITION: {condition}")
        print("=" * 60)

        pairs = find_pairs(input_dir)
        save_comparisons(model, pairs, output_dir, condition)

    print("\nAll requested conditions have been processed.")
