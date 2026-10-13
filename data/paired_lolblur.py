"""Paired LOL-Blur discovery and scene-aware multi-condition datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImagePair:
    key: str
    input_path: Path
    target_path: Path
    class_id: int
    class_name: str


def _index_images(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError("Image condition directory does not exist: {}".format(folder))
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    indexed = {}  # type: Dict[str, Path]
    duplicates = []  # type: List[str]
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in supported)
    for path in paths:
        key = path.relative_to(folder).as_posix().casefold()
        if key in indexed:
            duplicates.append(key)
        else:
            indexed[key] = path
    if duplicates:
        raise ValueError("Duplicate image keys in {}: {}".format(folder, duplicates[:10]))
    if not indexed:
        raise ValueError("No supported images found under {}".format(folder))
    return indexed


def discover_class_pairs(dataset_root: Path, class_configs: List[dict], target_dir: str) -> Dict[int, Dict[str, ImagePair]]:
    """Index every class by relative key and fail if any target/input is unmatched."""
    root = Path(dataset_root).expanduser().resolve()
    targets = _index_images(root / target_dir)
    class_indexes = {}
    print("LOL-Blur training root: {}".format(root))
    print("Target '{}' images: {}".format(target_dir, len(targets)))
    for class_cfg in class_configs:
        class_id = int(class_cfg["id"])
        class_name = str(class_cfg["name"])
        input_dir = str(class_cfg["input_dir"])
        inputs = _index_images(root / input_dir)
        missing_targets = sorted(inputs.keys() - targets.keys())
        missing_inputs = sorted(targets.keys() - inputs.keys())
        print(
            "Class {} '{}' input '{}': {} images, {} matched, {} missing targets, {} missing inputs".format(
                class_id, class_name, input_dir, len(inputs), len(inputs) - len(missing_targets),
                len(missing_targets), len(missing_inputs)
            )
        )
        if missing_targets or missing_inputs:
            raise ValueError(
                "Pairing failed for {}: inputs without targets {}, targets without inputs {}".format(
                    input_dir, missing_targets[:10], missing_inputs[:10]
                )
            )
        class_indexes[class_id] = {
            key: ImagePair(key, inputs[key], targets[key], class_id, class_name)
            for key in sorted(inputs)
        }
    return class_indexes


def select_and_split_classes(
    class_indexes: Dict[int, Dict[str, ImagePair]], class_configs: List[dict], seed: int
) -> Tuple[List[ImagePair], List[ImagePair], List[str], List[str]]:
    """Select shared keys first, then split keys once to prevent cross-class leakage.

    Categories may use the same source key, but all category variants of that key
    stay on the same side of the train/validation boundary.
    """
    first_id = int(class_configs[0]["id"])
    common_keys = set(class_indexes[first_id])
    for class_cfg in class_configs[1:]:
        common_keys.intersection_update(class_indexes[int(class_cfg["id"])])
    common_keys = sorted(common_keys)
    selected_key_count = max(int(cfg["samples"]) for cfg in class_configs)
    if len(common_keys) < selected_key_count:
        raise ValueError(
            "Need {} common scene/frame keys; found {}".format(selected_key_count, len(common_keys))
        )

    rng = random.Random(seed)
    selected_keys = rng.sample(common_keys, selected_key_count)
    rng.shuffle(selected_keys)
    max_train = max(int(cfg["train_samples"]) for cfg in class_configs)
    train_keys = selected_keys[:max_train]
    validation_keys = selected_keys[max_train:]
    if any(int(cfg["train_samples"]) != len(train_keys) for cfg in class_configs):
        raise ValueError("All categories must share one scene-level training key set")
    if any(int(cfg["validation_samples"]) > len(validation_keys) for cfg in class_configs):
        raise ValueError("Validation category counts exceed the shared validation key set")

    train_pairs = []
    validation_pairs = []
    for class_cfg in class_configs:
        class_id = int(class_cfg["id"])
        count = int(class_cfg["samples"])
        train_count = int(class_cfg["train_samples"])
        val_count = int(class_cfg["validation_samples"])
        if train_count + val_count != count:
            raise ValueError("Class {} counts do not sum to its selected count".format(class_id))
        train_pairs.extend(class_indexes[class_id][key] for key in train_keys[:train_count])
    # Interleave validation categories by key so a short sanity validation sees
    # all three classes while retaining the single shared key-level split.
    for key_index, key in enumerate(validation_keys):
        for class_cfg in class_configs:
            class_id = int(class_cfg["id"])
            if key_index < int(class_cfg["validation_samples"]):
                validation_pairs.append(class_indexes[class_id][key])

    if len(set(pair.key for pair in train_pairs) & set(pair.key for pair in validation_pairs)):
        raise RuntimeError("Scene/frame key leakage found between training and validation")
    return train_pairs, validation_pairs, train_keys, validation_keys


class PairedLolBlurDataset(Dataset):
    def __init__(self, pairs: List[ImagePair], crop_size: int, training: bool):
        self.pairs = pairs
        self.crop_size = int(crop_size)
        self.training = training

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        with Image.open(pair.input_path) as image:
            input_image = image.convert("RGB")
        with Image.open(pair.target_path) as image:
            target_image = image.convert("RGB")
        if input_image.size != target_image.size:
            raise ValueError(
                "Paired dimensions differ for {}: input {}, target {}".format(
                    pair.key, input_image.size, target_image.size
                )
            )

        width, height = input_image.size
        pad_w, pad_h = max(0, self.crop_size - width), max(0, self.crop_size - height)
        if pad_w or pad_h:
            input_image = ImageOps.expand(input_image, (0, 0, pad_w, pad_h))
            target_image = ImageOps.expand(target_image, (0, 0, pad_w, pad_h))
            width, height = input_image.size
        if self.training:
            left = random.randint(0, width - self.crop_size)
            top = random.randint(0, height - self.crop_size)
        else:
            left = (width - self.crop_size) // 2
            top = (height - self.crop_size) // 2
        box = (left, top, left + self.crop_size, top + self.crop_size)
        input_image, target_image = input_image.crop(box), target_image.crop(box)

        if self.training:
            if random.random() < 0.5:
                input_image = input_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                target_image = target_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                input_image = input_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                target_image = target_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        def to_tensor(image: Image.Image) -> torch.Tensor:
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
            return torch.from_numpy(array).permute(2, 0, 1)

        # class_id and key are metadata; the trainer never sends them to DarkIR.
        return to_tensor(input_image), to_tensor(target_image), pair.class_id, pair.key
