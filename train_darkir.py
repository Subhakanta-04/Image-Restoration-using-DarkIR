"""Minimal supervised trainer for the official DarkIR architecture and LOL-Blur."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from archs.DarkIR import DarkIR  # noqa: E402
from data.paired_lolblur import (  # noqa: E402
    PairedLolBlurDataset,
    discover_class_pairs,
    select_and_split_classes,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_loss": best_val,
        },
        path,
    )



def train_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled, max_batches=None):
    model.train()
    loss_sum = 0.0
    batch_count = 0

    for batch_count, (inputs, targets, class_ids, keys) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Check the data before running the model.
        if not torch.isfinite(inputs).all():
            raise FloatingPointError(
                f"Non-finite INPUT at batch {batch_count}; keys={keys}"
            )
        if not torch.isfinite(targets).all():
            raise FloatingPointError(
                f"Non-finite TARGET at batch {batch_count}; keys={keys}"
            )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            restored = model(inputs)
            loss = criterion(restored, targets)

        # Identify whether the forward pass produced invalid values.
        if not torch.isfinite(restored).all():
            raise FloatingPointError(
                f"Non-finite MODEL OUTPUT at batch {batch_count}; keys={keys}"
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite LOSS at batch {batch_count}; "
                f"loss={loss.item()}, keys={keys}"
            )

        scaler.scale(loss).backward()

        # Check gradients before allowing an optimizer update.
        scaler.unscale_(optimizer)

        bad_gradients = [
            name for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ]
        if bad_gradients:
            raise FloatingPointError(
                f"Non-finite GRADIENTS at batch {batch_count}: "
                f"{bad_gradients[:10]}"
            )

        scaler.step(optimizer)
        scaler.update()

        # Ensure the optimizer has not produced invalid model weights.
        bad_parameters = [
            name for name, parameter in model.named_parameters()
            if not torch.isfinite(parameter).all()
        ]
        if bad_parameters:
            raise FloatingPointError(
                f"Non-finite PARAMETERS after batch {batch_count}: "
                f"{bad_parameters[:10]}"
            )

        loss_sum += float(loss.detach().item())

        if max_batches is not None and batch_count >= max_batches:
            break

    return loss_sum / max(batch_count, 1), batch_count



@torch.no_grad()
def validate(model, loader, criterion, device, amp_enabled, max_batches=None):
    model.eval()
    loss_sum = 0.0
    batch_count = 0
    class_loss_sums = {}
    class_counts = {}
    for batch_count, (inputs, targets, class_ids, keys) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            restored = model(inputs)
            loss = criterion(restored, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation loss at batch {batch_count}: {loss.item()}")
        loss_sum += float(loss.item())
        per_sample_loss = F.l1_loss(restored.float(), targets.float(), reduction="none").mean(dim=(1, 2, 3))
        for class_id, sample_loss in zip(class_ids.tolist(), per_sample_loss.tolist()):
            class_loss_sums[class_id] = class_loss_sums.get(class_id, 0.0) + sample_loss
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        if max_batches is not None and batch_count >= max_batches:
            break
    class_means = {
        class_id: class_loss_sums[class_id] / class_counts[class_id]
        for class_id in class_loss_sums
    }
    return loss_sum / max(batch_count, 1), class_means, batch_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "options" / "train_darkir_lolblur.yml")
    parser.add_argument("--sanity", action="store_true", help="Run only two train batches and one validation batch")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument("--max-train-batches", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    seed = int(config["dataset"]["seed"])
    seed_everything(seed)
    dataset_cfg, data_cfg = config["dataset"], config["data"]
    dataset_root = Path(dataset_cfg["root"])
    if not dataset_root.is_absolute():
        dataset_root = (PROJECT_ROOT / dataset_root).resolve()
    class_configs = dataset_cfg["classes"]
    class_indexes = discover_class_pairs(dataset_root, class_configs, dataset_cfg["target_dir"])
    training_pairs, validation_pairs, train_keys, validation_keys = select_and_split_classes(
        class_indexes, class_configs, seed
    )
    expected_total = sum(int(item["samples"]) for item in class_configs)
    if len(training_pairs) + len(validation_pairs) != expected_total:
        raise RuntimeError("Selected sample total differs from configured total")
    if expected_total != 5000 or len(training_pairs) != 4500 or len(validation_pairs) != 500:
        raise RuntimeError(
            "Configured split must contain exactly 5,000 samples (4,500 train / 500 validation)"
        )
    train_key_set = {pair.key for pair in training_pairs}
    val_key_set = {pair.key for pair in validation_pairs}
    if train_key_set & val_key_set:
        raise RuntimeError("Train/validation scene-key leakage detected")
    print(
        f"Selected {expected_total} paired samples on {len(train_keys) + len(validation_keys)} "
        f"shared scene/frame keys with seed {seed}: {len(training_pairs)} training / "
        f"{len(validation_pairs)} validation"
    )
    for class_cfg in class_configs:
        class_id = int(class_cfg["id"])
        train_count = sum(pair.class_id == class_id for pair in training_pairs)
        val_count = sum(pair.class_id == class_id for pair in validation_pairs)
        if train_count != int(class_cfg["train_samples"]) or val_count != int(class_cfg["validation_samples"]):
            raise RuntimeError("Class {} split counts differ from configuration".format(class_id))
        print(f"Class {class_id} ({class_cfg['name']}): {train_count} train / {val_count} validation")

    crop_size = int(data_cfg["crop_size"])
    train_set = PairedLolBlurDataset(training_pairs, crop_size=crop_size, training=True)
    val_set = PairedLolBlurDataset(validation_pairs, crop_size=crop_size, training=False)
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg["workers"]),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    if config["training"]["mixed_precision"] and not amp_enabled:
        print("CUDA is unavailable; mixed precision is disabled for this run.")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"AMP enabled: {amp_enabled}")

    model = DarkIR(**config["model"]).to(device)
    param_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Existing DarkIR architecture: {param_count:,} parameters (width={config['model']['width']})")
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["training"]["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val = float("inf")
    start_epoch = 1
    resume_path = args.resume or (
        Path(config["training"]["resume_from"]) if config["training"].get("resume_from") else None
    )
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_validation_loss", best_val))
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (PROJECT_ROOT / checkpoint_dir).resolve()
    if args.sanity:
        checkpoint_dir = checkpoint_dir / "sanity_check"
    epochs = 1 if args.sanity else int(config["training"]["epochs"])
    val_frequency = max(1, int(config["training"]["validation_frequency"]))

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss, train_batches = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, amp_enabled,
            max_batches=2 if args.sanity else args.max_train_batches,
        )
        if epoch % val_frequency == 0 or args.sanity:
            val_loss, per_class_val_loss, val_batches = validate(
                model, val_loader, criterion, device, amp_enabled,
                max_batches=3 if args.sanity else None,
            )
        else:
            val_loss, per_class_val_loss, val_batches = float("nan"), {}, 0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{start_epoch + epochs - 1} | train_loss={train_loss:.6f} "
            f"| val_loss={val_loss:.6f} | lr={current_lr:.3e} "
            f"| train_batches={train_batches} | val_batches={val_batches}"
        )
        for class_cfg in class_configs:
            class_id = int(class_cfg["id"])
            value = per_class_val_loss.get(class_id)
            if value is not None:
                print(f"  Class {class_id} ({class_cfg['name']}) validation L1: {value:.6f}")
        scheduler.step()
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_val)
            print(f"Saved new best checkpoint (validation L1={best_val:.6f})")
        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, scheduler, scaler, epoch, best_val)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
