import torch
import torch.distributed as dist
import sys
import os
from lpips import LPIPS
import numpy as np

sys.path.append('../losses')
sys.path.append('../data/datasets/datapipeline')

from losses import *
from tqdm import tqdm


calc_SSIM = SSIM(data_range=1.)


# ---------------------------------------------------------
# Distributed helpers — compatible with single-GPU evaluation
# ---------------------------------------------------------

def setup(rank, world_size, Master_port='12355'):
    """
    Initialize the device for evaluation.

    For single-GPU evaluation, no distributed process group
    is required. Multi-process distributed initialization is
    intentionally not performed here.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup():
    """Destroy a process group only if one was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """
    Average tensors across processes when distributed evaluation
    is active. For a single process, return the tensor unchanged.
    """
    if world_size <= 1:
        return tensor

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Distributed reduction requested, but no process "
            "group has been initialized."
        )

    result = tensor.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result /= world_size

    return result


def save_model(model, path):
    """Save a model without requiring distributed initialization."""
    if not dist.is_available() or not dist.is_initialized():
        torch.save(model.state_dict(), path)
        return

    if dist.get_rank() == 0:
        torch.save(model.state_dict(), path)


def shuffle_sampler(samplers, epoch):
    """
    Shuffle distributed samplers for the specified epoch.
    """
    if not samplers:
        return

    for sampler in samplers:
        sampler.set_epoch(epoch)


# ---------------------------------------------------------
# Device helper
# ---------------------------------------------------------

def get_device(rank=0):
    """
    Return the appropriate device for the current evaluation.
    """
    if torch.cuda.is_available():
        return torch.device(f'cuda:{rank}')

    return torch.device('cpu')


# ---------------------------------------------------------
# Evaluate one model on one test loader
# ---------------------------------------------------------

def eval_one_loader(
    model,
    test_loader,
    metrics,
    rank=0,
    world_size=1,
    eta=False
):
    device = get_device(rank)

    calc_LPIPS = LPIPS(
        net='vgg',
        verbose=False
    ).to(device)

    mean_metrics = {
        'valid_psnr': [],
        'valid_ssim': [],
        'valid_lpips': []
    }

    pbar = None

    if eta:
        pbar = tqdm(
            total=len(test_loader),
            desc='Evaluating'
        )

    with torch.no_grad():

        for high_batch_valid, low_batch_valid in test_loader:

            high_batch_valid = high_batch_valid.to(device)
            low_batch_valid = low_batch_valid.to(device)

            # Generate restored image
            enhanced_batch_valid = model(low_batch_valid)

            # Mean squared error
            valid_loss_batch = torch.mean(
                (high_batch_valid - enhanced_batch_valid) ** 2
            )

            # SSIM
            valid_ssim_batch = calc_SSIM(
                enhanced_batch_valid,
                high_batch_valid
            )

            # LPIPS
            valid_lpips_batch = calc_LPIPS(
                enhanced_batch_valid,
                high_batch_valid
            )

            # PSNR
            valid_psnr_batch = 20 * torch.log10(
                1.0 / torch.sqrt(valid_loss_batch)
            )

            mean_metrics['valid_psnr'].append(
                valid_psnr_batch.item()
            )

            mean_metrics['valid_ssim'].append(
                valid_ssim_batch.item()
            )

            mean_metrics['valid_lpips'].append(
                torch.mean(valid_lpips_batch).item()
            )

            if eta:
                pbar.update(1)

    if not mean_metrics['valid_psnr']:
        if pbar is not None:
            pbar.close()

        raise RuntimeError(
            "The test loader contains no images. "
            "Check the test dataset path and image pairing."
        )

    # Calculate average metrics
    valid_psnr_tensor = torch.tensor(
        np.mean(mean_metrics['valid_psnr']),
        device=device
    )

    valid_ssim_tensor = torch.tensor(
        np.mean(mean_metrics['valid_ssim']),
        device=device
    )

    valid_lpips_tensor = torch.tensor(
        np.mean(mean_metrics['valid_lpips']),
        device=device
    )

    # Reduce only if distributed evaluation is active
    valid_psnr_tensor = reduce_tensor(
        valid_psnr_tensor,
        world_size
    )

    valid_ssim_tensor = reduce_tensor(
        valid_ssim_tensor,
        world_size
    )

    valid_lpips_tensor = reduce_tensor(
        valid_lpips_tensor,
        world_size
    )

    metrics['valid_psnr'] = valid_psnr_tensor.item()
    metrics['valid_ssim'] = valid_ssim_tensor.item()
    metrics['valid_lpips'] = valid_lpips_tensor.item()

    # Keep the last batch's input, output and ground truth
    imgs_dict = {
        'input': low_batch_valid[0],
        'output': enhanced_batch_valid[0],
        'gt': high_batch_valid[0]
    }

    if pbar is not None:
        pbar.close()

    return metrics, imgs_dict


# ---------------------------------------------------------
# Evaluate one model across one or more test loaders
# ---------------------------------------------------------

def eval_model(
    model,
    test_loader,
    metrics,
    rank=0,
    world_size=1,
    eta=False
):
    """
    Evaluate a model on one loader or a dictionary of loaders.
    """

    if not isinstance(test_loader, dict):
        test_loader = {'data': test_loader}

    if len(test_loader) > 1:

        all_metrics = {}
        all_imgs_dict = {}

        for key, loader in test_loader.items():

            current_metrics = {}

            current_metrics, imgs_dict = eval_one_loader(
                model=model,
                test_loader=loader['loader'],
                metrics=current_metrics,
                rank=rank,
                world_size=world_size,
                eta=eta
            )

            all_metrics[key] = current_metrics
            all_imgs_dict[key] = imgs_dict

        return all_metrics, all_imgs_dict

    else:

        loader = test_loader['data']

        metrics, imgs_dict = eval_one_loader(
            model=model,
            test_loader=loader,
            metrics=metrics,
            rank=rank,
            world_size=world_size,
            eta=eta
        )

        return metrics, imgs_dict


# ---------------------------------------------------------
# Evaluate two models on one test loader
# ---------------------------------------------------------

def eval_one_loader_two_models(
    model1,
    model2,
    test_loader,
    metrics,
    devices=None,
    eta=False
):
    """
    Run model1 followed by model2.

    The output of model1 is clamped to [0, 1] before
    being passed to model2.
    """

    if devices is None:
        devices = ['cuda:0', 'cuda:1']

    device1 = torch.device(devices[0])
    device2 = torch.device(devices[1])

    calc_LPIPS = LPIPS(
        net='vgg',
        verbose=False
    ).to(device1)

    mean_metrics = {
        'valid_psnr': [],
        'valid_ssim': [],
        'valid_lpips': []
    }

    pbar = None

    if eta:
        pbar = tqdm(
            total=len(test_loader),
            desc='Evaluating two models'
        )

    with torch.no_grad():

        for high_batch_valid, low_batch_valid in test_loader:

            high_batch_valid = high_batch_valid.to(device1)
            low_batch_valid = low_batch_valid.to(device1)

            # First restoration model
            enhanced_batch_valid = model1(low_batch_valid)

            enhanced_batch_valid = torch.clamp(
                enhanced_batch_valid,
                0.0,
                1.0
            )

            # Second restoration model
            enhanced_batch_valid = model2(
                enhanced_batch_valid.to(device2)
            )

            enhanced_batch_valid = enhanced_batch_valid.to(device1)

            # MSE
            valid_loss_batch = torch.mean(
                (high_batch_valid - enhanced_batch_valid) ** 2
            )

            # SSIM
            valid_ssim_batch = calc_SSIM(
                enhanced_batch_valid,
                high_batch_valid
            )

            # LPIPS
            valid_lpips_batch = calc_LPIPS(
                enhanced_batch_valid,
                high_batch_valid
            )

            # PSNR
            valid_psnr_batch = 20 * torch.log10(
                1.0 / torch.sqrt(valid_loss_batch)
            )

            mean_metrics['valid_psnr'].append(
                valid_psnr_batch.item()
            )

            mean_metrics['valid_ssim'].append(
                valid_ssim_batch.item()
            )

            mean_metrics['valid_lpips'].append(
                torch.mean(valid_lpips_batch).item()
            )

            if eta:
                pbar.update(1)

    if not mean_metrics['valid_psnr']:
        if pbar is not None:
            pbar.close()

        raise RuntimeError(
            "The test loader contains no images."
        )

    metrics['valid_psnr'] = float(
        np.mean(mean_metrics['valid_psnr'])
    )

    metrics['valid_ssim'] = float(
        np.mean(mean_metrics['valid_ssim'])
    )

    metrics['valid_lpips'] = float(
        np.mean(mean_metrics['valid_lpips'])
    )

    imgs_dict = {
        'input': low_batch_valid[0],
        'output': enhanced_batch_valid[0],
        'gt': high_batch_valid[0]
    }

    if pbar is not None:
        pbar.close()

    return metrics, imgs_dict


# ---------------------------------------------------------
# Evaluate two models across one or more test loaders
# ---------------------------------------------------------

def eval_model_two_models(
    model1,
    model2,
    test_loader,
    metrics,
    devices=None,
    eta=False
):
    """
    Evaluate two sequential models on one loader or
    a dictionary of loaders.
    """

    if devices is None:
        devices = ['cuda:0', 'cuda:1']

    if not isinstance(test_loader, dict):
        test_loader = {'data': test_loader}

    if len(test_loader) > 1:

        all_metrics = {}
        all_imgs_dict = {}

        for key, loader in test_loader.items():

            current_metrics = {}

            current_metrics, imgs_dict = eval_one_loader_two_models(
                model1=model1,
                model2=model2,
                test_loader=loader['loader'],
                metrics=current_metrics,
                devices=devices,
                eta=eta
            )

            all_metrics[key] = current_metrics
            all_imgs_dict[key] = imgs_dict

        return all_metrics, all_imgs_dict

    else:

        loader = test_loader['data']

        metrics, imgs_dict = eval_one_loader_two_models(
            model1=model1,
            model2=model2,
            test_loader=loader,
            metrics=metrics,
            devices=devices,
            eta=eta
        )

        return metrics, imgs_dict