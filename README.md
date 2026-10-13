# CNN-Based Low-Light Image Restoration and Deblurring Using DarkIR

This student project applies DarkIR-m to paired LOL-Blur images and evaluates restoration across three degradation categories: low-light blur, blur with noise, and low-light sharp images. DarkIR predicts restored pixels; category IDs are metadata for grouping samples and metrics, not model outputs.

> These are project results based on DarkIR, not a claim of reproducing the paper's exact training protocol or results.

## Motivation and problem

Low-light images can suffer from poor illumination, blur, and noise. This project uses paired restoration: a degraded image is input to DarkIR and the output is compared with its aligned clean, sharp target.

## Research paper

Daniel Feijoo, Juan C. Benito, Alvaro Garcia, and Marcos V. Conde, “DarkIR: Robust Low-Light Image Restoration,” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 10879–10889.

- [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.html)
- [Official DarkIR repository](https://github.com/Fundacion-Cidaut/DarkIR)
- [Official project page](https://huggingface.co/Cidaut/DarkIR)

## Features and objectives

- Official DarkIR architecture, DarkIR-m configuration: width 32, 3,321,638 parameters.
- Paired training and testing with `high_sharp_scaled` reference images.
- Three degradation categories, with no classification head.
- Deterministic selection and scene/frame-key-aware train/validation splitting.
- Overall and per-category validation L1 loss, PSNR/SSIM/LPIPS evaluation, folder inference, a Streamlit interface, a benchmark, and synthetic Gaussian-noise analysis.

## Architecture and workflow

DarkIR is a convolutional encoder/decoder restoration network. See [`archs/DarkIR.py`](archs/DarkIR.py), [`archs/arch_model.py`](archs/arch_model.py), and [`archs/arch_util.py`](archs/arch_util.py). The implementation includes skip connections, dilated depthwise convolution branches, and frequency processing. The project does not modify the architecture with a classifier.

```text
Degraded LOL-Blur image ──► DarkIR-m ──► Restored image
                                  ▲            │
                                  └── L1 loss ◄┴── Aligned clean target
```

| Category | Input | Target |
|---|---|---|
| 1 — Low-Light + Blur | `low_blur` | `high_sharp_scaled` |
| 2 — Low-Light + Blur + Noise | `low_blur_noise` | `high_sharp_scaled` |
| 3 — Low-Light + Sharp | `low_sharp` | `high_sharp_scaled` |

## Dataset

LOL-Blur is not included; obtain it separately and follow its terms. The verified dataset has 10,200 training keys and 1,800 test keys per condition. The three training input conditions share relative scene/frame keys and align with `high_sharp_scaled`.

Expected layout:

```text
data/datasets/LOLBlur/
├── train/{low_blur,low_blur_noise,low_sharp,high_sharp_scaled}/<scene>/<frame>.png
└── test/{low_blur,low_blur_noise,low_sharp,high_sharp_scaled}/<scene>/<frame>.png
```

The training script reads only the configured train root; testing reads only the separate test root. Pairs are matched by relative path and filename. Missing pairs and size mismatches raise errors. Use [`verify_lolblur.py`](verify_lolblur.py) to check the training set.

## Environment and installation

Project environment: Python 3.10, PyTorch 2.5.1, TorchVision 0.20.1. The recorded evaluation used `cuda:0`; GPU availability depends on the local driver and PyTorch build. In PowerShell, from this repository directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` reflects imported training, evaluation, inference, plotting, and optional Streamlit packages. This setup has not been verified on every operating system or GPU.

## Configuration

Training settings are in [`options/train_darkir_lolblur.yml`](options/train_darkir_lolblur.yml), with dataset root `data/datasets/LOLBlur/train`. The test configuration is [`options/test/LOLBlur.yml`](options/test/LOLBlur.yml). Checkpoint files are local and ignored by Git.

| Category | Selected pairs | Train | Validation |
|---|---:|---:|---:|
| Low-Light + Blur | 1,667 | 1,500 | 167 |
| Low-Light + Blur + Noise | 1,667 | 1,500 | 167 |
| Low-Light + Sharp | 1,666 | 1,500 | 166 |
| **Total** | **5,000** | **4,500** | **500** |

Seed 2025 selects 1,667 shared underlying frame keys, then assigns 1,500 keys to training and 167 to validation. Each key's degradation variants stay on the same side of the split. Class 3 omits one validation key to meet its 1,666-example allocation.

## Training

The configuration uses 65 epochs, crop size 128, batch size 1, zero DataLoader workers, AdamW, and L1 loss. The checked-in YAML currently sets `mixed_precision: false`; enable AMP only after confirming compatibility with the local PyTorch/CUDA stack and DarkIR's FFT operations. Checkpoints are saved under `models/training_lolblur/`.

The recorded run used a cap of 2,500 training batches per epoch:

```powershell
.\.venv\Scripts\python.exe train_darkir.py --config options/train_darkir_lolblur.yml --max-train-batches 2500
```

At batch size 1, this processes at most 2,500 of 4,500 training pairs per epoch, not a full pass. Validation reports overall and per-class L1 loss. For a short check rather than a full epoch:

```powershell
.\.venv\Scripts\python.exe verify_lolblur.py --root data/datasets/LOLBlur/train
.\.venv\Scripts\python.exe train_darkir.py --sanity
```

## Testing and evaluation

Place the official test split separately under `data/datasets/LOLBlur/test` and a compatible checkpoint at `models/training_lolblur/best_official_format_fixed.pt`.

```powershell
.\.venv\Scripts\python.exe testing.py -p options/test/LOLBlur.yml
.\.venv\Scripts\python.exe evaluate_darkir_rem.py
.\.venv\Scripts\python.exe baseline_evaluation.py
.\.venv\Scripts\python.exe evaluate_rem.py
```

`testing.py` evaluates the configured condition (default: `low_blur_noise`). `evaluate_darkir_rem.py` evaluates `low_blur` and `low_sharp` with DarkIR. The baseline scripts measure inputs without restoration. Set `DARKIR_LOLBLUR_TEST` to override the dataset path where supported. Full evaluation is compute-intensive; the test scripts have primarily been used with CUDA.

## Custom image inference

Put supported images directly in `images/inputs`, then run:

```powershell
.\.venv\Scripts\python.exe inference.py -p options/inference/LOLBlur.yml -i images/inputs
```

Images are written to `images/results`. The config matches DarkIR-m and the expected local checkpoint. The Streamlit interface command is:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app is included but was not runtime-verified from a clean checkout; it requires the excluded checkpoint. The optional paired custom evaluator expects `low1`/`gt1` through `low12`/`gt12` in `custom_images/`. Those example images are excluded until redistribution rights are confirmed.

## Noise-robustness experiment

[`noise_robustness.py`](noise_robustness.py) samples 100 matched test pairs with seed 2025. It adds Gaussian noise with standard deviations 5, 10, 20, and 30 on the 8-bit scale (`sigma / 255`) to already degraded `low_blur_noise` inputs. References are unchanged. These synthetic-noise measurements are not proof of robustness to all real-world noise.

Recorded aggregate values from [`noise_robustness_summary.csv`](results/noise_robustness/noise_robustness_summary.csv):

| Sigma | Input PSNR | Restored PSNR | Input SSIM | Restored SSIM | Input LPIPS | Restored LPIPS |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10.615937 | 20.943094 | 0.239453 | 0.467304 | 0.839342 | 0.670123 |
| 10 | 10.602482 | 18.488017 | 0.174177 | 0.286163 | 1.044412 | 0.895066 |
| 20 | 10.588171 | 16.381044 | 0.106071 | 0.145514 | 1.239911 | 1.125289 |
| 30 | 10.578518 | 14.893276 | 0.074738 | 0.090929 | 1.324845 | 1.234805 |

## Recorded test results

Results are recorded in `results/research/condition_metrics.csv` and the baseline text summaries. The test set has 1,800 pairs per condition. Baselines compare degraded inputs to the aligned target; DarkIR results compare restored outputs to that target.

| Condition | Baseline PSNR | Baseline SSIM | Baseline LPIPS | DarkIR PSNR | DarkIR SSIM | DarkIR LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Low-Light + Blur (`low_blur`) | 10.738910 | 0.313874 | 0.412786 | 23.792050 | 0.770613 | 0.313093 |
| Low-Light + Blur + Noise (`low_blur_noise`) | 10.734589 | 0.307881 | 0.444950 | 23.121011 | 0.714039 | 0.383748 |
| Low-Light + Sharp (`low_sharp`) | 10.721686 | 0.357869 | 0.245753 | 28.016862 | 0.941119 | 0.120857 |

These are the project's recorded metrics; they do not establish exact reproduction of the paper's protocol or benchmark.

## Result figures

- [DarkIR network diagram](assets/networks-scheme.png)
- [Condition metric comparison](results/research/metric_comparison.png)
- [PSNR](results/research/psnr_db.png) · [SSIM](results/research/ssim.png) · [LPIPS](results/research/lpips.png)
- Noise plots: [PSNR](results/noise_robustness/noise_psnr.png) · [SSIM](results/noise_robustness/noise_ssim.png) · [LPIPS](results/noise_robustness/noise_lpips.png)

Aggregate CSV files and metric plots are retained. Per-image renders containing test imagery remain local and are excluded from Git by default.

## Hardware and software

- DarkIR-m, width 32, 3,321,638 parameters.
- The recorded benchmark used `cuda:0`, 1,120 × 640 input, 10 warm-up and 50 measured runs: mean latency 804.78 ms and approximate throughput 1.24 images/s. The CSV records the CUDA index, not the GPU model.
- Project hardware: NVIDIA GeForce RTX 2050 with 4 GiB VRAM. Crop 128 and batch 1 are conservative for this limit.
- Python 3.10, PyTorch 2.5.1, TorchVision 0.20.1. See [`requirements.txt`](requirements.txt).

## Limitations and future improvements

- The recorded run caps training at 2,500 of 4,500 pairs per epoch.
- Checkpoints are excluded from Git; obtain a compatible weight file separately or train locally.
- Dataset access and licensing are the user's responsibility.
- Synthetic robustness tests cover only the noise model and levels listed above.
- The Streamlit app and clean-checkout workflow were not runtime-verified in this repository session.
- Future work could run uncapped epochs, test additional noise types and datasets, and evaluate on more hardware.

## Repository structure

```text
archs/                   DarkIR architecture and related model code
data/dataset_reader/     Dataset readers for testing
data/paired_lolblur.py   Multi-condition paired training loader/split
losses/                  Restoration losses and metric helpers
options/                 Training, testing, and inference configurations
results/research/        Aggregate metrics, timing, complexity, and plots
results/noise_robustness/ Synthetic-noise metrics and plots
models/                  Local checkpoints (ignored by Git)
custom_images/           Local paired examples (ignored by Git)
train_darkir.py          Training entry point
testing.py               LOL-Blur evaluation entry point
inference.py             Folder inference entry point
app.py                   Streamlit interface
```

## Credits, license, and citation

DarkIR was created by Daniel Feijoo, Juan C. Benito, Alvaro Garcia, and Marcos V. Conde at CIDAUT AI and the University of Wuerzburg. The upstream [`LICENSE`](LICENSE) is MIT. LOL-Blur and other datasets remain subject to their owners' terms; obtain them separately and do not commit them here.

```bibtex
@InProceedings{Feijoo_2025_CVPR,
  author    = {Feijoo, Daniel and Benito, Juan C. and Garcia, Alvaro and Conde, Marcos V.},
  title     = {DarkIR: Robust Low-Light Image Restoration},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference},
  month     = {June},
  year      = {2025},
  pages     = {10879--10889}
}
```

## GitHub publication

Review staged files first and confirm that datasets, user-supplied examples, weights, and test-image renders are excluded. Create an empty public repository on GitHub, then from this directory:

```powershell
git init -b main
git add .
git status --short
git commit -m "Prepare DarkIR restoration project for publication"
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

Alternatively, with GitHub CLI installed and authenticated:

```powershell
gh repo create <OWNER>/<REPOSITORY> --public --source=. --remote=origin --push
```

Do not add datasets or checkpoint weights unless distribution rights and hosting limits have been checked.
