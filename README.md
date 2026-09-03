\# NAFNet-SR: AI-Based Restoration of Degraded Semiconductor Images



\*\*Team ChipCrafters\*\* — Semicon India Hackathon 2026, KLA Problem Statement (Phase 2)



Joint 2x super-resolution and multi-source denoising for high-speed SEM wafer inspection, built on a NAFNet-style encoder-decoder.



<p align="left">

&#x20; <img alt="PSNR" src="https://img.shields.io/badge/PSNR-23.85%20dB%20avg%20(peak%2033%2B)-blueviolet">

&#x20; <img alt="SSIM" src="https://img.shields.io/badge/SSIM-0.8961-blueviolet">

&#x20; <img alt="LPIPS" src="https://img.shields.io/badge/LPIPS-0.3337-blueviolet">

&#x20; <img alt="Params" src="https://img.shields.io/badge/Params-4.88M-informational">

&#x20; <img alt="Framework" src="https://img.shields.io/badge/PyTorch-CUDA%2FAMP-ee4c2c">

&#x20; <img alt="Status" src="https://img.shields.io/badge/Phase-2%20Complete-success">

</p>



\---



\## Table of Contents

\- \[Problem Statement](#problem-statement)

\- \[Our Approach](#our-approach)

\- \[Architecture](#architecture)

\- \[Loss Function](#loss-function)

\- \[Training Strategy](#training-strategy)

\- \[Results](#results)

\- \[Repository Structure](#repository-structure)

\- \[Getting Started](#getting-started)

&#x20; - \[Installation](#installation)

&#x20; - \[Data Preparation](#data-preparation)

&#x20; - \[Training](#training)

&#x20; - \[Evaluation / Inference](#evaluation--inference)

\- \[Innovation Highlights](#innovation-highlights)

\- \[Tech Stack](#tech-stack)

\- \[References](#references)

\- \[Team](#team)

\- \[License](#license)



\---



\## Problem Statement



Semiconductor inspection systems rarely capture perfectly clean images — noise and resolution loss introduced during imaging hide the fine structural detail that downstream defect-inspection tasks depend on.



KLA's dataset pairs each clean ground-truth SEM (Scanning Electron Microscopy) image with a degraded counterpart corrupted by:

\- \*\*Speckle noise\*\*

\- \*\*Additive Gaussian noise\*\*

\- \*\*Downsampling\*\* (resolution loss, applied in an undisclosed combination/order)



\*\*Goal:\*\* Reverse these degradations and reconstruct an image as close as possible to the original — without hallucinating detail or destroying real structure — while generalizing to unseen (out-of-distribution) test content and running efficiently on GPU hardware (benchmarked end-to-end on an NVIDIA H100, including I/O and batching).



\## Our Approach



In Phase 1 we built a NAFNet-based U-Net that jointly denoises and super-resolves in a single forward pass. In Phase 2, we scaled that idea into \*\*NAFNet-SR\*\*, trained and validated directly on KLA's paired dataset (4,785 `NoisyLR`/`GT` `.npy` pairs, 128×128 → 256×256), closing the gap between the theoretical design and measured, reproducible performance.



Key design principle: the model predicts a \*\*residual correction on top of a bicubic-upsampled input\*\*, rather than reconstructing the image from scratch. This keeps training stable and convergence fast, even with added architectural depth.



<p align="center">

&#x20; <img src="Assets/Pipeline.png" alt="Restoration Pipeline">

</p>



\## Architecture



\*\*NAFNet-SR\*\* — a UNet-style encoder-decoder built entirely from Nonlinear-Activation-Free (NAF) blocks.



<p align="center">

&#x20; <img src="Assets/architecture.png" alt="NAFNet-SR Architecture">

</p>



| Component | Detail |

|---|---|

| Encoder | 3 stages, `\[2, 2, 4]` blocks, channels `48 → 96 → 192 → 384` |

| Bottleneck | 2 NAFBlocks on the compressed representation |

| Decoder | 3 stages, `\[2, 2, 2]` blocks, PixelShuffle upsampling + encoder skip connections |

| Block design | SimpleGate activation + Spatial/Channel Attention (squeeze-and-excitation) instead of conventional nonlinearities — better gradient flow |

| Upsampling | Sub-pixel PixelShuffle convolutions for artifact-free 2x super-resolution |

| Output | Residual added to a bicubic-upsampled version of the input |

| Parameters | ≈ 4.88M (vs. 2.65M in Phase 1) |



Restoration is fully end-to-end: \*\*one forward pass\*\* performs denoising and 2x super-resolution simultaneously.



\## Loss Function



A four-term composite loss, each term deliberately mapped to one of KLA's scored metrics:



| Term | Weight | Target Metric | Purpose |

|---|---|---|---|

| Charbonnier loss | 1.0 | PSNR | Robust pixel-level accuracy |

| MS-SSIM loss | 0.5 | SSIM | Multi-scale structural fidelity |

| FFT-domain loss | 0.1 | Edge sharpness | Recovers high-frequency detail |

| VGG16 perceptual loss | 0.05 | LPIPS | Visual/perceptual quality |



Improving this loss directly improves leaderboard score, rather than optimizing a loose proxy objective.



\## Training Strategy



\- \*\*Augmentation:\*\*

&#x20; - CutBlur (40% probability) — swaps a random patch between the upsampled-degraded and ground-truth images, teaching spatially adaptive restoration instead of uniform sharpening

&#x20; - Random flips / 90° rotations (50%)

&#x20; - Dynamic Gaussian noise injection (30%, σ = 0.01–0.05) for robustness to unseen noise strengths

\- \*\*Optimizer:\*\* AdamW (lr = 1e-3, weight decay = 1e-4), cosine-annealed to 1e-6 over 45 epochs

\- \*\*Stability:\*\* Gradient clipping (norm 1.0), mixed-precision (fp16/AMP) training

\- \*\*Weight averaging:\*\* Exponential Moving Average (decay = 0.999) — the EMA shadow model is what gets evaluated and checkpointed

\- \*\*Reproducibility:\*\* Fixed global seed (42) across Python, NumPy, PyTorch/CUDA

\- \*\*Data split:\*\* 90/10 train/validation, files auto-discovered and matched by numeric ID



\## Results



Evaluated at epoch 45 on the full held-out validation set, against KLA's three scored metrics:



| Metric | Result | Target | Notes |

|---|---|---|---|

| \*\*SSIM\*\* | 0.8961 | 0.90 | Within 0.4% of target |

| \*\*PSNR (avg)\*\* | 23.85 dB | 30+ dB | Peaks of 33.89 dB / 32.39 dB on strong cases |

| \*\*LPIPS\*\* | 0.3337 | \~0.3 | Close to target |

| Baseline (bicubic) | PSNR 20.10 / SSIM 0.501 | — | For comparison |



The model clearly outperforms the bicubic-upsampling baseline on structural similarity, confirming the multi-loss strategy recovers real structural detail rather than just averaging pixels. Average PSNR is pulled down by a harder subset of test images, while performance on recoverable-degradation images is strong.



Two experimental alternatives were also evaluated for comparison and ultimately outperformed by NAFNet-SR:



| Model | Params | PSNR | SSIM |

|---|---|---|---|

| ResidualSRModel (6 residual blocks, L1 loss) | 518K | 23.55 dB | 0.578 |

| SharpSRNet (12-block RRDBNet-style, tiled inference) | — | 23–24 dB | \~0.65 |

| \*\*NAFNet-SR (this repo)\*\* | \*\*4.88M\*\* | \*\*23.85 dB\*\* | \*\*0.8961\*\* |



<br>



\*\*Top Successful Cases (Best Restoration):\*\*

<p align="center">

&#x20; <img src="Assets/best\_restorations.png" alt="Best Restorations">

</p>



\*\*Failure Cases Analysis (Hardest Degradations):\*\*

<p align="center">

&#x20; <img src="Assets/worst\_failures.png" alt="Worst Failures">

</p>



\## Repository Structure

