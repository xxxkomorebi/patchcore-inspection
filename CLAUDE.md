# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands require `PYTHONPATH=src` since the package is not installed as a module:

```bash
export PYTHONPATH=src
```

**Train and evaluate:**
```bash
python bin/run_patchcore.py --gpu 0 --seed 123 --save_segmentation_images \
    --log_group <name> --log_project <project> results \
    patch_core --vq_num_hiddens 128 --vq_num_embeddings 128 --vq_embedding_dim 128 \
        --vq_commitment_cost 0.25 --vq_lr 0.001 --vq_epochs 50 \
        --vq_use_ema_codebook --patchsize 3 \
    dataset --resize 256 --imagesize 224 -d <class> mpdd MPDD

# Or use the provided example script:
bash sample_training.sh
```

**Evaluate a saved model:**
```bash
python bin/load_and_evaluate_patchcore.py --gpu -1 --seed 0 <save_path> \
    patch_core_loader -p <model_dir> --no-faiss_on_gpu \
    dataset --resize 366 --imagesize 320 -d <class> mvtec <datapath>
```

**Lint / format:**
```bash
flake8 src test bin
black src test bin --check   # check only
black src test bin           # auto-format
isort src test bin           # import sort (profile=black, force_single_line)
```

**Tests:**
```bash
pytest test/ -q --log-level ERROR        # all tests
pytest test/test_foo.py::test_bar        # single test
```

## Architecture

This project replaces the original PatchCore memory-bank approach with a **VQ-VAE trained from scratch on normal images only**. Anomaly scoring is entirely reconstruction-based — no pretrained ImageNet backbone is required by default.

### Training vs. inference

**Training** (`PatchCore.fit` → `_train_vqvae`): Adam optimizer minimises `MSE(x_recon, x) + vq_loss` over `vq_epochs` epochs. Only normal images are seen.

**Inference** (`PatchCore._predict`): Two complementary anomaly signals are computed and fused:
- **MAD map** (latent space): `mean(|z_e − z_q|, dim=channel)` — measures how far the encoder output strays from the nearest codebook vector.
- **SSIM map** (image space): per-pixel structural similarity between the input and its reconstruction.
- **Fused map**: `MAD_map × (1 − SSIM_map)` — both signals must be high to produce a strong anomaly score.
- **Image-level score**: mean of the top 0.1 % pixels of the fused map.

Post-processing in `RescaleSegmentor` (`common.py`): bilinear upsample to input resolution + Gaussian blur (σ = 4).

### Module responsibilities

| File | Responsibility |
|---|---|
| `src/patchcore/patchcore.py` | `PatchCore` class (train / predict / save / load); `PatchMaker` (patch extraction via `torch.nn.Unfold`); `calculate_mad_map`, `calculate_ssim_map` |
| `src/patchcore/vqvae_model.py` | `VQVAE`, `Encoder`, `Decoder`, and the three interchangeable quantizers (see below) |
| `src/patchcore/common.py` | `RescaleSegmentor`; `NetworkFeatureAggregator` (hook-based feature extraction when a pretrained backbone is used as encoder); legacy `FaissNN` / `NearestNeighbourScorer` kept for `load_and_evaluate_patchcore.py` |
| `src/patchcore/backbones.py` | Registry of ~30 pretrained backbones (ResNet, EfficientNet, ViT, DenseNet via torchvision / timm), loaded by name with `eval()` |
| `src/patchcore/metrics.py` | Image-level and pixel-level AUROC |
| `src/patchcore/datasets/` | `MVTecDataset`, `MPDDDataset`, `CarDDDataset` — all share the same layout: `train/good`, `test/{good,<defect_type>}`, `ground_truth/<defect_type>` |
| `bin/run_patchcore.py` | Training + evaluation entry point; uses `click` command chaining |
| `bin/load_and_evaluate_patchcore.py` | Load a saved model and evaluate only |

### VQ-VAE encoder modes

The `Encoder` supports two modes selected at construction time:

- **Default (custom CNN)**: 3 conv blocks with stride-2 downsampling → 4× total reduction; BatchNorm + ReLU throughout. `--vq_layers_to_extract_from` selects which intermediate feature maps to concatenate (multi-scale fusion with bilinear alignment).
- **Pretrained backbone** (`--vq_backbone_name`): any backbone from `backbones.py`; features are extracted via forward hooks at the layers named by `--vq_layers_to_extract_from`.

### Quantizer selection

| CLI flag | Class | Codebook update | Loss |
|---|---|---|---|
| *(default)* | `VectorQuantizer` | gradient descent | codebook loss + β × commitment loss |
| `--vq_use_ema_codebook` | `EMAVectorQuantizer` | exponential moving average (no gradient through codebook) | β × commitment loss only |
| `--vq_use_fsq --vq_fsq_levels L1,L2,...` | `ScalarQuantizer` | none — implicit codebook via tanh + round + STE | none |

EMA codebook (`--vq_use_ema_codebook`) is recommended for regular use to reduce codebook collapse. FSQ (`--vq_use_fsq`) eliminates dead codes entirely at the cost of removing VQ loss as a training signal.

### CLI command chaining

`bin/run_patchcore.py` uses `click` group chaining. Subcommands `patch_core` and `dataset` each return a `(key, callable)` tuple; the framework collects them into a `methods` dict before `run()` executes. **Order on the command line matters**: `results patch_core [...] dataset [...] <dataset_type> <datapath>`.

### Output layout

Results are saved to `results/<log_project>/<log_group>/` (auto-incremented if the directory exists):

- `results.csv` — per-subdataset `instance_auroc`, `full_pixel_auroc`, `anomaly_pixel_auroc`
- `perplexity_epoch.csv` — codebook perplexity per epoch (diagnostic for quantizer utilisation)
- `models/<dataset>/vqvae_model.pth` + `patchcore_params.pkl` — saved weights and hyperparams (requires `--save_patchcore_model`)
- `recon_segmentation_images/<dataset>/` — 4-panel PNGs (input / GT mask / anomaly map / reconstruction) and `reconstruction_metrics.csv` with per-image PSNR / SSIM (requires `--save_segmentation_images`)
