# Configuration Guide

This repository keeps a focused UMF mainline setup for:

- Stage 1 VAE training
- Stage 2 indi_denoiser training
- Stage 3 reac_denoiser training
- Evaluation with `test.py`

Supported data scope in this branch: **InterHuman + HumanML3D**.

## Kept Configuration Files

Core files:

- `base.yaml`
- `assets.yaml`
- `datasets.yaml`
- `config_vae.yaml`
- `config_pflow.yaml`
- `config_sflow.yaml`

Network module files (loaded via `model.target: modules`):

- `modules/motion_vae.yaml`
- `modules/motion_vae_inter.yaml`
- `modules/text_encoder.yaml`
- `modules/denoiser.yaml`
- `modules/react_denoiser.yaml`
- `modules/scheduler.yaml`
- `modules/evaluators.yaml`
- `modules/traj_encoder.yaml`

## How Config Composition Works

Runtime config is merged in this order:

1. `configs/base.yaml`
2. experiment config passed by `--cfg`
3. all module yamls under `configs/modules/` (selected by `model.target`)
4. assets config passed by `--cfg_assets`
5. optional base path overrides `configs/umf_base.yaml` (or `--cfg_umf_base`)

The implementation is in `umf/config.py`.

## Single-File Path Control

For open-source portability, prefer editing only:

- `configs/umf_base.yaml`

`PATHS` values in this file can override runtime roots (datasets/assets/output)
and key checkpoint fields without changing stage config files.

This repo currently keeps runnable defaults in `umf_base.yaml` for local
development. For open-source release, replace path values with your own.

To avoid cross-folder hard dependencies, lightweight HumanML3D stats are
vendored in-repo:

- `assets/runtime/humanml3d_meta/mean_org.npy`
- `assets/runtime/humanml3d_meta/std_org.npy`

Sentinel strings are still supported when you want to defer a field:

- `__FROM_STAGE_CONFIG__`
- `__FROM_DATASET_CFG__`
- `__FROM_DATASET_HUMANML3D_ROOT__`
- `__FROM_WORD_VECTORIZER_ROOT__`

Later, replace those sentinel values with concrete project paths.

## Typical Usage

Stage 1:

```bash
python train_UMF.py --cfg configs/config_vae.yaml --cfg_assets configs/assets.yaml --nodebug
```

Stage 2:

```bash
python train_UMF.py --cfg configs/config_pflow.yaml --cfg_assets configs/assets.yaml --nodebug
```

`config_pflow.yaml` must keep:
- `TRAIN.STAGE: diffusion`
- `TRAIN.DIFFUSION_MODE: indi`

Stage 3:

```bash
python train_react.py --cfg configs/config_sflow.yaml --cfg_assets configs/assets.yaml --nodebug
```

`config_sflow.yaml` must keep:
- `TRAIN.STAGE: diffusion`
- `TRAIN.DIFFUSION_MODE: react`

The repo no longer uses epoch-based switching between indi/react branches.

Evaluation:

```bash
python test.py --cfg configs/config_pflow.yaml --cfg_assets configs/assets.yaml
```
