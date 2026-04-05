<div align="center">
  <h1>Unified Number-Free Text-to-Motion Generation Via Flow Matching</h1>
  <p><strong>CVPR 2026 </strong></p>
  <p>
    <strong>Guanhe Huang, Oya Celiktutan</strong>
  </p>

  <h4>
    <a href="https://arxiv.org/abs/2603.27040">[Paper]</a>
    &nbsp;•&nbsp;
    <a href="https://githubhgh.github.io/umf/">[Project Page]</a>
    &nbsp;•&nbsp;
    <a href="#-citation">[Citation]</a>
  </h4>

  <p>
    <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a href="#"><img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9+-blue.svg"></a>
  </p>
</div>

---

### Abstract
> *Generative models excel at motion synthesis for a fixed number of agents but struggle to generalize with variable agents. Based on limited, domain-specific data, existing methods employ autoregressive models to generate motion recursively, which suffer from inefficiency and error accumulation. We propose Unified Motion Flow (UMF), which consists of Pyramid Motion Flow (P-Flow) and Semi-Noise Motion Flow (S-Flow). UMF decomposes the number-free motion generation into a single-pass motion prior generation stage and multi-pass reaction generation stages. Specifically, UMF utilizes a unified latent space to bridge the distribution gap between heterogeneous motion datasets, enabling effective unified training. For motion prior generation, P-Flow operates on hierarchical resolutions conditioned on different noise levels, thereby mitigating computational overheads. For reaction generation, S-Flow learns a joint probabilistic path that adaptively performs reaction transformation and context reconstruction, alleviating error accumulation. Extensive results and user studies demonstrate UMF’s effectiveness as a generalist model for multi-person motion generation from text. Project page: https://githubhgh.github.io/umf/.*

---

## 1. Installation

### System Configuration (Used in This Project)

The main experiments in this repository were run with the following setup:

- GPU: **One NVIDIA H200**
- CUDA: **12.8**
- Python: **3.9.x**
- OS: **Linux**

> Note: This project (except for the VAE training) uses less than 24GB GPU memory, so it should also work with CUDA 11.x and other GPUs. If you encounter CUDA-related issues, please check the PyTorch installation and CUDA compatibility first.

### Clone the Repository
```bash
git clone https://github.com/Githubhgh/UMF_CVPR.git
cd UMF_CVPR
```

### Environment Setup
We recommend using the existing `pyproject.toml` with `uv`, which has been validated for Python 3.9.

```bash
# From repository root
cd UMF_CVPR

# Create/sync environment from pyproject.toml
# Using 'pip install uv' if you don't have 'uv' installed globally.
uv init
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Quick check
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Alternative: Pure pip Installation

If you prefer a pip-only workflow, use the provided `requirements.txt` (configured for CUDA 12.8 wheels):

```bash
cd UMF_CVPR
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 2. Data and Pretrained Models

### Dataset Preparation
Place or symlink the **InterHuman** and **HUMANML3D** datasets into the `./data/` directory. The expected structure is:

[HumanML3D](https://github.com/EricGuo5513/HumanML3D)

[InterGen](https://github.com/tr3e/InterGen)
```
./data/InterHuman/
├── annots
├── motions
├── motions_processed
└── split

./data/HUMANML3D/
├── texts/
├── new_joint_vecs/
├── new_joints/
```




### Pretrained Models
Download the necessary checkpoints and place them in the specified locations.

1.  **Evaluation Checkpoint (`interclip.ckpt`)**:
    -   Place it at `./eval_model/interclip.ckpt`.

2.  **UMF & VAE Checkpoints**:
    -   [UMF Checkpoint on Google Drive](https://drive.google.com/file/d/1jmGD3wvXcig43uyV63BeqZPjFxd3uBV-/view?usp=drive_link)
    -   [VAE Checkpoint on Google Drive](https://drive.google.com/file/d/14OCETIMfrZf-ZCPTq4xWMhNK5w7dY7Eu/view?usp=sharing)

    After downloading, update your configuration files to point to the local paths of these checkpoints.

### Configuration Verification
Before proceeding, ensure the following paths are correctly set in your configuration files:

-   `configs/datasets.yaml`: Verify all dataset root paths.
-   `configs/assets.yaml`:
    -   `model.t2m_path`
    -   `DATASET.*.ROOT`


---

## 3. Training Pipeline

Training is divided into three sequential stages, as described in the paper.

### Stage 1: Motion Heterogeneous VAE
This stage trains the Variational Autoencoder for motion data.

```bash
python train_UMF.py \
  --cfg configs/config_vae.yaml \
  --cfg_assets configs/assets.yaml \
  --batch_size 64 \
  --nodebug
```
> **VS Code**: use the **Train VAE (stage1)** launcher.

### Stage 2: Pyramid Motion Flow: Individual Denoiser (`indi_denoiser`)
This stage trains the diffusion model for individual motion.

-   **Prerequisite**: Set `TRAIN.PRETRAINED_VAE` in your config to the checkpoint from Stage 1.
-   **Config**: Ensure `TRAIN.STAGE` is `diffusion` and `TRAIN.DIFFUSION_MODE` is `indi`.

```bash
python train_UMF.py \
  --cfg configs/config_pflow.yaml \
  --cfg_assets configs/assets.yaml \
  --batch_size 64 \
  --nodebug
```
> **VS Code**: use the **Train P-Flow (indi_denoiser)** launcher.

### Stage 3: Semi-Noise Motion Flow: Reactive Denoiser (`reac_denoiser`)
This stage trains the diffusion model for reactive motion in interactions.

-   **Prerequisites**:
    -   Set `TRAIN.PRETRAINED_VAE` to the Stage 1 checkpoint.
    -   Set `TRAIN.PRETRAINED_INDI` to the Stage 2 checkpoint (this is required).
-   **Config**: Ensure `TRAIN.STAGE` is `diffusion` and `TRAIN.DIFFUSION_MODE` is `react`.

```bash
python train_react.py \
  --cfg configs/config_sflow.yaml \
  --cfg_assets configs/assets.yaml \
  --batch_size 64 \
  --nodebug
```
> **VS Code**: use the **Train S-Flow (reac_denoiser)** launcher.

### Resume Training
When resuming an existing experiment, set both of the following:

- `TRAIN.RESUME`: path to the experiment directory (for config/logger recovery)
- `TRAIN.PRETRAINED`: explicit checkpoint file path used by `trainer.fit(..., ckpt_path=...)`

`TRAIN.RESUME` alone is not enough; checkpoint auto-selection is intentionally disabled to avoid loading the wrong file.

---

## 4. Evaluation

Specify the checkpoint path in `TEST.CHECKPOINTS` of the appropriate configuration file, then run:

| Model / Stage | Config | VS Code Launcher |
|---|---|---|
| VAE (Stage 1) | `configs/config_vae.yaml` | **Test VAE** |
| Individual denoiser (Stage 2) | `configs/config_pflow.yaml` | **Test P-Flow (indi_denoiser cfg)** |
| Reactive denoiser (Stage 3) | `configs/config_sflow.yaml` | **Test S-Flow (reac_denoiser cfg)** |

```bash
# Test VAE
python test.py --cfg configs/config_vae.yaml --cfg_assets configs/assets.yaml

# Test individual denoiser
python test.py --cfg configs/config_pflow.yaml --cfg_assets configs/assets.yaml

# Test reactive denoiser
python test.py --cfg configs/config_sflow.yaml --cfg_assets configs/assets.yaml
```

---

## CITATION

If you find this work useful for your research, please cite our paper:

```bibtex
@article{huang2026unified,
  title={Unified Number-Free Text-to-Motion Generation Via Flow Matching},
  author={Huang, Guanhe and Celiktutan, Oya},
  journal={arXiv preprint arXiv:2603.27040},
  year={2026}
}
```

---

## Contact
For questions or issues, please open an issue on the GitHub repository or contact us at [Email](mailto:guanhe.huang@kcl.ac.uk).


## Acknowledgments

Note that our code depends on other libraries, including:
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) by Guo et al.
- [InterGen](https://github.com/tr3e/InterGen) by Liang et al.
- [MotionLCM](https://github.com/Dai-Wenxun/MotionLCM) by Dai et al.
- [Pyramid-Flow](https://github.com/jy0205/Pyramid-Flow) by Jin et al.
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) by Max Planck Institute
Please follow their respective licenses when using this code.

---

## License

This project is distributed under the [MIT LICENSE](LICENSE).
