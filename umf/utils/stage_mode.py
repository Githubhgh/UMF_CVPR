from collections import OrderedDict
from collections.abc import Mapping


REACTION_STATE_DICT_PREFIXES = (
    "reac_denoiser.",
    "flow_scheduler_org.",
    "clipTransEncoder_react.",
    "clip_ln_react.",
)


def _get_field(obj, name, default=None):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_train_cfg(cfg):
    train_cfg = _get_field(cfg, "TRAIN")
    if train_cfg is None:
        raise ValueError("Config is missing TRAIN section.")
    return train_cfg


def get_diffusion_mode(cfg, default=None):
    mode = _get_field(_get_train_cfg(cfg), "DIFFUSION_MODE", default)
    if mode is None:
        return None
    return str(mode).strip().lower()


def get_train_stage(cfg):
    stage = _get_field(_get_train_cfg(cfg), "STAGE")
    if stage is None:
        raise ValueError("Config is missing TRAIN.STAGE.")
    return str(stage).strip().lower()


def validate_train_umf_mode(cfg):
    stage = get_train_stage(cfg)
    if stage != "diffusion":
        return

    mode = get_diffusion_mode(cfg)
    if mode != "indi":
        raise ValueError(
            "train_UMF.py requires TRAIN.DIFFUSION_MODE='indi' when TRAIN.STAGE='diffusion'."
        )


def validate_train_react_mode(cfg):
    stage = get_train_stage(cfg)
    if stage != "diffusion":
        raise ValueError(
            "train_react.py requires TRAIN.STAGE='diffusion'."
        )

    mode = get_diffusion_mode(cfg)
    if mode != "react":
        raise ValueError(
            "train_react.py requires TRAIN.DIFFUSION_MODE='react'."
        )


def prune_reaction_keys_for_indi_checkpoint(state_dict):
    pruned = OrderedDict()
    removed = []
    for key, value in state_dict.items():
        if key.startswith(REACTION_STATE_DICT_PREFIXES):
            removed.append(key)
            continue
        pruned[key] = value
    return pruned, removed
