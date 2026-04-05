import os
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from umf.callback import ProgressLogger
from umf.config import parse_args
from umf.data.get_data import get_datasets
from umf.models.get_model import get_model
from umf.utils.logger import create_logger
from umf.utils.stage_mode import validate_train_umf_mode

os.environ.setdefault("UMF", "")
#os.environ.setdefault("UMF", "debug") #os.environ.setdefault("UMF", "debug") # Uncomment this line to enable debug mode, which uses a tiny dataset and validates more frequently.
os.environ["umf_current_epoch"] = str(0)


def _apply_debug_overrides(cfg):
    if "debug" in os.environ.get("UMF"):
        print("[DEBUG] Debug mode is active.")
        cfg.NAME = "Debug9"
        cfg.LOGGER.WANDB.RESUME_ID = "Debug"
        cfg.DEVICE = [0]
        cfg.LOGGER.VAL_EVERY_STEPS = 1


def _resume_from_experiment(cfg):
    if not cfg.TRAIN.RESUME:
        return cfg

    resume = cfg.TRAIN.RESUME
    backcfg = cfg.TRAIN.copy()
    if not os.path.exists(resume):
        raise ValueError("Resume path is not right.")

    file_list = sorted(os.listdir(resume), reverse=True)
    for item in file_list:
        if item.endswith(".yaml"):
            cfg = OmegaConf.load(os.path.join(resume, item))
            cfg.TRAIN = backcfg
            break

    if not cfg.TRAIN.PRETRAINED:
        raise ValueError(
            "TRAIN.RESUME is set but TRAIN.PRETRAINED is empty. "
            "Please explicitly set TRAIN.PRETRAINED to the checkpoint path."
        )
    if not os.path.exists(cfg.TRAIN.PRETRAINED):
        raise ValueError(
            f"TRAIN.PRETRAINED does not exist: {cfg.TRAIN.PRETRAINED}"
        )

    if os.path.exists(os.path.join(resume, "wandb")):
        wandb_list = sorted(os.listdir(os.path.join(resume, "wandb")), reverse=True)
        for item in wandb_list:
            if "run-" in item:
                cfg.LOGGER.WANDB.RESUME_ID = item.split("-")[-1]
    return cfg


def _build_loggers(cfg):
    loggers = []
    if cfg.LOGGER.WANDB.PROJECT:
        wandb_run_id = cfg.LOGGER.WANDB.RESUME_ID or cfg.NAME
        wandb_logger = pl_loggers.WandbLogger(
            project=cfg.LOGGER.WANDB.PROJECT,
            offline=cfg.LOGGER.WANDB.OFFLINE,
            id=wandb_run_id,
            save_dir=cfg.FOLDER_EXP,
            version="",
            name=cfg.NAME,
            anonymous=False,
            log_model=False,
        )
        loggers.append(wandb_logger)

    if cfg.LOGGER.TENSORBOARD:
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=cfg.FOLDER_EXP,
            sub_dir="tensorboard",
            version="",
            name="",
        )
        loggers.append(tb_logger)
    return loggers


def _metric_monitor():
    return {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_1": "Metrics/gt_R_precision_top_1",
        "gt_R_TOP_2": "Metrics/gt_R_precision_top_2",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "GA_VEL": "Metrics/GA_VEL",
        "GA_BL": "Metrics/GA_BL",
        "GA_FC": "Metrics/GA_FC",
        "GB_VEL": "Metrics/GB_VEL",
        "GB_BL": "Metrics/GB_BL",
        "GB_FC": "Metrics/GB_FC",
        "Inter_DM": "Metrics/Inter_DM",
        "Inter_JA": "Metrics/Inter_JA",
        "Inter_RO": "Metrics/Inter_RO",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "gt_Diversity": "Metrics/gt_Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",
        "gt_Accuracy": "Metrics/gt_accuracy",
        "HM_R_TOP_1": "Metrics/HM_R_precision_top_1",
        "HM_R_TOP_2": "Metrics/HM_R_precision_top_2",
        "HM_R_TOP_3": "Metrics/HM_R_precision_top_3",
        "HM_gt_R_TOP_1": "Metrics/HM_gt_R_precision_top_1",
        "HM_gt_R_TOP_2": "Metrics/HM_gt_R_precision_top_2",
        "HM_gt_R_TOP_3": "Metrics/HM_gt_R_precision_top_3",
        "HM_FID": "Metrics/HM_FID",
        "HM_Diversity": "Metrics/HM_Diversity",
        "HM_gt_Diversity": "Metrics/HM_gt_Diversity",
        "HM_MM_dist": "Metrics/HM_Matching_score",
        "HM_gt_MM_dist": "Metrics/HM_gt_Matching_score",
    }


def _build_callbacks(cfg):
    metric_monitor = _metric_monitor()
    return [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            filename="{epoch}",
            monitor="step",
            mode="max",
            every_n_epochs=25,
            save_top_k=2,
            save_last=False,
            save_on_train_epoch_end=True,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            filename="epoch={epoch}-fid={Metrics/FID:.2f}",
            monitor="Metrics/FID",
            mode="min",
            verbose=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
    ]


def _build_trainer(cfg, loggers, callbacks):
    if len(cfg.DEVICE) > 1:
        ddp_strategy = "ddp_find_unused_parameters_true"
    else:
        ddp_strategy = "auto"

    return pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        strategy=ddp_strategy,
        default_root_dir=cfg.FOLDER_EXP,
        log_every_n_steps=cfg.LOGGER.VAL_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
    )


def _load_pretrained_vae_if_needed(cfg, model, logger):
    if not cfg.TRAIN.PRETRAINED_VAE:
        return

    logger.info("Loading pretrain vae from {}".format(cfg.TRAIN.PRETRAINED_VAE))
    state_dict = torch.load(
        cfg.TRAIN.PRETRAINED_VAE,
        weights_only=False,
        map_location="cpu",
    )["state_dict"]

    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "vae":
            name = k.replace("vae.", "")
            vae_dict[name] = v
    model.vae.load_state_dict(vae_dict, strict=True)


def _load_pretrained_model_if_needed(cfg, model, logger):
    if not cfg.TRAIN.PRETRAINED:
        return

    logger.info("Loading pretrain mode from {}".format(cfg.TRAIN.PRETRAINED))
    logger.info("Attention! VAE will be recovered")
    state_dict = torch.load(
        cfg.TRAIN.PRETRAINED,
        map_location="cpu",
        weights_only=False,
    )["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k not in ["denoiser.sequence_pos_encoding.pe"]:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)


def main():
    cfg = parse_args()
    validate_train_umf_mode(cfg)
    _apply_debug_overrides(cfg)

    logger = create_logger(cfg, phase="train")
    cfg = _resume_from_experiment(cfg)
    pl.seed_everything(cfg.SEED_VALUE)

    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    loggers = _build_loggers(cfg)
    logger.info(OmegaConf.to_yaml(cfg))

    datasets = get_datasets(cfg, logger=logger)
    logger.info("datasets module {} initialized".format("".join(cfg.TRAIN.DATASETS)))

    model = get_model(cfg, datasets[0])
    logger.info("model {} loaded".format(cfg.model.model_type))

    callbacks = _build_callbacks(cfg)
    os.environ["umf_plot_dir"] = os.path.join(cfg.FOLDER_EXP)
    logger.info("Callbacks initialized")

    trainer = _build_trainer(cfg, loggers, callbacks)
    logger.info("Trainer initialized")

    _load_pretrained_vae_if_needed(cfg, model, logger)
    _load_pretrained_model_if_needed(cfg, model, logger)

    if cfg.TRAIN.RESUME:
        trainer.fit(model, datamodule=datasets[0], ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datasets[0])

    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
