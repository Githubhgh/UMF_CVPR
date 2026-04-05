import os
from pathlib import Path
import numpy as np
import torch
from pytorch_lightning import LightningModule
from umf.models.metrics import (
    TM2TMetrics,
    TM2TMetrics_HM,
    MMMetrics,
)
from collections import defaultdict


class BaseModel(LightningModule):
    """
    BaseModel with integrated robust-validation support.
    Enable it by setting `robust_val_runs > 1` at initialization.
    """
    def __init__(self, robust_val_runs = 5, *args, **kwargs):
        """
        Initialize BaseModel.
        Args:
            robust_val_runs (int): Number of repeated validation passes.
                                   Default is 1 (standard single-pass validation).
                                   Values > 1 enable robust validation mode.
        """
        super().__init__(*args, **kwargs)
        self.times = []
        # Save the new argument to hparams for easy access and persistence.
        #robust_val_runs = 10
        self.save_hyperparameters('robust_val_runs')

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())

        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    # --- Helper methods for robust validation ---
    def _get_val_metrics_names(self):
        """Get validation metric names for the current configuration."""
        if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.metrics_dict:
            return ['MMMetrics']
        return self.metrics_dict

    def _compute_val_metrics(self):
        """Compute all current validation metrics without resetting them."""
        dico = {}
        metric_names = self._get_val_metrics_names()
        for metric_name in metric_names:
            metrics_obj = getattr(self, metric_name)
            metrics_values = metrics_obj.compute(sanity_flag=self.trainer.sanity_checking)
            # Add detailed metric prefixes to avoid key collisions (e.g., with losses).
            dico.update({
                f"Metrics/{key}": value.item()
                for key, value in metrics_values.items()
            })
        return dico

    def _reset_val_metrics(self):
        """Reset all validation metric states."""
        metric_names = self._get_val_metrics_names()
        for metric_name in metric_names:
            getattr(self, metric_name).reset()

    # --- End-of-epoch logic ---
    def allsplit_epoch_end(self, split: str, outputs):
        dico = {}
        os.environ['umf_current_epoch'] = str(self.trainer.current_epoch + 1)

        if split in ["train", "val"]:
            losses = self.losses[split]
            loss_dict = losses.compute(split)
            losses.reset()
            dico.update({
                losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items() if not torch.isnan(value)
            })
            if split == 'train':
                print('Epoch: ', self.trainer.current_epoch, {k: v for k, v in dico.items() if v != 0})

        if split == "test":
            metrics_dicts = self.metrics_dict
            for metric in metrics_dicts:
                metrics_dict = getattr(self, metric).compute(sanity_flag=self.trainer.sanity_checking)
                getattr(self, metric).reset()
                dico.update({
                    f"Metrics/{metric}_{key}": value.item()
                    for key, value in metrics_dict.items()
                })

        if split != "test":
            dico.update({
                "epoch": float(self.trainer.current_epoch),
                "step": float(self.trainer.global_step),
            })
            
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_train_epoch_end(self, outputs=None):
        return self.allsplit_epoch_end("train", outputs)

    def on_validation_epoch_end(self, outputs=None):
        # If robust_val_runs <= 1, run standard single-pass validation.
        if self.hparams.robust_val_runs <= 1:
            # Loss computation
            losses = self.losses["val"]
            loss_dict = losses.compute("val")
            losses.reset()
            dico = {losses.loss2logname(k, "val"): v.item() for k, v in loss_dict.items()}

            # Metric computation and reset
            val_metrics = self._compute_val_metrics()
            self._reset_val_metrics()
            dico.update(val_metrics)
            
            if not self.trainer.sanity_checking:
                self.log_dict(dico, sync_dist=True, rank_zero_only=True)
            return

        # --- Robust validation logic ---
        if self.trainer.sanity_checking:
            return

        print(f"\n--- Running Robust Validation ({self.hparams.robust_val_runs} runs) at global step: {self.trainer.global_step} ---")
        all_runs_metrics = []

        # **Run 1**: Use the pass automatically executed by PyTorch Lightning.
        metrics_run_1 = self._compute_val_metrics()
        self._reset_val_metrics()
        all_runs_metrics.append(metrics_run_1)
        print(f"  Run 1/{self.hparams.robust_val_runs} metrics computed.")

        # **Subsequent runs**: iterate manually.
        if self.trainer.current_epoch > 1000:
            runs_loop = self.hparams.robust_val_runs
        else:
            runs_loop = 1
        self.eval()
        with torch.no_grad():
            for i in range(1, runs_loop):
                for batch_idx, batch in enumerate(self.trainer.datamodule.val_dataloader()):
                    batch = self.transfer_batch_to_device(batch, self.device, batch_idx)
                    self.allsplit_step("val", batch, batch_idx)
                
                metrics_run_i = self._compute_val_metrics()
                self._reset_val_metrics()
                all_runs_metrics.append(metrics_run_i)
                print(f"  Run {i+1}/{self.hparams.robust_val_runs} metrics computed.")

        # **Aggregate results**: compute mean metrics across runs.
        robust_metrics_agg = defaultdict(list)
        for metric_dict in all_runs_metrics:
            for key, val in metric_dict.items():
                robust_metrics_agg[key].append(val)
        
        final_avg_metrics = {key: np.mean(val) for key, val in robust_metrics_agg.items()}
        print(f"--- Robust Validation Finished. Final average metrics: {final_avg_metrics} ---")

        # **Logging**: log final averaged metrics and the first-run validation loss.
        losses = self.losses["val"]
        loss_dict = losses.compute("val")
        losses.reset()
        final_log_dict = {losses.loss2logname(k, "val"): v.item() for k, v in loss_dict.items()}
        final_log_dict.update(final_avg_metrics)
        
        self.log_dict(final_log_dict, sync_dist=True, rank_zero_only=True)

    def on_test_epoch_end(self, outputs=None):
        self.cfg.TEST.REP_I = self.cfg.TEST.REP_I + 1
        return self.allsplit_epoch_end("test", outputs)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}

    def configure_metrics(self):
        for metric in self.metrics_dict:
            if metric == "TM2TMetrics":
                self.TM2TMetrics = TM2TMetrics(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            elif metric == "TM2TMetrics_HM":
                self.TM2TMetrics_HM = TM2TMetrics_HM(
                    diversity_times=30
                    if self.debug else self.cfg.TEST.DIVERSITY_TIMES,
                    dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            else:
                raise NotImplementedError(f"Do not support Metric Type {metric}")
        if "TM2TMetrics" in self.metrics_dict:
            self.MMMetrics = MMMetrics(
                mm_num_times=self.cfg.TEST.MM_NUM_TIMES,
                dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP,
            )

    def save_npy(self, outputs):
        cfg = self.cfg
        output_dir = Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.model_type),
                str(cfg.NAME),
                "samples_" + cfg.TIME,
            ))
        if cfg.TEST.SAVE_PREDICTIONS:
            lengths = [i[1] for i in outputs]
            outputs = [i[0] for i in outputs]
            if cfg.TEST.DATASETS[0].lower() in ["humanml3d", "kit"]:
                keyids = self.trainer.datamodule.test_dataset.name_list
                for i in range(len(outputs)):
                    for bid in range(min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        gen_joints = outputs[i][bid].cpu().numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}_{cfg.TEST.REP_I}.npy"
                        else:
                            name = f"{keyid}.npy"
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
