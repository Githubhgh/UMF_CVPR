import numpy as np
import torch
from tqdm import tqdm
import os
import logging


from umf.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric)

from .base import BASEDataModule
import codecs as cs
from torch.utils import data
import random
from rich.progress import track
from os.path import join as pjoin

from .humanml.data.dataset import Text2MotionDatasetV2#, TextOnlyDataset
from data_loaders.interhuman.interhuman import InterHumanDataset

LOGGER = logging.getLogger(__name__)


def _is_verbose_logging(cfg=None, debug=False):
    if debug:
        return True
    if cfg is not None and bool(getattr(cfg, "DEBUG", False)):
        return True
    umf_flag = os.environ.get("UMF", "")
    return isinstance(umf_flag, str) and "debug" in umf_flag


def _emit_log(logger, message):
    (logger or LOGGER).info(message)


def _cfg_path(cfg, key, default=None):
    if cfg is None:
        return default
    paths_cfg = getattr(cfg, "PATHS", None)
    if paths_cfg is None:
        return default
    value = getattr(paths_cfg, key, default)
    if value in (None, ""):
        return default
    value_text = str(value).strip()
    sentinel_values = {
        "__FROM_STAGE_CONFIG__",
        "__FROM_DATASET_CFG__",
        "__FROM_DATASET_HUMANML3D_ROOT__",
        "__FROM_WORD_VECTORIZER_ROOT__",
    }
    if value_text in sentinel_values:
        return default
    return value


class IH_HumanML3DDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "interhuman"
        self.njoints = 22
        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            self.Dataset = InterHumanDataset
            #self.Dataset = InterHuman_Single_Dataset
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }

        self.nfeats = 262 # 524

    def feats2joints(self, features): # [32,300,262]


        from data_loaders.interhuman.interhuman import MotionNormalizerTorch
        MN = MotionNormalizerTorch() # mean.shape [262]
        features_motion1 = MN.backward(features[...,:262].clone())
        #features_motion2 = MN.backward(features[...,262:].clone())
        joints1 = features_motion1[..., :22*3].reshape(features.shape[0:2]+(22,3)) # [32,300,22,3]
        #joints2 = features_motion2[..., :22*3].reshape(features.shape[0:2]+(22,3)) ## [32,300,22,3]
        #return torch.concat([joints1, joints2], -2) # [32,300,44,3]
        return joints1



    def joints2feats(self, features):
        raise Exception

        features = process_file(features, self.njoints)[0]
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = (features - mean) / std
        return features

    def renorm4t2m(self, features):
        raise Exception

        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std

        return features

    def mm_mode(self, mm_on=True):
    
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.data_list = self.test_dataset.data_list
            self.mm_list = np.random.choice(self.data_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.data_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.data_list = self.data_list




import torch.utils.data as data
import numpy as np
import random
from os.path import join as pjoin



class MergedRotationDataset(BASEDataModule):
    """Dataset that merges Text2MotionDatasetV2 and InterHuman datasets, outputs only rotations"""
    
    def __init__(
        self,
        mode='train',
        cfg=None,
        role_graph=False,
        max_motion_length=300,
        min_motion_length=40,
        max_text_len=20,
        unit_length=4,
        tiny=False,
        debug=False,
        progress_bar=True,
        dataset_mode='mixed',
        agd_enabled=False,  # AGD mode switch
        logger=None,
        verbose=False,
    ):
        super().__init__(None, None, None)  # Parent-class args are set in setup
        self.logger = logger or LOGGER
        self.verbose = bool(verbose or debug)
        self.cfg = cfg
        
        # AGD (Alternating Gradient Descent) configuration
        self.agd_enabled = agd_enabled
        self.agd_step_counter = 0  # AGD step counter

        # Initialize InterHuman dataset
        if mode == 'test' or mode == 'val':
            dataset_mode = 'mixed'  # Always use mixed mode during testing
        else:
            dataset_mode = 'ih' if not agd_enabled else 'mixed'  # AGD mode needs mixed to access both datasets
        if dataset_mode in ['mixed', 'ih']:
            ih_dataset_cfg = _cfg_path(self.cfg, "INTERHUMAN_DATASET_CFG", "configs/datasets.yaml")
            ih_data_root = _cfg_path(self.cfg, "INTERHUMAN_DATA_ROOT", None)
            self.ih_dataset = InterHumanDataset(
                mode=mode,
                role_graph=role_graph,
                dataset_cfg_path=ih_dataset_cfg,
                data_root=ih_data_root,
            )
        if dataset_mode in ['mixed', 't2m']:
            t2m_path_kwargs = {}
            path_key_map = {
                "HML3D_DATA_ROOT": "data_root",
                "HML3D_GLOVE_ROOT": "glove_root",
                "HML3D_MEAN_ORG_PATH": "mean_org_path",
                "HML3D_STD_ORG_PATH": "std_org_path",
                "HML3D_TEXT_DIR": "text_dir",
                "HML3D_MOTION_DIR": "motion_dir",
            }
            for cfg_key, kw_key in path_key_map.items():
                path_value = _cfg_path(self.cfg, cfg_key, None)
                if path_value:
                    t2m_path_kwargs[kw_key] = path_value

            # Fallbacks from merged runtime cfg for better single-file control.
            if "data_root" not in t2m_path_kwargs and self.cfg is not None:
                hm_cfg = getattr(getattr(self.cfg, "DATASET", None), "HUMANML3D", None)
                hm_root = getattr(hm_cfg, "ROOT", None) if hm_cfg is not None else None
                if hm_root:
                    t2m_path_kwargs["data_root"] = hm_root

            if "glove_root" not in t2m_path_kwargs and self.cfg is not None:
                word_root = getattr(getattr(self.cfg, "DATASET", None), "WORD_VERTILIZER_PATH", None)
                if word_root:
                    t2m_path_kwargs["glove_root"] = word_root

            self.t2m_dataset = Text2MotionDatasetV2(
                mode=mode,
                max_motion_length=max_motion_length,
                min_motion_length=min_motion_length,
                max_text_len=max_text_len,
                unit_length=unit_length,
                tiny=tiny,
                debug=debug,
                progress_bar=progress_bar,
                **t2m_path_kwargs,
            )

        self.dataset_mode = dataset_mode
        # Store parameters
        self.mode = mode
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        #self.nfeats = 21*6  # rotation features only
        #self.nfeats = 21*6 + 21*3 + 22*3 + 4 # 259
        self.nfeats = 21*6 + 21*3 + 22*3 + 4 + 3  # 262
        self.njoints = 22
        
        # Calculate total length and dataset proportions
        self.t2m_len = len(self.t2m_dataset) if hasattr(self, 't2m_dataset') else 0
        self.ih_len = len(self.ih_dataset) if hasattr(self, 'ih_dataset') else 0
        self.total_len = self.t2m_len + self.ih_len

        self.current_phase = 'train'

        # Set total length based on dataset_mode
    
        self._sync_dataset_lengths(dataset_mode, prefix="Loaded")
        


        self.temp_batch = [np.ones([22,300]), np.ones([22,150]), 16]

    def _log_info(self, message):
        _emit_log(self.logger, message)

    def _log_verbose(self, message):
        if self.verbose:
            self._log_info(message)

    def _sync_dataset_lengths(self, dataset_mode, prefix="Using"):
        if dataset_mode == 'mixed':
            self.total_len = self.t2m_len + self.ih_len
            self.t2m_prop = self.t2m_len / self.total_len
            self.ih_prop = self.ih_len / self.total_len
            self._log_info(f"{prefix} combined dataset with:")
            self._log_info(f"- {self.t2m_len} single-person samples ({self.t2m_prop:.2%})")
            self._log_info(f"- {self.ih_len} two-person samples ({self.ih_prop:.2%})")
            return
        if dataset_mode == 't2m':
            self.total_len = self.t2m_len
            self._log_info(f"{prefix} Text2Motion dataset with {self.t2m_len} samples")
            return
        if dataset_mode == 'ih':
            self.total_len = self.ih_len
            self._log_info(f"{prefix} InterHuman dataset with {self.ih_len} samples")
            return
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}, choose from 'mixed', 't2m', or 'ih'")

    def extract_t2m_rotation(self, motion):
        """Extract rotation from T2M motion"""
        start_point = 4
        rotation_start = start_point + 21*3 # 263-(21*6) starts at 137
        rotation = motion[..., rotation_start:rotation_start + 21*6]
        rot_hm = motion[..., :start_point]
        joint = motion[..., start_point:start_point+21*3]
        vel = motion[..., start_point+21*3+21*6:start_point+21*3+21*6+22*3]
        fc = motion[..., -7:-3]
        rot_pos = motion[..., -3:]
        
        out = np.concatenate([rot_pos, joint, vel, rotation, fc], axis=-1) # out[4:] -> [len, 262]

        #return rotation
        assert out.shape[-1] == 262
        return out, rot_hm

    def extract_ih_rotation(self, motion_all):
        """Extract rotation from IH motion"""
        motion1, motion2 = motion_all[..., :262], motion_all[..., 262:]
        if np.random.rand() > 0.5:
            motion = motion1
            flag = 1  
            rest = motion2
        else: 
            motion = motion2
            flag = 2
            rest = motion1

        return motion, flag, rest

    def __len__(self):
        return self.total_len

    # ==================== AGD (Alternating Gradient Descent) Methods ====================
    
    def get_agd_pure_batch_t2m(self, batch_size):
        """Get a pure HumanML3D batch for AGD training."""
        if not hasattr(self, 't2m_dataset'):
            raise ValueError("t2m_dataset not initialized. Set dataset_mode='mixed' or 't2m'")
        
        indices = np.random.choice(self.t2m_len, min(batch_size, self.t2m_len), replace=False)
        batch = []
        for idx in indices:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, _ = self.t2m_dataset[idx]
            self.temp_batch = [word_embeddings, pos_one_hots, sent_len]
            ih_motion, rot_hm = self.extract_t2m_rotation(motion.copy())
            rest = np.zeros([motion.shape[0], 262])
            motion_full = np.concatenate([motion[..., :-3], np.zeros([motion.shape[0], 524-263])], axis=-1)
            batch.append((ih_motion, caption, word_embeddings, pos_one_hots, sent_len, m_length, 't2m', motion_full, rest))
        return batch
    
    def get_agd_pure_batch_ih(self, batch_size):
        """Get a pure InterHuman batch for AGD training."""
        indices = np.random.choice(self.ih_len, min(batch_size, self.ih_len), replace=False)
        batch = []
        for idx in indices:
            _, _, text, _, motion, m_length, _, _ = self.ih_dataset[idx]
            rotation, flag, rest = self.extract_ih_rotation(motion.copy())
            batch.append((rotation, text, self.temp_batch[0], self.temp_batch[1], self.temp_batch[2], m_length, 'ih_'+str(flag), motion, rest))
        return batch
    
    def agd_step(self):
        """Increase AGD step count and return the dataset type for this step."""
        self.agd_step_counter += 1
        # Strict alternation: odd steps use t2m, even steps use ih
        return 't2m' if self.agd_step_counter % 2 == 1 else 'ih'
    
    def get_current_agd_dataset(self):
        """Get dataset type for the current AGD step (without incrementing counter)."""
        return 't2m' if self.agd_step_counter % 2 == 0 else 'ih'
    
    def reset_agd_counter(self):
        """Reset AGD step counter (typically called at each epoch start)."""
        self.agd_step_counter = 0

    def train_dataloader(self):

        return data.DataLoader(
            self,
            batch_size=self.cfg.EVAL.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.EVAL.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True
        )


    def val_dataloader(self):

        return data.DataLoader(
            self,
            batch_size=self.cfg.EVAL.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.EVAL.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):

        return data.DataLoader(
            self,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.TEST.NUM_WORKERS,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
  

    def set_phase(self, phase):
        """Set the current phase (train, val, test)"""
        if phase not in ['train', 'val', 'test']:
            raise ValueError(f"Unknown phase: {phase}")
        self.current_phase = phase
        self._log_verbose(f"Set current phase to: {phase}")

    def __getitem__(self, idx):

        # Decide how to fetch samples based on dataset_mode


        if self.dataset_mode == 'mixed':
            # Mixed mode - original logic
            if idx < self.t2m_len:
                # Get item from Text2MotionV2 dataset
                word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, _ = self.t2m_dataset[idx]
                self.temp_batch = [word_embeddings, pos_one_hots, sent_len]
                ih_motion, rot_hm = self.extract_t2m_rotation(motion.copy())
                motion = np.concatenate([motion[..., :-3], np.zeros([motion.shape[0], 524-263])], axis=-1)
                rest = np.zeros([motion.shape[0], 262])
                assert motion.shape[-1] == 524
                return ih_motion, caption, word_embeddings, pos_one_hots, sent_len, m_length, 't2m', motion, rest
                
            else:
                # Get item from InterHuman dataset
                ih_idx = idx - self.t2m_len
                _, _, text, _, motion, m_length, _, _ = self.ih_dataset[ih_idx]
                rotation, flag, rest = self.extract_ih_rotation(motion.copy())
                return rotation, text, self.temp_batch[0], self.temp_batch[1], self.temp_batch[2], m_length, 'ih_'+str(flag), motion, rest
        
        elif self.dataset_mode == 't2m':
            raise Exception("Text2Motion-only mode is not supported in __getitem__.")
            # Text2Motion-only mode
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, _ = self.t2m_dataset[idx]
            self.temp_batch = [word_embeddings, pos_one_hots, sent_len]
            ih_motion, rot_hm = self.extract_t2m_rotation(motion.copy())
            rest = np.zeros([motion.shape[0], 262])
            motion = np.concatenate([motion[..., :-3], np.zeros([motion.shape[0], 524-263])], axis=-1)
            assert motion.shape[-1] == 524
            return ih_motion, caption, word_embeddings, pos_one_hots, sent_len, m_length, 't2m', motion, rest
            
        elif self.dataset_mode == 'ih':
            # InterHuman-only mode
            _, _, text, _, motion, m_length, _, _ = self.ih_dataset[idx]
            rotation, flag, rest = self.extract_ih_rotation(motion.copy())
            return rotation, text, self.temp_batch[0], self.temp_batch[1], self.temp_batch[2], m_length, 'ih_'+str(flag), motion, rest


    def set_dataset_mode(self, dataset_mode):
        """Set the dataset mode and update total length accordingly"""
        self.dataset_mode = dataset_mode
        self._sync_dataset_lengths(dataset_mode, prefix="Using")

    def get_random_batch(self, batch_size):
        """Get a random batch with specified proportions of each dataset"""
        if self.dataset_mode == 'mixed':
            # Mixed mode - original logic
            # Calculate number of samples from each dataset
            t2m_samples = int(batch_size * self.t2m_prop)
            ih_samples = batch_size - t2m_samples
            
            # Get random indices for each dataset
            t2m_indices = np.random.choice(self.t2m_len, t2m_samples, replace=False)
            ih_indices = np.random.choice(self.ih_len, ih_samples, replace=False)
            
            # Get samples
            batch = []
            for idx in t2m_indices:
                batch.append(self.__getitem__(idx))
            for idx in ih_indices:
                if self.dataset_mode == 'mixed':
                    batch.append(self.__getitem__(idx + self.t2m_len))
                else:
                    batch.append(self.__getitem__(idx))
        
        elif self.dataset_mode == 't2m':
            # Text2Motion-only mode
            indices = np.random.choice(self.t2m_len, batch_size, replace=False)
            batch = [self.__getitem__(idx) for idx in indices]
            
        elif self.dataset_mode == 'ih':
            # InterHuman-only mode
            indices = np.random.choice(self.ih_len, batch_size, replace=False)
            batch = [self.__getitem__(idx) for idx in indices]
            
        return self.collate_fn(batch)

    def sample_from_dataset(self, dataset_type, idx=None):
        """Sample from a specific dataset"""
        if dataset_type == 't2m':
            if idx is None:
                idx = random.randint(0, self.t2m_len - 1)
            if self.dataset_mode == 'mixed':
                return self.__getitem__(idx)
            elif self.dataset_mode == 't2m':
                return self.__getitem__(idx)
            else:
                # When requesting a 't2m' sample in 'ih' mode
                temp_mode = self.dataset_mode
                self.dataset_mode = 't2m'  # Temporarily switch mode
                sample = self.__getitem__(idx)
                self.dataset_mode = temp_mode  # Restore original mode
                return sample
                
        elif dataset_type == 'ih':
            if idx is None:
                idx = random.randint(0, self.ih_len - 1)
            if self.dataset_mode == 'mixed':
                return self.__getitem__(idx + self.t2m_len)
            elif self.dataset_mode == 'ih':
                return self.__getitem__(idx)
            else:
                # When requesting an 'ih' sample in 't2m' mode
                temp_mode = self.dataset_mode
                self.dataset_mode = 'ih'  # Temporarily switch mode
                sample = self.__getitem__(idx)
                self.dataset_mode = temp_mode  # Restore original mode
                return sample
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")



class AGDAlternatingDataLoader:
    """AGD alternating dataloader that strictly alternates pure HM3D and IH batches.
    
    Difference from traditional Batch Mixing:
    - Batch Mixing: each batch mixes samples from different datasets, and the model
      may exploit data-format differences as shortcuts.
    - AGD: each batch contains samples from only one dataset, forcing semantic learning.
    
    Alternation policy: odd steps -> HumanML3D (single-person),
    even steps -> InterHuman (two-person interaction).
    """
    
    def __init__(self, dataset, batch_size, num_workers, collate_fn, shuffle=True, logger=None, verbose=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.logger = logger or LOGGER
        self.verbose = verbose
        
        # Create two independent index lists
        self.t2m_indices = list(range(dataset.t2m_len))
        self.ih_indices = list(range(dataset.ih_len))
        
        # Iterator state
        self.t2m_ptr = 0
        self.ih_ptr = 0
        self.step_count = 0
        
        # Total steps: one full traversal for each dataset
        self.t2m_steps = (dataset.t2m_len + batch_size - 1) // batch_size
        self.ih_steps = (dataset.ih_len + batch_size - 1) // batch_size
        self._len = 2 * max(self.t2m_steps, self.ih_steps)
        
        self._log_info("=" * 60)
        self._log_info("AGD (Alternating Gradient Descent) DataLoader Initialized")
        self._log_info("=" * 60)
        self._log_info(f"  HumanML3D (t2m) samples: {dataset.t2m_len}")
        self._log_info(f"  InterHuman (ih) samples: {dataset.ih_len}")
        self._log_info(f"  Batch size: {batch_size}")
        self._log_info(f"  T2M steps per epoch: {self.t2m_steps}")
        self._log_info(f"  IH steps per epoch: {self.ih_steps}")
        self._log_info(f"  Total steps per epoch: {self._len}")
        self._log_info("-" * 60)
        self._log_info("AGD vs Batch Mixing:")
        self._log_info("  - Batch Mixing: mixed batches; model may exploit format differences")
        self._log_info("  - AGD: alternating pure batches; forces semantic feature learning")
        self._log_info("  - Alternation: odd steps->T2M(single-person), even steps->IH(two-person)")
        self._log_info("=" * 60)

    def _log_info(self, message):
        _emit_log(self.logger, message)

    def _log_verbose(self, message):
        if self.verbose:
            self._log_info(message)
        
    def __iter__(self):
        # Reset state
        self.t2m_ptr = 0
        self.ih_ptr = 0
        self.step_count = 0
        
        # Shuffle indices
        if self.shuffle:
            random.shuffle(self.t2m_indices)
            random.shuffle(self.ih_indices)
        
        self._log_verbose("[AGD] New epoch started - indices shuffled")
        return self
    
    def __next__(self):
        if self.step_count >= self._len:
            raise StopIteration
        
        self.step_count += 1
        
        # Strict alternation: odd steps use t2m, even steps use ih
        if self.step_count % 2 == 1:
            # T2M batch
            batch = self._get_t2m_batch()
            dataset_name = "T2M (HumanML3D)"
        else:
            # IH batch
            batch = self._get_ih_batch()
            dataset_name = "IH (InterHuman)"
        
        # Log status every 100 steps
        if self.step_count % 100 == 1:
            self._log_verbose(f"[AGD] Step {self.step_count}/{self._len} - Pure {dataset_name} batch")
        
        return self.collate_fn(batch)
    
    def _get_t2m_batch(self):
        """Get one pure T2M batch."""
        batch = []
        for _ in range(self.batch_size):
            if self.t2m_ptr >= len(self.t2m_indices):
                # Reuse in a loop
                self.t2m_ptr = 0
                if self.shuffle:
                    random.shuffle(self.t2m_indices)
            
            idx = self.t2m_indices[self.t2m_ptr]
            self.t2m_ptr += 1
            
            # Fetch T2M sample
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, _ = self.dataset.t2m_dataset[idx]
            self.dataset.temp_batch = [word_embeddings, pos_one_hots, sent_len]
            ih_motion, rot_hm = self.dataset.extract_t2m_rotation(motion.copy())
            rest = np.zeros([motion.shape[0], 262])
            motion_full = np.concatenate([motion[..., :-3], np.zeros([motion.shape[0], 524-263])], axis=-1)
            batch.append((ih_motion, caption, word_embeddings, pos_one_hots, sent_len, m_length, 't2m', motion_full, rest))
        
        return batch
    
    def _get_ih_batch(self):
        """Get one pure IH batch."""
        batch = []
        for _ in range(self.batch_size):
            if self.ih_ptr >= len(self.ih_indices):
                # Reuse in a loop
                self.ih_ptr = 0
                if self.shuffle:
                    random.shuffle(self.ih_indices)
            
            idx = self.ih_indices[self.ih_ptr]
            self.ih_ptr += 1
            
            # Fetch IH sample
            _, _, text, _, motion, m_length, _, _ = self.dataset.ih_dataset[idx]
            rotation, flag, rest = self.dataset.extract_ih_rotation(motion.copy())
            batch.append((rotation, text, self.dataset.temp_batch[0], self.dataset.temp_batch[1],
                         self.dataset.temp_batch[2], m_length, 'ih_'+str(flag), motion, rest))
        
        return batch
    
    def __len__(self):
        return self._len


class MergedDataModule(BASEDataModule):
    def __init__(self,
                 cfg,
                 batch_size=64,
                 num_workers=3,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                        num_workers=num_workers,
                        collate_fn=collate_fn)
        
        # Save parameters
        self.save_hyperparameters(logger=False)
        self.name = "merged"  # Use "merged" as a distinct module name
        self.njoints = 22
        self._logger = kwargs.get("logger", LOGGER)
        self._verbose_logging = _is_verbose_logging(cfg)
        
        # AGD config - read from cfg or use default
        self.agd_enabled = getattr(cfg.TRAIN, 'AGD_ENABLED', False) if hasattr(cfg, 'TRAIN') else False
        
        # Set dataset class
        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            self.Dataset = MergedRotationDataset  # Use the merged rotation dataset
            
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        
        # Set feature dimension for rotation representation
        #self.nfeats = 21 * 6  # 只包含旋转数据
        #self.nfeats = 21*6 + 21*3 + 22*3 + 4 # 259
        self.nfeats = 21*6 + 21*3 + 22*3 + 4 + 3 # 262
    def feats2joints_single(self, features, type_motion): # [32,300,262]


        from data_loaders.interhuman.interhuman import MotionNormalizerTorch
        MN = MotionNormalizerTorch() # mean.shape [262]
        mask_type =  torch.tensor(['ih' in label for label in type_motion])
        features_mix = features.clone()
        features_mix[mask_type] = MN.backward(features[mask_type][...,:262].clone())
        joints1 = features_mix[..., :22*3].reshape(features.shape[0:2]+(22,3)).to(features.device) # [32,300,22,3]

        return joints1

        #return torch.concat([joints1, joints2], -2) # [32,300,44,3]
    
    def feats2joints(self, features): # [32,300,262]


        from data_loaders.interhuman.interhuman import MotionNormalizerTorch
        MN = MotionNormalizerTorch() # mean.shape [262]
        features_motion1 = MN.backward(features[...,:262].clone())
        #features_motion2 = MN.backward(features[...,262:].clone())
        joints1 = features_motion1[..., :22*3].reshape(features.shape[0:2]+(22,3)).to(features.device) # [32,300,22,3]
        #joints2 = features_motion2[..., :22*3].reshape(features.shape[0:2]+(22,3)).to(features.device) ## [32,300,22,3]

        #return torch.concat([joints1, joints2], -2) # [32,300,44,3]
        return joints1


    def joints2feats(self, features):
        """Convert joints to features"""
        features = process_file(features, self.njoints)[0]
        return features
        
    def mm_mode(self, mm_on=True):
        """Handle multimodal mode"""
        if mm_on:
            self.is_mm = True
            self.data_list = self.test_dataset.data_list
            self.mm_list = np.random.choice(self.data_list,
                                          self.cfg.TEST.MM_NUM_SAMPLES,
                                          replace=False)
            self.test_dataset.data_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.data_list = self.data_list
            
    def setup(self, stage=None):
        """Setup datasets for different stages"""
        self.stage = stage
        # Set datasets for different stages
        if stage in (None, "fit"):
            if not hasattr(self, "_train_dataset"):
                # AGD mode needs mixed to access both datasets
                dataset_mode = 'mixed' if self.agd_enabled else 'ih'
                self._train_dataset = self.Dataset(
                    mode='train',
                    cfg=self.cfg,
                    max_motion_length=300,
                    min_motion_length=15,
                    dataset_mode=dataset_mode,
                    agd_enabled=self.agd_enabled,
                    logger=self._logger,
                    verbose=self._verbose_logging,
                )
            if not hasattr(self, "_val_dataset"):
                self._val_dataset = self.Dataset(
                    mode='val',
                    cfg=self.cfg,
                    max_motion_length=300,
                    min_motion_length=15,
                    dataset_mode='mixed',  # Validation set always uses mixed
                    logger=self._logger,
                    verbose=self._verbose_logging,
                )
        if stage in (None, "test"):
            if not hasattr(self, "_test_dataset"):
                self._test_dataset = self.Dataset(
                    mode='test',
                    cfg=self.cfg,
                    max_motion_length=300,
                    min_motion_length=15,
                    dataset_mode='mixed',  # Test set always uses mixed
                    logger=self._logger,
                    verbose=self._verbose_logging,
                )

    def train_dataloader(self):
        """Return train dataloader; use alternating loader in AGD mode."""
        # Get collate_fn from dataloader_options
        collate_fn = self.dataloader_options.get("collate_fn", None)
        
        if self.agd_enabled:
            # AGD mode: use alternating dataloader
            _emit_log(self._logger, "AGD Mode Enabled: Using alternating dataloader for pure batches")
            return AGDAlternatingDataLoader(
                dataset=self._train_dataset,
                batch_size=self.cfg.TRAIN.BATCH_SIZE,
                num_workers=self.cfg.TRAIN.NUM_WORKERS,
                collate_fn=collate_fn,
                shuffle=True,
                logger=self._logger,
                verbose=self._verbose_logging,
            )
        else:
            # Normal mode: use standard DataLoader
            return data.DataLoader(
                self._train_dataset,
                batch_size=self.cfg.TRAIN.BATCH_SIZE,
                shuffle=True,
                num_workers=self.cfg.TRAIN.NUM_WORKERS,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True
            )

    def __getattr__(self, item):
        """Handle dynamic attribute access."""
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                self.__dict__[item_c] = self.Dataset(
                    mode=subset,
                    cfg=self.cfg,
                    max_motion_length=300,
                    min_motion_length=15,
                    logger=self._logger,
                    verbose=self._verbose_logging,
                )
            return getattr(self, item_c)
            
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")
