import os
import numpy as np
import torch
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from umf.config import instantiate_from_config
from umf.models.architectures import (
    t2m_motionenc,
    t2m_textenc,
)
from umf.models.losses.umf import UMFLosses
from umf.models.modeltype.base import BaseModel
from umf.utils.stage_mode import (
    get_diffusion_mode,
    prune_reaction_keys_for_indi_checkpoint,
)
import torch.nn.functional as F
from umf.data.humanml.scripts.motion_process import *

import math
from umf.models.flow import FlowModel
from umf.models.flow_org import FlowModel_org

import clip
from torch import nn

_ATOL = 1e-6
_RTOL = 1e-3
from torchdiffeq import odeint


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

from umf.data.utils import lengths_to_mask
import logging
logger = logging.getLogger(__name__)

from einops import rearrange


class UMF(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.original_list = ['6370', '6367', '6251', '5545', '3362', '2731', '4074', '4095', '250']
        self.current_list = self.original_list.copy()
        self.last_epoch = -1


        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.diffusion_mode = get_diffusion_mode(cfg, default="indi")
        self.react_training = (
            str(self.stage).lower() == "diffusion"
            and self.diffusion_mode == "react"
        )
        self.condition = 'text'
        self.is_vae = cfg.model.vae
        self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule


        if cfg.TEST.DATASETS[0] == 'interhuman':
            from data_loaders.interhuman.interhuman import MotionNormalizerTorch, MotionNormalizer
             # mean.shape [262]
            self.normalizer_ih = MotionNormalizerTorch()

        try:
            self.vae_type = cfg.model.vae_type
        except:
            self.vae_type = cfg.model.motion_vae.target.split(
                ".")[-1].lower().replace("vae", "")


        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self._dtype = clip_model.dtype


        # T2M-specific text encoder
        clipTransEncoderLayer_t2m = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.clipTransEncoder_t2m = nn.TransformerEncoder(
            clipTransEncoderLayer_t2m,
            num_layers=2)
        self.clip_ln_t2m = nn.LayerNorm(768)
        
        # IH-specific text encoder
        clipTransEncoderLayer_ih = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.clipTransEncoder_ih = nn.TransformerEncoder(
            clipTransEncoderLayer_ih,
            num_layers=2)
        self.clip_ln_ih = nn.LayerNorm(768)


        # Initialize reaction text encoder (separate from indi text encoder)
        clipTransEncoderLayer_react = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.clipTransEncoder_react = nn.TransformerEncoder(
            clipTransEncoderLayer_react,
            num_layers=2)
        self.clip_ln_react = nn.LayerNorm(768)


        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        if self.vae_type != "no":
            self.vae = instantiate_from_config(cfg.model.motion_vae)

        self.reac_denoiser = instantiate_from_config(cfg.model.react_denoiser)
        self.indi_denoiser = instantiate_from_config(cfg.model.denoiser)

        # Dataset embedding layer for conditioning on dataset type
        # 0: HumanML3D (t2m/hm), 1: InterHuman (ih)
        self.dataset_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=768  # Match CLIP text feature dimension
        )
        # Initialize with small values for training stability
        nn.init.normal_(self.dataset_embedding.weight, std=0.02)

        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            if self.vae_type in ["umf", "vposert","actor"]:
                self.vae.training = False
                for p in self.vae.parameters():
                    p.requires_grad = False
                if 'relative_cond' in os.environ.get('UMF'):
                    self.reac_denoiser.training = False
                    for p in self.reac_denoiser.parameters():
                        p.requires_grad = False

            else:
                self.motion_encoder.training = False
                for p in self.motion_encoder.parameters():
                    p.requires_grad = False
                self.motion_decoder.training = False
                for p in self.motion_decoder.parameters():
                    p.requires_grad = False

        
        if not self.predict_epsilon:
            cfg.model.scheduler.params['prediction_type'] = 'sample'
            cfg.model.noise_scheduler.params['prediction_type'] = 'sample'

        
        if self.condition in ["text", "text_uncond"]:
            self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "umf":
            self._losses = MetricCollection({
                split: UMFLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports umf losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        if self.condition in ['text', 'text_uncond']:
            self.feats2joints = datamodule.feats2joints


        if self.stage == "diffusion":
            self.num_stages = self.cfg.TRAIN.PSTAGES
            self.flow_scheduler = FlowModel(self.indi_denoiser, schedule = "linear", num_stages=self.num_stages, gamma = float(self.cfg.TRAIN.PYRAMID_GAMMA))
            self.flow_scheduler_org = FlowModel_org(self.reac_denoiser, schedule = "linear")
            
            motion_context_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim[-1],  # Action latent dimension (e.g., 256)
                nhead=4,                     # Number of attention heads; tune if needed
                dim_feedforward=1024,        # FFN dimension; tune if needed
                dropout=0.1,
                activation="gelu",
                batch_first=True             # Ensure input format is [batch, seq, feature]
            )
            self.motion_context_encoder = nn.TransformerEncoder(
                motion_context_encoder_layer,
                num_layers=2  # Number of context-encoder layers; tune if needed
            )




    def text_encoder(self, text, dataset_type='t2m'):
        """
        Text encoder with dataset-specific routing.
        
        Args:
            text: List of text strings
            dataset_type: 't2m' for HumanML3D, 'ih' for InterHuman
        
        Returns:
            cond_emb: [batch_size, 1, 768]
        """
        device = next(self.clip_transformer.parameters()).device
        raw_text = text

        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self._dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self._dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self._dtype)

        if dataset_type == 't2m':
            out = self.clipTransEncoder_t2m(clip_out)
            out = self.clip_ln_t2m(out)
        else:  # 'ih'
            out = self.clipTransEncoder_ih(clip_out)
            out = self.clip_ln_ih(out)

        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return cond.unsqueeze(1)
    
    def text_encoder_react(self, text):
        device = next(self.clip_transformer.parameters()).device
        raw_text = text

        with torch.no_grad():

            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self._dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self._dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self._dtype)  # shared CLIP features

        # Use separate transformer encoder and layer norm for reaction
        out = self.clipTransEncoder_react(clip_out)
        out = self.clip_ln_react(out)

        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return cond.unsqueeze(1)


    def _get_t2m_evaluator(self, cfg):

        dataname = cfg.TEST.DATASETS[0]

        if cfg.TEST.DATASETS[0] == 'interhuman':

            from data_loaders.interhuman.datasets.evaluator import EvaluatorModelWrapper
            paths_cfg = getattr(cfg, "PATHS", None)
            ih_eval_model = getattr(
                paths_cfg,
                "INTERHUMAN_EVAL_MODEL",
                "datasets/InterHuman/eval_model.yaml",
            )
            eval_device = torch.device("cpu")
            if str(getattr(cfg, "ACCELERATOR", "")).lower() == "gpu" and torch.cuda.is_available():
                raw_devices = getattr(cfg, "DEVICE", [0])
                if isinstance(raw_devices, (list, tuple)):
                    if not raw_devices:
                        gpu_index = 0
                    else:
                        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                        local_rank = min(local_rank, len(raw_devices) - 1)
                        gpu_index = int(raw_devices[local_rank])
                else:
                    gpu_index = int(raw_devices)
                eval_device = torch.device(f"cuda:{gpu_index}")
            self.evalution_wrapper_ih = EvaluatorModelWrapper(
                ih_eval_model,
                device=eval_device,
            )


        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size= 263 - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        #dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m"
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname, "text_mot_match/model/finest.tar"),
            map_location="cpu",
            weights_only=False,
        )
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False
            # Keep this placement for backward-compatible behavior in current release.
        return None


    def sample_block_noise(self, bs, ch, height, width, eps=1e-6):
        gamma = self.flow_scheduler.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 - gamma) + torch.ones(4, 4) * gamma + eps * torch.eye(4))
        block_number = bs * ch * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)]) # [block number, 4]
        noise = rearrange(noise, '(b c h w) (p q) -> b c (h p) (w q)', b=bs, c=self.latent_dim[-1], h=2, w=2, p=2, q=2)
        return noise

    
    def fm_reverse(self, encoder_hidden_states, egomotion, hint=None, lengths=None, dataset_type='t2m'):
        bsz = encoder_hidden_states.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2

        height, width = 4, 4
        init_factor = 2 ** (self.num_stages - 1) # stage = 2, init_factor = 2^1 = 2; stage = 1, init_factor = 2^0 = 1
        height, width =  height // init_factor, width // init_factor

        dtype, device = encoder_hidden_states.dtype, encoder_hidden_states.device
        if egomotion is None:
            latent_model_input = torch.randn(
                (bsz, height*width, self.latent_dim[-1]), #(bsz, self.latent_dim[0], self.latent_dim[-1]),
                device=encoder_hidden_states.device,
                dtype=torch.float,
            )
            ode_now = self.ode_fn_indi
        else:
            latent_model_input = egomotion
            ode_now = self.ode_fn

        def wrapped_ode_fn(t, y):
            if egomotion is None:
                return ode_now(y, t, encoder_hidden_states, sample_kwargs.get("cfg_scale", self.guidance_scale), dataset_type=dataset_type)
            else:
                return ode_now(y, t, encoder_hidden_states, sample_kwargs.get("cfg_scale", self.guidance_scale))

        num_steps = [10 // self.num_stages] * self.num_stages   #[10] * self.num_stages
        for stage_idx in range(self.num_stages):
            self.test_stage_idx = stage_idx
            if stage_idx > 0 and egomotion is None: # only for indi denoiser
                latents = self.patch_latents(latents, dim_expand=True) # [bs, 16, 64] -> [bs, 64, 2, 2]

                height, width = height * 2, width * 2
                latents = F.interpolate(latents, (height, width), mode="nearest")
                noise = self.sample_block_noise(*latents.shape)
                noise = noise.to(device=device, dtype=dtype)

                original_start_t = self.flow_scheduler.original_start_t[stage_idx]
                gamma = self.flow_scheduler.gamma
                alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
                beta = alpha * (1 - original_start_t) / math.sqrt(- gamma)

                latents = alpha * latents + beta * noise

                latents = self.patch_latents(latents, dim_expand=False) # [bs, 64, h, w] -> [bs, h*w, 64]
                

            else:
                latents = latent_model_input

            sample_kwargs = {'num_steps': num_steps[stage_idx], 'method': 'rk4', 'cfg_scale': self.guidance_scale, 'use_sde':False}

            t = torch.linspace(0, 1, sample_kwargs['num_steps'], dtype=dtype).to(device)

            results = odeint(
                wrapped_ode_fn,
                latents,
                t,
                method=sample_kwargs.get("method", "euler"),
                atol=sample_kwargs.get("atol", _ATOL),
                rtol=sample_kwargs.get("rtol", _RTOL)
            )
            latents = results[-1]

        return latents
    
    

    def ode_fn(self, latent_model_input, t, encoder_hidden_states, cfg_scale):
        latent = (torch.cat([latent_model_input] * 2) if self.do_classifier_free_guidance else latent_model_input)
        t = t
        model_uc, model_c = self.reac_denoiser(
            sample=latent,
            timestep=t,
            timestep_cond=None,
            encoder_hidden_states=encoder_hidden_states)[0].chunk(2)
        model_output = model_uc + cfg_scale * (model_c - model_uc)


        return model_output
    
    def ode_fn_indi(self, latent_model_input, t, encoder_hidden_states, cfg_scale, dataset_type='t2m'):
        latent = (torch.cat([latent_model_input] * 2) if self.do_classifier_free_guidance else latent_model_input)
        t = self.rescale_timesteps(t, self.test_stage_idx)
        model_uc, model_c = self.indi_denoiser(
            sample=latent,
            timestep=t,
            timestep_cond=None,
            encoder_hidden_states=encoder_hidden_states,
            dataset_type=dataset_type)[0].chunk(2)
        model_output = model_uc + cfg_scale * (model_c - model_uc)

        return model_output


    def patch_latents(self, pixel_values, dim_expand = True):
        bs = pixel_values.shape[0]

        if dim_expand:
            assert len(pixel_values.shape) == 3
            assert pixel_values.shape[-1] == self.latent_dim[-1]

            pixel_values = pixel_values.permute(0, 2, 1)
            if pixel_values.shape[-1] == 16:
                h = 4
            elif pixel_values.shape[-1] == 4:
                h = 2
            else:
                raise ValueError("pixel_values.shape[-1] should be 4 or 16")
            pixel_values = pixel_values.reshape(bs, self.latent_dim[-1], h, h)
            return pixel_values
        else:
            assert len(pixel_values.shape) == 4
            assert pixel_values.shape[1] == self.latent_dim[-1]
            assert pixel_values.shape[2] == pixel_values.shape[3]
            assert pixel_values.shape[2] in [2, 4]

            pixel_values = pixel_values.reshape(bs, self.latent_dim[-1], -1)
            pixel_values = pixel_values.permute(0, 2, 1)  # [bs, 64, h, w] -> [bs, h*w, 64]
            return pixel_values



    def fm_add_noise_pyramid(self, pixel_values, noise_values, stage_idx=0):
        bs, dev, dtype = pixel_values.shape[0], pixel_values.device, pixel_values.dtype
        pixel_values = self.patch_latents(pixel_values, dim_expand=True)

        t_sample = torch.rand(bs, device=dev, dtype=dtype)

        corrected_stage_idx = self.num_stages - stage_idx - 1
        stage_select_indices = torch.randint(0, self.flow_scheduler.num_train_timesteps, (bs,))

        orig_height, orig_width = pixel_values.shape[2], pixel_values.shape[3]
        pixel_values_select = pixel_values
        end_height, end_width = orig_height // (2 ** stage_idx), orig_width // (2 ** stage_idx)   
        start_t, end_t = self.flow_scheduler.start_t[corrected_stage_idx], self.flow_scheduler.end_t[corrected_stage_idx]
        pixel_values_end = pixel_values_select
        pixel_values_start = pixel_values_select

        if stage_idx > 0:
            for downsample_idx in range(1, stage_idx + 1):
                pixel_values_end = F.interpolate(pixel_values_end, (orig_height // (2 ** downsample_idx), orig_width // (2 ** downsample_idx)), mode="bilinear")
        for downsample_idx in range(1, stage_idx + 2):
            pixel_values_start = F.interpolate(pixel_values_start, (orig_height // (2 ** downsample_idx), orig_width // (2 ** downsample_idx)), mode="bilinear")
        pixel_values_start = F.interpolate(pixel_values_start, (end_height, end_width), mode="nearest")

        noise = torch.randn_like(pixel_values_end)
        pixel_values_end = end_t * pixel_values_end + (1.0 - end_t) * noise # end is data
        pixel_values_start = start_t * pixel_values_start + (1.0 - start_t) * noise # start is noise

        pixel_values_start = self.patch_latents(pixel_values_start, dim_expand=False) # [bs, 64, h, w] -> [bs, h*w, 64]
        pixel_values_end = self.patch_latents(pixel_values_end, dim_expand=False)   # [bs, 64, h, w] -> [bs, h*w, 64]

        target = pixel_values_end - pixel_values_start
        ut = target

        t_select = self.flow_scheduler.t_window_per_stage[corrected_stage_idx][stage_select_indices].flatten()
        while len(t_select.shape) < pixel_values_start.ndim:
            t_select = t_select.unsqueeze(-1)
        t_select = t_select.to(dev)
        xt = t_select.float() * pixel_values_end + (1.0 - t_select.float()) * pixel_values_start

        t = self.rescale_timesteps(t_select, corrected_stage_idx)

        return xt, ut, t.reshape(-1)
    




    def fm_add_noise_from_guassian(self, x1, x0, t= None, return_noise = False):

        """
        input:
        x1: [bs, 1, latent_dim] latents (ending point -> data)
        x0: [bs, 1, latent_dim] noises (starting point -> noise)

        output:
        xt: [bs, 1, latent_dim] latents (intermediate point)
        ut: [bs, 1, latent_dim] noises (vector field)
        """

        if x0 is None:
            x0 = torch.randn_like(x1)

        bs, dev, dtype = x1.shape[0], x1.device, x1.dtype

        # Sample time t from uniform distribution U(0, 1)
        t = torch.rand(bs, device=dev, dtype=dtype) if t is None else t

        # sample xt and ut
        xt = self.flow_scheduler_org.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.flow_scheduler_org.compute_ut(x0=x0, x1=x1, t=t)

        if return_noise:
            return xt, ut, t, x0
        return xt, ut, t

    

    def fm_add_noise(self, x0, x1, t=None):
            """
            input:
            x1: [bs, 1, latent_dim] latents (ending point -> data)
            x0: [bs, 1, latent_dim] noises (starting point -> noise)

            output:
            xt: [bs, 1, latent_dim] latents (intermediate point)
            ut: [bs, 1, latent_dim] noises (vector field)
            """

            if x0 is None:
                x0 = torch.randn_like(x1)

            bs, dev, dtype = x1.shape[0], x1.device, x1.dtype

            # Sample time t from uniform distribution U(0, 1)
            t = torch.rand(bs, device=dev, dtype=dtype) if t is None else t

            # sample xt and ut
            xt = self.flow_scheduler_org.compute_xt(x0=x0, x1=x1, t=t)
            ut = self.flow_scheduler_org.compute_ut(x0=x0, x1=x1, t=t)

            return xt, ut, t

    
    def rescale_timesteps(self, t, stage_idx):
        T_start = self.flow_scheduler.Timesteps_per_stage[stage_idx][0].item()
        T_end = self.flow_scheduler.Timesteps_per_stage[stage_idx][-1].item()
        T_1000 = self.time_linear_to_Timesteps(t, 0, 1, T_start, T_end)
        t_1 = T_1000/1000
        return t_1

    def time_linear_to_Timesteps(self, t, t_start, t_end, T_start, T_end):
        # T = k * t + b
        k = (T_end - T_start) / (t_end - t_start)
        b = T_start - t_start * k
        return k * t + b

    def _diffusion_process(self, latents, encoder_hidden_states, hint=None, lengths=None, dataset_type='t2m'):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]

        controlnet_cond = None
        if hint is not None:
            hint_mask = hint.sum(-1) != 0
            controlnet_cond = self.traj_encoder(hint, mask=hint_mask)
            controlnet_cond = controlnet_cond.permute(1, 0, 2)

        # latents = latents.permute(1, 0, 2)

        timestep_cond = None

        stage_idx = np.random.choice(self.num_stages) if not self.react_training else 0
        noisy_latents, noise, timesteps = self.fm_add_noise_pyramid(latents, None, stage_idx)

        ### classifier guidance
        # noisy_latents = self.guide(noisy_latents, timesteps, hint, train=True)

        noise_pred = self.indi_denoiser(
                sample=noisy_latents,
                timestep=timesteps,
                timestep_cond=timestep_cond,
                encoder_hidden_states=encoder_hidden_states,
                dataset_type=dataset_type)[0]
        
        
        

        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0

        n_set = {
                "noise": noise,
                "noise_prior": noise_prior,
                "noise_pred": noise_pred,
                "noise_pred_prior": noise_pred_prior,
            }

        if not self.predict_epsilon:
            n_set["pred"] = noise_pred
            n_set["latent"] = latents
        return n_set

    

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = [i+1 for i in batch["length"]]
        if self.vae_type in ["umf", "vposert", "actor"]:
            motion_z, dist_m = self.vae.encode(feats_ref, lengths)
            feats_rst = self.vae.decode(motion_z, lengths)
        else:
            raise TypeError("vae_type must be mcross or actor")

        recons_z, dist_rm = self.vae.encode(feats_rst, lengths)
        

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m



        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])

        masks_ih = lengths_to_mask(torch.tensor(lengths).to(feats_ref.device))
        feats_ref = feats_ref * masks_ih.unsqueeze(-1)
        feats_rst = feats_rst * masks_ih.unsqueeze(-1)

        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "dist_m": dist_m,
            "dist_ref": dist_ref,
            "mask": masks_ih.int().unsqueeze(-1).expand(-1, -1, 2).unsqueeze(-1),
        }
        return rs_set



    def _extract_dataset_labels(self, type_list):
        """
        Convert a type list into dataset-label tensor.
        't2m' or 'hm' -> 0 (HumanML3D)
        'ih_1', 'ih_2', ... -> 1 (InterHuman)
        
        Args:
            type_list: List of type strings from batch["type"]
            
        Returns:
            torch.Tensor: Dataset labels [batch_size] with values 0 or 1
        """
        labels = []
        for t in type_list:
            if 'ih' in t:
                labels.append(1)  # InterHuman
            else:
                labels.append(0)  # HumanML3D (t2m/hm)
        return torch.tensor(labels, dtype=torch.long)

    def _get_motion_context(self, zs: list[torch.Tensor], type_context = 'mean') -> torch.Tensor:
        """
        Use a Transformer to encode multiple motion latent vectors and produce
        a global context vector.

        Args:
            zs (list[torch.Tensor]): A list of motion latent vectors z.
                                     Each z should have shape [batch_size, 1, latent_dim].

        Returns:
            torch.Tensor: Global scene context vector c with shape
                          [batch_size, 1, latent_dim].
        """
        if not zs:
            # Handle empty-list edge case
            return None

        # 1. Stack z list into a sequence tensor
        # Input: list of [B, 1, D] -> Output: [B, N, D] (N = number of people)
        z_sequence = torch.cat(zs, dim=1)

        # 2. Fuse context with Transformer Encoder
        # Input: [B, N, D] -> Output: [B, N, D]
        # Each output vector now encodes information from all others
        context_encoded_sequence = self.motion_context_encoder(z_sequence)

        # 3. Aggregate multi-person information into global context vector c
        # Input: [B, N, D] -> Output: [B, 1, D]
        #c = context_encoded_sequence.mean(dim=1, keepdim=True)
        context_list = context_encoded_sequence.chunk(len(zs), dim=1)
        if type_context == "last":
            context_final = context_list[-1]
        elif type_context == "first":
            context_final = context_list[0]
        elif type_context == "mean":
            context_final = torch.mean(torch.stack(context_list, dim=0), dim=0)
        else:
            raise TypeError("type_context must be last, first or mean")
        return context_final

    def _update_react_training_flag(self):
        self.react_training = (
            str(self.stage).lower() == "diffusion" and self.diffusion_mode == "react"
        )

    def _set_diffusion_trainable_modules(self):
        if not self.react_training:
            set_requires_grad(self.reac_denoiser, False)
            set_requires_grad(self.indi_denoiser, True)
            set_requires_grad(self.clipTransEncoder_react, False)
            set_requires_grad(self.clip_ln_react, False)
            return

        set_requires_grad(self.reac_denoiser, True)
        set_requires_grad(self.indi_denoiser, False)
        self.indi_denoiser.eval()
        set_requires_grad(self.clipTransEncoder_ih, False)
        set_requires_grad(self.clip_ln_ih, False)
        set_requires_grad(self.clipTransEncoder_t2m, False)
        set_requires_grad(self.clip_ln_t2m, False)
        set_requires_grad(self.clipTransEncoder_react, True)
        set_requires_grad(self.clip_ln_react, True)

    def _get_diffusion_hint(self, batch):
        if 'relative_cond' in os.environ.get('UMF'):
            return batch['rela'].clone().detach()
        return None

    def _encode_primary_motion_latent(self, feats_ref, lengths):
        with torch.no_grad():
            if self.vae_type in ["umf", "vposert", "actor"]:
                z_indi, _ = self.vae.encode(feats_ref[..., :262], lengths)
                return z_indi
            if self.vae_type == "no":
                return feats_ref.permute(1, 0, 2)
            raise TypeError("vae_type must be mcross or actor")

    def _resolve_dataset_type(self, batch_types):
        t2m_count = sum(1 for t in batch_types if 't2m' in t)
        ih_count = sum(1 for t in batch_types if 'ih' in t)
        return 'ih' if ih_count > 0 and t2m_count == 0 else 't2m'

    def _apply_cfg_tokens(self, texts):
        if not self.do_classifier_free_guidance:
            return texts
        uncond_tokens = [""] * len(texts)
        if self.condition == 'text':
            uncond_tokens.extend(texts)
        elif self.condition == 'text_uncond':
            uncond_tokens.extend(uncond_tokens)
        return uncond_tokens

    def _append_dataset_embedding(self, text_emb, dataset_labels):
        dataset_emb = self.dataset_embedding(dataset_labels).unsqueeze(1)
        if self.do_classifier_free_guidance and dataset_emb.shape[0] == text_emb.shape[0] // 2:
            dataset_emb = dataset_emb.repeat(2, 1, 1)
        return torch.cat([text_emb, dataset_emb], dim=1)

    def _apply_length_mask(self, features, lengths):
        masks = lengths_to_mask(torch.tensor(lengths).to(features.device))
        return features * masks.unsqueeze(-1)

    def _prepare_condition_embeddings(self, batch):
        if self.condition in ["text", "text_uncond"]:
            text = batch["text"]
            text = ["" if np.random.rand(1) < self.guidance_uncodp else i for i in text]
            dataset_type = self._resolve_dataset_type(batch["type"])
            cond_emb_indi = self.text_encoder(text, dataset_type=dataset_type)
            cond_emb_react = self.text_encoder_react(text)

            dataset_labels = self._extract_dataset_labels(batch["type"]).to(cond_emb_indi.device)
            cond_emb_indi = self._append_dataset_embedding(cond_emb_indi, dataset_labels)
            return cond_emb_indi, cond_emb_react, dataset_type

        if self.condition in ['action']:
            action = batch['action']
            return action, action, 't2m'

        raise TypeError(f"condition type {self.condition} not supported")

    def _compute_reaction_branch(
        self,
        feats_ref,
        lengths,
        dataset_type,
        noise_indi,
        noise_pred_indi,
        sampling_noise,
        cond_emb_react,
    ):
        zero_noise = torch.zeros_like(noise_indi)
        zero_pred = torch.zeros_like(noise_pred_indi)

        react_loss_active = self.react_training and dataset_type == 'ih'
        if not react_loss_active:
            return zero_noise, zero_pred, react_loss_active

        z_indi_second, _ = self.vae.encode(feats_ref[..., 262:], lengths)
        z_indi_context = (
            noise_indi + sampling_noise
            if np.random.rand(1) > 0.5
            else noise_pred_indi + sampling_noise
        )

        z_indi_input = self._get_motion_context([z_indi_context])
        noisy_latents_pad, noise_pad, timesteps_pad = self.fm_add_noise_from_guassian(
            z_indi_input, None
        )
        noisy_latents_inter, noise_inter, _ = self.fm_add_noise(
            z_indi_input, z_indi_second, t=timesteps_pad
        )

        noisy_latents = torch.cat([noisy_latents_pad, noisy_latents_inter], dim=-1)
        noise = torch.cat([noise_pad, noise_inter], dim=-1)
        noise_pred = self.reac_denoiser(
            sample=noisy_latents,
            timestep=timesteps_pad,
            timestep_cond=None,
            encoder_hidden_states=cond_emb_react,
        )[0]
        return noise, noise_pred, react_loss_active


    def train_diffusion_forward(self, batch):
        self._update_react_training_flag()
        feats_ref = batch["native_motion"].clone().detach() if self.react_training else batch["motion"]
        lengths = batch["length"]
        self._set_diffusion_trainable_modules()
        hint = self._get_diffusion_hint(batch)
        z_indi = self._encode_primary_motion_latent(feats_ref, lengths)
        cond_emb_indi, cond_emb_react, dataset_type = self._prepare_condition_embeddings(batch)
        n_set = self._diffusion_process(
            z_indi, cond_emb_indi, hint, lengths, dataset_type=dataset_type
        )

        noise_indi = n_set["noise"]
        noise_pred_indi = n_set["noise_pred"]
        sampling_noise = noise_indi - noise_pred_indi
        noise, noise_pred, react_loss_active = self._compute_reaction_branch(
            feats_ref,
            lengths,
            dataset_type,
            noise_indi,
            noise_pred_indi,
            sampling_noise,
            cond_emb_react,
        )

        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0

        n_set = {
            "noise_prior": noise_prior,
            "noise_pred_prior": noise_pred_prior,
            "noise_indi": noise_indi,
            "noise_pred_indi": noise_pred_indi,
            "noise": noise,
            "noise_pred": noise_pred,
            "react_training": react_loss_active,
        }

        return n_set

    def on_save_checkpoint(self, checkpoint):
        if str(self.stage).lower() != "diffusion" or self.diffusion_mode != "indi":
            return

        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return

        pruned_state_dict, removed_keys = prune_reaction_keys_for_indi_checkpoint(
            state_dict
        )
        checkpoint["state_dict"] = pruned_state_dict
        if removed_keys:
            logger.info(
                "Pruned %d reaction keys from stage2 checkpoint.",
                len(removed_keys),
            )

    def t2m_eval(self, batch):
        # First split data by type
        t2m_indices = []
        ih_indices = []
        
        for idx, typ in enumerate(batch["type"]):
            if 't2m' in typ:
                t2m_indices.append(idx)
            else:  # 'ih' in typ
                ih_indices.append(idx)
        
        # Build t2m and ih batches separately
        def select_by_indices(tensor_or_list, indices):
            if isinstance(tensor_or_list, torch.Tensor):
                return tensor_or_list[indices]
            elif isinstance(tensor_or_list, list):
                return [tensor_or_list[i] for i in indices]
            return tensor_or_list
        
        # Build t2m batch
        t2m_batch = {
            "text": select_by_indices(batch["text"], t2m_indices),
            "motion": select_by_indices(batch["motion"], t2m_indices),
            "length": select_by_indices(batch["length"], t2m_indices),
            "word_embs": select_by_indices(batch["word_embs"], t2m_indices),
            "pos_ohot": select_by_indices(batch["pos_ohot"], t2m_indices),
            "text_len": select_by_indices(batch["text_len"], t2m_indices),
            "type": select_by_indices(batch["type"], t2m_indices),
            "native_motion": select_by_indices(batch["native_motion"], t2m_indices)
        }
        
        # Build ih batch
        ih_batch = {
            "text": select_by_indices(batch["text"], ih_indices),
            "motion": select_by_indices(batch["motion"], ih_indices),
            "length": select_by_indices(batch["length"], ih_indices),
            "word_embs": select_by_indices(batch["word_embs"], ih_indices),
            "pos_ohot": select_by_indices(batch["pos_ohot"], ih_indices),
            "text_len": select_by_indices(batch["text_len"], ih_indices),
            "type": select_by_indices(batch["type"], ih_indices),
            "native_motion": select_by_indices(batch["native_motion"], ih_indices)
        }
        
        # Start timing
        start = time.time()
        
        # Process each split and return results
        results = {}

        # Process T2M data
        if t2m_indices:
            results['t2m'] = self._process_t2m_batch(t2m_batch)

        # Process IH data
        if ih_indices:
            results['ih'] = self._process_ih_batch(ih_batch)
        
        # End timing
        end = time.time()
        self.times.append(end - start)
        
        return results

    def _process_t2m_batch(self, batch):
        """Process samples from the T2M dataset."""
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = [i for i in batch["length"]]
        word_embs = batch["word_embs"].detach().clone().to(torch.float32)
        pos_ohot = batch["pos_ohot"].detach().clone().to(torch.float32)
        text_lengths = batch["text_len"].detach().clone()
        native_motions = batch["native_motion"].detach().clone()[..., :263]
        

        texts = self._apply_cfg_tokens(texts)

        if self.stage in ['diffusion', 'vae_diffusion']:
            text_emb = self.text_encoder(texts, dataset_type='t2m')
            batch_size = len(texts) // 2 if self.do_classifier_free_guidance else len(texts)
            dataset_labels = torch.zeros(batch_size, dtype=torch.long, device=text_emb.device)
            text_emb = self._append_dataset_embedding(text_emb, dataset_labels)
            z = self.fm_reverse(text_emb, None, None, lengths, dataset_type='t2m')


        elif self.stage in ['vae']:
            if self.vae_type in ["umf", "vposert", "actor"]:
                z, dist_m = self.vae.encode(motions, lengths)
            if self.condition in ['text_uncond']:
                z = torch.randn_like(z)
                
        with torch.no_grad():
            if self.vae_type in ["umf", "vposert", "actor"]:
                feats_rst = self.vae.decode(z, lengths)
            elif self.vae_type == "no":
                feats_rst = z.permute(1, 0, 2)
             
        rot_info = native_motions.clone()[..., :4]


        hm_motion = self.hm_from_262(motions, rot_info)
        processed_motions = hm_motion
        hm_feats = self.hm_from_262(feats_rst, rot_info)
        processed_feats = hm_feats

        motions = self._apply_length_mask(processed_motions, lengths)
        feats_rst = self._apply_length_mask(processed_feats, lengths)
        
        motions = motions[:, :196]
        feats_rst = feats_rst[:, :196]


        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_rst": torch.zeros([motions.shape[0], 300, 44, 3]),
            "joints_ref": torch.zeros([motions.shape[0], 300, 44, 3]),
            "text": texts,
            "length": lengths
        }
        return rs_set
        
    def hm_from_262(self, raw_hm, rot_info, pad= True):
        assert raw_hm.shape[-1] == 262
        assert rot_info.shape[-1] == 4
        hm = raw_hm
        
        if pad:
            rot_info = self.hm_from_pos(raw_hm)

        rot_pos = hm[..., :3]
        joint = hm[..., 3:66] # 21*3 = 63
        vel = hm[..., 66:132] # 22*3 = 66
        rotation = hm[..., 132:258] # 21*6 = 126
        fc = hm[..., 258:]

        real_hm = torch.concatenate([rot_info, joint, rotation, vel, fc], axis=-1)

        real_hm = real_hm * torch.from_numpy(self.datamodule.val_dataset.t2m_dataset.std[:263]).to(raw_hm.device) + torch.from_numpy(self.datamodule.val_dataset.t2m_dataset.mean[:263]).to(raw_hm.device)

        real_hm= (real_hm-torch.from_numpy(self.datamodule.val_dataset.t2m_dataset.mean_org).to(hm.device))/torch.from_numpy(self.datamodule.val_dataset.t2m_dataset.std_org).to(hm.device)
        assert real_hm.shape[-1] == 263
        return real_hm


    def hm_from_pos(self, hm):
        assert hm.shape[-1] == 262
        batch_size, seq_len = hm.shape[:2]

        hm_rot = []
        for i in range(batch_size):
            sample = hm[i]
            pos_flat = sample[:, :66]

            # Keep only valid, non-padded frames for finite-difference features.
            finite_mask = torch.isfinite(pos_flat).all(dim=-1)
            non_zero_mask = pos_flat.abs().sum(dim=-1) > 1e-8
            valid_mask = finite_mask & non_zero_mask
            valid_idx = torch.where(valid_mask)[0]

            if len(valid_idx) < 2:
                fallback = torch.zeros((seq_len, 4), device=hm.device, dtype=hm.dtype)
                if len(valid_idx) > 0:
                    fallback[:, 3] = sample[valid_idx[-1], 1]
                hm_rot.append(fallback)
                continue

            valid_len = int(valid_idx[-1].item()) + 1
            root_pos = sample[:valid_len, :3]
            l_hip_positions = sample[:valid_len, 3:6]
            r_hip_positions = sample[:valid_len, 6:9]

            sample_rot = process_file_from_root_and_hips(root_pos, r_hip_positions, l_hip_positions, False)

            # Rebuild frame t=last from forward differences: keep last velocity and true last root_y.
            last_vel = sample_rot[-1, :3]
            last_root_y = root_pos[-1, 1]
            last_frame = torch.cat([last_vel, last_root_y.view(1)]).unsqueeze(0)
            sample_rot = torch.cat([sample_rot, last_frame], dim=0)

            if sample_rot.shape[0] < valid_len:
                pad_valid = sample_rot[-1:].repeat(valid_len - sample_rot.shape[0], 1)
                sample_rot = torch.cat([sample_rot, pad_valid], dim=0)

            full_rot = torch.zeros((seq_len, 4), device=hm.device, dtype=hm.dtype)
            full_rot[:valid_len] = sample_rot[:valid_len]
            if valid_len < seq_len:
                full_rot[valid_len:] = full_rot[valid_len - 1:valid_len].repeat(seq_len - valid_len, 1)
            hm_rot.append(full_rot)

        hm_rot = torch.stack(hm_rot, dim=0)
        return hm_rot

    def _process_ih_batch(self, batch):
        """Process samples from the InterHuman dataset."""
        self._update_react_training_flag()
        texts = batch["text"]
        motions = batch["motion"].detach().clone()
        lengths = [i+1 for i in batch["length"]]
        native_motions = batch["native_motion"].detach().clone()

        texts = self._apply_cfg_tokens(texts)

        if self.stage in ['diffusion', 'vae_diffusion']:
            text_emb = self.text_encoder(texts, dataset_type='ih')
            batch_size = len(texts) // 2 if self.do_classifier_free_guidance else len(texts)
            dataset_labels = torch.ones(batch_size, dtype=torch.long, device=text_emb.device)
            text_emb = self._append_dataset_embedding(text_emb, dataset_labels)
            if hasattr(self, 'react_training') and self.react_training:
                text_emb_react = self.text_encoder_react(texts)
            z1 = self.fm_reverse(text_emb, None, None, lengths, dataset_type='ih')


            if not self.react_training:
                z2 = self.vae.encode(native_motions[..., 262:], lengths)[0]
            else:
                z_indi1_input = self._get_motion_context([z1])  # [bs, 1, latent_dim]

                pad_input = torch.randn(
                    (z1.shape[0], self.latent_dim[0], self.latent_dim[-1]),
                    device=text_emb.device,
                    dtype=torch.float,
                    )
                z_indi_pad = torch.cat([pad_input, z_indi1_input], dim=-1)
                z_indi2 = self.fm_reverse(text_emb_react, z_indi_pad, None, lengths)
                z2 = z_indi2[..., self.latent_dim[-1]:]

        elif self.stage in ['vae']:
            if self.vae_type in ["umf", "vposert", "actor"]:
                z1, _ = self.vae.encode(native_motions[..., :262], lengths)
                z2, _ = self.vae.encode(native_motions[..., 262:], lengths)
            if self.condition in ['text_uncond']:
                z = torch.randn_like(z)
                
        with torch.no_grad():
            if self.vae_type in ["umf", "vposert", "actor"]:
                feats_rst1 = self.vae.decode(z1, lengths)
                feats_rst2 = self.vae.decode(z2, lengths)

        
        processed_feats = torch.concatenate([feats_rst1, feats_rst2], axis=-1)
        processed_motions = native_motions.clone().detach()

        processed_motions = self.normalizer_ih.backward(processed_motions.reshape(-1, 300, 2, 262).clone().detach()).reshape(-1, 300, 524)
        processed_feats = self.normalizer_ih.backward(processed_feats.reshape(-1, 300, 2, 262).clone().detach()).reshape(-1, 300, 524)

        processed_motions = self._apply_length_mask(processed_motions, lengths)
        processed_feats = self._apply_length_mask(processed_feats, lengths)

        text_emb, motion_emb = self.evalution_wrapper_ih.get_co_embeddings_524(
            processed_motions, batch["text"], batch["length"])
        recons_emb = self.evalution_wrapper_ih.get_motion_embeddings_524(
            processed_feats, batch["text"], batch["length"])
        
        return {
            "m_ref": processed_motions,
            "m_rst": processed_feats,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_rst": torch.zeros([motions.shape[0], 300, 44, 3]),
            "joints_ref": torch.zeros([motions.shape[0], 300, 44, 3]),
            "text": texts,
            "length": lengths
        }




    def allsplit_step(self, split: str, batch, batch_idx):
        # Extract batch dataset source (shared by AGD logging and loss scaling)
        dataset_source = None
        if hasattr(batch, '__contains__') and "type" in batch:
            batch_types = batch["type"]
            t2m_count = sum(1 for t in batch_types if 't2m' in t)
            ih_count = sum(1 for t in batch_types if 'ih' in t)
            if t2m_count > 0 and ih_count == 0:
                dataset_source = 't2m'
            elif ih_count > 0 and t2m_count == 0:
                dataset_source = 'ih'
            else:
                dataset_source = 'mixed'

            # AGD logging (training only)
            if split == "train":
                is_pure_batch = dataset_source in ('t2m', 'ih')
                if is_pure_batch:
                    self.log("ds", 0.0 if dataset_source == 't2m' else 1.0, prog_bar=True)
                    self.log("train/agd_dataset", 0.0 if dataset_source == 't2m' else 1.0, prog_bar=False)
                    self.log("train/agd_pure_batch", 1.0, prog_bar=False)
                else:
                    self.log("ds", t2m_count / (t2m_count + ih_count), prog_bar=True)
                    self.log("train/agd_pure_batch", 0.0, prog_bar=False)
        
        if split in ["train", "val"]:
            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]
            elif self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)
            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")

            # AGD loss scaling: use smaller effective updates for HM batches
            # to preserve single-person motion knowledge.
            if split == "train" and dataset_source == 't2m':
                hm_scale = getattr(self.cfg.TRAIN, 'AGD_HM_LOSS_SCALE', 1.0)
                loss = loss * hm_scale
                self.log("train/hm_loss_scale", hm_scale, prog_bar=False)

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:



            if self.condition in ['text', 'text_uncond']:
                # use t2m evaluators
                rs_set = self.t2m_eval(batch)
                rs_set_ = rs_set
                if 'ih' in rs_set_.keys():
                    rs_set = rs_set_['ih']
                else:
                    rs_set = rs_set_['t2m']
            elif self.condition == 'action':
                # use a2m evaluators
                rs_set = self.a2m_eval(batch)

            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
                #batch['text'] = list(batch['text']) * 30
            else:
                metrics_dicts = self.metrics_dict


            for metric in metrics_dicts:
                if metric == "TM2TMetrics":
                    if 'ih' in rs_set_.keys():
                        rs_set = rs_set_['ih']
                        getattr(self, metric).update(
                            # lat_t, latent encoded from diffusion-based text
                            # lat_rm, latent encoded from reconstructed motion
                            # lat_m, latent encoded from gt motion
                            # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                            rs_set["lat_t"],
                            rs_set["lat_rm"],
                            rs_set["lat_m"],
                            rs_set["length"],
                            rs_set['text'],
                            rs_set['m_rst'],
                            [batch, rs_set]
                        )
                    if 't2m' in rs_set_.keys():
                        rs_set = rs_set_['t2m']
                        getattr(self, "TM2TMetrics_HM").update(
                            # lat_t, latent encoded from diffusion-based text
                            # lat_rm, latent encoded from reconstructed motion
                            # lat_m, latent encoded from gt motion
                            # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                            rs_set["lat_t"],
                            rs_set["lat_rm"],
                            rs_set["lat_m"],
                            rs_set["length"]
                        )

                elif metric == "TM2TMetrics_HM": 
                    continue   
                elif metric == "MMMetrics":
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                 batch["length"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        return loss
    
