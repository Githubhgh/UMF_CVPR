import torch
import torch.nn as nn
from torch import  nn
from umf.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from umf.models.operator import PositionalEncoding
from umf.models.operator.cross_attention import (SkipTransformerEncoder,
                                                 TransformerDecoder,
                                                 TransformerDecoderLayer,
                                                 TransformerEncoder,
                                                 TransformerEncoderLayer)
from umf.models.operator.position_encoding import build_position_encoding
from umf.utils.temos_utils import lengths_to_mask
from typing import Optional, Union

class UmfDenoiser(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 263,
                 condition: str = "text",
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768,
                 nclasses: int = 10,
                 is_controlnet: bool = False,
                 hidden_dim: Optional[int] = None,
                 dh_depth: int = 0,
                 **kwargs) -> None:

        super().__init__()
        hidden_dim = 512
        self.latent_dim = latent_dim[-1] if hidden_dim is None else hidden_dim
        self.dh_depth = dh_depth
        add_pre_post_proj = True
        self.latent_pre = nn.Linear(latent_dim[-1], self.latent_dim) if add_pre_post_proj else nn.Identity()
        # When dh_depth > 0, each expert branch has its own latent_post; otherwise they share one.
        if self.dh_depth > 0:
            self.latent_post_t2m = nn.Linear(self.latent_dim, latent_dim[-1])
            self.latent_post_ih  = nn.Linear(self.latent_dim, latent_dim[-1])
        else:
            self.latent_post = nn.Linear(self.latent_dim, latent_dim[-1]) if add_pre_post_proj else nn.Identity()
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = ablation.SKIP_CONNECT
        self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.pe_type = ablation.DIFF_PE_TYPE

        if self.diffusion_only:
            # assert self.arch == "trans_enc", "only implement encoder for diffusion-only"
            self.pose_embd = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj = nn.Linear(self.latent_dim, nfeats)

        # emb proj
        if self.condition in ["text", "text_uncond"]:
            # text condition
            # project time from text_encoded_dim to latent_dim
            self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                    self.latent_dim)
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Sequential(
                    nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
        elif self.condition in ['action']:
            self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(self.latent_dim,
                                                    self.latent_dim)
            self.emb_proj = EmbedAction(nclasses,
                                        self.latent_dim,
                                        guidance_scale=guidance_scale,
                                        guidance_uncodp=guidance_uncondp)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        if self.pe_type == "actor":
            self.query_pos = PositionalEncoding(self.latent_dim, dropout)
            self.mem_pos = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "umf":
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")

        if self.arch == "trans_enc":
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
            # When dh_depth > 0, reduce backbone depth and allocate dh_depth layers to each expert branch.
            backbone_layers = num_layers - 2 * self.dh_depth if self.dh_depth > 0 else num_layers
            encoder_norm = None if is_controlnet else nn.LayerNorm(self.latent_dim)
            self.encoder = SkipTransformerEncoder(encoder_layer, backbone_layers, encoder_norm,
                                                    return_intermediate=is_controlnet)

            if self.dh_depth > 0:
                # Expert branches: standard TransformerEncoderLayer (no skip), each with dh_depth layers.
                def _make_expert(depth):
                    blocks = nn.ModuleList([
                        TransformerEncoderLayer(
                            self.latent_dim, num_heads, ff_size, dropout, activation, normalize_before)
                        for _ in range(depth)
                    ])
                    norm = nn.LayerNorm(self.latent_dim)
                    return blocks, norm
                self.expert_t2m_blocks, self.expert_t2m_norm = _make_expert(self.dh_depth)
                self.expert_ih_blocks,  self.expert_ih_norm  = _make_expert(self.dh_depth)
        else:
            raise ValueError(f"Not supported architechure{self.arch}!")
        
        self.is_controlnet = is_controlnet
        def zero_module(module):
            for p in module.parameters():
                nn.init.zeros_(p)
            return module

        
        if self.is_controlnet:
            self.controlnet_cond_embedding = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                zero_module(nn.Linear(self.latent_dim, self.latent_dim))
            )

            self.controlnet_down_mid_blocks = nn.ModuleList([
                zero_module(nn.Linear(self.latent_dim, self.latent_dim)) for _ in range(num_layers)])



    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                timestep_cond: Optional[torch.Tensor] = None,
                controlnet_cond: Optional[torch.Tensor] = None,
                controlnet_residuals: Optional[list[torch.Tensor]] = None,
                dataset_type: str = 't2m'):
        # 0.  dimension matching
        # sample [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        sample = sample.permute(1, 0, 2) # [bn, 1, 256]
        sample = self.latent_pre(sample)


        # 0. check lengths for no vae (diffusion only)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)

        # 1. check if controlnet
        if self.is_controlnet:
            controlnet_cond = controlnet_cond.permute(1, 0, 2)
            sample = sample + self.controlnet_cond_embedding(controlnet_cond)

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding

        # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
        encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
        text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
        # textembedding projection
        if self.text_encoded_dim != self.latent_dim:
            # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
            text_emb_latent = self.emb_proj(text_emb)
        else:
            text_emb_latent = text_emb

        emb_latent = torch.cat((time_emb, text_emb_latent), 0)



        # 4. transformer
        if self.arch == "trans_enc":
            #xseq = torch.cat((sample, emb_latent), axis=0)
            xseq = torch.cat((emb_latent, sample), axis=0)

            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq, controlnet_residuals=controlnet_residuals)

            if self.is_controlnet:
                control_res_samples = []
                for res, block in zip(tokens, self.controlnet_down_mid_blocks):
                    r = block(res)
                    control_res_samples.append(r)
                return control_res_samples



            #sample = tokens[:sample.shape[0]]
            sample = tokens[-sample.shape[0]:]
        else:
            raise NameError

        # 5. Route through expert branches (dh_depth > 0) or output directly.
        if self.dh_depth > 0:
            # Select the corresponding expert branch by dataset_type.
            if dataset_type == 'ih':
                expert_blocks = self.expert_ih_blocks
                expert_norm   = self.expert_ih_norm
                latent_post   = self.latent_post_ih
            else:  # 't2m' or others
                expert_blocks = self.expert_t2m_blocks
                expert_norm   = self.expert_t2m_norm
                latent_post   = self.latent_post_t2m

            for block in expert_blocks:
                sample = block(sample)
            sample = expert_norm(sample)

            sample = sample.permute(1, 0, 2)
            sample = latent_post(sample)
        else:
            # [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
            sample = sample.permute(1, 0, 2)
            sample = self.latent_post(sample)

        return (sample, )


class EmbedAction(nn.Module):

    def __init__(self,
                 num_actions,
                 latent_dim,
                 guidance_scale=7.5,
                 guidance_uncodp=0.1,
                 force_mask=False):
        super().__init__()
        self.nclasses = num_actions
        self.guidance_scale = guidance_scale
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))

        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        if not self.training and self.guidance_scale > 1.0:
            uncond, output = output.chunk(2)
            uncond_out = self.mask_cond(uncond, force=True)
            out = self.mask_cond(output)
            output = torch.cat((uncond_out, out))

        output = self.mask_cond(output)

        return output.unsqueeze(0)

    def mask_cond(self, output, force=False):
        bs, d = output.shape
        # classifer guidence
        if self.force_mask or force:
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=output.device) *
                self.guidance_uncodp).view(
                    bs, 1)  # 1-> use null_cond, 0-> use real cond
            return output * (1. - mask)
        else:
            return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
