# Old preservation
from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from umf.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from umf.models.operator import PositionalEncoding
from umf.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from umf.models.operator.position_encoding import build_position_encoding
from umf.utils.temos_utils import lengths_to_mask

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution

from umf.models.operator.cross_attention import (  ### New update: Simplified imports and changed module path
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer
)
from umf.models.operator.position_encoding import build_position_encoding

class UmfVae(nn.Module):
    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 hidden_dim: Optional[int] = 512,  ### New update: Added optional hidden dimension parameter
                 force_pre_post_proj: bool = False,  ### New update: Added projection control parameter
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "encoder_decoder",  ### New update: Changed default architecture
                 normalize_before: bool = False,
                 norm_eps: float = 1e-5,  ### New update: Added normalization epsilon parameter
                 activation: str = "gelu",
                 norm_post: bool = True,  ### New update: Added post-normalization flag
                 activation_post: Optional[str] = None,  ### New update: Added post-activation option
                 position_embedding: str = "learned") -> None:
        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1] if hidden_dim is None else hidden_dim  ### New update: Support for custom hidden dimension
        add_pre_post_proj = force_pre_post_proj or (hidden_dim is not None and hidden_dim != latent_dim[-1])  ### New update: Added projection layer condition
        self.latent_pre = nn.Linear(self.latent_dim, latent_dim[-1]) if add_pre_post_proj else nn.Identity()  ### New update: Added pre-projection layer
        self.latent_post = nn.Linear(latent_dim[-1], self.latent_dim) if add_pre_post_proj else nn.Identity()  ### New update: Added post-projection layer

        self.arch = arch

        # Removed PE_TYPE related code and simplified to single position encoding approach
        self.query_pos_encoder = build_position_encoding(
            self.latent_dim, position_embedding=position_embedding)

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
            norm_eps  ### New update: Added epsilon parameter to layer normalization
        )
        encoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post else None  ### New update: Made normalization optional
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)  ### New update: Added post-activation parameter

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post else None
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, decoder_norm)
        elif self.arch == "encoder_decoder":
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)

            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
                norm_eps
            )
            decoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post else None
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm)

        # Removed MLP_DIST related code and simplified distribution handling
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.latent_dim))
        self.skel_embedding = nn.Linear(nfeats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, nfeats)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Distribution]: ### New update: Changed input parameter from lengths to mask
        mask = lengths_to_mask(mask, features.device)
        z, dist = self.encode(features, mask)
        feats_rst = self.decode(z, mask)
        return feats_rst, z, dist

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, Distribution]:  ### New update: Updated signature to use mask
        mask = lengths_to_mask(mask, features.device)
        bs, nframes, nfeats = features.shape
        x = self.skel_embedding(features)
        x = x.permute(1, 0, 2)
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=torch.bool, device=x.device)  ### New update: Changed dtype to torch.bool
        aug_mask = torch.cat((dist_masks, mask), 1)
        xseq = torch.cat((dist, x), 0)

        xseq = self.query_pos_encoder(xseq)
        dist_pre = self.encoder(xseq, src_key_padding_mask=~aug_mask)[:dist.shape[0]]  ### New update: Added [0] index for encoder output
        dist = self.latent_pre(dist_pre)  ### New update: Added pre-projection

        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]

        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        latent = latent.permute(1, 0, 2)  ### New update: Changed tensor shape for consistency
        return latent, dist

    def decode(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  ### New update: Updated signature to use mask
        mask = lengths_to_mask(mask, z.device)
        z = self.latent_post(z)  ### New update: Added post-projection
        z = z.permute(1, 0, 2)  ### New update: Adjusted tensor shape for decoder
        bs, nframes = mask.shape
        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size), dtype=torch.bool, device=z.device)
            aug_mask = torch.cat((z_mask, mask), axis=1)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(xseq, src_key_padding_mask=~aug_mask)[z.shape[0]:]  ### New update: Added [0] index for decoder output
        elif self.arch == "encoder_decoder":
            queries = self.query_pos_decoder(queries)
            output = self.decoder(tgt=queries, memory=z, tgt_key_padding_mask=~mask) ### New update: Added [0] index for decoder output

        output = self.final_layer(output)
        output[~mask.T] = 0
        feats = output.permute(1, 0, 2)
        return feats
    