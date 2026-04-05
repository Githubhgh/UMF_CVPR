# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional
from numpy import block

import torch
import torch.nn.functional as F
from torch import Tensor, nn






class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int,
                 norm: Optional[nn.Module] = None, return_intermediate: bool = False) -> None:
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                controlnet_residuals: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        x = src
        intermediate = []
        index = 0
        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask,
                       src_key_padding_mask=src_key_padding_mask, pos=pos)

            if controlnet_residuals is not None:
                x = x + controlnet_residuals[index]
                index += 1

            xs.append(x)

            if self.return_intermediate:
                intermediate.append(x)

        x = self.middle_block(x, src_mask=mask,
                              src_key_padding_mask=src_key_padding_mask, pos=pos)

        if controlnet_residuals is not None:
            x = x + controlnet_residuals[index]
            index += 1

        if self.return_intermediate:
            intermediate.append(x)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                       src_key_padding_mask=src_key_padding_mask, pos=pos)

            if controlnet_residuals is not None:
                x = x + controlnet_residuals[index]
                index += 1

            if self.return_intermediate:
                intermediate.append(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return x
    


class SkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int,
                 norm: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.d_model = decoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = tgt

        xs = []
        for module in self.input_blocks:
            x = module(x, memory, tgt_mask=tgt_mask,
                       memory_mask=memory_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask,
                       pos=pos, query_pos=query_pos)
            xs.append(x)

        x = self.middle_block(x, memory, tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              pos=pos, query_pos=query_pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, tgt_mask=tgt_mask,
                       memory_mask=memory_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask,
                       pos=pos, query_pos=query_pos)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, act=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.act = _get_activation_fn(act)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm:
            output = self.act(self.norm(output)) 

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, act = None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.act = _get_activation_fn(act)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm:
            # output = self.norm(output)
            output = self.act(self.norm(output))  ### New update: Add activation after norm
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)



    



class TransformerEncoderLayer(nn.Module):
    # def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
    #              activation="relu", normalize_before=False):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False, 
                 norm_eps: float = 1e-5) -> None:  ### New update: Add type hints and norm_eps parameter
        super().__init__()
        self.d_model = d_model
        self.activation_name = activation  ### New update: Add activation tracking for GEGLU
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear1 = nn.Linear(d_model, dim_feedforward if activation != 'geglu' else dim_feedforward * 2)  ### New update: Support GEGLU with double dimension
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)  ### New update: Add eps parameter for stability
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)  ### New update: Add eps parameter for stability
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.activation = _get_activation_fn(activation) if activation != 'geglu' else nn.GELU()  ### New update: Special handling for GEGLU

    # def with_pos_embed(self, tensor, pos: Optional[Tensor]):
    #     return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None
                     ):  ### New update: Remove pos parameter
        # q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]  ### New update: Simplified attention without position
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        if self.activation_name == 'geglu':  ### New update: Add GEGLU path
            src2, gate = self.linear1(src).chunk(2, dim=-1)
            src2 = src2 * self.activation(gate)
        else:
            src2 = self.activation(self.linear1(src))
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None
                    ):  ### New update: Remove pos parameter
        src2 = self.norm1(src)
        # q = k = self.with_pos_embed(src2, pos)
        # src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
        src2 = self.self_attn(src2, src2, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]  ### New update: Simplified attention without position
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        if self.activation_name == 'geglu':  ### New update: Add GEGLU path
            src2, gate = self.linear1(src2).chunk(2, dim=-1)
            src2 = src2 * self.activation(gate)
        else:
            src2 = self.activation(self.linear1(src2))
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ):  ### New update: Remove pos parameter
        if self.normalize_before:
            # return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
            return self.forward_pre(src, src_mask, src_key_padding_mask)  ### New update: Update call signature
        # return self.forward_post(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask)  ### New update: Update call signature


class TransformerDecoderLayer(nn.Module):
    # def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
    #             activation="relu", normalize_before=False):
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 activation: str = "relu", 
                 normalize_before: bool = False,
                 norm_eps: float = 1e-5) -> None:  ### New update: Add type hints and norm_eps parameter
        super().__init__()
        self.d_model = d_model
        self.activation_name = activation  ### New update: Add activation tracking for GEGLU
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear1 = nn.Linear(d_model, dim_feedforward if activation != 'geglu' else dim_feedforward * 2)  ### New update: Support GEGLU with double dimension
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)  ### New update: Add eps parameter for stability
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)  ### New update: Add eps parameter for stability
        self.norm3 = nn.LayerNorm(d_model, eps=norm_eps)  ### New update: Add eps parameter for stability
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.activation = _get_activation_fn(activation) if activation != 'geglu' else nn.GELU()  ### New update: Special handling for GEGLU

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None
                     ):  ### New update: Remove pos and query_pos parameters
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        tgt2 = self.self_attn(tgt, tgt, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]  ### New update: Simplified attention without position
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
        #                           key=self.with_pos_embed(memory, pos),
        #                           value=memory, attn_mask=memory_mask,
        tgt2 = self.multihead_attn(query=tgt, key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]  ### New update: Simplified cross-attention without position
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        if self.activation_name == 'geglu':  ### New update: Add GEGLU path
            tgt2, gate = self.linear1(tgt).chunk(2, dim=-1)
            tgt2 = tgt2 * self.activation(gate)
        else:
            tgt2 = self.activation(self.linear1(tgt))
        tgt2 = self.linear2(self.dropout(tgt2))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None
                    ):  ### New update: Remove pos and query_pos parameters
        tgt2 = self.norm1(tgt)
        # q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]  ### New update: Simplified attention without position
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
        #                           key=self.with_pos_embed(memory, pos),
        #                           value=memory, attn_mask=memory_mask,
        tgt2 = self.multihead_attn(query=tgt2, key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]  ### New update: Simplified cross-attention without position
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        if self.activation_name == 'geglu':  ### New update: Add GEGLU path
            tgt2, gate = self.linear1(tgt2).chunk(2, dim=-1)
            tgt2 = tgt2 * self.activation(gate)
        else:
            tgt2 = self.activation(self.linear1(tgt2))
        tgt2 = self.linear2(self.dropout(tgt2))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None
                ):  ### New update: Remove pos and query_pos parameters
        if self.normalize_before:
            # return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
            #                       tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                   tgt_key_padding_mask, memory_key_padding_mask)  ### New update: Update call signature
        # return self.forward_post(tgt, memory, tgt_mask, memory_mask,
        #                        tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask)  ### New update: Update call signature

def _get_clone(module):
    return copy.deepcopy(module)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )



ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU()
}

def _get_activation_fn(act_fn: Optional[str] = None) -> nn.Module:
    if act_fn is None:
        return nn.Identity()
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
    

# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")