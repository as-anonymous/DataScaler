import math
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from mup import MuReadout
from open_clip.pos_embed import get_2d_sincos_pos_embed
from open_clip.utils import feature_take_indices, to_2tuple
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(
            x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob: float = 0.5, exclude_first_token: bool = True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        scaled_cosine: bool = False,
        scale_heads: bool = False,
        logit_scale_max: float = math.log(1.0 / 0.01),
        batch_first: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max
        self.batch_first = batch_first
        self.use_fsdpa = hasattr(nn.functional, "scaled_dot_product_attention")

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.reshape(L, N * self.num_heads, -1).transpose(0, 1)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)

        x = x.transpose(0, 1).reshape(L, N, C)

        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 256,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True
        )
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,  # type: ignore
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
        batch_first: bool = True,
        SP: bool = False,
        attn_mult: float = 1.0,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)

        if SP:
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        else:
            self.attn = MultiheadAttention(
                d_model, n_head, standparam=SP, batch_first=batch_first, attn_mult=attn_mult
            )

        self.ls_1 = (
            LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        )
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        )

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,  # type: ignore
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        scale_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            batch_first=batch_first,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = (
            LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        )

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("ln", norm_layer(mlp_width) if scale_fc else nn.Identity()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        )

    def get_reference_weight(self):
        return self.mlp.c_fc.weight  # type: ignore

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomTransformer(nn.Module):
    """A custom transformer that can use different block types."""

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,  # type: ignore
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        batch_first: bool = True,
        block_types: Union[str, List[str]] = "CustomResidualAttentionBlock",
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first  # run transformer stack in batch first (N, L, D)
        self.grad_checkpointing = False

        if isinstance(block_types, str):
            block_types = [block_types] * layers
        assert len(block_types) == layers

        def _create_block(bt: str):
            if bt == "CustomResidualAttentionBlock":
                return CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                )
            else:
                assert False

        self.resblocks = nn.ModuleList([_create_block(bt) for bt in block_types])

    def get_cast_dtype(self) -> torch.dtype:
        weight = self.resblocks[0].get_reference_weight()  # type: ignore
        if hasattr(weight, "int8_original_dtype"):
            return weight.int8_original_dtype
        return weight.dtype

    def forward_intermediates(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        indices: Optional[Union[int, List[int]]] = None,
        stop_early: bool = False,
    ):
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()  # NLD -> LND

        intermediates = []
        if (
            torch.jit.is_scripting() or not stop_early  # type: ignore
        ):  # can't slice blocks in torchscript
            blocks = self.resblocks
        else:
            blocks = self.resblocks[: max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
                x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)  # type: ignore
            else:
                x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x.transpose(0, 1) if not self.batch_first else x)

        if not self.batch_first:
            x = x.transpose(0, 1)  # LND -> NLD

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """Prune layers not required for specified intermediates."""
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[: max_index + 1]  # truncate blocks
        return take_indices

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)  # type: ignore
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,  # type: ignore
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        batch_first: bool = True,
        SP: bool = False,
        attn_mult: float = 1.0,
        output_mult: float = 1.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                    SP=SP,
                    attn_mult=attn_mult,
                )
                for _ in range(layers)
            ]
        )

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, "int8_original_dtype"):  # type: ignore
            return self.resblocks[0].mlp.c_fc.int8_original_dtype  # type: ignore
        return self.resblocks[0].mlp.c_fc.weight.dtype  # type: ignore

    def forward_intermediates(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        indices: Optional[Union[int, List[int]]] = None,
        stop_early: bool = False,
    ):
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()  # NLD -> LND

        intermediates = []
        if (
            torch.jit.is_scripting() or not stop_early  # type: ignore
        ):  # can't slice blocks in torchscript
            blocks = self.resblocks
        else:
            blocks = self.resblocks[: max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
                x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)  # type: ignore
            else:
                x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x.transpose(0, 1) if not self.batch_first else x)

        if not self.batch_first:
            x = x.transpose(0, 1)  # LND -> NLD

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """Prune layers not required for specified intermediates."""
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[: max_index + 1]  # truncate blocks
        return take_indices

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()  # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)  # type: ignore
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)  # LND -> NLD
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]  # type: ignore

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,  # type: ignore
        attentional_pool: bool = False,
        attn_pooler_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.0,
        no_ln_pre: bool = False,
        pos_embed_type: str = "learnable",
        pool_type: str = "tok",
        final_ln_after_pool: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
        SP: bool = False,
        attn_mult: float = 1.0,
        output_mult: float = 1.0,
    ):
        super().__init__()
        assert pool_type in ("tok", "avg", "none")
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim
        self.SP = SP

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == "learnable":
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
            )
        elif pos_embed_type == "sin_cos_2d":
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1], (
                "currently sin cos 2d pos embedding only supports square input"
            )
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width),
                requires_grad=False,
            )
            pos_embed_type = get_2d_sincos_pos_embed(  # type: ignore
                width, self.grid_size[0], cls_token=True
            )
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            SP=SP,
            attn_mult=attn_mult,
            output_mult=output_mult,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = "none"
                if attentional_pool in ("parallel", "cascade"):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ""
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)

        if self.SP:
            self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))
            # self.proj = nn.Linear(pool_dim, output_dim, bias=False)
            # nn.init.normal_(self.proj.weight, std=scale)
        else:
            self.proj = MuReadout(
                pool_dim,
                output_dim,
                bias=False,
                readout_zero_init=True,
                output_mult=output_mult,
            )
            nn.init.normal_(self.proj.weight, std=scale)

        self.init_parameters()

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore  # type: ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore  # type: ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {"positional_embedding", "class_embedding"}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == "avg":
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == "tok":
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def _embeds(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # patch dropout (if active)
        x = self.patch_dropout(x)

        # apply norm before transformer
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == "parallel":
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == "cascade"
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        return pooled, tokens

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int]]] = None,
        stop_early: bool = False,
        normalize_intermediates: bool = False,
        intermediates_only: bool = False,
        output_fmt: str = "NCHW",
        output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            intermediates_only: Only return intermediate features
            normalize_intermediates: Apply final norm layer to all intermediates
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both extra prefix class tokens
        Returns:

        """
        assert output_fmt in ("NCHW", "NLC"), "Output format must be one of NCHW or NLC."
        reshape = output_fmt == "NCHW"

        # forward pass
        B, _, height, width = x.shape
        x = self._embeds(x)
        x, intermediates = self.transformer.forward_intermediates(
            x,
            indices=indices,
            stop_early=stop_early,
        )

        # process intermediates
        if normalize_intermediates:
            # apply final norm to all intermediates
            intermediates = [self.ln_post(xi) for xi in intermediates]
        num_prefix_tokens = 1  # one class token that's always there (as of now)
        if num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, num_prefix_tokens:] for y in intermediates]
        else:
            prefix_tokens = None
        if reshape:
            # reshape to BCHW output format
            H, W = height // self.patch_size[0], width // self.patch_size[1]  # type: ignore
            intermediates = [
                y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates
            ]

        output = {"image_intermediates": intermediates}
        if prefix_tokens is not None and output_extra_tokens:
            output["image_intermediates_prefix"] = prefix_tokens

        if intermediates_only:
            return output  # type: ignore

        pooled, _ = self._pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj  # type: ignore

        output["image_features"] = pooled  # type: ignore

        return output  # type: ignore

    def prune_intermediate_layers(
        self,
        indices: Union[int, List[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ):
        """Prune layers not required for specified intermediates."""
        take_indices = self.transformer.prune_intermediate_layers(indices)
        if prune_norm:
            self.ln_post = nn.Identity()
        if prune_head:
            self.proj = None
        return take_indices

    def forward(self, x: torch.Tensor):
        x = self._embeds(x)
        x = self.transformer(x)
        pooled, tokens = self._pool(x)

        if self.proj is not None:
            if self.SP:
                pooled = pooled @ self.proj  # type: ignore
            else:
                pooled = pooled @ self.proj.weight.t()  # type: ignore

        if self.output_tokens:
            return pooled, tokens

        return pooled


def text_global_pool(
    x: torch.Tensor,
    text: Optional[torch.Tensor] = None,
    pool_type: str = "argmax",
) -> torch.Tensor:
    if pool_type == "first":
        pooled = x[:, 0]
    elif pool_type == "last":
        pooled = x[:, -1]
    elif pool_type == "argmax":
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    else:
        pooled = x

    return pooled


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]  # type: ignore

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,  # type: ignore
        output_dim: Optional[int] = 512,
        embed_cls: bool = False,
        no_causal_mask: bool = False,
        pad_id: int = 0,
        pool_type: str = "argmax",
        proj_type: str = "linear",
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
        SP: bool = False,
        attn_mult: float = 1.0,
        output_mult: float = 1.0,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type
        self.SP = SP

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            SP=SP,
            attn_mult=attn_mult,
            output_mult=output_mult,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer("attn_mask", self.build_causal_mask(), persistent=False)

        if proj_type == "none" or not output_dim:
            self.text_projection = None
        else:
            if proj_bias:
                if self.SP:
                    self.text_projection = nn.Linear(width, output_dim)
                else:
                    self.text_projection = MuReadout(
                        width,
                        output_dim,
                        readout_zero_init=True,
                        output_mult=output_mult,
                    )
            else:
                if self.SP:
                    self.text_projection = nn.Parameter(torch.empty(width, output_dim))
                    # self.text_projection = nn.Linear(width, output_dim, bias=False)
                else:
                    self.text_projection = MuReadout(
                        width,
                        output_dim,
                        bias=False,
                        readout_zero_init=True,
                        output_mult=output_mult,
                    )

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        if self.SP:
            attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            if self.SP:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)  # type: ignore
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)  # type: ignore
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)  # type: ignore
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)  # type: ignore

        if self.text_projection is not None:
            if isinstance(self.text_projection, (nn.Linear, MuReadout)):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width**-0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    @torch.jit.ignore  # type: ignore  # false positive
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore  # type: ignore  # false positive
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {"positional_embedding"}
        if self.cls_emb is not None:
            no_wd.add("cls_emb")
        return no_wd

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _embeds(self, text) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        return x, attn_mask

    def forward_intermediates(
        self,
        text: torch.Tensor,
        indices: Optional[Union[int, List[int]]] = None,
        stop_early: bool = False,
        normalize_intermediates: bool = False,
        intermediates_only: bool = False,
        output_fmt: str = "NCHW",
        output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            text: Input text ids
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply norm layer to all intermediates
            intermediates_only: Only return intermediate features
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both prefix and intermediate tokens
        Returns:

        """
        assert output_fmt in ("NLC",), "Output format must be NLC."
        # forward pass
        x, attn_mask = self._embeds(text)
        x, intermediates = self.transformer.forward_intermediates(
            x,
            attn_mask=attn_mask,
            indices=indices,
            stop_early=stop_early,
        )

        # process intermediates
        if normalize_intermediates:
            # apply final norm to all intermediates
            intermediates = [self.ln_final(xi) for xi in intermediates]

        output = {}

        if self.cls_emb is not None:
            seq_intermediates = [
                xi[:, :-1] for xi in intermediates
            ]  # separate concat'd class token from sequence
            if output_extra_tokens:
                # return suffix class tokens separately
                cls_intermediates = [xi[:, -1:] for xi in intermediates]
                output["text_intermediates_suffix"] = cls_intermediates
            intermediates = seq_intermediates
        output["text_intermediates"] = intermediates

        if intermediates_only:
            return output

        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled = text_global_pool(x, pool_type="last")
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        output["text_features"] = pooled

        return output

    def prune_intermediate_layers(
        self,
        indices: Union[int, List[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ):
        """Prune layers not required for specified intermediates."""
        take_indices = self.transformer.prune_intermediate_layers(indices)
        if prune_norm:
            self.ln_final = nn.Identity()
        if prune_head:
            self.text_projection = None
        return take_indices

    def forward(self, text):
        x, attn_mask = self._embeds(text)

        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled = text_global_pool(x, pool_type="last")
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
            tokens = x[:, :-1]
        else:
            x = self.ln_final(x)
            pooled = text_global_pool(x, text, pool_type=self.pool_type)
            tokens = x

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


# It's for mup only
def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    __annotations__ = {
        "bias_k": torch._jit_internal.Optional[torch.Tensor],  # type: ignore
        "bias_v": torch._jit_internal.Optional[torch.Tensor],  # type: ignore
    }
    __constants__ = [
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "in_proj_weight",
    ]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        attn_mult=1.0,
        encoder_var=1,
        standparam=False,
        batch_first=True,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn_mult = attn_mult
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.standparam = standparam
        self.batch_first = batch_first

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.init_method = init_method_normal((encoder_var / embed_dim) ** 0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            self.init_method(self.in_proj_weight)

            # zero initializing query head
            if not self.standparam:
                nn.init.constant_(self.in_proj_weight[: self.embed_dim], 0.0)
        else:
            if self.standparam:
                self.init_method(self.q_proj_weight)
            else:
                # zero initializing query head
                nn.init.constant_(self.q_proj_weight, 0.0)

            self.init_method(self.k_proj_weight)
            self.init_method(self.v_proj_weight)

        self.init_method(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.constant_(self.bias_k, 0.0)
        if self.bias_v is not None:
            nn.init.constant_(self.bias_v, 0.0)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    ):
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length.
            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """

        if self.batch_first:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            #### swapping pytorch's attn with ours ####
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                attn_mult=self.attn_mult,
                standparam=self.standparam,
            )
        else:
            #### swapping pytorch's attn with ours ####
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                attn_mult=self.attn_mult,
                standparam=self.standparam,
            )

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def multi_head_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    bias_k: torch.Tensor | None,
    bias_v: torch.Tensor | None,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    training: bool = True,
    key_padding_mask: torch.Tensor | None = None,
    need_weights: bool = True,
    attn_mask: torch.Tensor | None = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: torch.Tensor | None = None,
    k_proj_weight: torch.Tensor | None = None,
    v_proj_weight: torch.Tensor | None = None,
    static_k: torch.Tensor | None = None,
    static_v: torch.Tensor | None = None,
    attn_mult: float = 1,
    standparam: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    if standparam:
        scaling = float(head_dim) ** -0.5 * math.sqrt(attn_mult)
    else:
        # ------------------ mup begins ------------------------
        scaling = float(head_dim) ** -1 * math.sqrt(attn_mult)
        # ------------------ mup ends --------------------------

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)  # type: ignore
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)  # type: ignore
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)  # type: ignore
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)

    q = q * scaling

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])  # type: ignore
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])  # type: ignore
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)  # type: ignore

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [
                k,  # type: ignore
                torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device),  # type: ignore
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,  # type: ignore
                torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device),  # type: ignore
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # type: ignore
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)  # type: ignore
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
