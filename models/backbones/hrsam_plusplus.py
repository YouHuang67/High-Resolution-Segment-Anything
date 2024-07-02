import math
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from mmengine.utils.misc import to_2tuple
from mmengine.model import BaseModule
from mmseg.registry import MODELS

from timm.models.layers import drop_path as timm_drop_path
from engine.utils import rearrange, repeat, reduce


from xformers.ops import fmha  # noqa
import selective_scan_cuda_core  # noqa, build selective_scan_cuda_core from projects/vmamba/selective_scan


class SelectiveScan(torch.autograd.Function):

    """
    forked from vmamba
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx,
                u,
                delta,
                A,
                B,
                C,
                D=None,
                delta_bias=None,
                delta_softplus=False,
                nrows=1):
        """
        :param ctx:
        :param u: (B, G * D, L)
        :param delta: (B, G * D, L)
        :param A: (G * D, N)
        :param B: (B, G, N, L)
        :param C: (B, G, N, L)
        :param D: (G * D)
        :param delta_bias: (G * D)
        :param delta_softplus: bool
        :param nrows: int, default 1
        :return: (B, G * D, L)
        """
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        out, x, *_ = selective_scan_cuda_core.fwd(
            u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *_ = \
            selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias,
                dout, x, ctx.delta_softplus, 1)
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None


class Rearrange(nn.Module):

    def __init__(self, pattern, **kwargs):
        super(Rearrange, self).__init__()
        self.pattern = pattern
        self.kwargs = kwargs

    def forward(self, x):
        return rearrange(x, self.pattern, **self.kwargs)


class Permute(nn.Module):

    def __init__(self, *pattern):
        super(Permute, self).__init__()
        self.pattern = pattern

    def forward(self, x):
        return x.permute(*self.pattern)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


def compute_padding_size(size, window_size):
    pad = (window_size - size % window_size) % window_size
    return size + pad, pad


def expand_and_concat(tensors, dim=-1):
    assert all(tensors[0].ndim == tensor.ndim for tensor in tensors), \
        "All tensors must have the same number of dimensions"
    dim = tuple(range(tensors[0].ndim))[dim]
    tensors = [
        tensor.expand(*[
            max_size if i != dim else -1
            for i, max_size in enumerate(
                map(max, zip(*[tensor.shape for tensor in tensors]))  # noqa
            )
        ]) for tensor in tensors
    ]
    return torch.cat(tensors, dim=dim)


class VisionRotaryEmbedding(nn.Module):

    """
    forked from EVA02's rotary embedding, with some modifications
    """

    def __init__(self,
                 dim,
                 ft_seq_len,
                 pt_seq_len,
                 theta=10000):
        super(VisionRotaryEmbedding, self).__init__()
        self.register_buffer('freqs', theta ** (
            -torch.arange(0, dim, 2)[:dim // 2].float() / dim
        ))
        self.ft_seq_len = ft_seq_len
        self.pt_seq_len = pt_seq_len
        self.cache = dict()

    def forward(self, x, batch_size, hw_shapes, window_size, shift):  # noqa
        """
        :param x: shape (B, nW, Ws, Ws, N, C)
        :param hw_shapes: [(H1, W1), (H2, W2), ...]
        :return:
        """
        key = f'{batch_size}-{hw_shapes}-{window_size}-{shift}'
        if key in self.cache:
            freqs_cos, freqs_sin = self.cache[key]
            return x * freqs_cos + self.rotate_half(x) * freqs_sin
        elif len(self.cache) == 4:
            self.cache.clear()

        # compatible with multi-scale inputs
        freqs_list = []
        for H, W in hw_shapes:
            freqs_h = torch.einsum(
                '..., f -> ... f',
                torch.arange(H, device=x.device) /
                self.ft_seq_len * self.pt_seq_len,
                self.freqs)
            freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)  # H, C // 2

            freqs_w = torch.einsum(
                '..., f -> ... f',
                torch.arange(W, device=x.device) /
                self.ft_seq_len * self.pt_seq_len,
                self.freqs)
            freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)  # W, C // 2

            freqs = expand_and_concat(
                [rearrange(freqs_h, 'd ... -> d () ...'),
                 rearrange(freqs_w, 'd ... -> () d ...')],
                dim=-1
            )  # H, W, C
            if window_size is not None:
                Ws1, Ws2 = window_size
                freqs_list.append(rearrange(
                    freqs, '(h ws1) (w ws2) ... -> (h w ws1 ws2) ...',
                    ws1=Ws1, ws2=Ws2))
            else:
                freqs_list.append(rearrange(freqs, 'h w ... -> (h w) ...'))

        freqs = torch.cat(freqs_list, dim=0)
        freqs = repeat(freqs, 'l c -> (b l) () c', b=batch_size)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        self.cache[key] = (freqs_cos, freqs_sin)

        return x * freqs_cos + self.rotate_half(x) * freqs_sin

    @staticmethod
    def rotate_half(x):
        new_x = x.new_empty(x.shape)
        new_x[..., 0::2] = x[..., 1::2]
        new_x[..., 1::2] = -x[..., 0::2]
        return new_x


class DropPath(nn.Module):

    """
    Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return timm_drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 attn_drop=0.,
                 proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        hidden_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * hidden_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.padding = nn.Parameter(torch.zeros(1, self.embed_dim))

    def forward(self, x, indices, rope, attn_bias):
        """
        params x: (B * L, C)
        return:   (B * L, C)
        """
        BL, C = x.shape
        if indices is not None:
            x = torch.cat([x, self.padding], dim=0)[indices]

        q, k, v = rearrange(
            self.qkv(x),
            'bl (n3 n d) -> () bl n3 n d',
            n3=3, n=self.num_heads).unbind(dim=-3)

        q, k = rope(q), rope(k)

        x = fmha.memory_efficient_attention(
            q, k, v, attn_bias=attn_bias, scale=self.scale)
        x = rearrange(x, '() bl n d -> bl (n d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        if indices is not None:
            ori_x = x.new_empty(BL + 1, C)
            ori_x[indices] = x
            x = ori_x[:-1]

        return x


class Block(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = \
            DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_dim=embed_dim, hidden_dim=mlp_hidden_dim)

    def forward(self, x, **kwargs):
        """
        params x: (B * L, C)
        return:   (B * L, C)
        """
        x = x + self.drop_path(self.attn(self.norm1(x), **kwargs))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_dim=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        H, W = to_2tuple(img_size)
        pH, pW = to_2tuple(patch_size)
        num_patches = (W // pW) * (H // pH)
        self.patch_shape = (H // pH, W // pW)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):  # noqa
        P = self.patch_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b h w (c p1 p2)', p1=P, p2=P)
        weight = rearrange(self.proj.weight, 'd c p1 p2 -> d (c p1 p2)')
        bias = self.proj.bias
        x = F.linear(x, weight, bias)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class Mamba(nn.Module):

    def __init__(self,
                 embed_dim,
                 state_dim,
                 extend_ratio,

                 # init configs
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4):

        super(Mamba, self).__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.extend_ratio = extend_ratio

        C, D = embed_dim, state_dim
        self.in_norm = nn.LayerNorm(C)
        self.in_proj = nn.Sequential(nn.Linear(C, 2 * C, bias=True), nn.SiLU())
        self.ssm_proj = nn.Linear(C, C + 2 * D, bias=True)
        self.ssm_proj_bias = nn.Parameter(torch.zeros(C))
        self.As_log = nn.Parameter(
            repeat(torch.arange(1, D + 1, dtype=torch.float32),
                   'd -> c d', c=C).log())
        self.As_log._no_weight_decay = True
        self.Ds = nn.Parameter(torch.ones(C).float())
        self.Ds._no_weight_decay = True
        self.gamma = nn.Parameter(
            torch.ones(C, extend_ratio, 1).float())
        self.out_norm = nn.LayerNorm(C)
        self.out_proj = nn.Linear(C, C, bias=True)

        # initialize SSM weight
        dt_init_std = C ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(
                self.ssm_proj.weight[:C], dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(
                self.ssm_proj.weight[:C], -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        nn.init.zeros_(self.ssm_proj.bias)

        # initialize SSM bias
        dt = torch.exp(
            torch.rand(C) *
            (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.ssm_proj_bias.data.copy_(inv_dt)

    def forward(self, x, batch_size, split=None):  # noqa
        """
        :param x: (B * L, C)
        :param batch_size: int
        :return:  (B * L, C)
        """
        B = batch_size
        L, C = x.shape
        L //= B
        D = self.state_dim

        shortcut, x = x, self.in_norm(x)
        x, z = self.in_proj(x).chunk(2, dim=-1)

        ssm = self.ssm_proj(x)  # B * L, C + 2 * D

        As = -self.As_log.exp().float().contiguous()
        Ds = self.Ds.float().contiguous()
        delta_bias = self.ssm_proj_bias.float().contiguous()

        R = self.extend_ratio
        if split is not None and len(split) > 1:
            ys = []
            for x, ssm in zip(
                rearrange(x, '(b l) c -> b l c', b=B).split(split, dim=1),
                rearrange(ssm, '(b l) c -> b l c', b=B).split(split, dim=1)
            ):
                x = repeat(x, 'b l c -> b c (r l)', b=B, r=R)
                ssm = repeat(ssm, 'b l c -> b () c (r l)', b=B, r=R)
                x = x.float().contiguous()
                dts, Bs, Cs = ssm.split([C, D, D], dim=-2)
                dts = rearrange(
                    dts, 'b () c l -> b c l').float().contiguous()
                Bs = Bs.float().contiguous()
                Cs = Cs.float().contiguous()
                y = SelectiveScan.apply(
                    x, dts, As, Bs, Cs, Ds, delta_bias, True, 1)
                y = rearrange(y, 'b c (r l) -> b c r l', r=R)
                y = reduce(
                    self.gamma * y, 'b c r l -> b l c', reduction='sum')
                ys.append(y)
            y = rearrange(torch.cat(ys, dim=1), 'b l c -> (b l) c')
        else:
            x = repeat(x, '(b l) c -> b c (r l)', b=B, r=R)
            ssm = repeat(ssm, '(b l) c -> b () c (r l)', b=B, r=R)
            x = x.float().contiguous()
            dts, Bs, Cs = ssm.split([C, D, D], dim=-2)
            dts = rearrange(dts, 'b () c l -> b c l').float().contiguous()
            Bs = Bs.float().contiguous()
            Cs = Cs.float().contiguous()
            y = SelectiveScan.apply(
                x, dts, As, Bs, Cs, Ds, delta_bias, True, 1)
            y = rearrange(y, 'b c (r l) -> b c r l', r=R)
            y = reduce(
                self.gamma * y, 'b c r l -> (b l) c', reduction='sum')

        out = shortcut + self.out_proj(z * self.out_norm(y))

        return out


@MODELS.register_module()
class HRSAMPlusPlusViT(BaseModule):

    def __init__(self,

                 downsample_sizes,

                 window_size,
                 in_dim,
                 img_size,
                 patch_size,
                 depth,
                 embed_dim,
                 num_heads,
                 mlp_ratio,

                 drop_rate,
                 attn_drop_rate,
                 drop_path_rate,

                 out_indices,
                 final_embed_dim,

                 use_checkpoint=False,
                 pretrained=None,
                 init_cfg=None,

                 **mamba_cfg):

        super(HRSAMPlusPlusViT, self).__init__(init_cfg)

        self.downsample_sizes = downsample_sizes

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_dim=in_dim, embed_dim=embed_dim)
        self.patch_shape = self.patch_embed.patch_shape

        self.patch_size = patch_size
        self.window_size = window_size

        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        half_head_dim = embed_dim // num_heads // 2
        self.rope = VisionRotaryEmbedding(
            dim=half_head_dim,
            ft_seq_len=(img_size // patch_size),
            pt_seq_len=(img_size // patch_size) / 2.0)

        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=torch.linspace(0, drop_path_rate, depth)[i].item())
            for i in range(depth)])

        self.ss_mamba_blocks = nn.ModuleList([
            Mamba(embed_dim=embed_dim, **mamba_cfg) for _ in out_indices])
        self.ms_mamba_blocks = nn.ModuleList([
            Mamba(embed_dim=embed_dim, **mamba_cfg) for _ in out_indices])

        num_scales = len(downsample_sizes) + 1
        self.multi_scale_fusion = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                Rearrange('(ns b) ... d -> b ... (ns d)', ns=num_scales)
            ) for _ in out_indices])

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(nn.Linear(num_scales * embed_dim, final_embed_dim,
                                    bias=False),
                          nn.LayerNorm(final_embed_dim),
                          Rearrange('b h w c -> b c h w'))
            for _ in out_indices])
        self.out_conv = nn.Sequential(
            nn.Conv2d(final_embed_dim, final_embed_dim,
                      kernel_size=3, padding=1, bias=False,
                      groups=final_embed_dim),
            Transpose(1, -1),
            nn.LayerNorm(final_embed_dim),
            Transpose(1, -1))

        self.indices_cache = dict()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        ori_pos_embed = rearrange(
            self.pos_embed[:, 1:], '() (h w) c -> () c h w',
            h=self.patch_shape[0], w=self.patch_shape[1])

        xs, hw_shapes = [], []
        (H, W), ori_x = x.shape[-2:], x
        sizes = [(H, W)] + list(map(to_2tuple, self.downsample_sizes))
        for h, w in sizes:
            if (h, w) != (H, W):
                x = F.interpolate(
                    ori_x, size=(h, w), mode='bilinear',
                    align_corners=False)
            else:
                x = ori_x
            x = self.patch_embed(x)
            *_, Hp, Wp = x.shape
            if ori_pos_embed.shape[-2:] != (Hp, Wp):
                pos_embed = F.interpolate(
                    ori_pos_embed, size=(Hp, Wp),
                    mode='bilinear', align_corners=False)
            else:
                pos_embed = ori_pos_embed
            x = self.pos_drop(x + pos_embed)
            xs.append(rearrange(x, 'b c h w -> b (h w) c'))
            hw_shapes.append((Hp, Wp))
        x = torch.cat(xs, dim=1)

        Bs = len(x)
        Ws = self.window_size

        # make plain window arguments
        shift = 0
        plain_window_indices, pad_hw_shapes = self.make_indices(
            x,
            hw_shapes=hw_shapes,
            window_size=to_2tuple(Ws),
            shift=shift,
            indices_cache=self.indices_cache)
        plain_window_kwargs = dict(
            rope=partial(self.rope,
                         batch_size=Bs,
                         hw_shapes=pad_hw_shapes,
                         window_size=to_2tuple(Ws),
                         shift=shift),
            indices=plain_window_indices,
            attn_bias=fmha.BlockDiagonalMask.from_seqlens(
                (len(plain_window_indices) // Ws ** 2) * [Ws ** 2]))

        # make shift window arguments
        shift = self.window_size // 2
        shift_window_indices, pad_hw_shapes = self.make_indices(
            x,
            hw_shapes=hw_shapes,
            window_size=to_2tuple(Ws),
            shift=shift,
            indices_cache=self.indices_cache)
        shift_window_kwargs = dict(
            rope=partial(self.rope,
                         batch_size=Bs,
                         hw_shapes=pad_hw_shapes,
                         window_size=to_2tuple(Ws),
                         shift=shift),
            indices=shift_window_indices,
            attn_bias=fmha.BlockDiagonalMask.from_seqlens(
                (len(shift_window_indices) // Ws ** 2) * [Ws ** 2]))

        embeds = []
        x, out_index = rearrange(x, 'b l c -> (b l) c'), 0
        for i, block in enumerate(self.blocks):
            if i in self.out_indices:
                x = self.ss_mamba_blocks[out_index](
                    x, batch_size=Bs, split=[h * w for h, w in hw_shapes])
            if i % 2 == 0:
                kwargs = dict(plain_window_kwargs)
            else:
                kwargs = dict(shift_window_kwargs)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, **kwargs)
            else:
                x = block(x, **kwargs)
            if i in self.out_indices:
                x = self.ms_mamba_blocks[out_index](x, batch_size=Bs)
                outs = []
                for idx, out in enumerate(
                        rearrange(
                            x, '(b l) c -> b l c', b=Bs
                        ).split([h * w for h, w in hw_shapes], dim=1)
                ):
                    h, w = hw_shapes[idx]
                    if idx > 0:
                        out = rearrange(
                            out, 'b (h w) c -> b c h w', h=h, w=w)
                        out = F.interpolate(
                            out, size=hw_shapes[0], mode='bilinear',
                            align_corners=False)
                        out = rearrange(out, 'b c h w -> b h w c')
                    else:
                        out = rearrange(
                            out, 'b (h w) c -> b h w c', h=h, w=w)
                    outs.append(out)
                out = torch.cat(outs, dim=0)
                out = self.multi_scale_fusion[out_index](out)
                embeds.append(out)
                out_index += 1

        out = sum([
            lateral_conv(embed) for lateral_conv, embed in
            zip(self.lateral_convs, embeds)])
        out = self.out_conv(out)
        return (out, )

    @staticmethod
    def make_indices(x,
                     hw_shapes,
                     window_size,
                     shift,
                     indices_cache):
        B, device = len(x), x.device
        Ws1, Ws2 = to_2tuple(window_size)
        S1, S2 = to_2tuple(shift)
        key = f'{B}-{hw_shapes}-{Ws1}-{Ws2}-{S1}-{S2}'
        if key in indices_cache:
            return indices_cache[key]

        if Ws1 <= S1 or Ws2 <= S2:
            raise ValueError
        Ph, Pw = (Ws1 - S1) % Ws1, (Ws2 - S2) % Ws2
        if len(indices_cache) >= 8:
            indices_cache.clear()
        start, base_indices, pad_hw_shapes = 0, [], []
        INF = 2 * sum(max(h + Ws1, w + Ws2) for h, w in hw_shapes) ** 2
        for H, W in hw_shapes:
            PH, _ = compute_padding_size(Ph + H, Ws1)
            hs = torch.full((PH, ), -INF, device=device).long()
            PW, _ = compute_padding_size(Pw + W, Ws2)
            ws = torch.full((PW, ), -INF, device=device).long()
            hs[Ph:Ph + H] = torch.arange(H, device=device)
            ws[Pw:Pw + W] = torch.arange(W, device=device)
            hs, ws = torch.meshgrid(hs, ws, indexing='ij')
            idxs = hs * W + ws + start
            base_indices.append(rearrange(
                idxs, '(h ws1) (w ws2) -> (h w ws1 ws2)', ws1=Ws1, ws2=Ws2))
            start += H * W
            pad_hw_shapes.append((PH, PW))
        base_indices = torch.cat(base_indices, dim=0)
        indices = rearrange(torch.arange(B, device=device), 'b -> b ()')
        indices = indices * (sum(h * w for h, w in hw_shapes)) + \
                  rearrange(base_indices, 'l -> () l')
        indices = rearrange(indices, 'b l -> (b l)')
        indices[indices < 0] = B * sum(h * w for h, w in hw_shapes)
        indices_cache[key] = (indices, pad_hw_shapes)
        return indices, pad_hw_shapes


if __name__ == '__main__':
    model = HRSAMPlusPlusViT(

        downsample_sizes=[512],
        window_size=16,

        # mamba configs
        state_dim=32,
        extend_ratio=3,

        in_dim=3,
        img_size=224,
        patch_size=16,
        depth=12,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        use_checkpoint=False,
        out_indices=(2, 5, 8, 11),
        final_embed_dim=256)

    count = 0
    for param in model.parameters():
        count += param.numel()
    if count > 1e6:
        count = count / 1e6
        print(f'Number of parameters: {count:.2f}M')
    elif count > 1e3:
        count = count / 1e3
        print(f'Number of parameters: {count:.2f}K')
    else:
        print(f'Number of parameters: {count}')

    model = model.cuda()
    with torch.no_grad():
        _out = model(torch.randn(2, 3, 384, 384).cuda())
    if isinstance(_out, (tuple, list)):
        print([tuple(_.shape) for _ in _out])
    elif isinstance(_out, dict):
        print({k: tuple(v.shape) for k, v in _out.items()})
    elif isinstance(_out, torch.Tensor):
        print(tuple(_out.shape))
    else:
        raise NotImplementedError
