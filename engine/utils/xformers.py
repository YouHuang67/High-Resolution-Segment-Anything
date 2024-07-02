import os
import warnings

import torch
import torch.nn.functional as F

try:
    import xformers.ops as xops
    if not torch.cuda.is_available():
        warnings.warn(f'xformers is not available for cpu')
        xops = None
    if 'DISABLE_XFORMERS' in os.environ:
        warnings.warn(f'xformers is disabled')
        xops = None
except ImportError:
    warnings.warn(f'Not found xformers, use plain implementation instead')
    xops = None


def memory_efficient_attention(
        query, key, value, attn_bias=None, p=0.0, scale=None):
    if xops is not None:
        result = xops.memory_efficient_attention(
            query=(query.transpose(1, 2) if query.ndim == 4 else query),
            key=(key.transpose(1, 2) if key.ndim == 4 else key),
            value=(value.transpose(1, 2) if value.ndim == 4 else value),
            attn_bias=attn_bias,
            p=p,
            scale=scale)
        return result.transpose(1, 2) if result.ndim == 4 else result
    else:
        scale = scale or query.size(-1) ** -0.5
        attn = ((query * scale) @ key.transpose(-2, -1))
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        if p > 0.0:
            attn = F.dropout(attn, p=p)
        return attn @ value
