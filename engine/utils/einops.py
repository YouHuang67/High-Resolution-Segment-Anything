__all__ = ['rearrange', 'repeat', 'reduce']
import torch
from einops import rearrange as _rearrange
from einops import repeat as _repeat
from einops import reduce as _reduce


def rearrange(x, *args, **kwargs):
    if isinstance(x, torch.Tensor):
        return _rearrange(x, *args, **kwargs).contiguous()
    return _rearrange(x, *args, **kwargs)


def repeat(x, *args, **kwargs):
    if isinstance(x, torch.Tensor):
        return _repeat(x, *args, **kwargs).contiguous()
    return _repeat(x, *args, **kwargs)


def reduce(x, *args, **kwargs):
    if isinstance(x, torch.Tensor):
        return _reduce(x, *args, **kwargs).contiguous()
    return _reduce(x, *args, **kwargs)
