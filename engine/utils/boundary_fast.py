from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.utils.misc import to_2tuple
__all__ = ['fast_erode', 'fast_dilate']


def _tensor_ndarray_inputs_wrapper(func):
    """
    transform Union[torch.Tensor, np.ndarray] with shape (*, height, width) to
    torch.Tensor with shape (batch_size, 1, height, width), and return the
    result to the original shape/type
    """
    def wrapper(x: Union[torch.Tensor, np.ndarray], *args, **kwargs):
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            raise TypeError(f'Cannot handle type of mask: {type(x)}')
        ori_x = x
        ori_shape = tuple(x.shape)
        to_tensor = isinstance(x, torch.Tensor)
        x = x if to_tensor else torch.from_numpy(x)
        x = x.view(-1, 1, *ori_shape[-2:])
        x = func(x, *args, **kwargs)
        x = x.view(*ori_shape)
        x = x.to(ori_x) if to_tensor else x.numpy().astype(ori_x.dtype)
        return x

    return wrapper


@_tensor_ndarray_inputs_wrapper
def fast_erode(mask, kernel_size, iterations):
    if (mask == 0).sum().item() + (mask == 1).sum().item() != mask.nelement():
        raise ValueError(
            f'`mask` should be binary, but got values in '
            f'{tuple(map(lambda x: x.item(), mask.unique()))}')
    if kernel_size % 2 == 0:
        raise ValueError(f'`kernel_size` should be odd, but got {kernel_size}')
    ori_mask = mask
    mask = mask.float()
    kernel = torch.ones(1, 1, *to_2tuple(kernel_size)).to(mask)
    for _ in range(iterations):
        mask = F.pad(mask, [kernel_size // 2] * 4, mode='constant', value=0)
        mask = (F.conv2d(mask, kernel) == kernel.nelement()).to(mask)
    return mask.to(ori_mask)


@_tensor_ndarray_inputs_wrapper
def fast_dilate(mask, kernel_size, iterations):
    if (mask == 0).sum().item() + (mask == 1).sum().item() != mask.nelement():
        raise ValueError(
            f'`mask` should be binary, but got values in '
            f'{tuple(map(lambda x: x.item(), mask.unique()))}')
    if kernel_size % 2 == 0:
        raise ValueError(f'`kernel_size` should be odd, but got {kernel_size}')
    ori_mask = mask
    mask = mask.float()
    kernel = torch.ones(1, 1, *to_2tuple(kernel_size)).to(mask)
    for _ in range(iterations):
        mask = F.pad(mask, [kernel_size // 2] * 4, mode='constant', value=0)
        mask = (F.conv2d(mask, kernel) > 0).to(mask)
    return mask.to(ori_mask)
