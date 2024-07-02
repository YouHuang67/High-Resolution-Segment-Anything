from typing import Union
import numpy as np
import torch

from .zoom_in import get_bbox_from_mask
from .distance import mask_to_distance


def fast_mask_to_distance(mask: Union[torch.Tensor, np.ndarray],
                          boundary_padding: bool
                          ) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert a binary mask to a distance map.

    :param mask: A binary mask of shape (*, height, width).
    :param boundary_padding: A boolean flag indicating whether to pad the
    boundary of the mask before computing the distance map.
    :return: A distance map of the same shape as the input mask.
    """
    if not isinstance(mask, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(mask)}')
    if not boundary_padding:
        raise NotImplementedError(f'`boundary_padding` must be True')

    bbox = get_bbox_from_mask(mask)

    to_tensor = isinstance(mask, torch.Tensor)
    device = mask.device if to_tensor else None
    mask = mask.detach().cpu().numpy() if to_tensor else mask
    bbox = bbox.detach().cpu().numpy() if to_tensor else bbox

    ori_shape = mask.shape
    mask = mask.reshape((-1, ) + ori_shape[-2:])
    dist = np.zeros_like(mask, dtype=np.float32)
    for i, (left, up, right, bottom) in enumerate(bbox.reshape((-1, 4))):
        dist[i, up:bottom, left:right] = mask_to_distance(
            mask[i, up:bottom, left:right], boundary_padding=True)
    dist = dist.reshape(ori_shape)

    dist = torch.from_numpy(dist).to(device) if to_tensor else dist
    return dist
