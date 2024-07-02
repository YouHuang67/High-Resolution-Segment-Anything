import random
import os
import warnings
from typing import Optional, List, Tuple

import torch

from ..timers import Timer
from .distance_fast import fast_mask_to_distance
from .click import CLK_POSITIVE, CLK_NEGATIVE, CLK_MODES
from .click import generate_single_click, generate_clicks


@Timer('CLK')
def fast_generate_single_click(pre_label: torch.Tensor,
                               seg_label: torch.Tensor,
                               points: Optional[List[Tuple]] = None,
                               sfc_inner_k: float = 1.0,
                               ignore_mask=None):
    """
    Click function for image segmentation

    :param pre_label: predicted label with shape (height, width)
    :param seg_label: ground truth label with shape (height, width)
    :param points: list of tuples (y, x, mode) representing clicked points
                   with mode being either 1 (positive) or 2 (negative)
    :param sfc_inner_k: float representing the adjustment factor for click area,
                        where 1.0 indicates the center of the erroneous region
    :param ignore_mask: mask of points to ignore
    :return: tuple representing new click with format (y, x, mode)
    """
    with Timer('Check'):
        # Check types and shapes of input labels
        if not isinstance(pre_label, torch.Tensor):
            raise TypeError(f'Cannot handle type of pre_label: '
                            f'{type(pre_label)}, expected torch.Tensor')
        if not isinstance(seg_label, torch.Tensor):
            raise TypeError(f'Cannot handle type of seg_label: '
                            f'{type(seg_label)}, expected torch.Tensor')
        if tuple(pre_label.shape) != tuple(seg_label.shape):
            raise ValueError(f'`pre_label` and `seg_label` '
                             f'are of different shapes: '
                             f'{tuple(pre_label.shape)} and '
                             f'{tuple(seg_label.shape)}')
        if len(pre_label.shape) != 2:
            raise ValueError(f'Both `pre_label` and `seg_label` are expected '
                             f'to have the shape of (height, width), but got '
                             f'shape {tuple(pre_label.shape)}')

        # Check validity of input points
        if points is not None:
            height, width = seg_label.shape
            for y, x, mode in points:
                if isinstance(float(y), float) \
                        and isinstance(float(x), float) \
                        and (0 <= y < height) \
                        and (0 <= x < width) \
                        and (mode in CLK_MODES):
                    continue
                raise ValueError(f'Found invalid point {(y, x, mode)} '
                                 f'for {height}x{width} labels '
                                 f'among points: {points}')

        # Calculate the distance scale based on sfc_inner_k
        if sfc_inner_k >= 1.0:
            dist_scale = 1 / (sfc_inner_k + torch.finfo(torch.float).eps)
        elif sfc_inner_k < 0.0:
            dist_scale = 0.0  # whole object area
        else:
            raise ValueError(f'Invalid sfc_inner_k: {sfc_inner_k}')

    # Convert labels to numpy arrays
    with Timer('Convert'):
        pre_label = (pre_label == 1)
        seg_label = (seg_label == 1)

    # Create ignore mask based on points
    with Timer('IgnoreMask'):
        if ignore_mask is None:
            ignore_mask = torch.zeros_like(pre_label)
        else:
            ignore_mask = ignore_mask.clone()
        if points is not None:
            y_coords, x_coords = zip(*[(y, x) for y, x, _ in points])
            y_tensor = torch.LongTensor(y_coords).to(ignore_mask.device)
            x_tensor = torch.LongTensor(x_coords).to(ignore_mask.device)
            ignore_mask[y_tensor, x_tensor] = True

    # Calculate erroneous regions and perform distance transform
    with Timer('Logic'):
        fneg = (~pre_label) & seg_label & (~ignore_mask)
        fpos = pre_label & (~seg_label) & (~ignore_mask)
    with Timer('Dist'):
        ndist = fast_mask_to_distance(fneg, True)
        pdist = fast_mask_to_distance(fpos, True)

    # Calculate maximum distances
    with Timer('MaxDist'):
        ndmax, pdmax = ndist.max(), pdist.max()
        if ndmax.item() == pdmax.item() == 0:
            return None, None, None

    # Determine click mode and points based on maximum distances
    with Timer('SelectPoints'):
        if ndmax > pdmax:
            mode = CLK_POSITIVE
            points = torch.nonzero(ndist > dist_scale * ndmax, as_tuple=False)
        else:
            mode = CLK_NEGATIVE
            points = torch.nonzero(pdist > dist_scale * pdmax, as_tuple=False)

        if points.size(0) == 0:
            return None, None, None

        # Randomly choose a point from the points
        y, x = points[random.choice(range(points.size(0)))].tolist()

    return int(y), int(x), mode


def fast_generate_clicks(pre_labels: torch.Tensor,
                         seg_labels: torch.Tensor,
                         points_list: Optional[List[List[Tuple]]] = None,
                         sfc_inner_k: float = 1.0,
                         ignore_masks=None,
                         **kwargs):
    if points_list is None:
        points_list = [None] * len(pre_labels)
    clicks = []
    for idx, (pre_label, seg_label, points) in enumerate(zip(
            pre_labels, seg_labels, points_list)):
        if ignore_masks is None:
            ignore_mask = None
        else:
            ignore_mask = ignore_masks[idx].reshape(ignore_masks.shape[-2:])
        clicks.append(fast_generate_single_click(
            pre_label=pre_label.reshape(pre_label.shape[-2:]),
            seg_label=seg_label.reshape(seg_label.shape[-2:]),
            points=points,
            sfc_inner_k=sfc_inner_k,
            ignore_mask=ignore_mask,
            **kwargs))
    return clicks


if 'DISABLE_FAST_CLICK' in os.environ:
    warnings.warn('Fast click is disabled')
    fast_generate_single_click = generate_single_click
    fast_generate_clicks = generate_clicks
