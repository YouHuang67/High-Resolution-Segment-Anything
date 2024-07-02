import random
from typing import Optional, Union, List, Tuple

import numpy as np
import torch

from ..timers import Timer
from .distance import mask_to_distance
from .distance_fast import fast_mask_to_distance


CLK_POSITIVE = 'positive'
CLK_POSITIVE_ID = 253
CLK_NEGATIVE = 'negative'
CLK_NEGATIVE_ID = 254
CLK_MODES = (CLK_POSITIVE, CLK_NEGATIVE)
CLK_MODE_IDS = (CLK_POSITIVE_ID, CLK_NEGATIVE_ID)


def generate_multiple_clicks(num_clicks: int,
                             pre_label: Union[torch.Tensor, np.ndarray],
                             seg_label: Union[torch.Tensor, np.ndarray],
                             points: Optional[List[Tuple]] = None,
                             sfc_inner_k: float = 1.0,
                             dist_mode='plain'):
    """
    Click function for image segmentation

    :param num_clicks: number of clicks to generate
    :param pre_label: predicted label with shape (height, width)
    :param seg_label: ground truth label with shape (height, width)
    :param points: list of tuples (y, x, mode) representing clicked points
                   with mode being either 1 (positive) or 2 (negative)
    :param sfc_inner_k: float representing the adjustment factor for click area,
                        where 1.0 indicates the center of the erroneous region
    :param dist_mode: str representing the mode of distance transform,
                      either 'plain' or 'fast'
    :return: tuple representing new click with format (y, x, mode)
    """
    # Check types and shapes of input labels
    if not isinstance(pre_label, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of pre_label: {type(pre_label)}')
    if not isinstance(seg_label, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of seg_label: {type(seg_label)}')
    if type(pre_label) != type(seg_label):
        raise TypeError(f'`pre_label` and `seg_label` are of different types: '
                        f'{type(pre_label)} and {type(seg_label)}')
    if tuple(pre_label.shape) != tuple(seg_label.shape):
        raise ValueError(f'`pre_label` and `seg_label` '
                         f'are of different shapes: '
                         f'{tuple(pre_label.shape)} and '
                         f'{tuple(seg_label.shape)}')
    if len(pre_label.shape) != 2:
        raise ValueError(f'Both `pre_label` and `seg_label` are expected to '
                         f'have the shape of (height, width), but got shape '
                         f'{tuple(pre_label.shape)}')
    if dist_mode not in ('plain', 'fast'):
        raise ValueError(f'Invalid mode: {dist_mode}')

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
    if isinstance(pre_label, torch.Tensor):
        pre_label = pre_label.detach().cpu().numpy()
        seg_label = seg_label.detach().cpu().numpy()
    pre_label = (pre_label == 1)
    seg_label = (seg_label == 1)

    # Create ignore mask based on points
    ignore_mask = np.zeros_like(pre_label, dtype=bool)
    for y, x, _ in (list() if points is None else points):
        ignore_mask[y, x] = True

    # Calculate erroneous regions and perform distance transform
    with Timer('Logic'):
        fneg = np.logical_and(~pre_label, seg_label)
        fpos = np.logical_and(pre_label, ~seg_label)
        fneg = np.logical_and(fneg, ~ignore_mask)
        fpos = np.logical_and(fpos, ~ignore_mask)
    with Timer('Dist'):
        if dist_mode == 'plain':
            ndist = mask_to_distance(fneg, True)
            pdist = mask_to_distance(fpos, True)
        elif dist_mode == 'fast':
            ndist = fast_mask_to_distance(fneg, True)
            pdist = fast_mask_to_distance(fpos, True)
        else:
            raise NotImplementedError(f'Invalid mode: {dist_mode}')

    # Calculate maximum distances
    ndmax, pdmax = ndist.max(), pdist.max()
    if ndmax == pdmax == 0:
        return [(None, None, None) for _ in range(num_clicks)]

    # Determine click mode and points based on maximum distances
    if ndmax > pdmax:
        mode = CLK_POSITIVE
        points = np.argwhere(ndist > dist_scale * ndmax)
    else:
        mode = CLK_NEGATIVE
        points = np.argwhere(pdist > dist_scale * pdmax)

    if len(points) == 0:
        return [(None, None, None) for _ in range(num_clicks)]

    points = list(map(tuple, points.tolist()))
    random.shuffle(points)
    clicks = [(int(y), int(x), mode) for y, x in points[:num_clicks]]
    if len(clicks) < num_clicks:
        clicks += [(None, None, None) for _ in range(num_clicks - len(clicks))]
    return clicks


@Timer('CLK')
def generate_single_click(pre_label: Union[torch.Tensor, np.ndarray],
                          seg_label: Union[torch.Tensor, np.ndarray],
                          points: Optional[List[Tuple]] = None,
                          sfc_inner_k: float = 1.0,
                          dist_mode='plain'):
    """
    Click function for image segmentation

    :param pre_label: predicted label with shape (height, width)
    :param seg_label: ground truth label with shape (height, width)
    :param points: list of tuples (y, x, mode) representing clicked points
                   with mode being either 1 (positive) or 2 (negative)
    :param sfc_inner_k: float representing the adjustment factor for click area,
                        where 1.0 indicates the center of the erroneous region
    :param dist_mode: str representing the mode of distance transform,
                      either 'plain' or 'fast'
    :return: tuple representing new click with format (y, x, mode)
    """
    with Timer('Check'):
        # Check types and shapes of input labels
        if not isinstance(pre_label, (torch.Tensor, np.ndarray)):
            raise TypeError(f'Cannot handle type of pre_label: '
                            f'{type(pre_label)}')
        if not isinstance(seg_label, (torch.Tensor, np.ndarray)):
            raise TypeError(f'Cannot handle type of seg_label: '
                            f'{type(seg_label)}')
        if type(pre_label) != type(seg_label):
            raise TypeError(
                f'`pre_label` and `seg_label` are of different types: '
                f'{type(pre_label)} and {type(seg_label)}')
        if tuple(pre_label.shape) != tuple(seg_label.shape):
            raise ValueError(f'`pre_label` and `seg_label` '
                             f'are of different shapes: '
                             f'{tuple(pre_label.shape)} and '
                             f'{tuple(seg_label.shape)}')
        if len(pre_label.shape) != 2:
            raise ValueError(f'Both `pre_label` and `seg_label` are expected '
                             f'to have the shape of (height, width), but got '
                             f'shape {tuple(pre_label.shape)}')
        if dist_mode not in ('plain', 'fast'):
            raise ValueError(f'Invalid mode: {dist_mode}')

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
        if isinstance(pre_label, torch.Tensor):
            pre_label = pre_label.detach().cpu().numpy()
            seg_label = seg_label.detach().cpu().numpy()
        pre_label = (pre_label == 1)
        seg_label = (seg_label == 1)

    # Create ignore mask based on points
    with Timer('IgnoreMask'):
        ignore_mask = np.zeros_like(pre_label, dtype=bool)
        for y, x, _ in (list() if points is None else points):
            ignore_mask[y, x] = True

    # Calculate erroneous regions and perform distance transform
    with Timer('Logic'):
        fneg = np.logical_and(~pre_label, seg_label)
        fpos = np.logical_and(pre_label, ~seg_label)
        fneg = np.logical_and(fneg, ~ignore_mask)
        fpos = np.logical_and(fpos, ~ignore_mask)
    with Timer('Dist'):
        if dist_mode == 'plain':
            ndist = mask_to_distance(fneg, True)
            pdist = mask_to_distance(fpos, True)
        elif dist_mode == 'fast':
            ndist = fast_mask_to_distance(fneg, True)
            pdist = fast_mask_to_distance(fpos, True)
        else:
            raise NotImplementedError(f'Invalid mode: {dist_mode}')

    # Calculate maximum distances
    with Timer('MaxDist'):
        ndmax, pdmax = ndist.max(), pdist.max()
        if ndmax == pdmax == 0:
            return None, None, None

    # Determine click mode and points based on maximum distances
    with Timer('SelectPoints'):
        if ndmax > pdmax:
            mode = CLK_POSITIVE
            points = np.argwhere(ndist > dist_scale * ndmax)
        else:
            mode = CLK_NEGATIVE
            points = np.argwhere(pdist > dist_scale * pdmax)

        if len(points) == 0:
            return None, None, None

        # Randomly choose a point from the points
        y, x = random.choice(list(map(tuple, points.tolist())))
        return int(y), int(x), mode


def generate_clicks(pre_labels: Union[torch.Tensor, np.ndarray],
                    seg_labels: Union[torch.Tensor, np.ndarray],
                    points_list: Optional[List[List[Tuple]]] = None,
                    sfc_inner_k: float = 1.0,
                    **kwargs):
    if points_list is None:
        points_list = [None] * len(pre_labels)
    clicks = []
    for pre_label, seg_label, points in zip(
            pre_labels, seg_labels, points_list):
        clicks.append(generate_single_click(
            pre_label=pre_label.reshape(pre_label.shape[-2:]),
            seg_label=seg_label.reshape(seg_label.shape[-2:]),
            points=points,
            sfc_inner_k=sfc_inner_k,
            **kwargs))
    return clicks


def generate_clicks_list(num_clicks: int,
                         pre_labels: Union[torch.Tensor, np.ndarray],
                         seg_labels: Union[torch.Tensor, np.ndarray],
                         points_list: Optional[List[List[Tuple]]] = None,
                         sfc_inner_k: float = 1.0,
                         **kwargs):
    if points_list is None:
        points_list = [None] * len(pre_labels)
    clicks_list = []
    for pre_label, seg_label, points in zip(
            pre_labels, seg_labels, points_list):
        clicks_list.append(generate_multiple_clicks(
            num_clicks=num_clicks,
            pre_label=pre_label.reshape(pre_label.shape[-2:]),
            seg_label=seg_label.reshape(seg_label.shape[-2:]),
            points=points,
            sfc_inner_k=sfc_inner_k,
            **kwargs))
    return clicks_list
