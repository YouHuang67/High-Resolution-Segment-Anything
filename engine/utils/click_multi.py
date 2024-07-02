import random

import torch

from ..timers import Timer
from .distance_fast import fast_mask_to_distance
from .click import CLK_POSITIVE, CLK_NEGATIVE, CLK_MODES


def generate_multi_clicks(num_points,
                          pre_label: torch.Tensor,
                          seg_label: torch.Tensor,
                          points=None,
                          sfc_inner_k=1.0,
                          mode=None,
                          sample_mode='random'):
    """
    Click function for image segmentation

    :param num_points: number of points to sample
    :param pre_label: predicted label with shape (height, width)
    :param seg_label: ground truth label with shape (height, width)
    :param points: list of tuples (y, x, mode) representing clicked points
                   with mode being either 1 (positive) or 2 (negative)
    :param sfc_inner_k: float representing the adjustment factor for click area,
                        where 1.0 indicates the center of the erroneous region
    :param mode: (optional) specify positiveness of multi points
    :param sample_mode: the strategy to sample points, either 'random' or 'max_dist'
    :return: a list of tuples representing new click with format (y, x, mode)
    """
    if num_points < 1:
        raise ValueError(f'Invalid num_points: {num_points}')
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
    if mode not in [None, CLK_POSITIVE, CLK_NEGATIVE]:
        raise ValueError(f'Invalid mode: {mode}')
    if sample_mode not in ['random', 'max_dist']:
        raise NotImplementedError(f'Invalid sample_mode: {sample_mode}')

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
    pre_label = (pre_label == 1)
    seg_label = (seg_label == 1)

    # Create ignore mask based on points
    ignore_mask = torch.zeros_like(pre_label)
    if points is not None:
        y_coords, x_coords = zip(*[(y, x) for y, x, _ in points])
        y_tensor = torch.LongTensor(y_coords).to(ignore_mask.device)
        x_tensor = torch.LongTensor(x_coords).to(ignore_mask.device)
        ignore_mask[y_tensor, x_tensor] = True

    # Calculate erroneous regions and perform distance transform
    fneg = (~pre_label) & seg_label & (~ignore_mask)
    fpos = pre_label & (~seg_label) & (~ignore_mask)
    ndist = fast_mask_to_distance(fneg, True)
    pdist = fast_mask_to_distance(fpos, True)

    # Calculate maximum distances
    ndmax, pdmax = ndist.max(), pdist.max()
    if ndmax.item() == pdmax.item() == 0:
        return []

    if mode is None:
        # Determine click mode and points based on maximum distances
        if ndmax > pdmax:
            mode = CLK_POSITIVE
            points = torch.nonzero(ndist > dist_scale * ndmax, as_tuple=False)
        else:
            mode = CLK_NEGATIVE
            points = torch.nonzero(pdist > dist_scale * pdmax, as_tuple=False)

    else:
        if (mode == CLK_POSITIVE and ndmax > 0.0) or (pdmax == 0.0):
            mode = CLK_POSITIVE
            points = torch.nonzero(ndist > dist_scale * ndmax, as_tuple=False)
        else:
            mode = CLK_NEGATIVE
            points = torch.nonzero(pdist > dist_scale * pdmax, as_tuple=False)

    if points.size(0) == 0:
        return []

    with Timer('SamplePoints'):

        left_indices = list(range(len(points)))
        random.shuffle(left_indices)

        if sample_mode == 'random':
            indices = left_indices[:num_points]
        elif sample_mode == 'max_dist':
            indices = [left_indices[0]]
            for _ in range(num_points - 1):
                left_indices.remove(indices[-1])
                dist = torch.cdist(points[left_indices], points[indices], p=2)
                prob = dist.mean(dim=1)
                prob = prob / (prob.sum() + torch.finfo(torch.float).eps)
                indices.append(left_indices[torch.multinomial(prob, 1).item()])
        else:
            raise NotImplementedError(f'Invalid sample_mode: {sample_mode}')

    return [(y, x, mode) for y, x in points[indices].tolist()]


def batch_generate_multi_clicks(num_points,
                                pre_labels: torch.Tensor,
                                seg_labels: torch.Tensor,
                                points_list=None,
                                sfc_inner_k=1.0,
                                **kwargs):
    if points_list is None:
        points_list = [None] * len(pre_labels)
    batch_clicks = []
    for pre_label, seg_label, points in zip(
            pre_labels, seg_labels, points_list):
        batch_clicks.append(generate_multi_clicks(
            num_points=num_points,
            pre_label=pre_label.reshape(pre_label.shape[-2:]),
            seg_label=seg_label.reshape(seg_label.shape[-2:]),
            points=points,
            sfc_inner_k=sfc_inner_k,
            **kwargs))
    return batch_clicks
