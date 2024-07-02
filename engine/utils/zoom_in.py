import numpy as np
import torch
from mmengine.utils.misc import to_2tuple


def get_bbox_from_mask(mask):
    """
    Returns the bounding boxes (left, up, right, bottom) of each mask in the given tensor.

    :param mask: Union[torch.Tensor, np.ndarray], shape (*, height, width)
    :return: Union[torch.Tensor, np.ndarray], shape (*, 4)
    """
    if not isinstance(mask, (torch.Tensor, np.ndarray)):
        raise TypeError(f'Cannot handle type of mask: {type(mask)}')

    ori_shape = mask.shape
    to_tensor = isinstance(mask, torch.Tensor)

    rows, cols = mask.any(-1), mask.any(-2)
    rows = rows.detach().cpu().numpy() if to_tensor else rows  # convert to np.ndarray
    rows = rows.reshape((-1, rows.shape[-1]))
    cols = cols.detach().cpu().numpy() if to_tensor else cols
    cols = cols.reshape((-1, cols.shape[-1]))

    results = []
    for row, col in zip(rows, cols):
        if not row.any() or not col.any():
            left, up, right, bottom = 0, 0, ori_shape[-1], ori_shape[-2]
        else:
            row, col = np.where(row)[0], np.where(col)[0]
            left, up, right, bottom = col[0], row[0], col[-1] + 1, row[-1] + 1
        results.append((left, up, right, bottom))

    bbox = np.array(results).reshape(ori_shape[:-2] + (4, ))
    return torch.from_numpy(bbox).long().to(mask.device) if to_tensor else bbox


def expand_bbox(bbox, h_ratio, w_ratio, height, width, h_min=0, w_min=0):
    """
    Expand bounding box by given ratios and minimum sizes.

    :param bbox: A tensor or ndarray with shape (*, 4), each includes (left, up, right, bottom)
    :param h_ratio: The expand ratio of height
    :param w_ratio: The expand ratio of width
    :param height: The height of the image
    :param width: The width of the image
    :param h_min: The minimum height of the expanded bbox
    :param w_min: The minimum width of the expanded bbox
    :return: A tensor or ndarray with shape (*, 4), each includes (left, up, right, bottom)
    """

    # Check the type and shape of the input bbox
    if not isinstance(bbox, (torch.Tensor, np.ndarray)):
        raise TypeError(f"Cannot handle type of bbox: {type(bbox)}")
    if bbox.shape[-1] != 4:
        raise ValueError(f"`bbox` is expected to have the shape: (*, 4), "
                         f"but got `bbox` of shape: {tuple(bbox.shape)}")

    # Convert bbox to a tensor and split it into 4 tensors
    ori_bbox = bbox
    to_array = isinstance(bbox, np.ndarray)
    bbox = torch.from_numpy(bbox) if to_array else bbox
    left, up, right, bottom = torch.chunk(bbox, 4, dim=-1)

    if (left >= right).any().item() or (up >= bottom).any().item():
        raise ValueError(
            f"Invalid bbox: {ori_bbox} with left >= right or up >= bottom")

    # Compute the center coordinates of the bbox
    xc, yc = (left + right) / 2.0, (up + bottom) / 2.0

    # Expand the bbox according to the ratios
    left = torch.round(xc - w_ratio * (xc - left)).clip(0, None)
    right = torch.round(xc - w_ratio * (xc - right)).clip(None, width)
    up = torch.round(yc - h_ratio * (yc - up)).clip(0, None)
    bottom = torch.round(yc - h_ratio * (yc - bottom)).clip(None, height)

    # Apply minimum size constraints to the expanded bbox
    if w_min > 0:
        _left = torch.round(xc - w_min / 2.0).clip(0, None)
        left = torch.where(left < _left, left, _left)
        _right = torch.round(xc + w_min / 2.0).clip(None, width)
        right = torch.where(right > _right, right, _right)

    if h_min > 0:
        _up = torch.round(yc - h_min / 2.0).clip(0, None)
        up = torch.where(up < _up, up, _up)
        _bottom = torch.round(yc + h_min / 2.0).clip(None, height)
        bottom = torch.where(bottom > _bottom, bottom, _bottom)

    # Concatenate the expanded bbox tensors and convert it back to the original type
    bbox = torch.cat([left, up, right, bottom], dim=-1).to(bbox)
    return bbox.numpy() if to_array else bbox.to(ori_bbox)


def convert_bbox_to_mask(bbox, size, device='cpu'):
    """
    :param bbox: torch.Tensor or np.ndarray, shape (*, 4),
    each includes (left, up, right, bottom)
    :param size: height and width of the mask
    :param device:
    :return: torch.Tensor, shape (*, height, width)
    """
    if isinstance(bbox, np.ndarray):
        bbox = torch.from_numpy(bbox).long()
        to_array = True
    else:
        to_array = False
    if not isinstance(bbox, torch.Tensor):
        raise NotImplementedError(f'Cannot handle type of bbox: {type(bbox)}')
    if bbox.shape[-1] != 4:
        raise ValueError(f'`bbox` is expected to have the shape: (*, 4), '
                         f'but got `bbox` of shape: {tuple(bbox.shape)}')
    ori_shape = tuple(bbox.shape[:-1])
    bbox = bbox.view(-1, 4)
    bbox_mask = torch.zeros(
        bbox.size(0), *size, dtype=torch.float32, device=device)
    for i in range(len(bbox)):
        left, top, right, bottom = bbox[i].view(4).cpu().numpy().tolist()
        bbox_mask[i, ..., top:bottom, left:right] = 1.0
    bbox_mask = bbox_mask.view(*ori_shape, *size)
    if to_array:
        bbox_mask = bbox_mask.numpy()
    return bbox_mask


def convert_mask_to_bbox_mask(mask, expand_ratio):
    return convert_bbox_to_mask(
        expand_bbox(get_bbox_from_mask(mask),
                    *to_2tuple(expand_ratio),
                    *mask.shape[-2:]),
        mask.shape[-2:],
        mask.device).long()


def expand_bbox_divisible_by(bbox, mask_height, mask_width, divisor=32):
    """
    slightly expand the bbox to be divisible by `divisor`
    """
    bbox_shape = bbox.shape
    if not isinstance(bbox, (np.ndarray, torch.Tensor)):
        raise TypeError("bbox must be a numpy array or a torch tensor")

    bbox = bbox.reshape(-1, 4)
    for i in range(len(bbox)):
        left, up, right, bottom = bbox[i]

        width = right - left
        height = bottom - up

        extra_width = (divisor - width % divisor) % divisor
        extra_height = (divisor - height % divisor) % divisor

        left -= extra_width // 2
        up -= extra_height // 2

        left = left.clip(0, None)
        right = left + width + extra_width
        right = right.clip(None, mask_width)
        left = right - width - extra_width
        up = up.clip(0, None)
        bottom = up + height + extra_height
        bottom = bottom.clip(None, mask_height)
        up = bottom - height - extra_height

        bbox[i, 0] = left
        bbox[i, 1] = up
        bbox[i, 2] = right
        bbox[i, 3] = bottom

    bbox = bbox.reshape(bbox_shape)
    return bbox
