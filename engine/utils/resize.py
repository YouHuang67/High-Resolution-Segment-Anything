import cv2
import numpy as np
import torch
from torchvision.transforms.functional import resize


def resize_along_longest_side(hw_shape, target_length):
    scale = float(target_length) / max(hw_shape)
    return int(scale * hw_shape[0] + 0.5), \
           int(scale * hw_shape[1] + 0.5)


def resize_image_along_longest_size(image, target_length, **kwargs):
    if isinstance(image, torch.Tensor):
        hw_shape = image.shape[-2:]
        return resize(image,
                      resize_along_longest_side(hw_shape, target_length),
                      antialias=None)
    else:
        raise TypeError(f'Invalid type of image: {type(image)}')


def resize_coord_along_longest_size(coord, hw_shape, target_length):
    ori_h, ori_w = hw_shape
    h, w = resize_along_longest_side(hw_shape, target_length)
    if isinstance(coord, np.ndarray):
        coord = np.copy(coord)
        coord[..., 0] = coord[..., 0] * (w / ori_w)
        coord[..., 1] = coord[..., 1] * (h / ori_h)
    elif isinstance(coord, torch.Tensor):
        coord = coord.clone()
        coord[..., 0] = coord[..., 0] * (w / ori_w)
        coord[..., 1] = coord[..., 1] * (h / ori_h)
    else:
        raise TypeError(f'Invalid type of coord: {type(coord)}')
    return coord


def resize_binary_mask_by_polygons(mask, target_length, ignore_index=255):
    if isinstance(mask, np.ndarray):
        if len(mask.shape) != 2:
            raise ValueError(f'Invalid shape of mask: {mask.shape}')
        label_ids = set(np.unique(mask).tolist())
        if not label_ids.issubset({0, 1, ignore_index}):
            raise ValueError(f'Invalid label ids: {label_ids} '
                             f'given ignore_index={ignore_index}')

        if ignore_index in label_ids:
            ignore_mask = (mask == ignore_index)
            mask = mask.copy()
            mask[ignore_mask] = 0

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hw_shape = mask.shape
        zoom_factor = float(target_length) / max(hw_shape)
        new_height, new_width = \
            resize_along_longest_side(hw_shape, target_length)
        upsampled_mask = np.zeros((new_height, new_width), dtype=np.uint8)
        for cnt in contours:
            if cnt.size > 0:
                cnt_upsampled = (cnt * zoom_factor).astype(int)
                cnt_upsampled[:, 0, 0] = np.clip(
                    cnt_upsampled[:, 0, 0], 0, new_width - 1)
                cnt_upsampled[:, 0, 1] = np.clip(
                    cnt_upsampled[:, 0, 1], 0, new_height - 1)
                cv2.drawContours(upsampled_mask, [cnt_upsampled], -1, 1, -1)

        if ignore_index in label_ids:
            ignore_mask = cv2.resize(
                ignore_mask.astype(np.uint8), (new_width, new_height),
                interpolation=cv2.INTER_NEAREST)
            upsampled_mask[ignore_mask] = ignore_index

        return upsampled_mask
    else:
        raise NotImplementedError(f'Invalid type of mask: {type(mask)}')
