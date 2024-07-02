import cv2
import numpy as np
import warnings


def resize_binary_mask_by_polygons(mask, new_size, scale_factor):
    """
    :param mask: shape (ori_h, ori_w), only include binary values
    :param new_size: (new_w, new_h)
    :param scale_factor: float or int
    :return:
    """
    if isinstance(mask, np.ndarray):
        if len(mask.shape) != 2:
            raise ValueError(f'Invalid shape of mask: {mask.shape}')
        label_ids = set(np.unique(mask).tolist())
        if not label_ids.issubset({0, 1}):  # only include binary values
            raise ValueError(f'Invalid label ids: {label_ids}')

        new_w, new_h = new_size
        new_mask = np.zeros((new_h, new_w)).astype(np.uint8)
        if len(label_ids) == 1:
            warnings.warn(f'Only one label id: {label_ids}')
            new_mask[...] = label_ids.pop()
            return new_mask

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cnt.size > 0:
                cnt_resized = (cnt * scale_factor).astype(int)
                cnt_resized[:, 0, 0] = np.clip(
                    cnt_resized[:, 0, 0], 0, new_w - 1)
                cnt_resized[:, 0, 1] = np.clip(
                    cnt_resized[:, 0, 1], 0, new_h - 1)
                cv2.drawContours(new_mask, [cnt_resized], -1, 1, -1)

        return new_mask.astype(mask.dtype)
    else:
        raise NotImplementedError(f'Invalid type of mask: {type(mask)}')
