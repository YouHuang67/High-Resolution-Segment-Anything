import numpy as np
import warnings

import mmcv
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.image.geometric import rescale_size
from mmseg.registry import TRANSFORMS

from engine.utils.geometric import resize_binary_mask_by_polygons


@TRANSFORMS.register_module()
class ResizeByPolygons(MMCV_Resize):

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        for seg_key in results.get('seg_fields', []):
            if results.get(seg_key, None) is not None:
                if self.keep_ratio:
                    gt_seg = self._resize_seg_by_polygons(
                        results[seg_key], results['scale'])
                else:
                    raise NotImplementedError
                results[seg_key] = gt_seg

    def _resize_seg_by_polygons(self, gt_seg, scale):
        h, w = gt_seg.shape[:2]
        new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
        new_w, new_h = new_size
        resized_gt_seg = np.zeros((new_h, new_w) + gt_seg.shape[2:],
                                  dtype=gt_seg.dtype)

        label_ids = set(np.unique(gt_seg).tolist())
        if len(label_ids) > 2:
            warnings.warn(f'Multiple label ids: {label_ids}')
        elif len(label_ids) == 1:
            warnings.warn(f'Only one label id: {label_ids}, '
                          f'use simply resized gt_seg')
            return mmcv.imrescale(
                gt_seg, scale, interpolation='nearest', backend=self.backend)
        for label_id in sorted(list(label_ids)):
            if label_id != 0:
                mask = (gt_seg == label_id)
                resized_mask = resize_binary_mask_by_polygons(
                    mask, new_size, scale_factor)
                resized_gt_seg[resized_mask] = label_id
        if not resized_gt_seg.any():
            warnings.warn(
                f'Empty gt_seg after resize, use simply resized gt_seg')
            return mmcv.imrescale(
                gt_seg, scale, interpolation='nearest', backend=self.backend)
        return resized_gt_seg
