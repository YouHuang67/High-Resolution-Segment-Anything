import warnings
import numpy as np
import numpy.random as rng
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

from engine.datasets.base import InvalidInterSegSampleError


@TRANSFORMS.register_module()
class ObjectSampler(BaseTransform):

    def __init__(self,
                 max_num_merged_objects=1,
                 min_area_ratio=0.0,
                 ignore_index=None,
                 merge_prob=0.0,
                 include_other=False,
                 max_retry=100):
        self.max_num_merged_objects = max_num_merged_objects
        self.min_area_ratio = min_area_ratio
        self.ignore_index = ignore_index
        self.merge_prob = merge_prob
        self.include_other = include_other
        self.max_retry = max_retry

    def transform(self, results):
        gt_seg_map = results.pop('gt_seg_map')
        segments_info = []
        for info in results.pop('segments_info'):
            if np.any(gt_seg_map == info['id']):
                segments_info.append(info)
        if len(segments_info) == 0:
            raise InvalidInterSegSampleError('Not found valid annotation')

        merge_prob = self.merge_prob
        ignore_index = self.ignore_index
        min_area_ratio = self.min_area_ratio
        num_objects = len(segments_info)
        max_num_merged_objects = max(self.max_num_merged_objects, num_objects)
        for _ in range(self.max_retry):
            seg_label = np.zeros_like(gt_seg_map)
            object_idxs = rng.permutation(range(num_objects))
            if max_num_merged_objects > 1 and rng.rand() < merge_prob:
                num_merged_objects = rng.randint(2, max_num_merged_objects + 1)
            else:
                num_merged_objects = 1
            for idx in object_idxs[:num_merged_objects]:
                seg_label[gt_seg_map == segments_info[idx]['id']] = 1
            if np.mean(seg_label) > min_area_ratio:
                if ignore_index is not None:
                    seg_label[gt_seg_map == ignore_index] = ignore_index
                if self.include_other:
                    for idx in object_idxs[num_merged_objects:]:
                        seg_label[gt_seg_map == segments_info[idx]['id']] = 2
                results['gt_seg_map'] = seg_label
                return results
        else:
            raise InvalidInterSegSampleError('Failed to sample valid objects')

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'max_num_merged_objects={self.max_num_merged_objects}, ' \
               f'min_area_ratio={self.min_area_ratio}, ' \
               f'ignore_index={self.ignore_index}, ' \
               f'merge_prob={self.merge_prob}, ' \
               f'include_other={self.include_other}, ' \
               f'max_retry={self.max_retry})'


@TRANSFORMS.register_module()
class MultiObjectSampler(BaseTransform):

    """
    Sample multiple objects from the gt_seg_map,
    each represented by a unique label within the same gt_seg_map.
    """

    def __init__(self, num_objects, min_area_ratio=0.0):
        self.num_objects = num_objects
        self.min_area_ratio = min_area_ratio

    def transform(self, results):
        gt_seg_map = results.pop('gt_seg_map')
        segments_info = []
        for info in results.pop('segments_info'):
            if np.any(gt_seg_map == info['id']):
                segments_info.append(info)
        if len(segments_info) == 0:
            raise InvalidInterSegSampleError('Not found valid annotation')

        num_objects = 0
        seg_label = np.zeros_like(gt_seg_map)
        for idx in rng.permutation(len(segments_info)):
            mask = (gt_seg_map == segments_info[idx]['id'])
            if mask.astype(np.float32).mean() > self.min_area_ratio:
                num_objects += 1
                seg_label[mask] = num_objects
                if num_objects >= self.num_objects:
                    break
        else:
            warnings.warn(f'Failed to sample {self.num_objects} objects, '
                          f'only sampled {num_objects} objects.')
        if num_objects > 0:
            results['gt_seg_map'] = seg_label
            return results
        else:
            raise InvalidInterSegSampleError('Failed to sample valid objects')

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'num_objects={self.num_objects}, ' \
               f'min_area_ratio={self.min_area_ratio})'
