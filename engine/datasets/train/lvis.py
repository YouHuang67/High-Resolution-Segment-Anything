import random
import warnings
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import torch

import mmengine
from mmengine.dist import get_dist_info, barrier
from mmengine.utils.misc import to_2tuple
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS, DATASETS

from ..base import BaseInterSegDataset


@TRANSFORMS.register_module()
class LoadLVISAnnotations(LoadAnnotations):

    def __init__(self, max_overlap_ratio=0.5, scale_factor=None):
        super(LoadLVISAnnotations, self).__init__()
        self.max_overlap_ratio = max_overlap_ratio
        self.scale_factor = scale_factor

    def _load_seg_map(self, results):
        size = results['img_shape'][:2]
        gt_seg_map = np.zeros(size, dtype=np.uint8)
        ori_segments_info = results.pop('segments_info', None)
        if ori_segments_info is None:
            segments_info_file = results.pop('segments_info_file', None)
            if segments_info_file is None:
                raise ValueError(
                    f'Not found segments_info or segments_info_file for '
                    f'image {results["img_path"]}')
            segments_info = mmengine.load(segments_info_file)
            ori_segments_info = segments_info['segments_info']
        if 'scale_factor' in results:
            w_scale, h_scale = results['scale_factor']
            if self.scale_factor is not None:
                scale_factor = to_2tuple(self.scale_factor)
                if w_scale != scale_factor[0] or h_scale != scale_factor[1]:
                    warnings.warn(
                        f'LoadLVISAnnotations: '
                        f'scale_factor {scale_factor} is not consistent '
                        f'with the one in results {results["scale_factor"]}, '
                        f'use the latter.')
        else:
            w_scale, h_scale = 1.0, 1.0
            if self.scale_factor is not None:
                warnings.warn(f'LoadLVISAnnotations: '
                              f'The image has not been resized, '
                              f'but scale_factor {self.scale_factor} is set, '
                              f'ignore it.')

        segments_info = []
        areas = dict()
        random.shuffle(ori_segments_info)
        for idx, info in enumerate(ori_segments_info, 1):
            mask = self.polygons2mask(
                info['segmentation'], size, w_scale, h_scale)
            area = mask.astype(float).sum()
            if area < torch.finfo(torch.float).eps:
                continue
            mask = mask.astype(bool)
            overlap_areas = np.bincount(gt_seg_map[mask].flatten())
            overlap_ratio = overlap_areas[1:].sum() / area
            overlap_ratio = max(
                [overlap_ratio] +
                [overlap_areas[i] / areas[i]
                 for i in areas if i < len(overlap_areas)])
            if overlap_ratio > self.max_overlap_ratio:
                continue
            areas[idx] = area
            gt_seg_map[mask] = idx
            info = deepcopy(info)
            info.update(dict(id=idx))
            segments_info.append(info)

        results['gt_seg_map'] = gt_seg_map
        results['seg_fields'].append('gt_seg_map')
        results['segments_info'] = segments_info

    @staticmethod
    def polygons2mask(polygons, size, w_scale, h_scale):
        """
        Static method to convert polygons to binary mask.

        :param polygons: List of polygons.
        :param size: Size of the image.
        :param w_scale: Scale factor of width.
        :param h_scale: Scale factor of height.

        :return: array containing the binary mask.
        """
        mask = np.zeros(size, dtype=np.uint8)
        scale_factor = np.array([w_scale, h_scale]).reshape((1, 2))
        for polygon in polygons:
            points = np.array(polygon).reshape((-1, 2))
            points = np.round(points * scale_factor).astype(np.int32)
            cv2.fillPoly(mask, points[None, ...], 1)
        return mask

    def __repr__(self):
        return f'{self.__class__.__name__}' \
               f'(max_overlap_ratio={self.max_overlap_ratio}, ' \
               f'scale_factor={self.scale_factor})'


@DATASETS.register_module()
class LVISDataset(BaseInterSegDataset):

    default_meta_file = 'data/meta-info/lvis.json'
    default_meta_root = 'data/meta-info/lvis-segments-infos'
    default_ignore_file = 'data/meta-info/coco_invalid_image_names.json'  # shared with COCO

    def __init__(self,
                 data_root,
                 pipeline,
                 meta_file=None,
                 meta_root=None,
                 ignore_file=None,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        if ignore_file is None:
            ignore_file = self.default_ignore_file
        if Path(ignore_file).is_file():
            self.ignore_sample_indices = \
                set(map(int, mmengine.load(ignore_file)['train']))
        else:
            warnings.warn(f'Not found ignore_file {ignore_file}')
            self.ignore_names = set()
        self.meta_file = meta_file or self.default_meta_file
        super(LVISDataset, self).__init__(
            data_root=data_root,
            pipeline=pipeline,
            meta_root=meta_root,
            img_suffix='.jpg',
            ann_suffix=None,
            filter_cfg=None,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=255,
            backend_args=None)

    def load_data_list(self):
        data_root = Path(self.data_root)
        meta_file = Path(self.meta_file)
        if meta_file.is_file():
            data_list = mmengine.load(meta_file)['data_list']
        else:
            meta_root = Path(self.meta_root)
            data_list = []
            img_files = {
                int(p.stem): p for p in
                (data_root / 'train2017').rglob(f'*{self.img_suffix}')}
            prefixes = set(img_files.keys()) - self.ignore_sample_indices

            ann_infos = dict()
            meta_infos = mmengine.load(
                next(data_root.rglob(f'lvis_v1_train.json'))
            )
            for info in meta_infos['annotations']:
                if info['image_id'] in img_files:
                    ann_infos.setdefault(info['image_id'], list()).append(info)

            for info in meta_infos['images']:
                prefix = info['id']
                if prefix in prefixes and prefix in ann_infos:
                    segments_info_file = str(meta_root / f'{prefix}.json')
                    if get_dist_info()[0] == 0:
                        mmengine.dump(dict(segments_info=ann_infos[prefix]),
                                      segments_info_file)
                    data_list.append(
                        dict(img_path=str(img_files[prefix]),
                             seg_map_path=None,
                             segments_info_file=segments_info_file,
                             seg_fields=[], reduce_zero_label=False)
                    )
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_file)
            barrier()
        return data_list
