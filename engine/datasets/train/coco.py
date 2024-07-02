import warnings
from pathlib import Path
from copy import deepcopy

import mmengine
import numpy as np
from mmengine.dist import get_dist_info, barrier
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.registry import TRANSFORMS, DATASETS

from ..base import BaseInterSegDataset


@TRANSFORMS.register_module()
class LoadCOCOPanopticAnnotations(LoadAnnotations):

    def _load_seg_map(self, results):
        super(LoadCOCOPanopticAnnotations, self)._load_seg_map(results)
        gt_seg_map = results.pop('gt_seg_map')
        segments_info = results.pop('segments_info')
        gt_seg_map = (gt_seg_map[..., 0] * (1 << 16) +
                      gt_seg_map[..., 1] * (1 << 8) +
                      gt_seg_map[..., 2] * (1 << 0))
        results['gt_seg_map'] = np.zeros_like(gt_seg_map).astype(np.uint8)
        results['segments_info'] = []
        for i, info in enumerate(segments_info, 1):
            results['gt_seg_map'][gt_seg_map == info['id']] = i
            info = deepcopy(info)
            info['id'] = i
            results['segments_info'].append(info)


@DATASETS.register_module()
class COCOPanopticDataset(BaseInterSegDataset):

    default_meta_root = 'data/meta-info/coco.json'
    default_ignore_file = 'data/meta-info/coco_invalid_image_names.json'

    def __init__(self,
                 data_root,
                 pipeline,
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
        super(COCOPanopticDataset, self).__init__(
            data_root=data_root,
            pipeline=pipeline,
            img_suffix='.jpg',
            ann_suffix='.png',
            filter_cfg=None,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=255,
            backend_args=None)

    def load_data_list(self):
        data_root = Path(self.data_root)
        meta_root = Path(self.meta_root)
        if meta_root.is_file():
            data_list = mmengine.load(meta_root)['data_list']
        else:
            data_list = []
            img_files = {int(p.stem): p for p in
                         data_root.rglob(f'*{self.img_suffix}')}
            ann_files = {int(p.stem): p for p in
                         data_root.rglob(f'*{self.ann_suffix}')}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            prefixes = prefixes - self.ignore_sample_indices

            ann_infos = mmengine.load(
                next(data_root.rglob(f'*panoptic_train2017.json'))
            )['annotations']
            for info in ann_infos:
                prefix = int(Path(info.pop('file_name')).stem)
                if prefix in prefixes:
                    data_list.append(
                        dict(img_path=str(img_files[prefix]),
                             seg_map_path=str(ann_files[prefix]),
                             seg_fields=[], reduce_zero_label=False, **info)
                    )
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_root)
            barrier()
        return data_list

    def get_data_info(self, idx):
        data_info = super(BaseInterSegDataset, self).get_data_info(idx)
        data_info['dataset'] = 'coco'
        return data_info
