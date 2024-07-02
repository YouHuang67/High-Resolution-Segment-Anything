from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np
import mmcv
import mmengine
from mmengine.dist import get_dist_info, barrier
from mmseg.datasets.transforms import LoadAnnotations, Resize
from mmseg.registry import TRANSFORMS, DATASETS

from .base import BaseFewShotInterSegDataset


@TRANSFORMS.register_module()
class LoadExtHQSeg44kAnnotations(LoadAnnotations):

    def _load_seg_map(self, results):
        with Image.open(results['seg_map_path'], 'r') as gt_seg_map:
            gt_seg_map = np.array(gt_seg_map, dtype=np.uint8)  # noqa

        gt_seg_map[gt_seg_map > 0] = 1
        results['gt_seg_map'] = gt_seg_map
        results['segments_info'] = [dict(id=1)]
        results['seg_fields'].append('gt_seg_map')


@TRANSFORMS.register_module()
class LoadExtHQSeg44kTrainAnnotations(LoadAnnotations):

    def __init__(self, binary_mask=True):
        super(LoadExtHQSeg44kTrainAnnotations, self).__init__()
        self.binary_mask = binary_mask

    def _load_seg_map(self, results):
        with Image.open(results['seg_map_path'], 'r') as gt_seg_map:
            gt_seg_map = np.array(gt_seg_map, dtype=np.uint8)  # noqa

        if len(gt_seg_map.shape) == 3:
            gt_seg_map = gt_seg_map[..., 0]
        elif len(gt_seg_map.shape) != 2:
            raise ValueError(
                f'Invalid shape of semantic segmentation label, '
                f'expected 2 or 3, got shape {gt_seg_map.shape}')

        if self.binary_mask:
            gt_seg_map[gt_seg_map > 0] = 1
            results['gt_seg_map'] = gt_seg_map
            results['segments_info'] = [dict(id=1)]
        else:
            results['gt_seg_map'] = gt_seg_map
            results['segments_info'] = []  # multi-value mask has no ids
        results['seg_fields'].append('gt_seg_map')


@TRANSFORMS.register_module()
class ResizeHQSeg44k(Resize):

    def __init__(self, target_size):
        self.target_size = target_size
        super(ResizeHQSeg44k, self).__init__(
             scale=(target_size, target_size),
             scale_factor=None, keep_ratio=True)

    def _resize_seg(self, results):
        for seg_key in results.get('seg_fields', []):
            if results.get(seg_key, None) is not None:
                if self.keep_ratio:
                    gt_seg, _ = mmcv.imrescale(
                        results[seg_key],
                        results['scale'],
                        interpolation=self.interpolation,
                        return_scale=True,
                        backend=self.backend)
                else:
                    raise NotImplementedError
                results[seg_key] = gt_seg

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(target_size={self.target_size})'
        return repr_str


@DATASETS.register_module()
class ExtHQSeg44kValDataset(BaseFewShotInterSegDataset):

    default_meta_root = 'data/meta-info/ext/hqseg44k-val.json'
    
    def __init__(self,
                 pipeline,
                 data_root='data/sam-hq',
                 subdirs=('thin_object_detection/COIFT',
                          'thin_object_detection/HRSOD',
                          'thin_object_detection/ThinObject5K/',  # will be filtered by 'train' in path
                          'DIS5K/DIS-VD'),
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        self.subdirs = subdirs
        super(ExtHQSeg44kValDataset, self).__init__(
            num_instances_percls=1000_000_000,
            pipeline=pipeline,
            data_root=data_root,
            img_suffix='.jpg',
            ann_suffix='.png',
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=255,
            backend_args=None
        )

    def load_data_list(self):
        data_root = Path(self.data_root)
        meta_root = Path(self.meta_root)
        
        if meta_root.is_file():
            data_list = mmengine.load(meta_root)['data_list']
        else:
            data_list = []
            for subdir in self.subdirs:
                img_files = {
                    p.stem: p for p in
                    (data_root / subdir).rglob(f'*{self.img_suffix}')
                    if 'train' not in str(p)}
                ann_files = {
                    p.stem: p for p in
                    (data_root / subdir).rglob(f'*{self.ann_suffix}')
                    if 'train' not in str(p)}
                prefixes = set(img_files.keys()) & set(ann_files.keys())
                for prefix in tqdm(sorted(img_files.keys())):
                    if prefix in prefixes:
                        data_list.append(dict(
                            img_path=str(img_files[prefix]),
                            seg_map_path=str(ann_files[prefix]),
                            seg_fields=[],
                            class_id=0))
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_root)
            barrier()
        return data_list


@DATASETS.register_module()
class ExtHQSeg44kTrainDataset(BaseFewShotInterSegDataset):

    default_meta_root = 'data/meta-info/ext/hqseg44k-train.json'

    def __init__(self,
                 pipeline,
                 data_root='data/sam-hq',
                 subdirs=('DIS5K/DIS-TR',
                          'thin_object_detection/ThinObject5K',
                          'cascade_psp/fss_all',
                          'cascade_psp/DUTS-TR',
                          'cascade_psp/DUTS-TE',
                          'cascade_psp/ecssd',
                          'cascade_psp/MSRA_10K'),
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        self.subdirs = subdirs
        super(ExtHQSeg44kTrainDataset, self).__init__(
            num_instances_percls=1000_000_000,
            pipeline=pipeline,
            data_root=data_root,
            img_suffix='.jpg',
            ann_suffix='.png',
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=255,
            backend_args=None
        )

    def load_data_list(self):
        data_root = Path(self.data_root)
        meta_root = Path(self.meta_root)

        if meta_root.is_file():
            data_list = mmengine.load(meta_root)['data_list']
        else:
            data_list = []
            for subdir in self.subdirs:
                img_files = {
                    p.stem: p for p in
                    (data_root / subdir).rglob(f'*{self.img_suffix}')
                    if 'images_test' not in str(p) and
                       'masks_test' not in str(p)}
                ann_files = {
                    p.stem: p for p in
                    (data_root / subdir).rglob(f'*{self.ann_suffix}')
                    if 'images_test' not in str(p) and
                       'masks_test' not in str(p)}
                prefixes = set(img_files.keys()) & set(ann_files.keys())
                for prefix in tqdm(sorted(img_files.keys())):
                    if prefix in prefixes:
                        data_list.append(dict(
                            img_path=str(img_files[prefix]),
                            seg_map_path=str(ann_files[prefix]),
                            seg_fields=[],
                            class_id=0))
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_root)
            barrier()
        return data_list
