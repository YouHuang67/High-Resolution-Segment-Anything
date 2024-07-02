from pathlib import Path

import mmengine
from mmengine.dist import get_dist_info, barrier
from mmseg.registry import DATASETS

from ..base import BaseInterSegDataset


@DATASETS.register_module()
class DAVISDataset(BaseInterSegDataset):

    default_meta_root = 'data/meta-info/davis.json'

    def __init__(self,
                 data_root,
                 pipeline,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        super(DAVISDataset, self).__init__(
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
            img_files = {f'{p.stem}#001': p for p in
                         data_root.rglob(f'*{self.img_suffix}')}
            ann_files = {p.stem: p for p in
                         data_root.rglob(f'*{self.ann_suffix}')}
            prefixes = set(img_files.keys()) & set(ann_files.keys())
            for prefix in sorted(list(prefixes)):
                data_list.append(
                    dict(img_path=str(img_files[prefix]),
                         seg_map_path=str(ann_files[prefix]),
                         seg_fields=[], segments_info=[dict(id=1)],
                         reduce_zero_label=False)
                )
            if get_dist_info()[0] == 0:
                mmengine.dump(dict(data_list=data_list), meta_root, indent=4)
            barrier()
        return data_list
