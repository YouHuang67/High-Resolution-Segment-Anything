from pathlib import Path

import mmengine
from mmengine.dist import get_dist_info
from engine.datasets import BaseInterSegDataset


class BaseFewShotInterSegDataset(BaseInterSegDataset):

    """
    Additional args:
        num_instances_percls (int):
            number of instances per class, if
            - > 0: include `num_instances_percls` previous instances in each class
            - < 0: exclude `num_instances_percls` previous instances in each class
    """

    def __init__(self,
                 num_instances_percls,
                 pipeline,
                 data_root,
                 meta_root=None,
                 img_suffix='.jpg',
                 ann_suffix='.png',
                 filter_cfg=None,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000,
                 ignore_index=255,
                 backend_args=None):
        if num_instances_percls == 0:
            raise ValueError('`num_instances_percls` should not be 0')
        self.num_instances_percls = num_instances_percls
        super(BaseFewShotInterSegDataset, self).__init__(
            pipeline=pipeline,
            data_root=data_root,
            meta_root=meta_root,
            img_suffix=img_suffix,
            ann_suffix=ann_suffix,
            filter_cfg=filter_cfg,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            ignore_index=ignore_index,
            backend_args=backend_args)

    def full_init(self):
        if self._fully_initialized:
            return
        self.data_list, self.categories = self.load_data_list_and_categorize()  # noqa
        self.data_list = self.filter_data()
        self.index_map = self.build_index_map()  # noqa
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()  # noqa

        self._fully_initialized = True

    def __getitem__(self, idx):
        idx = self.index_map[idx]
        return super(BaseFewShotInterSegDataset, self).__getitem__(idx)

    def __len__(self):
        return len(self.index_map)

    def load_data_list_and_categorize(self):
        data_list = self.load_data_list()
        meta_root = Path(self.meta_root)
        meta_info = mmengine.load(meta_root)
        if 'categories' in meta_info:
            categories = meta_info['categories']
        else:
            categories = dict()
            for idx, data_info in enumerate(data_list):
                if 'class_id' not in data_info:
                    raise ValueError(
                        f'`class_id` is not provided in {data_info}')
                class_id = data_info['class_id']
                if not isinstance(class_id, (int, str)):
                    raise ValueError(
                        f'`class_id` should be int or str, '
                        f'but got {class_id} of type {type(class_id)}')
                categories.setdefault(str(class_id), []).append(idx)
            if get_dist_info()[0] == 0:
                meta_info['categories'] = categories
                mmengine.dump(meta_info, meta_root)
        return data_list, categories

    def build_index_map(self):
        categories = dict()
        for class_id, idxs in self.categories.items():
            if self.num_instances_percls > 0:
                idxs = idxs[:self.num_instances_percls]
            elif self.num_instances_percls < 0:
                idxs = idxs[abs(self.num_instances_percls):]
            else:
                raise ValueError('`num_instances_percls` should not be 0')
            categories[class_id] = set(idxs)
            
        index_map = dict()
        valid_sample_index = 0
        for idx, data_info in enumerate(self.data_list):
            if idx in categories[str(data_info['class_id'])]:
                index_map[valid_sample_index] = idx
                valid_sample_index += 1
        return index_map

    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
             data_list (list[dict]): All data info of dataset, each info includes:
                - img_path (str):
                - seg_map_path (str):
                - seg_fields (list[str]):
                - reduce_zero_label (bool, optional):
                - segments_info (list[dict]):
                - class_id (int, str): additional info to indicate the class,
                                       allowing selection of prompt instances
        """
        raise NotImplementedError
