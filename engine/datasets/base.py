import copy
import logging

import numpy as np
from mmengine.dataset import BaseDataset, Compose
from mmengine.logging import print_log


class InvalidInterSegSampleError(Exception):
    """Invalid InterSeg sample."""
    pass


class NotFoundValidInterSegSampleError(Exception):
    """Not found valid InterSeg sample."""
    pass


class BaseInterSegDataset(BaseDataset):

    default_meta_root = ''

    def __init__(self,
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
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.ignore_index = ignore_index
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.meta_root = meta_root or self.default_meta_root
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = None
        self.serialize_data = serialize_data
        self.max_refetch = max_refetch
        self.data_list = []
        self.data_bytes: np.ndarray

        self.pipeline = Compose(pipeline)
        if not lazy_init:
            self.full_init()

    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        info = None
        for _ in range(self.max_refetch):
            try:
                data = self.prepare_data(idx)
            except InvalidInterSegSampleError as error:
                info = str(error)
                continue
            return data
        else:
            raise NotFoundValidInterSegSampleError(
                f'Cannot find valid InterSeg sample after '
                f'{self.max_refetch} retries, due to {info}')

    def get_data_info(self, idx):
        data_info = super(BaseInterSegDataset, self).get_data_info(idx)
        data_info['dataset'] = \
            self.__class__.__name__.lower().replace('dataset', '')
        return data_info

    @property
    def metainfo(self):
        return dict()
