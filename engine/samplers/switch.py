import itertools
import random

import torch
from torch.utils.data import Sampler
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SwitchSampler(Sampler):

    # note: always shuffle, batch_size is the number of samples "per worker"
    def __init__(self, dataset, batch_size, seed=None):
        if not isinstance(dataset, _ConcatDataset):
            raise TypeError(f'dataset should be a ConcatDataset, '
                            f'but got {type(dataset)}')
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_batch_size = batch_size * world_size
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.sizes = tuple(map(len, dataset.datasets))
        if any(s <= self.total_batch_size for s in self.sizes):
            raise ValueError('batch_size * world_size should be smaller than '
                             'the number of samples of each dataset, but '
                             f'got datasets with sizes {self.sizes} '
                             f'for total_batch_size={self.total_batch_size}')
        self.cumulative_sizes = dataset.cumulative_sizes
        self.indices = self._indices_of_rank()

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        random.seed(self.seed)

        def grouper(iterable, n):
            return itertools.zip_longest(*[iter(iterable)] * n, fillvalue=None)

        while True:
            indices = [(s + torch.randperm(n, generator=g)).tolist()
                       for s, n in zip([0]+self.cumulative_sizes, self.sizes)]
            batch_indices = [list(grouper(idxs, self.total_batch_size))[:-1]
                             for idxs in indices]
            shuffled_batch_indices = []
            for batch_idxs in batch_indices:
                shuffled_batch_indices += batch_idxs
            random.shuffle(shuffled_batch_indices)
            shuffled_indices = []
            for idxs in shuffled_batch_indices:
                shuffled_indices += idxs
            yield from shuffled_indices

    def _indices_of_rank(self):
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.world_size)

    def __iter__(self):
        yield from self.indices

    def __len__(self):
        return sum(self.sizes)

    def set_epoch(self, epoch):
        pass
