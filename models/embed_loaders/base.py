__all__ = ['BaseEmbedLoader', 'EMBED_LOADERS']
from pathlib import Path

import numpy as np
import torch

import mmengine
from mmengine.registry import Registry
from mmengine.dist import get_dist_info
from engine.segmentors import EmptyBackbone
from .utils import collect_strings, broadcast_strings


EMBED_LOADERS = Registry('embed_loader')


@EMBED_LOADERS.register_module()
class BaseEmbedLoader(object):

    def __init__(self,
                 backbone,
                 embed_dir,
                 suffix='.npy',
                 list_format=True,
                 update_prefixes_each_step=True,
                 string_broadcast_length=2048,
                 max_num_prefixes=-1):
        self.backbone = backbone
        self.embed_dir = Path(embed_dir)
        self.suffix = suffix
        self.list_format = list_format
        self.update_prefixes_each_step = update_prefixes_each_step
        self.string_broadcast_length = string_broadcast_length
        self.max_num_prefixes = max_num_prefixes
        self.rank, self.world_size = get_dist_info()
        self.embed_dir.mkdir(parents=True, exist_ok=True)
        if self.meta_file.is_file():
            self.prefixes = set(mmengine.load(self.meta_file)['prefixes'])
        else:
            self.prefixes = set()

    @torch.no_grad()
    def __call__(self, inputs, data_samples):
        embeds = [None for _ in data_samples]
        prefixes = set()
        not_found_idxs = []
        for idx, prefix in enumerate(map(self.to_prefix, data_samples)):
            if prefix in self.prefixes:
                embeds[idx] = torch.from_numpy(
                    np.load(self.prefix_to_embed_file(prefix))).to(inputs)
                prefixes.add(prefix)
            else:
                not_found_idxs.append(idx)

        max_num_prefixes = self.max_num_prefixes
        update_flag = self.update_prefixes_each_step \
            and (max_num_prefixes < 0 or len(self.prefixes) < max_num_prefixes)
        if len(not_found_idxs) > 0:
            if isinstance(self.backbone, EmptyBackbone):
                raise ValueError(
                    '`backbone` is EmptyBackbone, but not found embeds of {}'.
                    format([
                        data_samples[idx].metainfo["img_path"]
                        for idx in not_found_idxs])
                )
            not_found_idxs = torch.tensor(
                not_found_idxs, dtype=torch.long, device=inputs.device)
            not_found_embeds = self.backbone(inputs[not_found_idxs])
            if isinstance(not_found_embeds, (tuple, list)):
                if len(not_found_embeds) == 1:
                    not_found_embeds = not_found_embeds[0]
                else:
                    raise NotImplementedError(
                        f'Not support multiple embeds, '
                        f'got {len(not_found_embeds)} embeds.')
            elif not isinstance(not_found_embeds, torch.Tensor):
                raise NotImplementedError(
                    f'Not support type {type(not_found_embeds)} of embeds.')
            for idx, embed in zip(not_found_idxs, not_found_embeds.detach()):
                prefix = self.to_prefix(data_samples[idx])
                if update_flag:
                    np.save(
                        self.prefix_to_embed_file(prefix), embed.cpu().numpy())
                embeds[idx] = embed
                prefixes.add(prefix)
        else:
            update_flag = False

        if update_flag:
            self.update_prefixes(prefixes, inputs.device)

        embeds = torch.stack(embeds, dim=0)
        if self.list_format:
            embeds = [embeds]
        return embeds

    @staticmethod
    def to_prefix(data_sample):
        return Path(data_sample.metainfo['img_path']).stem

    def prefix_to_embed_file(self, prefix):
        return str(self.embed_dir / f'{prefix}{self.suffix}')

    @property
    def meta_file(self):
        return self.embed_dir / 'meta-info.json'

    def update_prefixes(self, prefixes, device):
        prefixes = list(prefixes)
        if self.world_size > 1:
            prefixes = collect_strings(prefixes)
            prefixes = broadcast_strings(prefixes)
        update_flag = False
        for prefix in prefixes:
            if prefix not in self.prefixes:
                update_flag = True
                self.prefixes.add(prefix)
        if update_flag:
            self.dump_meta_info()

    def dump_meta_info(self):
        if self.rank == 0 and len(self.prefixes) > 0:
            mmengine.dump(dict(prefixes=self.prefixes), self.meta_file)
