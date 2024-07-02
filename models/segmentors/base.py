import warnings
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from mmengine.config import ConfigDict
from mmengine.utils.misc import is_list_of
from engine.utils.click_fast import (CLK_POSITIVE, fast_generate_single_click,
                                     fast_generate_clicks)
from engine.utils.resize import resize_along_longest_side
from engine.utils.resize import resize_image_along_longest_size
from engine.utils.resize import resize_coord_along_longest_size
from engine.segmentors import BaseInterSegmentor, EmptyBackbone
from ..embed_loaders import EMBED_LOADERS


class BaseClickSegmentor(BaseInterSegmentor):

    """
    Implement helper functions for ClickSegmentor.
    """

    def __init__(self,
                 backbone,
                 neck,
                 decode_head,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None,
                 remove_backbone=False,
                 freeze_backbone=True,
                 freeze_neck=True,
                 freeze_decode_head=False,
                 image_embed_loader=None):

        super(BaseClickSegmentor, self).__init__(
            backbone=backbone,
            neck=neck,
            decode_head=decode_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            remove_backbone=remove_backbone)

        self.freeze_backbone = freeze_backbone
        self.freeze_neck = freeze_neck
        self.freeze_decode_head = freeze_decode_head

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif backbone is None or remove_backbone:
            raise ValueError('Try to train backbone, but `backbone` is None or '
                             'set `remove_backbone = True`')
        if freeze_neck:
            for param in self.neck.parameters():
                param.requires_grad = False
        if freeze_decode_head:
            for param in self.decode_head.parameters():
                param.requires_grad = False
        if image_embed_loader is not None:
            self.image_embed_loader = EMBED_LOADERS.build(
                image_embed_loader, default_args=dict(backbone=self.backbone))
        else:
            self.image_embed_loader = None

    def interact_train(self, inputs, data_samples, prompts):  # noqa
        raise NotImplementedError

    def interact_test(self, inputs, data_samples, prompts):  # noqa
        raise NotImplementedError

    def parse_train_cfg(self, dataset):
        cfg = self.train_cfg
        if hasattr(cfg, 'interact_params'):
            interact_params = cfg.interact_params
        else:
            warnings.warn(f'Not found interact_params in train_cfg')
            interact_params = dict()
        if dataset in interact_params:
            params = interact_params[dataset]
            max_num_clicks = params.get('max_num_clicks', cfg.max_num_clicks)
            gamma = params.get('gamma', cfg.gamma)
        else:
            warnings.warn(f'Not found interact_params of {dataset}')
            max_num_clicks = cfg.max_num_clicks
            gamma = cfg.gamma
        if hasattr(cfg, 'sfc_inner_k'):
            sfc_inner_k = cfg.sfc_inner_k
        else:
            sfc_inner_k = 1.7
        return ConfigDict(max_num_clicks=max_num_clicks,
                          gamma=gamma, sfc_inner_k=sfc_inner_k)

    @staticmethod
    def check_gt_validity(data_samples, train):
        gt_sem_seg = torch.stack([
            sample.gt_sem_seg.data for sample in data_samples], dim=0)
        if train:
            # Check if gt_semantic_seg is multi-label
            if gt_sem_seg.ndim == 4 and gt_sem_seg.size(1) != 1:
                raise ValueError(f'Cannot handle multi `gt_sem_seg` '
                                 f'with shape {tuple(gt_sem_seg.shape)}')
            elif gt_sem_seg.ndim not in (3, 4):
                raise ValueError(f'`gt_sem_seg` is expected to have '
                                 f'shape (batch_size, height, width) or '
                                 f'(batch_size, 1, height, width), but '
                                 f'got shape {tuple(gt_sem_seg.shape)}')
        else:
            # Check if only one sample is present in the batch.
            if gt_sem_seg.size(0) > 1:
                raise ValueError(f'Only a single sample per batch is allowed, '
                                 f'but got {gt_sem_seg.size(0)} samples '
                                 f'in this batch')
            # Check if gt_semantic_seg has the correct shape.
            if gt_sem_seg.ndim not in (3, 4) or \
                    gt_sem_seg[..., 0, 0].nelement() > 1:
                raise ValueError(f'`gt_sem_seg` is expected to have '
                                 f'shape (1, height, width) or '
                                 f'(1, 1, height, width), but '
                                 f'got shape {tuple(gt_sem_seg.shape)}')
        if gt_sem_seg.ndim == 3:
            gt_sem_seg = gt_sem_seg.unsqueeze(1)
        return gt_sem_seg

    @staticmethod
    def sample_num_clicks(max_num_clicks, gamma):
        probs = gamma ** np.arange(max_num_clicks + 1)
        probs /= probs.sum()
        return np.random.choice(range(len(probs)), p=probs)

    def preprocess_inputs(self, inputs, data_samples):
        if self.image_embed_loader is not None:
            if not self.freeze_backbone:
                warnings.warn(f'backbone is trainable, but '
                              f'`image_embed_loader` is not None, '
                              f'`backbone` will be ignored.')
            return self.image_embed_loader(inputs, data_samples)
        else:
            if isinstance(self.backbone, EmptyBackbone):
                raise ValueError(f'`image_embed_loader` is None, '
                                 f'but `backbone` is EmptyBackbone.')
            if self.freeze_backbone:
                with torch.no_grad():
                    return self.backbone(inputs)
            else:
                return self.backbone(inputs)

    def encode_decode(self, inputs, data_samples,
                      image_embeds=None, **inter_info):
        if image_embeds is None:
            image_embeds = self.preprocess_inputs(inputs, data_samples)
        logits = self.decode_head(self.neck(image_embeds, **inter_info))
        return logits

    @staticmethod
    def redistribute_tensor(inputs, data_samples):
        """Redistribute tensor inputs and data_samples according to dataset"""
        inputs_dict, data_samples_dict, sample_idxs2idxs = {}, {}, {}
        for idx, (x, data_sample) in enumerate(zip(inputs, data_samples)):
            sample_idx = data_sample.metainfo['sample_idx']
            sample_idxs2idxs[sample_idx] = idx
            dataset = data_sample.metainfo['dataset']
            inputs_dict.setdefault(dataset, []).append(x)
            data_samples_dict.setdefault(dataset, []).append(data_sample)
        inputs_dict = {dataset: torch.stack(inputs, dim=0)
                       for dataset, inputs in inputs_dict.items()}
        return inputs_dict, data_samples_dict, sample_idxs2idxs

    @staticmethod
    def merge_tensors(x_list):
        if not isinstance(x_list, (list, tuple)):
            raise TypeError(
                f'x_list must be list or tuple, but got {type(x_list)}')
        x_list = list(x_list)
        if isinstance(x_list[0], torch.Tensor):
            assert is_list_of(x_list, torch.Tensor)
            return torch.cat(x_list, dim=0)
        elif isinstance(x_list[0], (list, tuple)):
            assert is_list_of(x_list, (list, tuple))
            return list(map(partial(torch.cat, dim=0), zip(*x_list)))
        elif isinstance(x_list[0], dict):
            assert is_list_of(x_list, dict)
            return {k: torch.cat([x[k] for x in x_list], dim=0)
                    for k in x_list[0].keys()}
        else:
            raise NotImplementedError(f'Unknown type {type(x_list[0])}')

    @staticmethod
    def split_tensors(x, split, dim=0):
        split = partial(torch.split, split_size_or_sections=split, dim=dim)
        if isinstance(x, torch.Tensor):
            return split(x)
        elif isinstance(x, (list, tuple)):
            return list(zip(*map(split, x)))
        elif isinstance(x, dict):
            # todo
            raise NotImplementedError(f'Not support type {type(x)}')
        else:
            raise NotImplementedError(f'Unknown type {type(x)}')

    @staticmethod
    def update_clicks(pre_label,
                      seg_label,
                      points_list,
                      sfc_inner_k=1.0,
                      downsample_factor=1.0,
                      ignore_masks=None):

        ori_pre_label = pre_label.detach().clone()
        ori_seg_label = seg_label.detach().clone()

        if points_list is None:
            points_list = [None for _ in pre_label]
        ori_points_list = deepcopy(points_list)
        scale = 1.0
        points_list = deepcopy(ori_points_list)
        if downsample_factor > 1.0:
            warnings.warn(f'`downsample_factor` > 1.0, ignored')
        elif downsample_factor < 1.0:
            scale = 1.0 / downsample_factor
            ori_h, ori_w = pre_label.shape[-2:]
            tar_h = int(ori_h * downsample_factor)
            tar_w = int(ori_w * downsample_factor)
            pre_label = F.interpolate(
                pre_label.float(), size=(tar_h, tar_w),
                mode='bilinear', align_corners=False)
            pre_label = (pre_label > 0.5).long()
            seg_label = F.interpolate(
                seg_label.float(), size=(tar_h, tar_w),
                mode='bilinear', align_corners=False)
            seg_label = (seg_label > 0.5).long()

            def clamp(y_, x_):
                return max(0, min(y_, pre_label.shape[-2] - 1)), \
                    max(0, min(x_, pre_label.shape[-1] - 1))

            for points in points_list:
                if points is not None:
                    for idx, (y, x, mode) in list(enumerate(points)):
                        points[idx] = \
                            clamp(
                                int(downsample_factor * y),
                                int(downsample_factor * x)
                            ) + (mode,)

        clicks = fast_generate_clicks(
            pre_labels=pre_label, seg_labels=seg_label,
            points_list=points_list, sfc_inner_k=sfc_inner_k,
            ignore_masks=ignore_masks)
        points_list = deepcopy(ori_points_list)
        for idx, (y, x, mode) in enumerate(clicks):
            if mode is not None:
                if points_list[idx] is None:
                    points_list[idx] = []
                points_list[idx].append((int(scale * y), int(scale * x), mode))
            elif points_list[idx] is None:
                warnings.warn(
                    f'No clicks generated for sample {idx}, '
                    f'please increase the downsample_factor '
                    f'{downsample_factor}')
                if ignore_masks is None:
                    ignore_mask = None
                else:
                    ignore_mask = ignore_masks[idx].squeeze()
                y, x, mode = fast_generate_single_click(
                    pre_label=ori_pre_label[idx].squeeze(),
                    seg_label=ori_seg_label[idx].squeeze(),
                    sfc_inner_k=sfc_inner_k,
                    ignore_mask=ignore_mask)
                if mode is None:
                    raise ValueError(f'Still not found valid clicks in '
                                     f'sample {idx} for the original size, '
                                     f'please check the input data')
                points_list[idx] = [(y, x, mode)]
        return points_list

    @staticmethod
    def point_lists_to_coords(
            point_lists, device, scale=1.0, max_num_points=0):
        coords, labels = list(), list()
        for points in point_lists:
            coord = torch.Tensor([[x, y] for y, x, _ in points]).view(1, -1, 2)
            label = torch.Tensor([
                int(mode == CLK_POSITIVE) for *_, mode in points]).view(1, -1)
            coords.append(coord)
            labels.append(label)
        max_num_points = max(
            [len(points) for points in point_lists] + [max_num_points])
        if max_num_points > 0:
            for idx, (coord, label) in list(enumerate(zip(coords, labels))):
                if max_num_points > coord.size(1):
                    coord_pad = torch.zeros(
                        1, max_num_points - coord.size(1), 2).to(coord)
                    coords[idx] = torch.cat([coord, coord_pad], dim=1)
                    label_pad = -torch.ones(
                        1, max_num_points - label.size(1)).to(label)
                    labels[idx] = torch.cat([label, label_pad], dim=1)
            coords = torch.cat(coords, dim=0).float().to(device)
            labels = torch.cat(labels, dim=0).to(device)
        else:
            coords = torch.zeros(len(point_lists), 1, 2).float().to(device)
            labels = -torch.ones(len(point_lists), 1).to(device)
        coords = coords * scale
        return coords, labels

    @staticmethod
    def resize_and_pad_to_target_size(x, target_size):
        x = resize_image_along_longest_size(x, target_size)
        h, w = x.shape[-2:]
        x = F.pad(x, (0, target_size - w, 0, target_size - h))
        return x

    def crop_and_resize_to_original_size(
            self, x, ori_hw, target_size, mode='bilinear'):
        h, w = resize_along_longest_side(ori_hw, target_size)
        x = self.interpolate(x[..., :h, :w], ori_hw, mode=mode)
        return x

    @staticmethod
    def resize_coord_to_target_size(points, cur_hw, target_size, device):
        coords = torch.Tensor([[[x, y] for y, x, _ in points]]).to(device)
        coords = resize_coord_along_longest_size(coords, cur_hw, target_size)
        labels = torch.Tensor([
            [int(mode == CLK_POSITIVE) for *_, mode in points]]).to(device)
        return coords, labels

    def _forward(self, inputs, data_samples=None):
        raise NotImplementedError

    def extract_feat(self, inputs):
        raise NotImplementedError

    def predict(self, inputs, data_samples=None):
        raise NotImplementedError

    def intertest_init(self):
        return

    def intertest_predict(self,
                          inputs,
                          data_samples,
                          resized_padded_inputs,
                          image_embeds,
                          ori_image_embeds,
                          step,
                          prev_logits,
                          points):
        raise NotImplementedError

    @staticmethod
    def copy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone()
        elif isinstance(x, (list, tuple)):
            out = [_.detach().clone() for _ in x]
            if isinstance(x, tuple):
                out = tuple(out)
            return out
        elif isinstance(x, dict):
            return {k: v.detach().clone() for k, v in x.items()}
        else:
            raise NotImplementedError(f'Unknown type {type(x)}')
