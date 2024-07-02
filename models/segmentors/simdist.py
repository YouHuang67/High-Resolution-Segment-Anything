import warnings
from collections import OrderedDict

import torch
from mmseg.registry import MODELS
from mmseg.utils import add_prefix

from engine.timers import Timer
from engine.utils import repeat
from .simseg import SimpleSegmentor


@MODELS.register_module()
class SimpleDistillation(SimpleSegmentor):

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

        if remove_backbone:
            warnings.warn('remove_backbone is True, '
                          'but it is not supported in SimpleDistillation')
            remove_backbone = False

        if freeze_backbone:
            warnings.warn('freeze_backbone is True, '
                          'but it is not supported in SimpleDistillation')
            freeze_backbone = False

        super(SimpleDistillation, self).__init__(
            backbone=backbone,
            neck=neck,
            decode_head=decode_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            remove_backbone=remove_backbone,
            freeze_backbone=freeze_backbone,
            freeze_neck=freeze_neck,
            freeze_decode_head=freeze_decode_head,
            image_embed_loader=image_embed_loader)

    @Timer('SimpleDistillationTrain')
    def interact_train(self, inputs, data_samples, prompts=None):
        if self.image_embed_loader is None:
            raise ValueError('image_embed_loader should be provided')

        inputs_dict, data_samples_dict, _ = \
            self.redistribute_tensor(inputs, data_samples)
        if len(inputs_dict) > 1:
            raise ValueError(f'Not support multiple datasets, '
                             f'but got {list(inputs_dict.keys())}')
        dataset = next(iter(inputs_dict.keys()))
        inputs, data_samples = inputs_dict[dataset], data_samples_dict[dataset]

        target_embeds = self.image_embed_loader(inputs, data_samples)
        image_embeds = self.backbone(inputs)

        if hasattr(self.train_cfg, 'repeat_ratio'):
            K = self.train_cfg.repeat_ratio + 1
            if isinstance(image_embeds, torch.Tensor):
                target_embeds = repeat(
                    target_embeds, 'b ... -> (k b) ...', k=K)
            elif isinstance(image_embeds, (tuple, list)):
                target_embeds = [
                    repeat(embed, 'b ... -> (k b) ...', k=K)
                    for embed in target_embeds]
            else:
                raise NotImplementedError

        losses = OrderedDict()
        if isinstance(image_embeds, torch.Tensor):
            losses['mse_loss'] = (
                (image_embeds - target_embeds).pow(2).flatten(1).mean(dim=-1))
        elif isinstance(image_embeds, (tuple, list)):
            losses['mse_loss'] = sum(
                (embed - target_embed).pow(2).flatten(1).mean(dim=-1)
                for embed, target_embed in zip(image_embeds, target_embeds))
        else:
            raise NotImplementedError
        return losses

    @torch.no_grad()
    @Timer('SimpleSegmentorTest')
    def interact_test(self, inputs, data_samples):
        cfg = self.test_cfg
        gt_sem_seg = self.check_gt_validity(data_samples, train=False)

        if hasattr(cfg, 'sfc_inner_k'):
            sfc_inner_k = cfg.sfc_inner_k
        else:
            sfc_inner_k = 1.0

        if isinstance(self.test_cfg.target_size, (int, float)):
            target_size = int(self.test_cfg.target_size)
        elif isinstance(self.test_cfg.target_size, (list, tuple)):
            max_length = max(inputs.shape[-2:])
            for ts in sorted(list(self.test_cfg.target_size)):
                if max_length <= ts:
                    target_size = ts
                    break
            else:
                target_size = max(self.test_cfg.target_size)
        else:
            raise ValueError(
                f'test_cfg.target_size should be int or list of int, '
                f'but got {type(self.test_cfg.target_size)}')

        self.eval()
        resized_padded_inputs = \
            self.resize_and_pad_to_target_size(inputs, target_size)
        image_embeds = self.backbone(resized_padded_inputs)
        ori_image_embeds = image_embeds
        pre_labels = torch.zeros_like(gt_sem_seg)
        seg_labels = gt_sem_seg
        prev_logits = None

        self.intertest_init()
        results, points, prev_logits_list = [], None, []
        for step in range(cfg.num_clicks):
            points, *_ = self.update_clicks(
                pre_labels, seg_labels, [points], sfc_inner_k)
            image_embeds, logits = self.intertest_predict(
                inputs=inputs,
                data_samples=data_samples,
                resized_padded_inputs=resized_padded_inputs,
                image_embeds=image_embeds,
                ori_image_embeds=ori_image_embeds,
                step=step,
                prev_logits=prev_logits,
                points=points,
                target_size=target_size)
            prev_logits = logits
            logits = self.interpolate(logits, resized_padded_inputs.shape[-2:])
            logits = self.crop_and_resize_to_original_size(
                logits, inputs.shape[-2:], target_size)
            pre_labels = (logits > 0.0).to(pre_labels)
            h, w = seg_labels.shape[-2:]
            pre_labels = pre_labels[..., :h, :w]
            results.append(pre_labels.squeeze().detach().cpu().numpy())
            prev_logits_list.append(
                prev_logits.squeeze().detach().cpu().numpy())
        gt_sem_seg = gt_sem_seg.squeeze().detach().cpu().numpy()
        return points, results, gt_sem_seg, prev_logits_list
