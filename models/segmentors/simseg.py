import warnings

import torch
from mmseg.registry import MODELS
from mmseg.utils import add_prefix

from engine.timers import Timer
from .base import BaseClickSegmentor


@MODELS.register_module()
class SimpleSegmentor(BaseClickSegmentor):

    """
    SimpleSegmentor that only performs interactive simulations and forward passes
    """

    @Timer('SimpleSegmentorTrain')
    def interact_train(self, inputs, data_samples, prompts=None):
        inputs_dict, data_samples_dict, _ = \
            self.redistribute_tensor(inputs, data_samples)
        if len(inputs_dict) > 1:
            raise ValueError(f'Not support multiple datasets, '
                             f'but got {list(inputs_dict.keys())}')
        dataset = next(iter(inputs_dict.keys()))
        inputs, data_samples = inputs_dict[dataset], data_samples_dict[dataset]

        device = inputs.device
        cfg = self.parse_train_cfg(dataset)
        gt_sem_seg = self.check_gt_validity(data_samples, train=True)

        self.eval()

        pre_labels = torch.zeros_like(gt_sem_seg)
        seg_labels = gt_sem_seg.detach().clone()

        if not set(
            torch.unique(seg_labels).detach().cpu().numpy().tolist()
        ).issubset({0, 1}):
            warnings.warn(f'gt_sem_seg has values other than 0 and 1, '
                          f'convert it to binary mask')
            seg_labels = (seg_labels > 0).to(seg_labels)

        points_list = self.update_clicks(pre_labels, seg_labels, None, 1.0)
        prev_logits = None

        if hasattr(self.train_cfg, 'target_image_size'):
            h, w = inputs.shape[-2:]
            if h != w:
                scale = float(self.train_cfg.target_image_size) / max(h, w)
                target_shape = (int(scale * h + 0.5), int(scale * w + 0.5))
                warnings.warn(
                    f'The image has different height and width: '
                    f'{h} and {w}, resize the image to {target_shape} '
                    f'as large as possible')
            else:
                scale = float(self.train_cfg.target_image_size) / h
                target_shape = (self.train_cfg.target_image_size, ) * 2
            if tuple(inputs.shape[-2:]) != target_shape:
                inputs = self.interpolate(inputs, target_shape)
        else:
            scale = 1.0

        with torch.no_grad():
            image_embeds = self.preprocess_inputs(inputs, data_samples)
            for _ in range(self.sample_num_clicks(
                    cfg.max_num_clicks, cfg.gamma)):
                logits = self.encode_decode(
                    inputs, data_samples,
                    image_embeds=image_embeds,
                    prev_logits=prev_logits,
                    points=self.point_lists_to_coords(
                        points_list, device, scale=scale)
                )
                prev_logits = logits
                logits = self.interpolate(logits, pre_labels.shape[-2:])
                pre_labels = (logits > 0.0).to(pre_labels)
                points_list = self.update_clicks(
                    pre_labels, seg_labels, points_list, cfg.sfc_inner_k)

        self.train()

        losses = dict()

        if not self.freeze_backbone:
            image_embeds = self.backbone(inputs)
        seg_logits = self.encode_decode(
            inputs, data_samples,
            image_embeds=image_embeds,
            prev_logits=prev_logits,
            points=self.point_lists_to_coords(
                points_list, device, scale=scale))
        seg_logits = self.interpolate(seg_logits, gt_sem_seg.shape[-2:])
        loss = self.loss_by_decode(seg_logits, gt_sem_seg)
        if self.with_metric:
            loss.update(self.metric_by_decode(seg_logits, gt_sem_seg))
        losses.update(add_prefix(loss, dataset))
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
        image_embeds = \
            self.preprocess_inputs(resized_padded_inputs, data_samples)
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

    def intertest_predict(self,
                          inputs,
                          data_samples,
                          resized_padded_inputs,
                          image_embeds,
                          ori_image_embeds,
                          step,
                          prev_logits,
                          points,
                          target_size,
                          **extra_kwargs):
        points = self.resize_coord_to_target_size(
            points, inputs.shape[-2:], target_size, inputs.device)
        logits = self.decode_head(
            self.neck(image_embeds, prev_logits=prev_logits, points=points),
            **extra_kwargs)
        return image_embeds, logits
