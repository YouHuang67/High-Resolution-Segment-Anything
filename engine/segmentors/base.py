import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmseg.models.builder import MODELS
from mmseg.models.segmentors.base import BaseSegmentor


@MODELS.register_module()
class EmptyBackbone(BaseModule):

    def __init__(self):
        super(EmptyBackbone, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'{self.__class__.__name__} is empty')


class _BaseInterSegmentor(BaseSegmentor):

    """
    Base class for interactive segmentors,
    with API for interactive training and testing.
    """

    def __init__(self,
                 backbone,
                 neck,
                 decode_head,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None):
        raise NotImplementedError

    def _init_decode_head(self, decode_head):
        raise NotImplementedError

    def interact_train(self, inputs, data_samples, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def interact_test(self, inputs, data_samples, **kwargs):
        raise NotImplementedError

    def forward(self, inputs, data_samples, mode='loss', **kwargs):  # noqa
        if mode == 'loss':
            return self.interact_train(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.interact_test(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               f'Only supports loss and predict')

    def loss(self, inputs, data_samples):
        raise NotImplementedError(f'Deprecated loss function.')

    @property
    def with_losses(self):
        return hasattr(self, 'losses') and len(self.losses) > 0

    @property
    def with_metric(self):
        return hasattr(self, 'metric') and len(self.metric) > 0


class BaseInterSegmentor(_BaseInterSegmentor):

    def __init__(self,
                 backbone,
                 neck,
                 decode_head,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None,
                 remove_backbone=False):
        super(_BaseInterSegmentor, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if backbone is not None and not remove_backbone:
            if pretrained is not None:
                assert backbone.get('pretrained') is None, \
                    'both backbone and segmentor set pretrained weight'
                backbone.pretrained = pretrained
        else:
            backbone = dict(type='EmptyBackbone')
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if decode_head is not None:
            self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.vis_results = []

    def _init_decode_head(self, decode_head):
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        if isinstance(self.decode_head.loss_decode, nn.ModuleList):
            losses = list(self.decode_head.loss_decode)
        else:
            losses = [self.decode_head.loss_decode]
        self.losses = nn.ModuleDict()
        self.metric = nn.ModuleDict()
        for loss in losses:
            if 'loss' in loss.loss_name:
                self.losses[loss.loss_name] = loss
            else:
                self.metric[loss.loss_name] = loss

    def loss_by_decode(self, seg_logits, seg_labels):
        if tuple(seg_logits.shape[-2:]) != tuple(seg_labels.shape[-2:]):
            seg_logits = self.interpolate(seg_logits, seg_labels.shape[-2:])
        if seg_labels.ndim == 4:
            seg_labels = seg_labels.squeeze(1)
        losses = dict()
        for name, loss in self.losses.items():
            losses[f'los.{name}'] = loss(seg_logits, seg_labels)
        return losses

    def metric_by_decode(self, seg_logits, seg_labels):
        if tuple(seg_logits.shape[-2:]) != tuple(seg_labels.shape[-2:]):
            seg_logits = self.interpolate(seg_logits, seg_labels.shape[-2:])
        if seg_labels.ndim == 4:
            seg_labels = seg_labels.squeeze(1)
        losses = dict()
        for name, metric in self.metric.items():
            losses[f'met.{name}'] = metric(seg_logits, seg_labels)
        return losses

    def interpolate(self, x, size, mode='bilinear'):
        """Resize with backward compatibility."""
        return F.interpolate(
            x, size, mode=mode, align_corners=self.align_corners)
