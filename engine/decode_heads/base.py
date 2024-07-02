import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.models.builder import build_loss


class BaseDecodeHead(BaseModule):

    def __init__(self,
                 *,
                 num_classes=2,
                 align_corners=False,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 init_cfg=None):
        super(BaseDecodeHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.align_corners = align_corners

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict, '
                            f'but got {type(loss_decode)}')
