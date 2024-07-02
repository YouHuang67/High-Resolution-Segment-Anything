# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings
import functools
import logging

from torch.distributed.fsdp import FullyShardedDataParallel
from mmengine.dist import get_dist_info
from mmengine.logging import print_log as _print_log
from mmengine.optim import DefaultOptimWrapperConstructor

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


print_log = functools.partial(_print_log, logger='current')


def get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.layers') or \
            var_name.startswith('backbone.blocks'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ViTLearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for ConvNeXt,
    BEiT and MAE.
    """

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        print_log(f'Build ViTLearningRateDecayOptimizerConstructor for '
                  f'model \n{module}')

        backbone_type = self.paramwise_cfg.get('backbone_type', 'vit')
        if backbone_type != 'vit':
            warnings.warn(f'Got backbone_type {backbone_type}, '
                          f'but it will use get_layer_id_for_vit '
                          f'function to get layer id.')
        parameter_groups = {}
        print_log(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        print_log('Build LearningRateDecayOptimizerConstructor  '
                  f'{decay_type} {decay_rate} - {num_layers}')
        ignore_keys = self.paramwise_cfg.get('ignore_keys', [])
        if len(ignore_keys) > 0:
            print_log(f'Ignore keys: {ignore_keys} in learning rate decay')
        weight_decay = self.base_wd
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or \
                    name in ('pos_embed', 'cls_token'):
                group_name = 'no_weight_decay'
                this_weight_decay = 0.
            else:
                group_name = 'weight_decay'
                this_weight_decay = weight_decay
            if 'layer_wise' in decay_type:
                backbone = module.backbone
                if isinstance(backbone, FullyShardedDataParallel):
                    backbone = backbone._fsdp_wrapped_module  # noqa
                    unwrap_name = name.replace('_fsdp_wrapped_module.', '')
                else:
                    unwrap_name = name
                if backbone_type in backbone.__class__.__name__.lower():
                    layer_id = get_layer_id_for_vit(unwrap_name, num_layers)
                    print_log(f'set param {unwrap_name} as id {layer_id}')
                else:
                    print_log(f'Cannot get layer id for Backbone: '
                              f'{backbone.__class__.__name__}, given '
                              f'backbone_type {backbone_type}')
                    raise NotImplementedError(
                        f'Cannot get layer id for Backbone: '
                        f'{backbone.__class__.__name__}, given '
                        f'backbone_type {backbone_type}')
            else:
                raise NotImplementedError(f'Unknown decay type {decay_type}')
            group_name = f'layer_{layer_id}_{group_name}'

            if any(key in name for key in ignore_keys):
                group_name = f'{group_name}_no_lr_decay'
                if group_name not in parameter_groups:
                    scale = 1.0
                    parameter_groups[group_name] = {
                        'weight_decay': this_weight_decay,
                        'params': [],
                        'param_names': [],
                        'lr_scale': scale,
                        'group_name': group_name,
                        'lr': scale * self.base_lr,
                    }

            else:
                group_name = f'{group_name}_lr_decay'
                if group_name not in parameter_groups:
                    scale = decay_rate ** (num_layers - layer_id - 1)
                    parameter_groups[group_name] = {
                        'weight_decay': this_weight_decay,
                        'params': [],
                        'param_names': [],
                        'lr_scale': scale,
                        'group_name': group_name,
                        'lr': scale * self.base_lr,
                    }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print_log(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())
