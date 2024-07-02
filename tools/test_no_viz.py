# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner, set_random_seed
from mmengine.dist import get_dist_info


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('-c', '--extra-config',
                        help='extra config file', default=[], nargs='*')
    parser.add_argument('-w', '--extra-weight',
                        help='extra weight file', default=[], nargs='*')
    parser.add_argument('--use-embed-loader', action='store_true')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42,
                        help='To eliminate very slight randomness')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_extra_weights(args, checkpoint):
    rank, _ = get_dist_info()
    if len(args.extra_weight) == 0:
        return checkpoint

    merged_file = 'merged'
    for file in list(args.extra_weight) + [checkpoint]:
        merged_file = f'{merged_file}-{Path(file).stem}'
    merged_file = Path(args.checkpoint).parent / f'{merged_file}.pth'
    if merged_file.is_file():
        print(f'{str(merged_file)} already exists, skip merging')
        return str(merged_file)

    weight = OrderedDict()
    for file in list(args.extra_weight) + [checkpoint]:
        checkpoint = torch.load(file, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        weight.update(OrderedDict((k, v) for k, v in state_dict.items()))

    if rank == 0:
        torch.save(weight, str(merged_file))
    else:
        while not merged_file.is_file():
            time.sleep(1)
    return str(merged_file)


def main():
    args = parse_args()

    # load config
    checkpoint = Path(merge_extra_weights(args, args.checkpoint))
    root = checkpoint.parent
    cfg = Config.fromfile(next(root.glob('*.py')))
    cfg.merge_from_dict(Config.fromfile(args.config).to_dict())
    work_dir = str(root /
                   f'{Path(args.config).stem}-'
                   f'{checkpoint.stem}')
    for config in args.extra_config:
        cfg.merge_from_dict(Config.fromfile(config).to_dict())
        work_dir = f'{work_dir}-{Path(config).stem}'
    cfg.work_dir = work_dir
    cfg.launcher = args.launcher
    cfg.load_from = str(checkpoint)
    cfg.resume = False

    if hasattr(cfg.model, 'remove_backbone') and cfg.model.remove_backbone:
        cfg.model.remove_backbone = False

    if hasattr(cfg.model, 'image_embed_loader') and \
            cfg.model.image_embed_loader is not None:
        if not args.use_embed_loader:
            cfg.model.image_embed_loader = None

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # to eliminate very slight randomness
    set_random_seed(args.seed)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
