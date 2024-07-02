import argparse
import os.path as osp
from collections import OrderedDict

import torch
import mmengine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input-file')
    parser.add_argument('-p', '--prefix')
    parser.add_argument('-of', '--output-file')
    args = parser.parse_args()

    weight = OrderedDict()
    checkpoint = torch.load(args.input_file, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    for k, v in state_dict.items():
        if k.startswith(args.prefix):
            weight[k] = v
    mmengine.mkdir_or_exist(osp.dirname(args.output_file))
    torch.save(weight, args.output_file)


if __name__ == '__main__':
    main()
