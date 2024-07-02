import argparse
import os.path as osp
from collections import OrderedDict

import torch
import mmengine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input-files',
                        nargs='+', help='src model path or url')
    parser.add_argument('-of', '--output-file', help='save path')
    args = parser.parse_args()

    weight = OrderedDict()
    for file in args.input_files:
        checkpoint = torch.load(file, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        weight.update(OrderedDict((k, v) for k, v in state_dict.items()))
    mmengine.mkdir_or_exist(osp.dirname(args.output_file))
    torch.save(weight, args.output_file)


if __name__ == '__main__':
    main()
