import numpy as np
import torch

from engine.utils import mask_to_distance
from engine.utils import fast_mask_to_distance


def main():
    seed = 1024

    # test numpy
    np.random.seed(seed)
    mask = np.random.randint(0, 11, size=(100, 512, 512))
    mask = (mask > 9).astype(np.uint8)
    dist = mask_to_distance(mask, boundary_padding=True)
    fast_dist = fast_mask_to_distance(mask, boundary_padding=True)
    if np.allclose(dist, fast_dist):
        print('numpy test passed')

    # test torch
    torch.manual_seed(seed)
    mask = torch.randint(0, 11, size=(100, 512, 512))
    mask = (mask > 9).to(torch.uint8)
    dist = mask_to_distance(mask, boundary_padding=True)
    fast_dist = fast_mask_to_distance(mask, boundary_padding=True)
    if torch.allclose(dist, fast_dist):
        print('torch test passed')


if __name__ == '__main__':
    main()
