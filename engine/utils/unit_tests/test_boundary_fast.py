import numpy as np
import torch

from engine.utils import erode, dilate
from engine.utils import fast_erode, fast_dilate


def main():
    seed = 1024

    # test numpy
    np.random.seed(seed)
    mask = np.random.randint(0, 11, size=(100, 512, 512))
    mask = (mask > 9).astype(np.uint8)
    eroded = erode(mask, kernel_size=3, iterations=3)
    fast_eroded = fast_erode(mask, kernel_size=3, iterations=3)
    dilated = dilate(mask, kernel_size=3, iterations=3)
    fast_dilated = fast_dilate(mask, kernel_size=3, iterations=3)
    if np.allclose(eroded, fast_eroded) and np.allclose(dilated, fast_dilated):
        print('numpy test passed')
    else:
        print(
            f'numpy test failed, '
            f'average diff of erode: {np.abs(eroded - fast_eroded).mean()}, '
            f'average diff of dilate: {np.abs(dilated - fast_dilated).mean()}')

    # test torch
    torch.manual_seed(seed)
    mask = torch.randint(0, 11, size=(100, 512, 512))
    mask = (mask > 9).to(torch.uint8)
    eroded = erode(mask, kernel_size=3, iterations=3)
    fast_eroded = fast_erode(mask, kernel_size=3, iterations=3)
    dilated = dilate(mask, kernel_size=3, iterations=3)
    fast_dilated = fast_dilate(mask, kernel_size=3, iterations=3)
    if torch.allclose(dilated, fast_dilated):
        print('torch test passed')
    else:
        print(
            f'torch test failed, '
            f'average diff of erode: {(eroded - fast_eroded).abs.mean()}, '
            f'average diff of dilate: {(dilated - fast_dilated).abs.mean()}')


if __name__ == '__main__':
    main()
