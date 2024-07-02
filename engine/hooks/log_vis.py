import functools
import matplotlib.pyplot as plt
from pathlib import Path

from mmengine.hooks import Hook
from mmengine.model.wrappers.utils import is_model_wrapper
from mmengine.logging import print_log as _print_log
from mmseg.registry import HOOKS

from ..utils import CLK_POSITIVE

print_log = functools.partial(_print_log, logger='current')


@HOOKS.register_module()
class SimpleVisLoggerHook(Hook):

    """
    loop in model.vis_results, and save the results in visualization folder
    each item in model.vis_results is a pair like
        ([(image, title), ...], filename) or
        ([(image, points, title), ...], filename), where
            - the image is a numpy array with shape (H, W, 3),
            - the title is a string,
            - the filename is a string
    """

    def __init__(self, num_figures_per_row=4, figure_size=5):
        self.num_figures_per_row = num_figures_per_row
        self.figure_size = figure_size

    def _after_iter(self, runner, *args, **kwargs):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if not hasattr(model, 'vis_results'):
            print_log(
                f'Not found vis_results in {model.__class__.__name__}, '
                f'ignore visualization')
            return
        if len(model.vis_results) == 0:
            print_log(
                f'Empty vis_results in {model.__class__.__name__}, '
                f'ignore visualization')
            return
        figure_dir = Path(runner.log_dir) / 'visualization'
        figure_dir.mkdir(parents=True, exist_ok=True)
        for vis_list, filename in model.vis_results:
            self._plot_single_sample(vis_list, str(figure_dir / filename))
        model.vis_results.clear()

    def _plot_single_sample(self, vis_list, filename):
        """
        vis_list: [(image, title), ...]
        """
        num_figures = len(vis_list)
        num_cols = min(num_figures, self.num_figures_per_row)
        num_rows = (num_figures + num_cols - 1) // num_cols

        # set aspect ratio by the first image
        aspect_ratio = vis_list[0][0].shape[1] / vis_list[0][0].shape[0]

        width = num_cols * aspect_ratio * self.figure_size
        height = num_rows * self.figure_size
        _, axes = plt.subplots(num_rows, num_cols, figsize=(width, height))

        for i, axis in enumerate(axes.flat):
            if i >= num_figures:
                axis.axis('off')
                continue
            if len(vis_list[i]) == 2:
                image, title = vis_list[i]
            elif len(vis_list[i]) == 3:
                image, points, title = vis_list[i]
                for y, x, mode in points:
                    color = 'green' if mode == CLK_POSITIVE else 'red'
                    axis.plot(x, y, marker='*', color=color)
            else:
                raise NotImplementedError(
                    f'Cannot handle {len(vis_list[i])} items '
                    f'like {vis_list[i]}')
            axis.imshow(image)
            axis.set_title(title)
            axis.axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
