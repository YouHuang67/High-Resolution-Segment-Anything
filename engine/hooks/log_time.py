from mmengine.dist import get_dist_info
from mmengine.hooks import Hook

from mmseg.registry import HOOKS
from ..timers import Timer


@HOOKS.register_module()
class SimpleTimeLoggerHook(Hook):

    def _after_iter(self, runner, *args, **kwargs):
        if get_dist_info()[0] == 0:
            with open(f'{runner.log_dir}/time_info.txt', 'w') as file:
                file.write(Timer.get_time_info())
