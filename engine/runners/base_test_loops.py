import warnings

import torch
import mmengine
from mmengine.runner import BaseLoop
from mmengine.dist import get_dist_info, collect_results_gpu
from engine.timers import Timer


class BaseTestLoop(BaseLoop):

    def __init__(self, runner, dataloader, evaluator=None):
        super(BaseTestLoop, self).__init__(runner, dataloader)
        if evaluator is not None and evaluator:
            warnings.warn(f'evaluator {evaluator} will be ignored')

    def run(self):
        self.runner.model.eval()
        results = []
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmengine.ProgressBar(len(self.dataloader.dataset))
        for idx, data_batch in enumerate(self.dataloader):
            result = self.process(self.run_iter(idx, data_batch), data_batch)
            results.append(result)
            if rank == 0:
                for _ in range(world_size):
                    prog_bar.update()
        if world_size > 1:
            results = collect_results_gpu(
                results, len(self.dataloader.dataset))
        if rank == 0:
            self.compute_metrics(results)
            with open(f'{self.runner.log_dir}/time_info.txt', 'w') as file:
                file.write(Timer.get_time_info())

    @torch.no_grad()
    def run_iter(self, idx, data_batch):
        outputs = self.runner.model.test_step(data_batch)
        return outputs

    def process(self, outputs, data_batch):
        raise NotImplementedError

    def compute_metrics(self, results):
        raise NotImplementedError
