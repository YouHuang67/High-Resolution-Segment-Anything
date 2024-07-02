import numpy as np
import mmengine
from mmengine import print_log
from mmengine.registry import LOOPS

from engine.utils import calculate_binary_iou_single_sample
from engine.runners import BaseTestLoop


@LOOPS.register_module()
class ClickTestLoop(BaseTestLoop):

    def __init__(self,
                 runner,
                 dataloader,
                 evaluator=None,
                 metrics=('noc85', 'noc90', 'noc95'),
                 click_indices=(1, 2, 3, 4, 5, 10, 15, 20)):
        super(ClickTestLoop, self).__init__(runner, dataloader, evaluator)
        self.metrics = metrics
        self.click_indices = click_indices

    def process(self, outputs, data_batch):
        point_lists, results, gt_sem_seg, *_ = outputs
        return [calculate_binary_iou_single_sample(res, gt_sem_seg)
                for res in results]

    def compute_metrics(self, results):
        work_dir = self.runner.log_dir
        mmengine.dump(dict(results=results),
                      f'{work_dir}/click_test_res.json', indent=4)

        results = np.array(results)
        metrics = dict()
        for metric in self.metrics:
            if metric.startswith('noc'):
                threshold = float(metric[3:]) / 100.0
                metrics[metric] = self.compute_noc(results, threshold)
            else:
                raise ValueError(f'Unknown metric {metric}')
            print_log(f'{metric}: {metrics[metric]:.2f}', logger='current')
        for click_idx in self.click_indices:
            metrics[f'click{click_idx:02d}'] = \
                float(np.mean(results[:, click_idx - 1]))
        mmengine.dump(metrics, f'{work_dir}/click_test_met.json', indent=4)

    @staticmethod
    def compute_noc(results, threshold):
        results = np.copy(results)
        results[:, -1] = 1.0
        return float(np.mean(
            np.argmax(results >= threshold, axis=-1).astype(np.float32)
        )) + 1.0
