from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):

    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())

import time
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class EpochIterTimerHook(Hook):
    """Iter-level timer hook for EpochBasedRunner.

    This hook manually records time and data_time
    so that TextLoggerHook can print them.
    """

    def __init__(self):
        self._iter_start_time = None
        self._data_time = 0.0

    def before_train_iter(self, runner):
        # time since last iter finished -> data loading time
        now = time.time()
        if self._iter_start_time is not None:
            self._data_time = now - self._iter_start_time
        self._iter_start_time = now

    def after_train_iter(self, runner):
        if self._iter_start_time is None:
            return

        time_cost = time.time() - self._iter_start_time

        runner.log_buffer.update(
        dict(
            time=time_cost,
            data_time=self._data_time
        ),
        count=1
        )


from mmcv.runner import HOOKS
from mmcv.runner.hooks.logger import TextLoggerHook


@HOOKS.register_module()
class EpochTimeTextLoggerHook(TextLoggerHook):
    """TextLoggerHook with epoch time for mmcv 1.4"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ⭐ 关键：不要过滤 time / data_time
        self.ignore_last = ()

    def _log_info(self, log_dict, runner):
        log_items = []
        for name, val in log_dict.items():
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')

        log_str = ', '.join(log_items)

        prefix = f'Epoch({runner.epoch + 1}) [{runner.inner_iter + 1}]'

        runner.logger.info(f'{prefix}\t{log_str}')
