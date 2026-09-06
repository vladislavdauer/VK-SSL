import math

import torch


class NoamAnnealing(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 10000,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self._normalize = d_model ** (-0.5)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        out = []

        for initial_lr in self.base_lrs:
            mult = self._normalize * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
            lr = initial_lr * mult

            if step > self.warmup_steps:
                lr = max(lr, self.min_lr)

            out.append(lr)

        return out


class LinearWarmupDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        if step < self.warmup_steps:
            scale = step / float(self.warmup_steps)
        else:
            remain = max(self.total_steps - self.warmup_steps, 1)
            scale = max(0.0, (self.total_steps - step) / float(remain))

        return [base_lr * scale for base_lr in self.base_lrs]


class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        warmup_steps: int,
        hold_steps: int,
        decay_steps: int,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1,
    ):
        self.init_lr = float(init_lr)
        self.warmup_steps = max(1, int(warmup_steps))
        self.hold_steps = max(0, int(hold_steps))
        self.decay_steps = max(1, int(decay_steps))
        self.final_lr_scale = float(final_lr_scale)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        if step <= self.warmup_steps:
            scale = step / float(self.warmup_steps)
        elif step <= self.warmup_steps + self.hold_steps:
            scale = 1.0
        else:
            decay_step = step - self.warmup_steps - self.hold_steps
            progress = min(decay_step / float(self.decay_steps), 1.0)
            scale = 1.0 - progress * (1.0 - self.final_lr_scale)

        lr = self.init_lr * scale
        return [lr for _ in self.base_lrs]


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self._step_count / self.warmup_steps * base_lr for base_lr in self.base_lrs]
        else:
            decay_steps = self.total_steps - self.warmup_steps
            return [
                0.5 * base_lr * (1 + math.cos(math.pi * (self._step_count - self.warmup_steps) / decay_steps))
                for base_lr in self.base_lrs
            ]