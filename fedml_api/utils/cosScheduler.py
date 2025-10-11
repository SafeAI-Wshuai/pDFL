import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

class CosineSchedulerPair:
    """
    同时管理 head/body 两个优化器的余弦调度器
    - 每个阶段内部按本地 epoch 余弦退火
    - 不含 warmup，不依赖 SequentialLR
    """
    def __init__(self,
                 head_optimizer,
                 body_optimizer,
                 head_epochs: int,
                 body_epochs: int,
                 eta_min_head: float = 0.0,
                 eta_min_body: float = 0.0):

        self.head_optimizer = head_optimizer
        self.body_optimizer = body_optimizer

        # Head scheduler
        if head_optimizer is not None:
            self.head_sched = CosineAnnealingLR(
                head_optimizer,
                T_max=max(1, head_epochs),
                eta_min=eta_min_head
            )
        else:
            self.head_sched = None

        # Body scheduler
        if body_optimizer is not None:
            self.body_sched = CosineAnnealingLR(
                body_optimizer,
                T_max=max(1, body_epochs),
                eta_min=eta_min_body
            )
        else:
            self.body_sched = None

    def step_head(self):
        if self.head_sched is not None:
            self.head_sched.step()

    def step_body(self):
        if self.body_sched is not None:
            self.body_sched.step()

    def get_lrs(self):
        lr_h = self.head_optimizer.param_groups[0]['lr'] if self.head_optimizer else None
        lr_b = self.body_optimizer.param_groups[0]['lr'] if self.body_optimizer else None
        return lr_h, lr_b
