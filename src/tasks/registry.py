from __future__ import annotations

from src.tasks.common import TaskBatcher
from src.tasks.copy.batch import make_batch as copy_batch
from src.tasks.copy.metrics import loss_and_metrics as copy_loss
from src.tasks.induction_associative_recall.batch import make_batch as iar_batch
from src.tasks.induction_associative_recall.metrics import loss_and_metrics as iar_loss
from src.tasks.lra_listops.batch import make_batch as listops_batch
from src.tasks.lra_listops.metrics import loss_and_metrics as listops_loss
from src.tasks.selective_copy.batch import make_batch as selective_batch
from src.tasks.selective_copy.metrics import loss_and_metrics as selective_loss


TASK_BATCHERS: dict[str, TaskBatcher] = {
    "copy": copy_batch,
    "selective_copy": selective_batch,
    "induction_associative_recall": iar_batch,
    "lra_listops": listops_batch,
}

TASK_LOSSES = {
    "copy": copy_loss,
    "selective_copy": selective_loss,
    "induction_associative_recall": iar_loss,
    "lra_listops": listops_loss,
}


def get_batcher(task: str) -> TaskBatcher:
    return TASK_BATCHERS[task]


def get_loss(task: str):
    return TASK_LOSSES[task]

