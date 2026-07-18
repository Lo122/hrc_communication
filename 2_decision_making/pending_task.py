"""Pending task pool for refused or unanswered robot tasks."""

from models import RobotTask


class PendingTaskPool:
    """Stores pending tasks without dispatching or mutating their state."""

    def __init__(self):
        self.pending_tasks: dict[str, RobotTask] = {}

    def add(self, task: RobotTask) -> None:
        self.pending_tasks[task.task_instance_id] = task

    def get(self, task_instance_id: str) -> RobotTask | None:
        return self.pending_tasks.get(task_instance_id)

    def remove(self, task_instance_id: str) -> RobotTask | None:
        return self.pending_tasks.pop(task_instance_id, None)

    def list_all(self) -> list[RobotTask]:
        return list(self.pending_tasks.values())

    def contains(self, task_instance_id: str) -> bool:
        return task_instance_id in self.pending_tasks
