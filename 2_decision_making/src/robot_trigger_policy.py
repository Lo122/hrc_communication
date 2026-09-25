"""When the robot may offer a task on its own, from the database's trigger rules.

A robot task is offered automatically only when all of these hold:
  1. the database lets the robot do it and a robot action is configured for it;
  2. it is still open (not WORKING, not DONE) and not yet offered for that piece;
  3. the human's reference task (the one being worked on, or the one just
     confirmed done) is one of the rule's previous tasks, at >= its progress --
     or, for a "Done signal" rule, one of them is confirmed done on that piece.

Support tasks (Bring Tool etc.) belong to the piece the human is on. A task the
human also does per piece (Lift, Pull Cables) moves on to the next piece once it
is done on this one, so the robot can prepare the next panel while the human is
still finishing the current one. A robot action on no piece's task list (Leave
from the panel) is offered once for the piece the human is on.

Whether the robot can act at all right now -- it can only leave a panel it is
holding -- is TaskManager's call, not this module's.

A human asking the robot directly (H_REQUEST_ROBOT_TASK) does not go through here.
"""

from events import TaskStatus
from task_database import ROBOT

OPEN_STATUSES = (TaskStatus.NOT_DONE, TaskStatus.PENDING)


class RobotTriggerPolicy:
    def __init__(self, database, tracker, robot_tasks: dict[str, int], logger=None):
        self.database = database
        self.tracker = tracker
        self.robot_tasks = robot_tasks
        self.logger = logger
        self._offered: set[tuple[int, str]] = set()
        self._skip_logged: set[str] = set()

    def candidates(self) -> list[tuple[str, int]]:
        """(task_name, piece_id) pairs the robot may offer now: this piece's tasks
        first, each piece in task-list order."""
        result = []
        piece_id = self._reference_piece()
        if piece_id is None:
            return result
        order = {name: index for index, name in enumerate(self._task_list(piece_id))}
        for rule in self.database.trigger_rules.values():
            if not self._rule_met(rule):
                continue
            target = self.target_piece(rule.task_name)
            if target is None or (target, rule.task_name) in self._offered:
                continue
            if self._robot_action(rule.task_name):
                result.append((rule.task_name, target))
        return sorted(result, key=lambda item: (item[1] != piece_id, order.get(item[0], len(order))))

    def allows(self, task_name: str, piece_id: int) -> bool:
        """Re-check before a queued offer is made: is the task still open?"""
        task = self.tracker.get(task_name, piece_id)
        return task is not None and task.status in OPEN_STATUSES

    def mark_offered(self, task_name: str, piece_id: int) -> None:
        self._offered.add((piece_id, task_name))
        self.tracker.mark_pending(task_name, piece_id)

    def target_piece(self, task_name: str) -> int | None:
        piece_id = self._reference_piece()
        if not self.database.in_task_lists(task_name):
            # Not tracked per piece; the offer itself (mark_offered) keeps it to once.
            return piece_id
        task = self.tracker.get(task_name, piece_id)
        if task is not None and task.status in OPEN_STATUSES:
            return piece_id
        if task is not None and task.status == TaskStatus.DONE and self.tracker.is_recognized(task_name):
            next_piece = self.tracker.next_piece_id(piece_id)
            next_task = self.tracker.get(task_name, next_piece) if next_piece is not None else None
            if next_task is not None and next_task.status in OPEN_STATUSES:
                return next_piece
        return None

    def _reference_piece(self) -> int | None:
        piece_id = self.tracker.reference_piece_id
        return self.tracker.human_piece_id if piece_id is None else piece_id

    def _rule_met(self, rule) -> bool:
        if rule.done_signal:
            # Any previous task confirmed done on the piece the human is on -- not only
            # the reference task, since recognition may already have moved on to the
            # next step when "screw done" arrives. Recognized progress never counts: it
            # is clamped to 1.0 near the end of a task and would pass for the signal.
            piece_id = self._reference_piece()
            tasks = (self.tracker.get(name, piece_id) for name in rule.previous_tasks)
            return any(task is not None and task.status == TaskStatus.DONE for task in tasks)
        return (self.tracker.reference_task in rule.previous_tasks
                and self.tracker.reference_progress >= rule.progress)

    def _robot_action(self, task_name: str) -> bool:
        if not self.database.can_execute(task_name, ROBOT):
            reason = "Trigger rule skipped: the database does not let the robot do this task."
        elif task_name not in self.robot_tasks:
            reason = "Trigger rule skipped: no robot action configured for this task."
        else:
            return True
        if task_name not in self._skip_logged and self.logger is not None:
            self._skip_logged.add(task_name)
            self.logger.log_message(reason, {"task_name": task_name})
        return False

    def _task_list(self, piece_id: int) -> tuple[str, ...]:
        for piece in self.database.pieces:
            if piece.piece_id == piece_id:
                return piece.task_list
        return ()
