"""When the robot may offer a task on its own, from the database's trigger rules.

With n the piece the human is on, a robot task is offered automatically for piece t
only when all of these hold:
  1. the database lets the robot do it and a robot action is configured for it;
  2. a previous task of its rule triggers it on n: the human's reference task (the
     one being worked on, or the one just confirmed done) is that previous task, at
     >= the rule's progress -- or, for a "Done signal" rule, it is done on n;
  3. the rule's "Condition" holds on n (task_database.py);
  4. t is n plus that previous task's "Piece id" offset, and the task is still open
     on t (not WORKING, not DONE) and not yet offered for t.

A robot action on no piece's task list (Leave from the panel) has no status of its
own: the offer itself (mark_offered) keeps it to once per piece.

Whether the robot can act at all right now -- it can only leave a panel it is
holding -- is TaskManager's call, not this module's. So are the later tasks of a
"robot task" chain: TaskManager offers each when the robot finishes the one before.

A human asking the robot directly (H_REQUEST_ROBOT_TASK) does not go through here.
"""

from events import TaskStatus
from task_database import DONE_SIGNAL, ROBOT

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
        """(task_name, piece_id) pairs the robot may offer now: the human's piece
        first, each piece in task-list order."""
        piece_id = self._reference_piece()
        if piece_id is None:
            return []
        result = []
        for rule in self.database.trigger_rules.values():
            for target in self.targets(rule, piece_id):
                if (target, rule.task_name) not in self._offered and self._robot_action(rule.task_name):
                    result.append((rule.task_name, target))
        order = {name: index for index, name in enumerate(self._task_list(piece_id))}
        return sorted(result, key=lambda item: (item[1] != piece_id, item[1],
                                                order.get(item[0], len(order))))

    def targets(self, rule, piece_id: int) -> list[int]:
        """The pieces the rule offers its task for now, with the human on piece_id."""
        if not self.conditions_met(rule, piece_id):
            return []
        targets = []
        for previous in self._triggering(rule, piece_id):
            target = self._offset_piece(piece_id, rule.piece_offset(previous))
            if target is not None and target not in targets and self._open(rule.task_name, target):
                targets.append(target)
        return targets

    def conditions_met(self, rule, piece_id: int) -> bool:
        """Every "Condition" task meets at least one of its alternatives on piece_id."""
        return all(any(self._holds(task_name, piece_id, alternative) for alternative in alternatives)
                   for task_name, alternatives in rule.conditions.items())

    def explain(self) -> list[dict]:
        """Why each rule does or does not offer its task right now, for the live view
        (decision_view.py). Reads only; offers nothing."""
        piece_id = self._reference_piece()
        rows = []
        for rule in self.database.trigger_rules.values():
            action = self.database.can_execute(rule.task_name, ROBOT) and rule.task_name in self.robot_tasks
            triggering = self._triggering(rule, piece_id) if piece_id is not None else []
            conditions_met = piece_id is not None and self.conditions_met(rule, piece_id)
            targets = self.targets(rule, piece_id) if piece_id is not None else []
            offered = sorted(piece for piece, name in self._offered if name == rule.task_name)
            offers = [target for target in targets if target not in offered] if action else []
            if not action:
                verdict = "no robot action configured"
            elif piece_id is None:
                verdict = "no piece left"
            elif not triggering:
                verdict = "waiting for its previous task"
            elif not conditions_met:
                verdict = "waiting for its condition"
            elif offers:
                verdict = "offers it for piece " + ", ".join(map(str, offers))
            elif targets:
                verdict = "already offered"
            else:
                verdict = "task not open on the target piece"
            condition_state = {}
            for task_name in rule.conditions:
                task = self.tracker.get(task_name, piece_id) if piece_id is not None else None
                condition_state[task_name] = (None if task is None else
                                              {"status": task.status.name, "signals": dict(task.signals)})
            rows.append({
                "task": rule.task_name,
                "previous": list(rule.previous_tasks),
                "progress": rule.progress,
                "done_signal": rule.done_signal,
                "piece_offsets": dict(rule.piece_offsets),
                "conditions": {task: [dict(item) for item in items] for task, items in rule.conditions.items()},
                "condition_state": condition_state,
                "robot_tasks": list(rule.robot_tasks),
                "robot_action": action,
                "piece_n": piece_id,
                "triggering": triggering,
                "conditions_met": conditions_met,
                "targets": targets,
                "offered": offered,
                "offers": offers,
                "verdict": verdict,
            })
        return rows

    def allows(self, task_name: str, piece_id: int) -> bool:
        """Re-check before a queued offer is made: is the task still open?"""
        task = self.tracker.get(task_name, piece_id)
        return task is not None and task.status in OPEN_STATUSES

    def mark_offered(self, task_name: str, piece_id: int) -> None:
        self._offered.add((piece_id, task_name))
        self.tracker.mark_pending(task_name, piece_id)

    def _reference_piece(self) -> int | None:
        piece_id = self.tracker.reference_piece_id
        return self.tracker.human_piece_id if piece_id is None else piece_id

    def _triggering(self, rule, piece_id: int) -> list[str]:
        """The previous tasks that trigger the rule on piece_id now."""
        if rule.done_signal:
            # Any previous task confirmed done on the piece the human is on -- not only
            # the reference task, since recognition may already have moved on to the
            # next step when "screw done" arrives. Recognized progress never counts: it
            # is clamped to 1.0 near the end of a task and would pass for the signal.
            return [name for name in rule.previous_tasks
                    if (task := self.tracker.get(name, piece_id)) is not None
                    and task.status == TaskStatus.DONE]
        reference = self.tracker.reference_task
        if reference in rule.previous_tasks and self.tracker.reference_progress >= rule.progress:
            return [reference]
        return []

    def _holds(self, task_name: str, piece_id: int, requirement: dict) -> bool:
        task = self.tracker.get(task_name, piece_id)
        if task is None:
            return False
        for name, wanted in requirement.items():
            if name == DONE_SIGNAL:
                value = task.status == TaskStatus.DONE
            else:
                value = task.signals.get(name)
            if isinstance(wanted, bool):
                if bool(value) != wanted:
                    return False
            elif value is None or value < wanted:
                return False
        return True

    def _offset_piece(self, piece_id: int, offset: int) -> int | None:
        """The piece offset places after piece_id in database order, if there is one."""
        ids = self.tracker.piece_ids
        index = ids.index(piece_id) + offset if piece_id in ids else -1
        return ids[index] if 0 <= index < len(ids) else None

    def _open(self, task_name: str, piece_id: int) -> bool:
        if not self.database.in_task_lists(task_name):
            return True  # not tracked per piece; the offer itself keeps it to once
        task = self.tracker.get(task_name, piece_id)
        return task is not None and task.status in OPEN_STATUSES

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
