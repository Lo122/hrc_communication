"""The demo's scripted opening: the first panel starts from the dialogue.

Recognition cannot reliably tell when the assembly begins, so with --demo
(run_communication.py) the robot opens by offering to pull the cables of the first
panel, and TaskManager ignores recognition and the task detectors until the panel's
lift is settled:

  "Would you like me to pull the cables?"
    yes -> the robot pulls them. Its pull takes a fixed time, so
           DEMO_LIFT_ASK_AFTER_PULL_START_S after it starts, the robot asks about the
           lift while still pulling: a yes lifts the panel the moment the cables are
           pulled (TaskManager._offer_in_advance).
    no  -> the human pulls them (not pending: the robot does not ask again). After
           Pull Cables' duration limit (the tracker's, p95 of the annotations) plus
           DEMO_HUMAN_PULL_BUFFER_S the robot asks about the lift.
    later -> the pull is pending until the human asks for it ("pull the cables"); the
           opening is over, and the lift follows the pull as its chain's next task.
  "Would you like me to lift the panel?"
    yes -> the usual lift: free drive, holding, screw done, leave.
    no  -> the human lifts it.
    later -> the lift is pending until the human asks for it ("lift the panel").

Once the lift is answered (or left unanswered: pending too) the opening is over:
recognition, the trigger rules and the task detectors take over -- screw detection,
the pipe connector, the tool, the next panel -- and the human can ask for robot
tasks as always. Yes, no and later are the usual H_ACCEPT / H_REFUSE / H_DEFER.
"""

import sys
from pathlib import Path

# src/ holds this layer's helpers (task database, transition table, trigger policy).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config
from events import RobotTaskState as S
from task_database import HUMAN

PULL = "Pull Cables"
LIFT = "Lift"
TASK_NAMES = {task_id: name for name, task_id in config.TRACKED_TO_ROBOT_TASK.items()}

# The lift question settled one way or another: the opening is over.
LIFT_SETTLED = {S.R_ACCEPTED, S.R_DEFER, S.R_REFUSED, S.R_PENDING}
# The robot's pull stopped short, or waits pending (later, or no answer): nothing left
# to script.
PULL_STOPPED = {S.R_RECOVERY_EVALUATING, S.R_CANCELED, S.R_PENDING}


class DemoOpening:
    def __init__(self, *, lift_after_robot_pull_s: float, human_pull_buffer_s: float):
        self.lift_after_robot_pull_s = lift_after_robot_pull_s
        self.human_pull_buffer_s = human_pull_buffer_s
        self.active = False
        self.piece_id: int | None = None
        self._lift_scheduled = False

    def start(self, manager) -> None:
        self.piece_id = manager.tracker.first_open_piece_for(PULL)
        if self.piece_id is None:
            manager.logger.log_message("Demo opening skipped: no piece has cables left to pull.", {})
            return
        self.active, self._lift_scheduled = True, False
        manager.logger.log_message("Demo opening started; recognition is ignored until the lift is answered.",
                                   {"piece_id": self.piece_id})
        manager.offer_robot_task(PULL, self.piece_id)

    def on_transition(self, manager, task, event) -> None:
        """TaskManager calls this after every robot task state change."""
        if not self.active or task.piece_id != self.piece_id:
            return
        name, state = TASK_NAMES.get(task.task_id), task.state
        if name == PULL:
            if state == S.R_EXECUTING and not self._lift_scheduled:
                self._schedule_lift(manager, self.lift_after_robot_pull_s, "the robot started pulling")
            elif state == S.R_REFUSED:
                # "No, I'll do it."
                manager.tracker.start_task(PULL, self.piece_id, HUMAN)
                wait = manager.tracker.duration_limits.get(PULL, 0.0) + self.human_pull_buffer_s
                self._schedule_lift(manager, wait, "the human pulls the cables")
            elif state in PULL_STOPPED:
                self._end(manager, "the robot's cable pull is pending" if state == S.R_PENDING
                          else "the robot's cable pull was stopped")
        elif name == LIFT and state in LIFT_SETTLED:
            if state == S.R_REFUSED:
                manager.tracker.start_task(LIFT, self.piece_id, HUMAN)
            self._end(manager, f"lift {state.name}")

    def _schedule_lift(self, manager, delay: float, why: str) -> None:
        self._lift_scheduled = True
        manager.schedule_offer(LIFT, self.piece_id, delay)
        manager.logger.log_message("Demo opening: lift question scheduled.",
                                   {"piece_id": self.piece_id, "delay_s": round(delay, 1), "because": why})

    def _end(self, manager, why: str) -> None:
        self.active = False
        manager.cancel_scheduled_offer(LIFT, self.piece_id)
        manager.logger.log_message("Demo opening over; recognition, trigger rules and detectors take over.",
                                   {"piece_id": self.piece_id, "because": why})


def build_demo_opening() -> DemoOpening:
    return DemoOpening(lift_after_robot_pull_s=config.DEMO_LIFT_ASK_AFTER_PULL_START_S,
                       human_pull_buffer_s=config.DEMO_HUMAN_PULL_BUFFER_S)
