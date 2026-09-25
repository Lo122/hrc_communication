"""Simulated robot for running the watch UI without ROS, UR10e or Grasshopper.

It hooks into the existing HRCSystem objects instead of replacing them, so the
real TaskManager + StateMachine still decide everything. The simulator only
plays the role of the robot side: it answers a Grasshopper dispatch with
ROBOT_RUNNING, advances a fake progress value while the task is executing
(respecting pause and speed), and finally emits ROBOT_SUCCESS / ROBOT_HOMED.
"""

from __future__ import annotations

import time

import config
from events import Event, EventType, RobotTaskState


class SimRobot:
    def __init__(self, system, *, task_seconds: float = 16.0, start_delay: float = 1.5,
                 home_seconds: float = 2.5):
        self.system = system
        self.task_seconds = task_seconds
        self.start_delay = start_delay
        self.home_seconds = home_seconds
        self.auto_robot = True
        self.progress: float | None = None
        self._running_at: float | None = None
        self._homed_at: float | None = None
        self._last_tick = time.time()
        self._trigger_counter = 0
        self.set_safe_home(True)
        self._wrap()

    # -- hooks ---------------------------------------------------------------
    def _wrap(self) -> None:
        gh = self.system.gh_dispatcher
        ros = self.system.ros
        original_dispatch = gh.dispatch_task
        original_restart = ros.publish_restart
        original_home = ros.publish_return_home

        def dispatch_task(task):
            result = original_dispatch(task)
            self._schedule_start()
            return result

        def publish_restart():
            original_restart()
            self._schedule_start()

        def publish_return_home():
            original_home()
            if self.auto_robot:
                self._homed_at = time.time() + self.home_seconds

        gh.dispatch_task = dispatch_task
        ros.publish_restart = publish_restart
        ros.publish_return_home = publish_return_home

    def _schedule_start(self) -> None:
        self.progress = 0.0
        if self.auto_robot:
            self._running_at = time.time() + self.start_delay

    # -- controls from the simulator panel ----------------------------------
    def set_safe_home(self, safe: bool) -> None:
        """Pretend the arm stopped inside / outside the validated home zone."""
        ros = self.system.ros
        if safe:
            ros.latest_joint_positions = [0.0] * len(config.SAFE_RETURN_JOINT_RANGES)
            ros.latest_gripper_open = True
        else:
            ros.latest_joint_positions = None
            ros.latest_gripper_open = None

    @property
    def safe_home(self) -> bool:
        return self.system.ros.latest_joint_positions is not None

    def emit(self, event_type: EventType, payload: dict | None = None) -> None:
        self.system.event_queue.put(
            Event(event_type=event_type, source="sim_robot", payload=payload or {})
        )

    def trigger(self, step_id: int) -> None:
        self._trigger_counter += 1
        self.emit(EventType.RECOGNITION_TRIGGER, {
            "step_id": step_id,
            "piece_id": self._trigger_counter,
            "round_id": 0,
            "progress": 1.0,
            "confidence": 1.0,
        })

    # -- called every loop iteration ----------------------------------------
    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        dt, self._last_tick = max(0.0, now - self._last_tick), now
        task = self.system.task_manager.active_task

        if self._running_at is not None and now >= self._running_at:
            self._running_at = None
            self.emit(EventType.ROBOT_RUNNING)

        if self._homed_at is not None and now >= self._homed_at:
            self._homed_at = None
            self.emit(EventType.ROBOT_HOMED)

        if task is None or task.state not in {RobotTaskState.R_EXECUTING, RobotTaskState.R_PAUSED}:
            if task is None or task.state not in {RobotTaskState.R_ACCEPTED, RobotTaskState.R_REDO}:
                self.progress = None
            return

        if task.state is RobotTaskState.R_EXECUTING and self.progress is not None:
            speed_factor = task.speed / config.DEFAULT_SPEED
            self.progress = min(1.0, self.progress + dt * speed_factor / self.task_seconds)
            if self.progress >= 1.0 and self.auto_robot and not task.robot_success_received:
                self.progress = None
                self.emit(EventType.ROBOT_SUCCESS)
