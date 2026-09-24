"""Central HRC task state-management skeleton."""

import time
from collections import deque

import config
from events import Event, EventType, RobotTaskState, TaskStatus
from models import RobotTask
from task_database import HUMAN, ROBOT

S = RobotTaskState

# How a robot task's state shows up on the tracked task it performs.
ROBOT_STATE_TO_TASK_STATUS = {
    S.R_WAITING_RESPONSE: TaskStatus.PENDING,
    S.R_REFUSED: TaskStatus.PENDING,
    S.R_PENDING: TaskStatus.PENDING,
    S.R_CANCELED: TaskStatus.PENDING,
    S.R_ACCEPTED: TaskStatus.WORKING,
    S.R_DEFER: TaskStatus.WORKING,
    S.R_EXECUTING: TaskStatus.WORKING,
    S.R_PAUSED: TaskStatus.WORKING,
    S.R_REDO: TaskStatus.WORKING,
    S.R_DONE: TaskStatus.DONE,
}
# The lift itself is over once the robot reports success; free drive and holding
# belong to Place and Screw.
LIFT_FINISHED_STATES = {S.R_WAITING_FREE_DRIVE, S.R_FREE_DRIVE, S.R_HOLDING, S.R_DONE}
ROBOT_TASK_TO_TRACKED = {task_id: name for name, task_id in config.TRACKED_TO_ROBOT_TASK.items()}


class TaskManager:
    """Owns active task state, pending tasks, and all valid transitions."""

    def __init__(
        self,
        state_machine,
        pending_pool,
        timer_manager,
        message_manager,
        cli,
        gh_dispatcher,
        ros_communication,
        logger,
        task_tracker=None,
        trigger_policy=None,
        status_callback=None,
    ):
        self.state_machine = state_machine
        self.pending_pool = pending_pool
        self.timer = timer_manager
        self.message_manager = message_manager
        self.cli = cli
        self.gh_dispatcher = gh_dispatcher
        self.ros = ros_communication
        self.logger = logger
        # Assembly-task tracking (task database + transition table). Without them no
        # robot task is offered automatically; human requests and follow-ups still work.
        self.tracker = task_tracker
        self.policy = trigger_policy
        self.status_callback = status_callback

        self.active_task: RobotTask | None = None
        # Robot offers waiting for the active task to finish:
        # {"task_name", "piece_id", "requested"}.
        self.waiting_triggers = deque()
        self._task_instance_counts = {}
        self._round_id = 0
        self._last_status_line = None

        self.ros.publish_speed(config.DEFAULT_SPEED)
        print({"Initialize the speed to:": config.DEFAULT_SPEED})

    def handle_event(self, event: Event) -> None:
        """Route an event to the corresponding handler."""
        self.logger.log_event(event)

        handlers = {
            EventType.RECOGNITION_TRIGGER: self._handle_human_task_update,
            EventType.HUMAN_TASK_UPDATE: self._handle_human_task_update,
            EventType.HUMAN_LOCATION_UPDATE: self._handle_human_location_update,
            EventType.H_TASK_DONE: self._handle_task_done,
            EventType.H_REQUEST_ROBOT_TASK: self._handle_robot_request,
            EventType.H_NEXT_PIECE: self._handle_next_piece,
            EventType.H_ACCEPT: self._handle_accept,
            EventType.H_REFUSE: self._handle_refuse,
            EventType.H_DEFER: self._handle_defer,
            EventType.RESPONSE_TIMEOUT: self._handle_response_timeout,
            EventType.DEFER_TIMEOUT: self._handle_defer_timeout,
            EventType.H_EXECUTE_PENDING_TASK: self._handle_execute_pending,
            EventType.H_PAUSE: self._handle_pause,
            EventType.H_RESUME: self._handle_resume,
            EventType.H_RESTART: self._handle_restart,
            EventType.H_CANCEL: self._handle_cancel,
            EventType.H_SPEEDUP: self._handle_speedup,
            EventType.H_SLOWDOWN: self._handle_slowdown,
            EventType.H_FREE_GO: self._handle_free_go,
            EventType.H_RETURN_HOME: self._handle_return_home,
            EventType.H_MANUAL_RECOVERY: self._handle_manual_recovery,
            EventType.H_DONE: self._handle_human_done,
            EventType.H_SCREW_DONE: self._handle_screw_done,
            EventType.ROBOT_RUNNING: self._handle_robot_running,
            EventType.ROBOT_SUCCESS: self._handle_robot_success,
            EventType.ROBOT_HOMED: self._handle_robot_homed,
            # EventType.HOLD_WHEN_DISASSEMBLE: self._handle_hold_when_disassemble,
        }

        handler = handlers.get(event.event_type)
        if handler is None:
            self._log_invalid(event, "No handler registered for event.")
            return

        handler(event)
        self._start_next_waiting()
        self._report_status()

    # -- assembly-task tracking ----------------------------------------------

    def _handle_human_task_update(self, event: Event) -> None:
        """Recognition reports the human's current task; offer what it unlocks."""
        step_id = event.payload.get("step_id")
        if not isinstance(step_id, int) or not 0 <= step_id < len(config.STEP_NAMES):
            self.logger.log_message("Human task update with unknown step id.", event.payload)
            return
        self._round_id = int(event.payload.get("round_id", self._round_id))
        if self.tracker is None:
            self.logger.log_message("No task tracker; human task update not tracked.", event.payload)
            return
        self.tracker.on_task_recognized(config.STEP_NAMES[step_id], float(event.payload.get("progress", 0.0)))
        self._offer_triggered_tasks()

    def _offer_triggered_tasks(self) -> None:
        """Offer (or queue) every robot task the trigger rules allow right now."""
        if self.policy is None:
            return
        for task_name, piece_id in self.policy.candidates():
            self.policy.mark_offered(task_name, piece_id)
            if self.active_task is not None or self.waiting_triggers or self._leave_pending():
                self.waiting_triggers.append({"task_name": task_name, "piece_id": piece_id, "requested": False})
                self.logger.log_message("Queued robot offer until the current task is released.",
                                        {"task_name": task_name, "piece_id": piece_id})
                continue
            self._start_robot_task(task_name, piece_id, requested=False)

    def _start_next_waiting(self) -> None:
        while self.active_task is None and self.waiting_triggers and not self._leave_pending():
            entry = self.waiting_triggers.popleft()
            task_name, piece_id = entry["task_name"], entry["piece_id"]
            if not self._tracked_task_open(task_name, piece_id):
                self.logger.log_message("Dropped queued robot offer: task no longer open.",
                                        {"task_name": task_name, "piece_id": piece_id})
                continue
            pooled = self._pooled_task(task_name, piece_id)
            if entry["requested"] and pooled is not None:
                self._execute_pooled(pooled, "Requested pending task dispatched.")
            else:
                self._start_robot_task(task_name, piece_id, requested=entry["requested"])

    def _start_robot_task(self, task_name: str, piece_id: int, requested: bool) -> None:
        task_id = config.TRACKED_TO_ROBOT_TASK[task_name]
        if self.policy is not None:
            self.policy.mark_offered(task_name, piece_id)
        reference = self.tracker.reference_task if self.tracker is not None else None
        context = {
            "step_id": config.STEP_NAMES.index(reference) if reference in config.STEP_NAMES else -1,
            "piece_id": piece_id,
            "round_id": self._round_id,
            "progress": self.tracker.reference_progress if self.tracker is not None else 0.0,
        }
        if not requested:
            self._propose_task(context, task_id)
            return
        # The human asked for it: that request is the permission.
        self._propose_task(context, task_id, ask_permission=False)
        self._handle_accept(Event(EventType.H_ACCEPT, "human_request",
                                  task_instance_id=self.active_task.task_instance_id))

    def _handle_robot_request(self, event: Event) -> None:
        """Human asks the robot for a task: no trigger rule, no permission question."""
        task_name = event.payload.get("task_name")
        if task_name not in config.TRACKED_TO_ROBOT_TASK or (
                self.tracker is not None and not self.tracker.database.can_execute(task_name, ROBOT)):
            self.cli.show_message(f"Sorry, I cannot do {task_name or 'that task'}.",
                                  speech="Sorry, I cannot do that.")
            return
        piece_id = self._request_piece(task_name)
        if piece_id is None:
            self.cli.show_message(f"{task_name} is already done or in progress.",
                                  speech=f"{task_name} is already done or in progress.")
            return
        if self.active_task is not None or self._leave_pending():
            self.waiting_triggers.appendleft({"task_name": task_name, "piece_id": piece_id, "requested": True})
            self.cli.show_message(f"Okay, I will do {task_name} after the current task.",
                                  speech=f"I will do {task_name} next.")
            return
        pooled = self._pooled_task(task_name, piece_id)
        if pooled is not None:
            self._execute_pooled(pooled, "Human requested the pending task; dispatched.")
            return
        self.logger.log_message("Human requested robot task.", {"task_name": task_name, "piece_id": piece_id})
        self._start_robot_task(task_name, piece_id, requested=True)

    def _request_piece(self, task_name: str) -> int | None:
        if self.tracker is None:
            return self.active_task.piece_id if self.active_task is not None else 0
        piece_id = self.tracker.first_open_piece_for(task_name)
        if piece_id is None or not self._tracked_task_open(task_name, piece_id):
            return None
        return piece_id

    def _handle_task_done(self, event: Event) -> None:
        """Human confirms a task (or, with no task name, the one being worked on)."""
        if self.tracker is None:
            self._log_invalid(event, "No task tracker to confirm tasks with.")
            return
        task = self.tracker.confirm_done(event.payload.get("task_name"), HUMAN)
        if task is None:
            name = event.payload.get("task_name") or "working task"
            self.cli.show_message(f"There is no open {name} to mark done.", speech="Nothing to mark done.")
            return
        self._withdraw_finished_offers()
        self.cli.show_message(f"Okay, {task.task_name} is done.", speech=f"{task.task_name} done.")
        self._offer_triggered_tasks()

    def _handle_next_piece(self, event: Event) -> None:
        if self.tracker is None:
            self._log_invalid(event, "No task tracker.")
            return
        piece_id = self.tracker.advance_piece()
        text = "All pieces are done." if piece_id is None else f"Moving on to piece {piece_id}."
        self.cli.show_message(text, speech=text)

    def _withdraw_finished_offers(self) -> None:
        """Drop robot offers for tasks that are done now (by the human, or inferred)."""
        if self.tracker is None:
            return
        for task in self.pending_pool.list_all():
            name = ROBOT_TASK_TO_TRACKED.get(task.task_id)
            if name is not None and not self._tracked_task_open(name, task.piece_id, allow_working=True):
                self.pending_pool.remove(task.task_instance_id)
                self.logger.log_message("Removed pending robot task: task already done.",
                                        {"task_instance_id": task.task_instance_id})
        task = self.active_task
        name = ROBOT_TASK_TO_TRACKED.get(task.task_id) if task is not None else None
        if (name is not None and task.state == S.R_WAITING_RESPONSE
                and not self._tracked_task_open(name, task.piece_id, allow_working=True)):
            self.timer.cancel_response_timer()
            self._transition(task, S.R_CANCELED,
                             Event(EventType.H_TASK_DONE, "task_manager", task.task_instance_id),
                             "Offer withdrawn: the human already did the task.")
            self.active_task = None

    def _tracked_task_open(self, task_name: str, piece_id: int, allow_working=False) -> bool:
        if self.tracker is None:
            return True
        task = self.tracker.get(task_name, piece_id)
        if task is None:
            return False
        return task.status != TaskStatus.DONE and (allow_working or task.status != TaskStatus.WORKING)

    def _pooled_task(self, task_name: str, piece_id: int) -> RobotTask | None:
        task_id = config.TRACKED_TO_ROBOT_TASK.get(task_name)
        for task in self.pending_pool.list_all():
            if task.task_id == task_id and task.piece_id == piece_id:
                return task
        return None

    def _leave_pending(self) -> bool:
        # A pending leave task still owns the held panel.
        return any(task.task_id == config.TASK_LEAVE for task in self.pending_pool.list_all())

    def _sync_tracker(self, task: RobotTask, event: Event) -> None:
        """Mirror a robot task's state onto the tracked task it performs."""
        name = ROBOT_TASK_TO_TRACKED.get(task.task_id)
        if self.tracker is None or name is None:
            return
        if task.task_id == config.TASK_LIFT_PANEL and task.state in LIFT_FINISHED_STATES:
            self.tracker.set_robot_status(name, task.piece_id, TaskStatus.DONE)
            if task.state == S.R_HOLDING:
                # Adjusted by the human in free drive, or left where the robot put it.
                executor = HUMAN if event.event_type == EventType.H_DONE else ROBOT
                self.tracker.confirm_done("Place", executor, task.piece_id)
            return
        status = ROBOT_STATE_TO_TASK_STATUS.get(task.state)
        if status is not None:
            self.tracker.set_robot_status(name, task.piece_id, status)

    def _report_status(self) -> None:
        if self.tracker is None:
            return
        line = self.tracker.status_line()
        if line != self._last_status_line:
            self._last_status_line = line
            self.logger.log_message("Task status.", {"line": line})
            if self.status_callback is not None:
                self.status_callback(line)

    # -- robot tasks ------------------------------------------------------------

    def _propose_task(self, context: dict, task_id: int, ask_permission: bool = True) -> None:
        """Ask permission for one robot action, retaining its human context."""
        step_id = context["step_id"]
        piece_id = context["piece_id"]
        round_id = context["round_id"]
        now = time.time()

        task = RobotTask(
            task_instance_id=self._build_task_instance_id(round_id, task_id, piece_id),
            step_id=step_id,
            task_id=task_id,
            piece_id=piece_id,
            round_id=round_id,
            state=RobotTaskState.R_WAITING_RESPONSE,
            speed=config.DEFAULT_SPEED,
            progress=context.get("progress", 0.0),
            created_at=now,
            updated_at=now,
        )
        self.active_task = task
        self._sync_tracker(task, Event(EventType.HUMAN_TASK_UPDATE, "task_manager"))
        self.logger.log_message("Task entered R_WAITING_RESPONSE.", {"task_instance_id": task.task_instance_id})
        if not ask_permission:
            return

        message = self.message_manager.get_permission_message(task.task_id)
        self.cli.show_permission_request(
            message, speech=self.message_manager.get_permission_message(task.task_id, spoken=True),
        )
        duration = self._task_duration(task, "response_timeout_seconds", config.RESPONSE_TIMEOUT_SECONDS)
        self.timer.start_response_timer(task.task_instance_id, duration)

    def _task_duration(self, task: RobotTask, key: str, default: float) -> float:
        return config.TASK_TIMINGS.get(task.task_id, {}).get(key, default)

    def _propose_followup(self, task: RobotTask, task_id: int) -> None:
        name = ROBOT_TASK_TO_TRACKED.get(task_id)
        if name is not None:
            if not self._tracked_task_open(name, task.piece_id):
                self.logger.log_message("Skipped follow-up: task no longer open.",
                                        {"task_name": name, "piece_id": task.piece_id})
                return
            if self.policy is not None:
                self.policy.mark_offered(name, task.piece_id)
            self.waiting_triggers = deque(
                entry for entry in self.waiting_triggers
                if (entry["task_name"], entry["piece_id"]) != (name, task.piece_id))
        self._propose_task({
            "step_id": config.HUMAN_SCREW_DONE,
            "piece_id": task.piece_id,
            "round_id": task.round_id,
            "progress": 1.0,
        }, task_id)

    def _handle_human_location_update(self, event: Event) -> None:
        """Forward the human's world-frame position (plus, if this frame
        had one, their pelvis-relative posture keypoints) to both
        consumers -- Grasshopper (visualization) and ROS (path planning).
        Stateless: doesn't touch active_task/the state machine, just
        relays."""
        xyz = (event.payload["x"], event.payload["y"], event.payload["z"])
        timestamp = event.payload.get("timestamp")
        keypoints = event.payload.get("keypoints")
        # UDPEventReceiver is message transfer
        # self.gh_dispatcher.dispatch_human_location(xyz, timestamp)
        # ROS message transfer
        self.ros.publish_human_location(xyz, timestamp, keypoints)

    def _handle_accept(self, event: Event) -> None:
        if self.active_task is not None:
            if self.active_task.state == RobotTaskState.R_WAITING_FREE_DRIVE:
                self._handle_free_go(event)
                return
            if self.active_task.state == RobotTaskState.R_WAITING_HOME_PERMISSION:
                self._handle_return_home(event)
                return
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_ACCEPTED, event, "Human accepted task.")
        self.gh_dispatcher.dispatch_task(task)
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_refuse(self, event: Event) -> None:
        task = self._require_active_in(
            event,
            {
                RobotTaskState.R_WAITING_RESPONSE,
                RobotTaskState.R_WAITING_FREE_DRIVE,
                RobotTaskState.R_WAITING_HOME_PERMISSION,
            },
        )
        if task is None:
            return

        if task.state == RobotTaskState.R_WAITING_FREE_DRIVE:
            self._enter_holding(task, event)
            return

        if task.state == RobotTaskState.R_WAITING_HOME_PERMISSION:
            self._enter_manual_recovery(task, event)
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_REFUSED, event, "Human refused task.")
        task.pending_reason = "refused"
        self.pending_pool.add(task)
        self.active_task = None
        self.cli.show_message(
            self.message_manager.get_pending_message(task),
            speech=self.message_manager.get_pending_message(task, spoken=True),
        )

    def _handle_defer(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_DEFER, event, "Human deferred task.")
        duration = self._task_duration(task, "defer_seconds", config.DEFER_SECONDS)
        self.cli.show_message(
            self.message_manager.get_defer_message(task, duration),
            speech=self.message_manager.get_defer_message(task, duration, spoken=True),
        )
        self.timer.start_defer_timer(task.task_instance_id, duration)

    def _handle_response_timeout(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self._transition(task, RobotTaskState.R_PENDING, event, "Response timeout.")
        task.pending_reason = "timeout"
        self.pending_pool.add(task)
        self.active_task = None
        self.cli.show_message(
            self.message_manager.get_pending_message(task),
            speech=self.message_manager.get_pending_message(task, spoken=True),
        )

    def _handle_defer_timeout(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_DEFER)
        if task is None:
            return

        self.gh_dispatcher.dispatch_task(task)
        self._transition(task, RobotTaskState.R_ACCEPTED, event, "Deferred task dispatched; waiting for robot running status.")

    def _handle_execute_pending(self, event: Event) -> None:
        if self.active_task is not None:
            self._log_invalid(event, "Cannot execute pending task while active task exists.")
            return
        if event.task_instance_id is None or not self.pending_pool.contains(event.task_instance_id):
            self._log_invalid(event, "Pending task id not found.")
            return

        if any(
            task.task_id == config.TASK_LEAVE and task.task_instance_id != event.task_instance_id
            for task in self.pending_pool.list_all()
        ):
            self._log_invalid(event, "Execute the pending leave task before starting another task.")
            return

        self._execute_pooled(self.pending_pool.get(event.task_instance_id),
                             "Pending task dispatched; waiting for robot running status.", event)

    def _execute_pooled(self, task: RobotTask, message: str, event: Event | None = None) -> None:
        """Dispatch a refused/timed-out robot task from the pending pool."""
        if event is None:
            event = Event(EventType.H_EXECUTE_PENDING_TASK, "task_manager", task.task_instance_id)
        self.pending_pool.remove(task.task_instance_id)
        self.active_task = task
        self.gh_dispatcher.dispatch_task(task)
        self._transition(task, RobotTaskState.R_ACCEPTED, event, message)

    def _handle_pause(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        self.ros.publish_pause()
        self._transition(task, RobotTaskState.R_PAUSED, event, "ROS pause published.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_resume(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_PAUSED)
        if task is None:
            return

        self.ros.publish_resume(task.speed)
        self._transition(
            task,
            RobotTaskState.R_EXECUTING,
            event,
            f"Robot resumed at saved speed {task.speed}.",
        )
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_restart(self, event: Event) -> None:
        task = self._require_active_in(
            event,
            {RobotTaskState.R_EXECUTING, RobotTaskState.R_PAUSED},
        )
        if task is None:
            return

        self.ros.publish_restart()
        task.robot_running_received = False
        task.robot_success_received = False
        self._transition(task, RobotTaskState.R_REDO, event, "ROS restart published; waiting for robot running status.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_cancel(self, event: Event) -> None:

        task = self._require_active_in(
            event,
            {
                RobotTaskState.R_ACCEPTED,
                RobotTaskState.R_EXECUTING,
                RobotTaskState.R_PAUSED,
                RobotTaskState.R_DEFER,
                RobotTaskState.R_REDO,
                RobotTaskState.R_WAITING_FREE_DRIVE,
                RobotTaskState.R_FREE_DRIVE,
                RobotTaskState.R_HOLDING,
                RobotTaskState.R_MANUAL_RECOVERY,
            },
        )
        if task is None:
            return

        if task.state == RobotTaskState.R_DEFER:
            self.timer.cancel_defer_timer()
            if task.task_id == config.TASK_LEAVE:
                self._enter_holding(task, event)
            else:
                self._finish_canceled_task(task, event, "Deferred task canceled.")
            return

        was_holding = task.state == RobotTaskState.R_HOLDING
        self.ros.publish_cancel()
        if task.free_drive_active:
            self.ros.publish_free_drive(False)
            task.free_drive_active = False
            self._finish_canceled_task(task, event, "Free-drive disabled; task canceled.")
            return

        if task.state == RobotTaskState.R_WAITING_FREE_DRIVE:
            self._finish_canceled_task(task, event, "Task canceled before free-drive started.")
            return

        self._transition(
            task,
            RobotTaskState.R_RECOVERY_EVALUATING,
            event,
            "Stop published; evaluating recovery strategy.",
        )
        time.sleep(config.RECOVERY_STOP_DELAY_SECONDS)
        joint_positions = self.ros.get_latest_joint_positions()
        gripper_has_object = self.ros.get_latest_gripper_has_object()
        if was_holding:
            # The workflow still owns a held panel; an old open-gripper sample
            # must not authorize returning home from this state.
            gripper_has_object = True

        return_check = self._can_return_home(joint_positions, gripper_has_object)

        #can return home directly with permission
        if return_check:
            decision_event = Event(
                event_type=EventType.RECOVERY_HOME_AVAILABLE,
                source="task_manager",
                task_instance_id=task.task_instance_id,
            )
            self._transition(
                task,
                RobotTaskState.R_WAITING_HOME_PERMISSION,
                decision_event,
                "Recovery conditions allow return home; waiting for permission.",
            )
            self.cli.show_message(
                self.message_manager.get_return_home_permission_message(),
                speech=self.message_manager.get_return_home_permission_message(spoken=True),
            )
            return

        decision_event = Event(
            event_type=EventType.RECOVERY_MANUAL_REQUIRED,
            source="task_manager",
            task_instance_id=task.task_instance_id,
        )
        self._enter_manual_recovery(task, decision_event)

    def _handle_speedup(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        # Speed Control Formula: increase speed by SPEED_STEP, but do not exceed MAX_SPEED
        task.speed = min(task.speed + config.SPEED_STEP, config.MAX_SPEED)

        self.ros.publish_speed(task.speed)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot speed increased.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_slowdown(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        # Speed Control Formula: decrease speed by SPEED_STEP, but do not go below MIN_SPEED
        task.speed = max(task.speed - config.SPEED_STEP, config.MIN_SPEED)

        self.ros.publish_speed(task.speed)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot speed decreased.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _handle_free_go(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_FREE_DRIVE)
        if task is None:
            return

        self.ros.publish_free_drive(True)
        task.free_drive_active = True
        self._transition(
            task,
            RobotTaskState.R_FREE_DRIVE,
            event,
            "Human approved free-drive mode; free-drive enabled.",
        )
        self.cli.show_message(
            self.message_manager.get_acknowledgement(EventType.H_FREE_GO),
            speech=self.message_manager.get_acknowledgement(EventType.H_FREE_GO, spoken=True),
        )

    def _handle_return_home(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_HOME_PERMISSION)
        if task is None:
            return

        self.ros.publish_return_home()
        self._transition(
            task,
            RobotTaskState.R_RETURNING_HOME,
            event,
            "Return-home command published.",
        )
        self.cli.show_message(
            self.message_manager.get_acknowledgement(EventType.H_RETURN_HOME),
            speech=self.message_manager.get_acknowledgement(EventType.H_RETURN_HOME, spoken=True),
        )

    def _handle_manual_recovery(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_HOME_PERMISSION)
        if task is None:
            return

        self._enter_manual_recovery(task, event)

    def _enter_manual_recovery(self, task: RobotTask, event: Event) -> None:
        self.ros.publish_free_drive(True)
        task.free_drive_active = True
        self._transition(
            task,
            RobotTaskState.R_MANUAL_RECOVERY,
            event,
            "Manual recovery required; free-drive enabled.",
        )
        self.cli.show_message(
            self.message_manager.get_manual_recovery_message(),
            speech=self.message_manager.get_manual_recovery_message(spoken=True),
        )

    def _can_return_home(
        self,
        joint_positions: list[float] | None,
        gripper_has_object: bool | None,
    ) -> bool:
        return (
            config.RETURN_HOME_RECOVERY_ENABLED
            and gripper_has_object is False
            and self._is_in_safe_return_zone(joint_positions)
        )

    def _is_in_safe_return_zone(
        self,
        joint_positions: list[float] | None,
    ) -> bool:
        ranges = config.SAFE_RETURN_JOINT_RANGES
        if joint_positions is None or ranges is None or len(joint_positions) != len(ranges):
            return False

        return all(
            lower <= position <= upper
            for position, (lower, upper) in zip(joint_positions, ranges)
        )

    def _finish_canceled_task(self, task: RobotTask, event: Event, message: str) -> None:
        self._transition(task, RobotTaskState.R_CANCELED, event, message)
        self.active_task = None
        self.cli.show_message(
            self.message_manager.get_acknowledgement(EventType.H_CANCEL),
            speech=self.message_manager.get_acknowledgement(EventType.H_CANCEL, spoken=True),
        )

    def _handle_human_done(self, event: Event) -> None:
        if self.active_task is None and self.tracker is not None:
            # No robot task to report to: "done" confirms the human's own task.
            self._handle_task_done(Event(EventType.H_TASK_DONE, event.source))
            return
        task = self._require_active_in(
            event,
            {
                RobotTaskState.R_EXECUTING,
                RobotTaskState.R_FREE_DRIVE,
                RobotTaskState.R_MANUAL_RECOVERY,
            },
        )
        if task is None:
            return

        if task.state in {RobotTaskState.R_FREE_DRIVE, RobotTaskState.R_MANUAL_RECOVERY}:
            is_manual_recovery = task.state == RobotTaskState.R_MANUAL_RECOVERY
            self.ros.publish_free_drive(False)
            task.free_drive_active = False
            if not is_manual_recovery:
                self._enter_holding(task, event)
                return
            self._finish_canceled_task(task, event, "Manual recovery completed; free-drive disabled.")
            return

        self.ros.publish_human_done()
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Human-done published.")
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _enter_holding(self, task: RobotTask, event: Event) -> None:
        self._transition(task, RobotTaskState.R_HOLDING, event, "Panel held; waiting for screw done.")
        message = self.message_manager.get_holding_message()
        speech = self.message_manager.get_holding_message(spoken=True)
        if event.event_type == EventType.H_CANCEL:
            message = "The delayed leave action is canceled. " + message
            speech = "Leave canceled. " + speech
        self.cli.show_message(message, speech=speech)

    def _handle_screw_done(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_HOLDING)
        if task is None:
            return
        self._transition(task, RobotTaskState.R_DONE, event, "Screwing finished; proposing leave action.")
        if self.tracker is not None:
            self.tracker.confirm_done("Screw", HUMAN, task.piece_id)
            self._withdraw_finished_offers()
        self._propose_followup(task, config.TASK_LEAVE)
        # Screw done can unlock support tasks; they queue behind the leave action.
        self._offer_triggered_tasks()

    def _handle_robot_running(self, event: Event) -> None:
        # time.sleep(config.RECOVERY_STOP_DELAY_SECONDS)
        task = self._require_active_in(
            event,
            {RobotTaskState.R_ACCEPTED, RobotTaskState.R_REDO, RobotTaskState.R_EXECUTING},
        )
        if task is None:
            return

        if task.robot_running_received:
            #send default speed from config.py
            # self.ros.publish_speed(config.DEFAULT_SPEED)
            # print({"Initialize the speed to:": config.DEFAULT_SPEED})
            self.logger.log_message(
                "Ignored repeated robot running status.",
                {"task_instance_id": task.task_instance_id},
            )
            return

        task.robot_running_received = True
        if task.state != RobotTaskState.R_EXECUTING:
            self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot physical status running.")
        self._show_execution_dialogue(task)

    def _handle_robot_homed(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_RETURNING_HOME)
        if task is None:
            return

        self._finish_canceled_task(
            task,
            event,
            "Robot homed; cancellation recovery completed.",
        )

    def _handle_robot_success(self, event: Event) -> None:
        task = self._require_active_in(
            event,
            {RobotTaskState.R_EXECUTING, RobotTaskState.R_PAUSED},
        )
        if task is None:
            return

        if task.robot_success_received:
            self.logger.log_message(
                "Ignored repeated robot success.",
                {"task_instance_id": task.task_instance_id},
            )
            return
        task.robot_success_received = True

        next_state = self.state_machine.get_next_state(task.state, event.event_type, task.task_id)
        if next_state == RobotTaskState.R_WAITING_FREE_DRIVE:
            self._transition(
                task,
                RobotTaskState.R_WAITING_FREE_DRIVE,
                event,
                "Robot success received; waiting for free-drive permission.",
            )
            self.logger.log_message(
                "Asked human for permission to enable free-drive mode.",
                {"task_instance_id": task.task_instance_id},
            )
            self.cli.show_message(
                self.message_manager.ask_permission_for_free_drive(),
                speech=self.message_manager.ask_permission_for_free_drive(spoken=True),
            )
            return

        self._transition(task, RobotTaskState.R_DONE, event, "Robot success received.")
        self.active_task = None
        if task.task_id == config.TASK_LEAVE:
            self._propose_followup(task, config.TASK_BRING_CONNECTOR)
            return
        self.cli.show_message(
            self.message_manager.get_acknowledgement(event.event_type),
            speech=self.message_manager.get_acknowledgement(event.event_type, spoken=True),
        )

    def _show_execution_dialogue(self, task: RobotTask) -> None:
        self.cli.show_message(
            self.message_manager.get_execution_message(task),
            speech=self.message_manager.get_execution_message(task, spoken=True),
        )

    def _transition(
        self,
        task: RobotTask,
        new_state: RobotTaskState,
        event: Event,
        message: str | None = None,
    ) -> None:
        """Apply every state change through one logging path."""
        old_state = task.state
        expected = self.state_machine.get_next_state(old_state, event.event_type, task.task_id)
        if expected != new_state:
            raise ValueError(
                f"Invalid transition for task {task.task_id}: "
                f"{old_state.name} + {event.event_type.name} -> {new_state.name}"
            )
        task.state = new_state
        task.updated_at = time.time()
        self.logger.log_transition(task, event, old_state, new_state, message)
        self._sync_tracker(task, event)

    def _build_task_instance_id(self, round_id: int, task_id: int, piece_id: int) -> str:
        base = f"round_{round_id}_task_{task_id}_piece_{piece_id}"
        attempt = self._task_instance_counts.get(base, 0) + 1
        self._task_instance_counts[base] = attempt
        return base if attempt == 1 else f"{base}_attempt_{attempt}"

    def _require_active(self, event: Event, state: RobotTaskState) -> RobotTask | None:
        return self._require_active_in(event, {state})

    def _require_active_in(self, event: Event, states: set[RobotTaskState]) -> RobotTask | None:
        task = self.active_task
        if task is None:
            self._log_invalid(event, "No active task.")
            return None
        if event.task_instance_id is not None and event.task_instance_id != task.task_instance_id:
            self._log_invalid(event, "Event task id does not match active task.")
            return None
        if task.state not in states:
            self._log_invalid(event, "Invalid event for current task state.")
            return None
        return task

    def _log_invalid(self, event: Event, message: str) -> None:
        state = self.active_task.state if self.active_task is not None else None
        self.logger.log_message(
            message,
            {
                "event_type": event.event_type.name,
                "state": state.name if state is not None else None,
            },
        )
        self.cli.show_message(
            self.message_manager.get_invalid_event_message(state, event.event_type),
            speech=self.message_manager.get_invalid_event_message(state, event.event_type, spoken=True),
        )
