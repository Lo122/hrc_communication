"""Central HRC task state-management skeleton."""

import time

import config
import task_catalog
from events import Event, EventType, RobotTaskState
from models import RobotTask

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
    ):
        self.state_machine = state_machine
        self.pending_pool = pending_pool
        self.timer = timer_manager
        self.message_manager = message_manager
        self.cli = cli
        self.gh_dispatcher = gh_dispatcher
        self.ros = ros_communication
        self.logger = logger

        self.active_task: RobotTask | None = None

        self.ros.publish_speed(config.DEFAULT_SPEED)
        print({"Initialize the speed to:": config.DEFAULT_SPEED})

    def handle_event(self, event: Event) -> None:
        """Route an event to the corresponding handler."""
        self.logger.log_event(event)

        handlers = {
            EventType.RECOGNITION_TRIGGER: self._handle_recognition_trigger,
            EventType.HUMAN_LOCATION_UPDATE: self._handle_human_location_update,
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
            EventType.H_SECURED: self._handle_secured,
            EventType.H_NOT_YET: self._handle_not_yet,
            EventType.ROBOT_RUNNING: self._handle_robot_running,
            EventType.ROBOT_SUCCESS: self._handle_robot_success,
            EventType.ROBOT_HOMED: self._handle_robot_homed,
        }

        handler = handlers.get(event.event_type)
        if handler is None:
            self._log_invalid(event, "No handler registered for event.")
            return

        handler(event)

    def _handle_recognition_trigger(self, event: Event) -> None:
        """Create a task and ask the human for permission."""
        if self.active_task is not None:
            self.logger.log_message("Ignored trigger because an active task exists.", event.payload)
            return

        step_id = event.payload["step_id"]
        piece_id = event.payload["piece_id"]
        round_id = event.payload["round_id"]
        task_key = event.payload.get("task_key", "")
        now = time.time()

        task = RobotTask(
            task_instance_id=self._build_task_instance_id(round_id, step_id, piece_id),
            step_id=step_id,
            task_key=task_key,
            piece_id=piece_id,
            round_id=round_id,
            state=RobotTaskState.R_WAITING_RESPONSE,
            speed=config.DEFAULT_SPEED,
            progress=event.payload.get("progress", 0.0),
            created_at=now,
            updated_at=now,
        )
        self.active_task = task

        self.cli.show_permission_request(self.message_manager.get_task_proposal(task))
        self.timer.start_response_timer(task.task_instance_id, config.RESPONSE_TIMEOUT_SECONDS)
        self.logger.log_message("Task entered R_WAITING_RESPONSE.", {"task_instance_id": task.task_instance_id})

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
        # An announced task can also be confirmed early, so the human
        # does not have to wait out a window they already agree with.
        task = self._require_active_in(
            event, {RobotTaskState.R_WAITING_RESPONSE, RobotTaskState.R_ANNOUNCED}
        )
        if task is None:
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_ACCEPTED, event, "Human accepted task.")
        self.gh_dispatcher.dispatch_task(task)
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

    def _handle_refuse(self, event: Event) -> None:
        task = self._require_active_in(
            event,
            {
                RobotTaskState.R_WAITING_RESPONSE,
                RobotTaskState.R_ANNOUNCED,
                RobotTaskState.R_WAITING_FREE_DRIVE,
                RobotTaskState.R_WAITING_HOME_PERMISSION,
            },
        )
        if task is None:
            return

        if task.state == RobotTaskState.R_WAITING_FREE_DRIVE:
            self._transition(
                task,
                RobotTaskState.R_DONE,
                event,
                "Human declined free-drive mode; task completed.",
            )
            self.active_task = None
            self.cli.show_message(self.message_manager.get_acknowledgement(EventType.ROBOT_SUCCESS))
            return

        if task.state == RobotTaskState.R_WAITING_HOME_PERMISSION:
            self._enter_manual_recovery(task, event)
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_REFUSED, event, "Human refused task.")
        task.pending_reason = "refused"
        self.pending_pool.add(task)
        self.active_task = None
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

    def _handle_defer(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_DEFER, event, "Human deferred task.")
        self.timer.start_defer_timer(task.task_instance_id, config.DEFER_SECONDS)
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

    def _handle_response_timeout(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self._transition(task, RobotTaskState.R_PENDING, event, "Response timeout.")
        task.pending_reason = "timeout"
        self.pending_pool.add(task)
        self.active_task = None

    def _handle_defer_timeout(self, event: Event) -> None:
        # Fires for both R_DEFER (human asked to wait) and R_ANNOUNCED (the
        # veto window closed). Same outcome, different meaning in the log.
        task = self._require_active_in(
            event, {RobotTaskState.R_DEFER, RobotTaskState.R_ANNOUNCED}
        )
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

        task = self.pending_pool.remove(event.task_instance_id)
        self.active_task = task
        self.gh_dispatcher.dispatch_task(task)
        self._transition(task, RobotTaskState.R_ACCEPTED, event, "Pending task dispatched; waiting for robot running status.")

    def _handle_pause(self, event: Event) -> None:
        # Inside a veto window "stop" means refuse, not pause. Rewritten
        # as a real H_REFUSE so the log records a refusal and the human
        # hears the refusal wording, not "say resume when you are ready".
        if self.active_task is not None and self.active_task.state == RobotTaskState.R_ANNOUNCED:
            self._handle_refuse(
                Event(
                    event_type=EventType.H_REFUSE,
                    source=event.source,
                    task_instance_id=event.task_instance_id,
                    payload={"rewritten_from": event.event_type.name},
                )
            )
            return

        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        self.ros.publish_pause()
        self._transition(task, RobotTaskState.R_PAUSED, event, "ROS pause published.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

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
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

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
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

    def _handle_cancel(self, event: Event) -> None:
        # Confirmation gate first -- before anything is published to ROS.
        # Only load-bearing tasks are gated (task_catalog.carries_load, plus
        # anything already in R_HOLDING): cancelling a trip to fetch a tool
        # is harmless, cancelling a panel hold drops a panel.
        if not self._cancel_confirmed(event):
            return

        # self.ros.publish_pause()
        self.ros.publish_cancel()
        task = self._require_active_in(
            event,
            {
                RobotTaskState.R_ACCEPTED,
                RobotTaskState.R_EXECUTING,
                RobotTaskState.R_PAUSED,
                RobotTaskState.R_DEFER,
                RobotTaskState.R_ANNOUNCED,
                RobotTaskState.R_REDO,
                RobotTaskState.R_WAITING_FREE_DRIVE,
                RobotTaskState.R_FREE_DRIVE,
                RobotTaskState.R_MANUAL_RECOVERY,
            },
        )
        if task is None:
            return

        if task.state in {RobotTaskState.R_DEFER, RobotTaskState.R_ANNOUNCED}:
            self.timer.cancel_defer_timer()
            self._finish_canceled_task(task, event, "Deferred task canceled.")
            return

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
        #region CHANGE HERE IN ROBOLAB!!!!
        # gripper_has_object = self.ros.get_latest_gripper_has_object()
        gripper_has_object = False

        return_check = self._can_return_home(joint_positions, gripper_has_object)

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
            self.cli.show_message(self.message_manager.get_return_home_permission_message())
            return

        decision_event = Event(
            event_type=EventType.RECOVERY_MANUAL_REQUIRED,
            source="task_manager",
            task_instance_id=task.task_instance_id,
        )
        self._enter_manual_recovery(task, decision_event)

    def _cancel_needs_confirmation(self, task: RobotTask) -> bool:
        """True when letting go right now would drop something."""
        if task.state == RobotTaskState.R_HOLDING:
            return True
        spec = task_catalog.task_by_key(task.task_key)
        return bool(spec is not None and spec.carries_load)

    def _cancel_confirmed(self, event: Event) -> bool:
        """Gate a cancel behind a second confirmation where it matters.

        Returns True when the cancel should go through. Returns False after
        asking, having recorded the request -- a second cancel inside
        CANCEL_CONFIRM_WINDOW_SECONDS passes the gate.
        """
        task = self.active_task
        if task is None or not self._cancel_needs_confirmation(task):
            return True

        now = time.time()
        asked_at = task.cancel_requested_at
        if asked_at is not None and (now - asked_at) <= config.CANCEL_CONFIRM_WINDOW_SECONDS:
            task.cancel_requested_at = None
            self.logger.log_message(
                "Cancel confirmed by a second request.",
                {"task_instance_id": task.task_instance_id, "task_key": task.task_key},
            )
            return True

        task.cancel_requested_at = now
        self.logger.log_message(
            "Cancel held for confirmation on a load-bearing task.",
            {
                "task_instance_id": task.task_instance_id,
                "task_key": task.task_key,
                "state": task.state.name,
            },
        )
        self.cli.show_message(self.message_manager.get_cancel_confirmation(task))
        return False

    def _handle_speedup(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        # Speed Control Formula: increase speed by SPEED_STEP, but do not exceed MAX_SPEED
        task.speed = min(task.speed + config.SPEED_STEP, config.MAX_SPEED)

        self.ros.publish_speed(task.speed)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot speed increased.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

    def _handle_slowdown(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        # Speed Control Formula: decrease speed by SPEED_STEP, but do not go below MIN_SPEED
        task.speed = max(task.speed - config.SPEED_STEP, config.MIN_SPEED)

        self.ros.publish_speed(task.speed)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot speed decreased.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

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
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

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
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

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
        self.cli.show_message(self.message_manager.get_manual_recovery_message())

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
        self.cli.show_message(self.message_manager.get_acknowledgement(EventType.H_CANCEL))

    def _handle_human_done(self, event: Event) -> None:
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
            final_state = RobotTaskState.R_CANCELED if is_manual_recovery else RobotTaskState.R_DONE
            message = (
                "Manual recovery completed; free-drive disabled."
                if is_manual_recovery
                else "Human alignment completed; free-drive disabled."
            )
            self._transition(task, final_state, event, message)
            acknowledgement = EventType.H_CANCEL if is_manual_recovery else EventType.ROBOT_SUCCESS
            self.cli.show_message(self.message_manager.get_acknowledgement(acknowledgement, task))
            if is_manual_recovery:
                self.active_task = None
                return
            # Finishing the alignment is not the end of the sequence -- the
            # panel still has to be held while it is screwed.
            self._propose_followup(task, event)
            return

        self.ros.publish_human_done()
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Human-done published.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))

    def _handle_robot_running(self, event: Event) -> None:
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

        spec = task_catalog.task_by_key(task.task_key)
        mode = spec.mode if spec is not None else task_catalog.MODE_SIMPLE

        if mode == task_catalog.MODE_FREE_DRIVE:
            self._transition(
                task,
                RobotTaskState.R_WAITING_FREE_DRIVE,
                event,
                "Robot success received; waiting for free-drive permission.",
            )
            self.cli.show_message(self.message_manager.ask_permission_for_free_drive())
            return

        if mode == task_catalog.MODE_HOLD_UNTIL_SECURED:
            # The robot has the panel against the frame. It does NOT finish
            # here -- it parks under load until the human says it is fixed.
            # There is deliberately no timeout that releases: a timer that
            # dropped a panel would be the worst failure this system has.
            self._transition(
                task,
                RobotTaskState.R_HOLDING,
                event,
                "Robot reached hold pose; holding until the human confirms the panel is secured.",
            )
            self.cli.show_message(self.message_manager.get_hold_message(task))
            self.timer.start_response_timer(task.task_instance_id, config.HOLD_REMINDER_SECONDS)
            return

        self._transition(task, RobotTaskState.R_DONE, event, "Robot success received.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type, task))
        self._propose_followup(task, event)

    def _handle_secured(self, event: Event) -> None:
        """Human confirms the panel is fixed; the robot may let go."""
        task = self._require_active(event, RobotTaskState.R_HOLDING)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self.ros.publish_human_done()
        self._transition(task, RobotTaskState.R_DONE, event, "Human confirmed the panel is secured; hold released.")
        self.cli.show_message(self.message_manager.get_release_message(task))
        self._propose_followup(task, event)

    def _handle_not_yet(self, event: Event) -> None:
        """Human asks the robot to keep holding. Restarts the reminder only."""
        task = self._require_active(event, RobotTaskState.R_HOLDING)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self.timer.start_response_timer(task.task_instance_id, config.HOLD_REMINDER_SECONDS)
        self.cli.show_message(self.message_manager.get_acknowledgement(EventType.H_NOT_YET, task))

    def _propose_followup(self, task: RobotTask, event: Event) -> None:
        """Offer the next task in the workflow, if the catalog defines one.

        This is where the system's anticipation actually lives. Recognition
        says when a sequence STARTS; the workflow model carries it forward,
        so the robot does not need to re-detect a step it already knows is
        coming.
        """
        self.active_task = None

        spec = task_catalog.task_by_key(task.task_key)
        if spec is None or spec.next_task is None:
            return

        following = task_catalog.task_by_key(spec.next_task)
        if following is None:
            self.logger.log_message(
                "Follow-up task key is not in the catalog.",
                {"task_key": task.task_key, "next_task": spec.next_task},
            )
            return

        now = time.time()
        next_task = RobotTask(
            task_instance_id=self._build_task_instance_id(
                task.round_id, task.step_id, task.piece_id
            ) + f"_{following.key}",
            step_id=task.step_id,
            piece_id=task.piece_id,
            round_id=task.round_id,
            state=RobotTaskState.R_WAITING_RESPONSE,
            speed=config.DEFAULT_SPEED,
            progress=1.0,
            task_key=following.key,
            created_at=now,
            updated_at=now,
        )
        self.active_task = next_task
        self.logger.log_message(
            "Proposed follow-up task from the workflow catalog.",
            {
                "after": task.task_key,
                "task_key": following.key,
                "task_instance_id": next_task.task_instance_id,
                "permission": following.permission,
                "reason": following.reason,
            },
        )

        if following.permission == task_catalog.PERMISSION_ANNOUNCE:
            # The human already agreed to the situation this task continues,
            # so a second question is friction, not consent. Say what is
            # about to happen, leave a window to stop it, then go.
            next_task.state = RobotTaskState.R_ANNOUNCED
            self.cli.show_message(self.message_manager.get_announcement(following))
            self.timer.start_defer_timer(
                next_task.task_instance_id, config.ANNOUNCE_VETO_SECONDS
            )
            return

        self.cli.show_permission_request(self.message_manager.get_followup_proposal(following))
        self.timer.start_response_timer(next_task.task_instance_id, config.RESPONSE_TIMEOUT_SECONDS)

    def _show_execution_dialogue(self, task: RobotTask) -> None:
        self.cli.show_message(self.message_manager.get_execution_message(task))

    def _transition(
        self,
        task: RobotTask,
        new_state: RobotTaskState,
        event: Event,
        message: str | None = None,
    ) -> None:
        """Apply every state change through one logging path."""
        old_state = task.state
        task.state = new_state
        task.updated_at = time.time()
        self.logger.log_transition(task, event, old_state, new_state, message)

    def _build_task_instance_id(self, round_id: int, step_id: int, piece_id: int) -> str:
        return f"round_{round_id}_step_{step_id}_piece_{piece_id}"

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
        self.cli.show_message(self.message_manager.get_invalid_event_message(state, event.event_type))
