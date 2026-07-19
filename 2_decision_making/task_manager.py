"""Central HRC task state-management skeleton."""

import time

import config
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

    def handle_event(self, event: Event) -> None:
        """Route an event to the corresponding handler."""
        self.logger.log_event(event)

        handlers = {
            EventType.RECOGNITION_TRIGGER: self._handle_recognition_trigger,
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
            EventType.H_DONE: self._handle_human_done,
            EventType.ROBOT_SUCCESS: self._handle_robot_success,
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
        now = time.time()

        task = RobotTask(
            task_instance_id=self._build_task_instance_id(round_id, step_id, piece_id),
            step_id=step_id,
            piece_id=piece_id,
            round_id=round_id,
            state=RobotTaskState.R_WAITING_RESPONSE,
            speed=config.DEFAULT_SPEED,
            progress=event.payload.get("progress", 0.0),
            created_at=now,
            updated_at=now,
        )
        self.active_task = task

        message = self.message_manager.get_permission_message(task.step_id)
        self.cli.show_permission_request(message)
        self.timer.start_response_timer(task.task_instance_id, config.RESPONSE_TIMEOUT_SECONDS)
        self.logger.log_message("Task entered R_WAITING_RESPONSE.", {"task_instance_id": task.task_instance_id})

    def _handle_accept(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_ACCEPTED, event, "Human accepted task.")
        self.gh_dispatcher.dispatch_task(task)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Task dispatched to Grasshopper.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type))
        self._show_execution_dialogue(task)

    def _handle_refuse(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_REFUSED, event, "Human refused task.")
        task.pending_reason = "refused"
        self.pending_pool.add(task)
        self.active_task = None
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type))

    def _handle_defer(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self.timer.cancel_response_timer()
        self._transition(task, RobotTaskState.R_DEFER, event, "Human deferred task.")
        self.timer.start_defer_timer(task.task_instance_id, config.DEFER_SECONDS)
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type))

    def _handle_response_timeout(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_WAITING_RESPONSE)
        if task is None:
            return

        self._transition(task, RobotTaskState.R_PENDING, event, "Response timeout.")
        task.pending_reason = "timeout"
        self.pending_pool.add(task)
        self.active_task = None

    def _handle_defer_timeout(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_DEFER)
        if task is None:
            return

        self.gh_dispatcher.dispatch_task(task)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Deferred task dispatched.")
        self._show_execution_dialogue(task)

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
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Pending task dispatched.")
        self._show_execution_dialogue(task)

    def _handle_pause(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        self.ros.publish_pause()
        self._transition(task, RobotTaskState.R_PAUSED, event, "ROS pause published.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type))

    def _handle_resume(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_PAUSED)
        if task is None:
            return

        self.ros.publish_resume()
        self._transition(task, RobotTaskState.R_RESUME, event, "ROS resume published.")
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Task executing after resume.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type))
        self._show_execution_dialogue(task)

    def _handle_restart(self, event: Event) -> None:
        task = self._require_active_in(event, {RobotTaskState.R_EXECUTING, RobotTaskState.R_PAUSED})
        if task is None:
            return

        self.ros.publish_restart()
        self._transition(task, RobotTaskState.R_REDO, event, "ROS restart published.")
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Task executing after restart.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type))
        self._show_execution_dialogue(task)

    def _handle_cancel(self, event: Event) -> None:
        task = self._require_active_in(
            event,
            {RobotTaskState.R_EXECUTING, RobotTaskState.R_PAUSED, RobotTaskState.R_DEFER},
        )
        if task is None:
            return

        if task.free_drive_active:
            self.ros.publish_free_drive(False)
            task.free_drive_active = False

        if task.state == RobotTaskState.R_DEFER:
            self.timer.cancel_defer_timer()
        else:
            self.ros.publish_cancel()

        self._transition(task, RobotTaskState.R_CANCELED, event, "Task canceled.")
        self.active_task = None
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type))

    def _handle_speedup(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        task.speed = min(task.speed + config.SPEED_STEP, config.MAX_SPEED)
        self.ros.publish_speed(task.speed)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot speed increased.")

    def _handle_slowdown(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        task.speed = max(task.speed - config.SPEED_STEP, config.MIN_SPEED)
        self.ros.publish_speed(task.speed)
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Robot speed decreased.")

    def _handle_human_done(self, event: Event) -> None:
        task = self._require_active(event, RobotTaskState.R_EXECUTING)
        if task is None:
            return

        if task.free_drive_active:
            self.ros.publish_free_drive(False)
            task.free_drive_active = False
            self._transition(task, RobotTaskState.R_DONE, event, "Human alignment completed; free-drive disabled.")
            self.active_task = None
            self.cli.show_message(self.message_manager.get_acknowledgement(EventType.ROBOT_SUCCESS))
            return

        self.ros.publish_human_done()
        self._transition(task, RobotTaskState.R_EXECUTING, event, "Human-done published.")
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type))

    def _handle_robot_success(self, event: Event) -> None:
        task = self._require_active_in(event, {RobotTaskState.R_EXECUTING, RobotTaskState.R_PAUSED})
        if task is None:
            return

        if task.step_id == config.STEP_LIFT_PANEL:
            self.ros.publish_free_drive(True)
            task.free_drive_active = True
            self.logger.log_message(
                "Robot success received; free-drive enabled for human alignment.",
                {"task_instance_id": task.task_instance_id},
            )
            self.cli.show_message(self.message_manager.get_free_drive_alignment_message())
            return

        self._transition(task, RobotTaskState.R_DONE, event, "Robot success received.")
        self.active_task = None
        self.cli.show_message(self.message_manager.get_acknowledgement(event.event_type))

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
