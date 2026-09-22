# HRC System Architecture for Codex Implementation

## 1. Purpose

This document defines the confirmed software architecture and implementation boundaries for a lightweight, event-driven human–robot collaboration system.

The architecture follows the four conceptual layers used in the thesis system diagram:

```text
Recognition Layer
→ Decision-making Layer
→ Communication Layer
→ Execution Layer
```

The code structure should directly correspond to these layers.

The first implementation should remain simple. Do not add robot internal task phases, motion progress tracking, multi-task concurrency, or unnecessary distributed infrastructure.

---

# 2. System Scope

The Python system is responsible for:

- Receiving human activity recognition results.
- Detecting robot-assistance trigger conditions.
- Creating and managing robot task instances.
- Requesting human permission through CLI.
- Handling human responses and runtime commands.
- Sending `task_id + piece_id` to Grasshopper through an existing UDP sender.
- Publishing pause, resume, restart, cancel, speed, and human-done commands through ROS/rosbridge.
- Receiving a final robot success signal from a ROS topic.
- Managing response timeout, deferred execution, and pending tasks.
- Logging events and task-state transitions.

The Python system is not responsible for:

- Robot trajectory generation.
- Robot internal task phases.
- Motion progress tracking.
- Free-drive phase tracking.
- Detailed robot execution feedback before final success.
- Reimplementing the existing UDP sender.
- Reimplementing the Grasshopper task templates.

Grasshopper selects and sends the predefined robot task template.

ROS/rosbridge updates the behavior of the currently running task and returns the final success signal.

---

# 3. Confirmed Discrete Event Logic

## 3.1 Recognition trigger

```text
Recognition trigger
→ R_WAITING_RESPONSE
```

## 3.2 Human accepts the proposed task

```text
H_ACCEPT
→ R_ACCEPTED
→ send task_id + piece_id to GH
→ R_EXECUTING
```

## 3.3 Human refuses the proposed task

```text
H_REFUSE
→ R_REFUSED
→ add task to pending pool
```

## 3.4 Human defers the task

```text
H_DEFER
→ R_DEFER
→ wait 5 seconds
→ DEFER_TIMEOUT
→ send task_id + piece_id to GH
→ R_EXECUTING
```

## 3.5 Human gives no response

```text
RESPONSE_TIMEOUT
→ R_PENDING
→ add task to pending pool
```

## 3.6 Human executes a pending task

```text
H_EXECUTE_PENDING_TASK(id)
→ find pending task
→ remove task from pending pool
→ send task_id + piece_id to GH
→ R_EXECUTING
```

## 3.7 Human pauses the current task

```text
R_EXECUTING + H_PAUSE
→ publish ROS pause
→ R_PAUSED
```

## 3.8 Human resumes the paused task

```text
R_PAUSED + H_RESUME
→ publish ROS resume
→ R_RESUME
→ R_EXECUTING
```

`R_RESUME` may be a short transition state used mainly for logging.

## 3.9 Human restarts the current task

```text
R_EXECUTING or R_PAUSED + H_RESTART
→ publish ROS restart
→ R_REDO
→ R_EXECUTING
```

`R_REDO` may be a short transition state used mainly for logging.

## 3.10 Human cancels the current task

```text
R_EXECUTING or R_PAUSED or R_DEFER + H_CANCEL
→ publish ROS cancel when applicable
→ R_CANCELED
```

For `R_DEFER`, the robot has not started yet. Cancel the defer timer; ROS cancel is normally unnecessary.

## 3.11 Human changes robot speed

```text
R_EXECUTING + H_SPEEDUP
→ publish new speed
→ remain R_EXECUTING
```

```text
R_EXECUTING + H_SLOWDOWN
→ publish new speed
→ remain R_EXECUTING
```

## 3.12 Human finishes the human task

For the first implementation:

```text
R_EXECUTING + H_DONE
→ publish human-done signal through ROS
→ remain R_EXECUTING
```

`H_DONE` does not set the robot task to done.

## 3.13 Robot finishes the robot task

```text
ROBOT_SUCCESS
→ R_DONE
```

The only confirmed robot execution feedback available to Python is the final success signal.

---

# 4. High-Level Architecture

```text
Human / Reality
      │
      ▼
Recognition Layer
      │
      │ RECOGNITION_TRIGGER
      ▼
Decision-making Layer
      ▲
      │ human events / timer events / robot success
      │
Communication Layer
      │
      ├── human-facing CLI messages
      └── CLI command events
      │
      ▼
Execution Layer
      ├── Grasshopper task dispatch
      └── ROS runtime control and success feedback
```

More explicitly:

```text
Recognition Result
step_id / progress / confidence / piece_id / round_id
                         │
                         ▼
                  Trigger Manager
                         │
                         ▼
                    Event Queue
                         │
                         ▼
                   Task Manager
                    / State Machine
              ┌──────────┼──────────┐
              ▼          ▼          ▼
       CLI Messaging   GH Dispatch  ROS Runtime Control
              ▲          │          │
              │          ▼          ▼
          Human CLI   Grasshopper   Robot / UR
                                      │
                                      ▼
                               ROBOT_SUCCESS topic
                                      │
                                      └──▶ Event Queue
```

---

# 5. Project Structure

```text
hrc_system/
│
├── main.py
├── config.py
│
├── core/
│   ├── __init__.py
│   ├── events.py
│   ├── models.py
│   ├── event_queue.py
│   └── logger.py
│
├── recognition/
│   ├── __init__.py
│   ├── recognition_manager.py
│   ├── step_stabilizer.py
│   └── trigger_manager.py
│
├── decision_making/
│   ├── __init__.py
│   ├── task_manager.py
│   ├── state_machine.py
│   ├── pending_task_pool.py
│   └── timer_manager.py
│
├── communication/
│   ├── __init__.py
│   ├── cli_interface.py
│   ├── command_parser.py
│   └── message_manager.py
│
├── execution/
│   ├── __init__.py
│   ├── gh_dispatcher.py
│   ├── udp_sender.py
│   └── ros_communication.py
│
└── tests/
    ├── test_trigger_manager.py
    ├── test_state_machine.py
    ├── test_task_manager.py
    ├── test_command_parser.py
    └── test_execution_interfaces.py
```

The first version should use one shared in-process event queue.

Do not introduce Kafka, Redis, RabbitMQ, or another external event broker.

---

# 6. Shared Core

## 6.1 `core/events.py`

### Responsibility

Define:

- Robot task states.
- Event types.
- Common event object.

### Robot task states

```python
class RobotTaskState(Enum):
    R_WAITING_RESPONSE = auto()

    R_ACCEPTED = auto()
    R_REFUSED = auto()
    R_DEFER = auto()
    R_PENDING = auto()

    R_EXECUTING = auto()
    R_PAUSED = auto()
    R_RESUME = auto()
    R_REDO = auto()

    R_CANCELED = auto()
    R_DONE = auto()
```

`R_ROUND` must not be a state. Store it as `round_id` in `RobotTask`.

### Event types

```python
class EventType(Enum):
    RECOGNITION_TRIGGER = auto()

    H_ACCEPT = auto()
    H_REFUSE = auto()
    H_DEFER = auto()
    H_EXECUTE_PENDING_TASK = auto()

    H_CANCEL = auto()
    H_PAUSE = auto()
    H_RESUME = auto()
    H_RESTART = auto()

    H_SPEEDUP = auto()
    H_SLOWDOWN = auto()

    H_DONE = auto()

    RESPONSE_TIMEOUT = auto()
    DEFER_TIMEOUT = auto()

    ROBOT_SUCCESS = auto()
```

### Common event object

```python
@dataclass
class Event:
    event_type: EventType
    source: str
    task_instance_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
```

### Rules

- Events are instantaneous.
- States persist.
- Event producers must not directly mutate task state.
- State mutation belongs to the Decision-making Layer.

---

## 6.2 `core/models.py`

### Responsibility

Define shared data structures.

### `RecognitionResult`

```python
@dataclass
class RecognitionResult:
    round_id: int
    step_id: int
    progress: float
    piece_id: int
    confidence: float
    timestamp: float
```

### `RobotTask`

```python
@dataclass
class RobotTask:
    task_instance_id: str

    task_id: int
    piece_id: int
    round_id: int

    state: RobotTaskState

    speed: float
    pending_reason: str | None = None

    created_at: float | None = None
    updated_at: float | None = None
```

### Field meanings

- `task_instance_id`: unique ID for this specific occurrence.
- `task_id`: Grasshopper task-template ID.
- `piece_id`: target piece or element ID.
- `round_id`: workflow iteration number.
- `state`: current robot task lifecycle state.
- `speed`: current runtime speed value.
- `pending_reason`: `"refused"` or `"timeout"`.

Example:

```text
round_2_task_1_piece_4
```

---

## 6.3 `core/event_queue.py`

### Responsibility

Provide one shared queue for all event producers.

Event sources:

```text
Recognition Layer
CLI input thread
Timer callbacks
ROS success callback
```

Consumer:

```text
TaskManager
```

Recommended implementation:

```python
from queue import Queue
```

A thin wrapper is optional.

Required operations:

```python
put(event)
get()
empty()
```

---

## 6.4 `core/logger.py`

### Responsibility

Record:

- Incoming events.
- State transitions.
- GH task-dispatch messages.
- ROS runtime-control actions.
- Timer events.
- Robot success events.

### Suggested fields

```text
timestamp
event_type
source
task_instance_id
task_id
piece_id
round_id
old_state
new_state
message
payload
```

### Required methods

```python
log_event(event)

log_transition(
    task,
    event,
    old_state,
    new_state,
    message=None,
)
```

All state changes should go through one transition method in `TaskManager` so logging is never skipped.

---

# 7. Recognition Layer

```text
recognition/
├── recognition_manager.py
├── step_stabilizer.py
└── trigger_manager.py
```

## 7.1 `recognition/recognition_manager.py`

### Responsibility

Integrate the existing recognition pipeline and output a standardized `RecognitionResult`.

Inputs may include:

- Skeleton or feature window.
- Model inference output.
- Current `piece_id`.
- Current `round_id`.

Conceptual flow:

```text
input features
→ model inference
→ step probabilities
→ progress prediction
→ step stabilization
→ RecognitionResult
```

### Main method

```python
update(input_data) -> RecognitionResult
```

### Boundaries

This file must not:

- Create robot tasks.
- Ask for permission.
- Send GH or ROS commands.
- Modify robot task state.

---

## 7.2 `recognition/step_stabilizer.py`

### Responsibility

Contain the existing step-stabilization logic.

Possible responsibilities:

- Probability smoothing.
- Confirmation count.
- Confidence threshold.
- Minimum margin.
- Stable step selection.

### Main method

```python
update(step_probabilities) -> stable_step_id
```

### Boundary

This component improves recognition output only.

It does not create `RECOGNITION_TRIGGER`.

---

## 7.3 `recognition/trigger_manager.py`

### Responsibility

Convert continuous recognition results into one-time `RECOGNITION_TRIGGER` events.

### Internal state

```python
previous_progress = {}
triggered_keys = set()
```

### Trigger key

```text
(round_id, task_id, piece_id)
```

### Main method

```python
update(recognition_result) -> list[Event]
```

### Logic

For each configured trigger rule:

1. Check matching `step_id`.
2. Check minimum confidence.
3. Detect threshold crossing.
4. Check that the trigger key has not fired.
5. Mark the key as fired.
6. Create `RECOGNITION_TRIGGER`.

### Threshold crossing

```text
previous_progress <= threshold
and
current_progress > threshold
```

Do not repeatedly fire while `progress > threshold`.

### Event payload

```python
{
    "task_id": ...,
    "piece_id": ...,
    "round_id": ...,
    "step_id": ...,
    "progress": ...,
}
```

### Boundary

This file must not:

- Create `RobotTask` directly.
- Ask for permission.
- Add tasks to pending pool.
- Dispatch a task to GH.

---

# 8. Decision-making Layer

```text
decision_making/
├── task_manager.py
├── state_machine.py
├── pending_task_pool.py
└── timer_manager.py
```

## 8.1 `decision_making/state_machine.py`

### Responsibility

Centralize transition validity.

It answers:

```text
Is this event valid in the current state?
What is the next state?
```

It does not perform external actions.

### Suggested interface

```python
is_valid_transition(
    current_state,
    event_type,
) -> bool
```

Optional:

```python
get_next_state(
    current_state,
    event_type,
) -> RobotTaskState | None
```

### Confirmed valid transitions

```text
R_WAITING_RESPONSE + H_ACCEPT
→ R_ACCEPTED

R_WAITING_RESPONSE + H_REFUSE
→ R_REFUSED

R_WAITING_RESPONSE + H_DEFER
→ R_DEFER

R_WAITING_RESPONSE + RESPONSE_TIMEOUT
→ R_PENDING

R_DEFER + DEFER_TIMEOUT
→ R_EXECUTING

R_DEFER + H_CANCEL
→ R_CANCELED

R_EXECUTING + H_PAUSE
→ R_PAUSED

R_PAUSED + H_RESUME
→ R_RESUME
→ R_EXECUTING

R_EXECUTING or R_PAUSED + H_RESTART
→ R_REDO
→ R_EXECUTING

R_EXECUTING or R_PAUSED + H_CANCEL
→ R_CANCELED

R_EXECUTING + H_SPEEDUP
→ R_EXECUTING

R_EXECUTING + H_SLOWDOWN
→ R_EXECUTING

R_EXECUTING + H_DONE
→ R_EXECUTING

R_EXECUTING or R_PAUSED + ROBOT_SUCCESS
→ R_DONE
```

Pending-task execution is handled using the pending pool rather than a normal active-task transition.

### Boundary

`state_machine.py` should not:

- Publish ROS commands.
- Send GH messages.
- Start timers.
- Print CLI messages.

---

## 8.2 `decision_making/pending_task_pool.py`

### Responsibility

Manage tasks that entered the pending pool because of:

```text
R_REFUSED
R_PENDING
```

### Internal structure

Recommended:

```python
pending_tasks: dict[str, RobotTask]
```

Key:

```text
task_instance_id
```

### Required operations

```python
add(task)
get(task_instance_id)
remove(task_instance_id)
list_all()
contains(task_instance_id)
```

### Boundary

This component stores pending tasks only.

It does not dispatch them or change their state.

---

## 8.3 `decision_making/timer_manager.py`

### Responsibility

Manage:

- Human response timer.
- Deferred-execution timer.

Because only one active task is supported, the first version may use one of each.

### Required methods

```python
start_response_timer(
    task_instance_id,
    duration,
)

cancel_response_timer()

start_defer_timer(
    task_instance_id,
    duration,
)

cancel_defer_timer()
```

### Response timer

Started when:

```text
task enters R_WAITING_RESPONSE
```

Cancelled on:

```text
H_ACCEPT
H_REFUSE
H_DEFER
```

Expiry event:

```text
RESPONSE_TIMEOUT
```

### Defer timer

Started on:

```text
H_DEFER
→ R_DEFER
```

Cancelled on:

```text
R_DEFER + H_CANCEL
```

Expiry event:

```text
DEFER_TIMEOUT
```

### Important requirement

Each timeout event must include the associated `task_instance_id`.

This prevents a stale timer callback from affecting a newer active task.

### Boundary

Timer callbacks only generate events.

They do not modify task state.

---

## 8.4 `decision_making/task_manager.py`

### Responsibility

This is the central coordinator and the only module that owns task-state changes.

### Dependencies

```text
StateMachine
PendingTaskPool
TimerManager
MessageManager
CLIInterface
GHDispatcher
ROSCommunication
Logger
```

### Internal data

```python
active_task: RobotTask | None
pending_pool: PendingTaskPool
```

### System rule

```text
At most one active robot task.
```

### Main handler structure

```python
handle_event(event)
```

Dispatch to:

```text
_handle_recognition_trigger
_handle_accept
_handle_refuse
_handle_defer
_handle_response_timeout
_handle_defer_timeout
_handle_execute_pending
_handle_pause
_handle_resume
_handle_restart
_handle_cancel
_handle_speedup
_handle_slowdown
_handle_human_done
_handle_robot_success
```

### Recognition trigger handler

If `active_task is None`:

```text
create RobotTask
→ set active_task
→ transition to R_WAITING_RESPONSE
→ show permission request
→ start response timer
```

If an active task already exists:

```text
ignore the new trigger in the first version
→ log ignored trigger
```

Do not create a second active task.

### Accept handler

Valid only in:

```text
R_WAITING_RESPONSE
```

Steps:

```text
cancel response timer
→ R_ACCEPTED
→ GHDispatcher.dispatch_task(task)
→ R_EXECUTING
```

### Refuse handler

Valid only in:

```text
R_WAITING_RESPONSE
```

Steps:

```text
cancel response timer
→ R_REFUSED
→ pending_reason = "refused"
→ pending_pool.add(task)
→ active_task = None
```

### Defer handler

Valid only in:

```text
R_WAITING_RESPONSE
```

Steps:

```text
cancel response timer
→ R_DEFER
→ start defer timer
```

The task remains active during defer.

### Response-timeout handler

Checks:

- Active task exists.
- Event task ID matches active task.
- Current state is `R_WAITING_RESPONSE`.

Steps:

```text
R_PENDING
→ pending_reason = "timeout"
→ pending_pool.add(task)
→ active_task = None
```

### Defer-timeout handler

Checks:

- Active task exists.
- Event task ID matches.
- Current state is `R_DEFER`.

Steps:

```text
GHDispatcher.dispatch_task(task)
→ R_EXECUTING
```

### Execute-pending handler

Checks:

- No active task exists.
- Requested task exists in pending pool.

Steps:

```text
pending_pool.remove(id)
→ active_task = task
→ GHDispatcher.dispatch_task(task)
→ R_EXECUTING
```

### Pause handler

Valid only in:

```text
R_EXECUTING
```

Steps:

```text
ROSCommunication.publish_pause()
→ R_PAUSED
```

### Resume handler

Valid only in:

```text
R_PAUSED
```

Steps:

```text
ROSCommunication.publish_resume()
→ R_RESUME
→ R_EXECUTING
```

### Restart handler

Valid in:

```text
R_EXECUTING
R_PAUSED
```

Steps:

```text
ROSCommunication.publish_restart()
→ R_REDO
→ R_EXECUTING
```

### Cancel handler

For `R_EXECUTING` or `R_PAUSED`:

```text
ROSCommunication.publish_cancel()
→ R_CANCELED
→ active_task = None
```

For `R_DEFER`:

```text
cancel defer timer
→ R_CANCELED
→ active_task = None
```

### Speed-up handler

Valid only in:

```text
R_EXECUTING
```

Steps:

```text
speed = min(speed + SPEED_STEP, MAX_SPEED)
→ ROSCommunication.publish_speed(speed)
→ remain R_EXECUTING
```

### Slow-down handler

Valid only in:

```text
R_EXECUTING
```

Steps:

```text
speed = max(speed - SPEED_STEP, MIN_SPEED)
→ ROSCommunication.publish_speed(speed)
→ remain R_EXECUTING
```

### Human-done handler

For the first implementation, valid only in:

```text
R_EXECUTING
```

Steps:

```text
ROSCommunication.publish_human_done()
→ remain R_EXECUTING
```

### Robot-success handler

Recommended valid states:

```text
R_EXECUTING
R_PAUSED
```

Steps:

```text
R_DONE
→ active_task = None
```

Allowing success from `R_PAUSED` handles a possible timing race between pause and completion.

### Unified transition method

All state changes must pass through:

```python
_transition(
    task,
    new_state,
    event,
    message=None,
)
```

Responsibilities:

```text
store old state
update task.state
update timestamp
write transition log
```

### Invalid event behavior

If an event is invalid in the current state:

```text
do not change state
log invalid event
optionally display short CLI feedback
```

---

# 9. Communication Layer

```text
communication/
├── cli_interface.py
├── command_parser.py
└── message_manager.py
```

## 9.1 `communication/cli_interface.py`

### Responsibility

- Display messages to the human.
- Read raw CLI text.
- Send raw text to `CommandParser`.

### Suggested methods

```python
show_message(message)
show_permission_request(message)
read_input() -> str
```

### Concurrency

For the integrated system:

```text
CLI thread
→ read text
→ CommandParser
→ Event Queue
```

Blocking `input()` is acceptable only during isolated first tests.

### Boundary

This file does not validate task state.

---

## 9.2 `communication/command_parser.py`

### Responsibility

Convert raw human CLI text into standardized events.

### Confirmed mapping

```text
yes / accept / okay
→ H_ACCEPT

no / refuse
→ H_REFUSE

later / defer
→ H_DEFER

pause
→ H_PAUSE

continue / resume
→ H_RESUME

restart / redo
→ H_RESTART

cancel
→ H_CANCEL

faster / speed up
→ H_SPEEDUP

slower / slow down
→ H_SLOWDOWN

done / finished
→ H_DONE
```

### Pending-task syntax

Use explicit ID in the first implementation:

```text
execute <task_instance_id>
```

Output:

```python
Event(
    event_type=H_EXECUTE_PENDING_TASK,
    source="human_cli",
    task_instance_id=...,
)
```

### Boundary

The parser does not decide whether the event is valid.

It always returns the standardized event if the text is recognized.

---

## 9.3 `communication/message_manager.py`

### Responsibility

Centralize human-facing text.

Types of messages:

- Permission request.
- Command acknowledgement.
- Invalid-command feedback.
- Pending-task information.
- Completion feedback.

### Suggested interface

```python
get_permission_message(task_id) -> str

get_acknowledgement(event_type) -> str

get_invalid_event_message(
    current_state,
    event_type,
) -> str
```

### Examples

```text
TASK_LIFT_PANEL
→ "Would you like me to lift the panel?"

H_PAUSE processed
→ "The robot task has been paused."

Invalid H_RESUME
→ "There is no paused robot task to resume."
```

### Boundary

This file contains text only.

It does not print directly and does not modify state.

---

# 10. Execution Layer

```text
execution/
├── gh_dispatcher.py
├── udp_sender.py
└── ros_communication.py
```

## 10.1 `execution/gh_dispatcher.py`

### Responsibility

Start a complete predefined robot task through Grasshopper.

Input:

```text
task_id
piece_id
```

Flow:

```text
RobotTask
→ build GH message
→ existing UDP sender
→ Grasshopper task template
```

### Suggested interface

```python
dispatch_task(task) -> str
```

### Message builder

```python
build_message(
    task_id,
    piece_id,
) -> str
```

Use the already confirmed GH message format.

### Boundary

This file handles only task start.

It does not handle:

- Pause.
- Resume.
- Restart.
- Cancel.
- Speed update.
- Human done.
- Robot success.

---

## 10.2 `execution/udp_sender.py`

### Responsibility

Contain or import the existing UDP implementation.

Responsibilities:

- Socket setup.
- Host and port.
- Message encoding.
- Sending.

Suggested interface:

```python
send(message: str)
```

### Boundary

This file does not understand:

- Robot task states.
- Human events.
- Task IDs semantically.
- Pending tasks.

---

## 10.3 `execution/ros_communication.py`

### Responsibility

Publish runtime-control commands and subscribe to final robot success.

### Required publish methods

```python
publish_pause()
publish_resume()
publish_restart()
publish_cancel()
publish_speed(value)
publish_human_done()
```

Acceleration support may be added later, but it is not part of the current human-event list.

### Success subscriber

```text
robot success topic
→ ROS callback
→ ROBOT_SUCCESS event
→ Event Queue
```

### Suggested setup

```python
set_event_callback(callback)
```

When success is received:

```python
event = Event(
    event_type=ROBOT_SUCCESS,
    source="ros",
    payload={"message": raw_message},
)

event_callback(event)
```

### Boundary

ROS callbacks must not directly modify task state.

---

# 11. `config.py`

### Responsibility

Store all configurable values and mappings.

### Suggested contents

```python
RESPONSE_TIMEOUT_SECONDS = 5.0
DEFER_SECONDS = 5.0

DEFAULT_SPEED = 0.5
SPEED_STEP = 0.1
MIN_SPEED = 0.1
MAX_SPEED = 1.0
```

### Task definitions

```python
TASK_LIFT_PANEL = 0
TASK_BRING_JOINT = 1
TASK_BRING_NEXT_PANEL = 2
```

### Trigger rules

```python
TRIGGER_RULES = {
    TASK_LIFT_PANEL: {
        "step_id": 0,
        "progress_threshold": 0.5,
        "min_confidence": 0.8,
    },

    TASK_BRING_JOINT: {
        "step_id": 2,
        "progress_threshold": 0.8,
        "min_confidence": 0.8,
    },
}
```

### Additional contents

- Permission-message mapping.
- ROS topic names.
- UDP host and port.
- Log path.
- CLI aliases if desired.

Avoid hard-coded values in handler functions.

---

# 12. `main.py`

### Responsibility

Initialize all modules, connect callbacks, start the CLI thread, and process events.

Do not put task-state rules in `main.py`.

### Initialization outline

```python
event_queue = EventQueue()

logger = EventLogger(...)
message_manager = MessageManager()
cli = CLIInterface(...)
command_parser = CommandParser(...)

udp_sender = ExistingUDPSender(...)
gh_dispatcher = GHDispatcher(udp_sender)

ros = ROSCommunication(...)
timer_manager = TimerManager(
    event_callback=event_queue.put
)

pending_pool = PendingTaskPool()
state_machine = StateMachine()

task_manager = TaskManager(
    state_machine=state_machine,
    pending_pool=pending_pool,
    timer_manager=timer_manager,
    message_manager=message_manager,
    cli=cli,
    gh_dispatcher=gh_dispatcher,
    ros_communication=ros,
    logger=logger,
)

recognition_manager = RecognitionManager(...)
trigger_manager = TriggerManager(...)

ros.set_event_callback(event_queue.put)
```

### CLI thread outline

```text
read raw CLI text
→ command_parser.parse(text)
→ if event exists: event_queue.put(event)
```

### Main loop outline

```text
get latest recognition result
→ trigger_manager.update(result)
→ put trigger events into queue

process queued events
→ task_manager.handle_event(event)
```

Conceptual pseudocode:

```python
while system_running:

    result = recognition_manager.update(...)

    trigger_events = trigger_manager.update(result)

    for event in trigger_events:
        event_queue.put(event)

    while not event_queue.empty():
        event = event_queue.get()
        task_manager.handle_event(event)
```

### Event producers

```text
Recognition Layer
CLI thread
Timer callbacks
ROS callback
```

All publish to the same queue.

---

# 13. Layer Boundaries

## Recognition Layer

Responsible for:

```text
What is the human doing?
Has a trigger condition been crossed?
```

Outputs:

```text
RecognitionResult
RECOGNITION_TRIGGER
```

Must not dispatch tasks.

## Decision-making Layer

Responsible for:

```text
What is the current task state?
Is the event valid?
What action should happen next?
```

Owns:

```text
active_task
pending tasks
state transitions
timers
```

## Communication Layer

Responsible for:

```text
What did the human type?
What should the system say to the human?
```

Outputs:

```text
Human events
Human-facing messages
```

## Execution Layer

Responsible for:

```text
How is the decision sent to GH or ROS?
```

Contains:

```text
GH task start
ROS runtime control
ROS success feedback
```

---

# 14. Valid State/Event Summary

| Current state | Valid events |
|---|---|
| `R_WAITING_RESPONSE` | `H_ACCEPT`, `H_REFUSE`, `H_DEFER`, `RESPONSE_TIMEOUT` |
| `R_DEFER` | `DEFER_TIMEOUT`, `H_CANCEL` |
| Pending pool task in `R_REFUSED` or `R_PENDING` | `H_EXECUTE_PENDING_TASK(id)` |
| `R_EXECUTING` | `H_PAUSE`, `H_RESTART`, `H_CANCEL`, `H_SPEEDUP`, `H_SLOWDOWN`, `H_DONE`, `ROBOT_SUCCESS` |
| `R_PAUSED` | `H_RESUME`, `H_RESTART`, `H_CANCEL`, `ROBOT_SUCCESS` |
| `R_DONE` | Terminal |
| `R_CANCELED` | Terminal |

---

# 15. Recommended Implementation Order

## Phase 1: Pure logic simulation

Implement:

```text
core/events.py
core/models.py
decision_making/state_machine.py
decision_making/pending_task_pool.py
decision_making/task_manager.py
communication/command_parser.py
communication/cli_interface.py
communication/message_manager.py
```

Use:

```text
manual recognition trigger
print-only GH dispatcher
print-only ROS communication
manual ROBOT_SUCCESS
```

## Phase 2: Timers

Add:

```text
decision_making/timer_manager.py
```

Test:

```text
response timeout
5-second defer
cancel during defer
```

## Phase 3: Existing UDP sender

Connect:

```text
execution/udp_sender.py
execution/gh_dispatcher.py
```

## Phase 4: ROS runtime communication

Connect:

```text
pause
resume
restart
cancel
speed
human done
```

## Phase 5: ROS success subscriber

Connect:

```text
ROBOT_SUCCESS
```

## Phase 6: Real recognition

Connect:

```text
RecognitionManager
StepStabilizer
TriggerManager
```

---

# 16. Minimum Test Scenarios

## Scenario A: Accept and complete

```text
manual trigger
→ H_ACCEPT
→ GH message sent
→ R_EXECUTING
→ H_DONE
→ remain R_EXECUTING
→ ROBOT_SUCCESS
→ R_DONE
```

## Scenario B: Refuse and execute pending

```text
manual trigger
→ H_REFUSE
→ R_REFUSED
→ pending pool
→ H_EXECUTE_PENDING_TASK(id)
→ GH message sent
→ R_EXECUTING
→ ROBOT_SUCCESS
→ R_DONE
```

## Scenario C: No response

```text
manual trigger
→ no CLI response
→ RESPONSE_TIMEOUT
→ R_PENDING
→ pending pool
```

## Scenario D: Defer

```text
manual trigger
→ H_DEFER
→ R_DEFER
→ wait 5 seconds
→ DEFER_TIMEOUT
→ GH message sent
→ R_EXECUTING
```

## Scenario E: Cancel during defer

```text
manual trigger
→ H_DEFER
→ R_DEFER
→ H_CANCEL
→ cancel defer timer
→ R_CANCELED
```

## Scenario F: Pause and resume

```text
R_EXECUTING
→ H_PAUSE
→ ROS pause
→ R_PAUSED
→ H_RESUME
→ ROS resume
→ R_RESUME
→ R_EXECUTING
```

## Scenario G: Restart

```text
R_EXECUTING or R_PAUSED
→ H_RESTART
→ ROS restart
→ R_REDO
→ R_EXECUTING
```

## Scenario H: Speed changes

```text
R_EXECUTING
→ H_SPEEDUP
→ bounded new speed
→ ROS speed publish
→ remain R_EXECUTING
```

```text
R_EXECUTING
→ H_SLOWDOWN
→ bounded new speed
→ ROS speed publish
→ remain R_EXECUTING
```

---

# 17. Non-Goals for the First Version

Do not add:

- Robot internal task phases.
- Detailed robot execution-state tracking.
- Robot start acknowledgements.
- Pause/resume/cancel acknowledgement states.
- Multiple simultaneous active tasks.
- LLM-based command parsing.
- Complex persistent databases.
- External event brokers.
- Automatic task prioritization.
- Automatic pending-task selection without explicit ID.
- Python-side free-drive state tracking.
- Additional human-workflow state machines.

The first implementation should remain a small, testable event-driven coordination system aligned with the thesis architecture.
