# Task workflow and dialogue draft

This describes the implemented workflow. The root README is the original
architecture proposal and does not describe all current runtime behavior.

## Human and robot IDs

| Input | Robot action |
|---|---|
| H0: pull cables | R1: lift panel, offer free drive, then hold |
| H1: lift panel | No new task |
| H2: adjust panel | No new task |
| `screw done` while holding | Ask permission for R2: release panel and leave |
| R2 physical `success` after `running` | Ask permission for R3: bring pipe connector |
| H4: connect pipes | R4: bring clamping tool |
| H5: clamp with tool | R5: take clamping tool back |

H3 recognition is not enabled. R0 has no configured trigger.
Recognition keeps its existing confidence/progress threshold mechanism. New
H4/H5 thresholds provisionally use the existing values: 0.1 progress and 0.1
confidence. Model training and other sensors are outside this change.

## Dialogue for review

Every task proposal ends with:
“Say yes, no, or later after the beep, or type your reply.”

| Moment | Draft system message | Human replies |
|---|---|---|
| R1 proposal | Would you like me to lift the panel? | yes / no / later |
| Lift complete | The panel is lifted. Would you like free drive for manual adjustment? | yes / free drive / no |
| Free drive enabled | Free drive is on. Adjust the panel, then say or type "done". I will then keep holding the panel while you screw it in place. | done / adjustment done |
| Holding, including after declining free drive | I will keep holding the panel. When screwing is finished, say or type "screw done". I will then ask before releasing the panel. | screw done / screwing done / finished screwing |
| R2 proposal | Screwing is finished. Would you like me to release the panel and move away? | yes / no / later |
| R2 complete; R3 proposal | I have moved away from the panel. Would you like me to bring the pipe connector? | yes / no / later |
| R4 proposal | Would you like me to bring the clamping tool? | yes / no / later |
| R5 proposal | Would you like me to take the clamping tool back? | yes / no / later |

CLI and voice use the same parser and human events. The runtime supplies all
parser aliases to the voice command vocabulary. `yes` is interpreted by the
task manager according to the current question; `done` and `screw done` are
different events. `screw done` outside holding is rejected and sends no action.

R2 and R3 each have their own permission request and timer. R3 is not proposed
when R2 is accepted, deferred, refused, timed out, or canceled; it is proposed
only after R2 completes. Each action waits for robot `running` after dispatch.

## Replies and timing

- `yes`: dispatch this action to GH.
- `no`: put this action in the existing pending pool.
- No reply within 20 seconds: put this action in the pending pool.
- `later`: start this action automatically after 5 seconds. Voice and CLI
  `cancel` are accepted during the delay.
- Canceling a delayed R2 returns to `R_HOLDING` without sending a robot stop
  command or starting queued tasks. Say/type `screw done` again to ask about R2.
  Each retry receives a distinct instance ID so old timeout events cannot
  affect the new request.
- Pending execution retains the existing CLI command:
  `execute round_7_task_2_piece_12`, for example. The prompt prints the actual
  task instance ID. Speaking arbitrary pending IDs is not added by this change.
- If R2 is refused or times out, the prompt explicitly says the panel remains
  held. Other task starts wait until this pending leave action is handled.
- Free-drive permission and the hold stage have no automatic release timeout.

State transitions are validated against the task-aware state table. R1 success
enters `R_WAITING_FREE_DRIVE`; R2–R5 success enters `R_DONE`. Canceling from hold
uses manual recovery rather than offering return home, even if the latest
gripper sample says open. Other execution cancellations use gripper feedback;
unknown or occupied gripper status does not permit return home. Invalid cancel
events are rejected before sending a robot command.

Defaults remain in `config.py`. Per-task overrides can be added without changing
handlers, for example:

```python
TASK_TIMINGS = {
    TASK_LEAVE: {"response_timeout_seconds": 20.0, "defer_seconds": 5.0},
    TASK_BRING_CONNECTOR: {"response_timeout_seconds": 30.0},
}
```

The response timer starts after the permission prompt finishes. The defer timer
starts after the delay announcement finishes so speech does not consume the
human's cancellation window. R2→R3 uses completion feedback, not a
fixed-duration estimate of the robot motion.

## Execution interface

`RobotTask.step_id` retains the originating human step; `RobotTask.task_id` is
the robot action. Task instance IDs include the robot ID, so R2 and R3 are
distinct despite both being associated with H3.

**GH's existing outbound `step_id` field now carries the robot action ID.**
The new `human_step_id` field preserves the recognition/command context. GH
templates must match these IDs and action names:

| GH step_id | suggested_action |
|---|---|
| 1 | assist_lifting |
| 2 | leave |
| 3 | bring_pipe_connector |
| 4 | bring_clamping_tool |
| 5 | return_clamping_tool |

Example R2 dispatch:

```json
{"step_id":2,"human_step_id":3,"progress":1.0,"piece_id":12,"round_id":7,"suggested_action":"leave"}
```

The Python hold stage disables free drive after adjustment and waits for the
human command. It does not issue a new gripping/holding trajectory. Maintaining
the panel after free drive is disabled, and releasing it as part of R2, must be
provided by the robot/GH implementation. ROS topics and motion control commands
are unchanged.

Only one task runs at a time. Recognition triggers arriving while busy are
retained in arrival order and deduplicated; R2's R3 follow-up is proposed first.
Each queued action still asks permission. The final H5 frames retain their
round/piece IDs; the round advances at the next H0 after all configured trigger
steps have been seen. The existing piece-ID-equals-round convention is retained.

## Offline verification

`--debug-step-id` is a **human** step ID, not a robot task ID. To start the
lift/hold/screw-done workflow, run:

```powershell
.venv/Scripts/python.exe run_communication.py --debug-trigger --debug-step-id 0
```

Other configured debug human steps are 4 and 5. H3 is command-only, so debug
step 3 is rejected before startup with an explanation instead of silently
waiting. This startup command uses the normal execution interfaces; it does
not simulate robot feedback. R1 still requires `running` and `success` feedback
before the free-drive question and hold stage become available.

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

These tests simulate voice callbacks, timers and robot feedback without opening
a microphone or sending network commands. Live speech recognition, GH templates,
physical holding and release still require integration testing.
