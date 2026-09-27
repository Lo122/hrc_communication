# Task workflow and dialogue draft

This describes the implemented workflow. The root README is the original
architecture proposal and does not describe all current runtime behavior.

## Human and robot IDs

| Input | Robot action |
|---|---|
| H0: pull cables | R1: lift panel, free drive for adjustment, then hold |
| H1: lift panel | No new task |
| H2: adjust panel | No new task |
| `screw done` while holding, or the panel secured (screw count or TCP force, see below) | Ask permission for R2: release panel and leave |
| R2 physical `success` after `running` | Ask permission for R3: bring pipe connector |
| H4: connect pipes | R4: bring clamping tool |
| H5: clamp with tool | R5: take clamping tool back |

H3 recognition is not enabled. R0 has no configured trigger.
Recognition keeps its existing confidence/progress threshold mechanism. New
H4/H5 thresholds provisionally use the existing values: 0.1 progress and 0.1
confidence. Model training and other sensors are outside this change.

## Demo opening

Recognition cannot reliably tell when the assembly begins, so
`run_communication.py --demo` (or `run_system.py --demo ...`) opens the first panel
with a scripted dialogue (`2_decision_making/demo_opening.py`). Until the lift is
answered, recognition updates and detector signals are logged and ignored.

The first question waits until recognition is active, so that it is ready to take
over: recognition's model needs time to recognize motion, so its task updates
count -- in the demo or not -- only `RECOGNITION_ACTIVATION_S` (20 s) after the
recognition process first reports in (a human location or a task update). Until
then they are logged and ignored, and the screw count does not see them. Typed
updates (`--fake-recognition`, `--manual-trigger`) and `--debug-trigger` are not
held back. Set it to 0 to start at once, e.g. `--demo` without recognition running.

| Question | yes | no ("I'll do it") |
|---|---|---|
| "Would you like me to pull the cables?" | The robot pulls. `DEMO_LIFT_ASK_AFTER_PULL_START_S` after it starts, while it is still pulling, it asks about the lift; a yes starts the lift the moment the pull reports success. | The human pulls (Pull Cables working, human). After Pull Cables' duration limit (p95, 6 s) + `DEMO_HUMAN_PULL_BUFFER_S` the robot asks about the lift. |
| "Would you like me to lift the panel?" | The usual lift, free drive and holding. | The human lifts (Lift working, human). |

Asked while the robot still pulls, "later" defers the lift until after the pull,
and no answer asks again once the pull is done. Once the lift is answered, the
opening is over: recognition, the trigger rules and the task detectors (on in the
demo, `DEMO_DETECTORS_MODE`) take over -- screw detection, the pipe connector, the
tool, the next panel -- and robot tasks can be requested as always. A stopped pull
also ends the opening. Only the first panel is scripted.

## Recognition moves the human's task

The human works on one task at a time; robot tasks can run alongside it.
Recognition moves the human to another task only if the transition table expects
it after the reference task (the task last recognized or confirmed): it is pending
(P >= `PENDING_MIN_PROBABILITY`), or a done step likely enough to be repeated.
The task the human worked on before then goes back to pending, or to not done if
the table does not expect it any more, so the robot's trigger rules may offer it.
Any other recognized task is ignored and logged once.

So that a wrong reference cannot stall tracking, the gate opens once the tracker
has stayed on the reference task (working on it, or done and not moved on) longer
than `TASK_OVERRUN_STAT` (p95) of that task's duration per block in
`TASK_DURATION_STATS_PATH`; the next recognized task is then believed whatever the
table says. The decision view shows the time on the reference, its limit, and the
task last ignored.

## A robot lift leads its piece

Once the human says yes to a robot lift, the lift -- not recognition -- says where
that piece's tasks are, until the holding ends:

| Robot lift | Lift | Place | Align | Screw |
|---|---|---|---|---|
| accepted / executing | working (robot) | working (robot) | | |
| arrived: free drive | done | done | working (human) | |
| holding, after "adjustment done" | | | done | working (human) |
| holding, after declining free drive (only with `LIFT_ASKS_FREE_DRIVE`) | | | done (inferred: no adjustment needed) | working (human) |
| panel secured (all 6 screws counted, or the TCP force): the leave is asked | | | | still working |
| screw done (voice, or recognition once it counts again) | | | | done |

Pull Cables on that piece counts as done once the lift starts, and a human starting
Align moves the human to that piece. Recognition's task updates in between are
logged and ignored; they count again once the holding ends (screw done, or the
panel secured), and then decide when Screw is done if the human does not say so. Its
Screw progress still feeds the screw count (below), which is Screw's progress while
the robot holds the panel.

## Trigger rules and task detectors

When the robot offers a task on its own now comes from the task database's
"Robot task trigger info" (`2_decision_making/src/task_database.py` explains the
format). Besides "previous task" and "Progress" / "Done signal", a rule has:

- "Piece id": which piece the robot task is for, relative to the piece n the
  human is on -- "n", "n + 1", or per previous task. Clamp Coupling on piece 1
  lets the robot pull the cables of piece 2 and lift its panel.
- "robot task": what the robot does, in order. Pull Cables is followed by Lift
  for the same piece without waiting for recognition, but the robot still asks
  "May I lift the panel?" first.
- "Condition": what must also hold, e.g. Screw done. A list gives alternatives:
  Bring Connector needs Screw done, half the screws counted, or a TCP weight change.

Screw counts and weight changes come from detectors
(`2_decision_making/task_transition_detector.py`):

- The screw count reads recognition's Screw progress: per screw it climbs to about
  0.7 and drops back near 0 as the next one starts, so each climb and fall is one
  screw (`SCREW_PROGRESS_HIGH` / `SCREW_PROGRESS_LOW`). All `SCREWS_PER_PANEL` (6)
  counted means the panel is secured. The last screw has no next one to drop into;
  it is counted when recognition moves on from Screw.
- The force detector reads the TCP force while the robot holds the panel: it
  notices the panel's weight going over to the frame, and reports the panel
  secured once the pushing has stopped and at least half the screws are counted,
  or once the load change has stayed in `TCP_WEIGHT_RANGE_N` for
  `TCP_WEIGHT_STEADY_S` (10 s) with no push -- whatever recognition counted.

"panel secured" is when the robot may leave, not Screw being done: while the robot
holds that panel it ends the holding and asks to release the panel and leave (R2),
and Screw stays open until recognition or the human's `screw done` confirms it.
`screw done` while holding still does both.
`config.TASK_DETECTORS_MODE` is "log" until the thresholds are tuned (force:
`4_execution/eval/force_logger.py` and `force_log_review.py`); set it to "on" to
let the signals count.

## Dialogue for review

Every task proposal ends with:
“Say yes, no, or later after the beep, or type your reply.”

| Moment | Draft system message | Human replies |
|---|---|---|
| R1 proposal | Would you like me to lift the panel? | yes / no / later |
| Lift complete: free drive on, no question | The panel is in position and free drive is on. Adjust the panel, then say or type "done". I will then keep holding the panel while you screw it in place. | done / adjustment done |
| Holding | I will keep holding the panel. When screwing is finished, say or type "screw done". I will then ask before releasing the panel. | screw done / screwing done / finished screwing |
| Panel secured (detector) | The panel looks secured. | — |
| R2 proposal | Would you like me to release the panel and move away? | yes / no / later |
| R2 complete; R3 proposal | I have moved away from the panel. Would you like me to bring the pipe connector? | yes / no / later |
| R4 proposal | Would you like me to bring the clamping tool? | yes / no / later |
| R3/R4 arrived | Can I hand over the pipe coupling / tool? Yes opens the gripper, so hold it first. Say yes or no after the beep, or type your reply. | yes / give me the … / no |
| Not ready (no) | Okay, I will keep holding the tool. Let me know when you are ready to receive it: say or type "give me the tool". | give me the tool / give me the (pipe) coupling |
| Pipe coupling handed over; R7 starts in 2 s | Opening the gripper. Here is the pipe coupling. I will move away in 2 seconds. Say or type cancel to keep me here. | cancel |
| Tool handed over; R7 proposal | Opening the gripper. Here is the tool. May I leave the hand-over position and move away? | yes / no / later |
| R5 proposal | Would you like me to take the clamping tool back? | yes / no / later |

CLI and voice use the same parser and human events. The runtime supplies all
parser aliases to the voice command vocabulary. `yes` is interpreted by the
task manager according to the current question; `done` and `screw done` are
different events. `screw done` while lifting or adjusting is rejected and sends no
action; after the holding it only confirms Screw.

R2 and R3 each have their own permission request and timer. R3 is not proposed
when R2 is accepted, deferred, refused, timed out, or canceled; it is proposed
only after R2 completes. Each action waits for robot `running` after dispatch.

### Hand-over after R3 and R4

R3 (pipe connector) and R4 (clamping tool) are not done when the robot arrives:
it holds the item out and asks to hand it over (`R_WAITING_HANDOVER`; the
items and their names are `config.HANDOVER_ITEMS`).

- `yes`, or asking for the item ("give me the tool"): the gripper opens
  (`true` on `/Robot/gripper`, `std_msgs/Bool`), the bring task is `R_DONE`, and R7
  follows: leave the hand-over position. After the pipe coupling the robot does
  not ask: R7 starts 2 s after it has said so (`config.HANDOVER_LEAVE_DELAY_S`),
  a delayed start like `later`. After the tool it asks first.
- `no`: the robot keeps holding the item (`R_HOLDING_HANDOVER`) and listens
  until the human says "give me the tool" / "give me the (pipe) coupling"; then
  the gripper opens and R7 follows as above.
- `cancel` while holding the item goes to manual recovery, as from `R_HOLDING`.

R7 is an ordinary robot action with its own permission, timer, pending and
delay messages, like R2: refused or unanswered, it stays pending ("I will stay
at the hand-over position") and other task starts wait for it; canceling a
delayed R7 -- including the 2 s after the pipe coupling -- puts it back in the
pending pool. Anything that follows the bring
task in its "robot task" chain is offered once R7 succeeds.

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
- Free drive and the hold stage have no automatic release timeout.

State transitions are validated against the task-aware state table. R1 success
enters `R_FREE_DRIVE` and switches free drive on (`R_WAITING_FREE_DRIVE`, asking
first, with `config.LIFT_ASKS_FREE_DRIVE = True`); R3/R4 success enters `R_WAITING_HANDOVER`; the
other actions' success enters `R_DONE`. Canceling while holding the panel or a
brought item uses manual recovery rather than offering return home, even if the latest
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
| 7 | leave_handover |

Example R2 dispatch:

```json
{"step_id":2,"human_step_id":3,"progress":1.0,"piece_id":12,"round_id":7,"suggested_action":"leave"}
```

The Python hold stage disables free drive after adjustment and waits for the
human command. It does not issue a new gripping/holding trajectory. Maintaining
the panel after free drive is disabled, and releasing it as part of R2, must be
provided by the robot/GH implementation. The hand-over opens the gripper by
publishing `true` on `/Robot/gripper` (`std_msgs/Bool`, true = open), the topic
the runtime also reads the gripper state from.

Only one task runs at a time. Recognition triggers arriving while busy are
retained in arrival order and deduplicated; R2's R3 follow-up is proposed first.
Each queued action still asks permission. The final H5 frames retain their
round/piece IDs; the round advances at the next H0 after all configured trigger
steps have been seen. The existing piece-ID-equals-round convention is retained.

## Run logs

Every run of `run_system.py` logs to `logs/runs/run_<date>_<time>/` (`--log-dir`,
`--run-name`; `--no-run-log` to turn it off). Both processes write there, all on the
wall clock, so what recognition saw and what communication did line up:

| File | Written by | Holds |
|---|---|---|
| `timeline.csv` | communication | Readable: every event, transition and message with its time, without the human-location stream. A task update shows when recognition saw it (`recognized_at`) and how long it took to arrive (`delay_ms`); the rows after it are the reaction. |
| `communication_events.jsonl` | communication | Everything, one JSON line each: `t` / `time` when written, `event_t` when the sender created the event. |
| `events.csv` | recognition | Every task update sent (`time`, `epoch_s`, task, progress). |
| `frames.csv` | recognition | Every processed frame: raw and stable step, confidence, progress, location. |
| `run.json`, `run.log` | recognition | Settings and summary; recognition's own log. |

`run_communication.py` and `run_recognition.py` log the same way on their own, each
to its own run directory unless given the same `--run-name`. Without a run
directory, communication appends to `hrc_communication_events.log` as before, now
timestamped too.

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
before free drive and the hold stage become available.

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

These tests simulate voice callbacks, timers and robot feedback without opening
a microphone or sending network commands. Live speech recognition, GH templates,
physical holding and release still require integration testing.
