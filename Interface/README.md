# HRC watch interface

The watch is how the human answers the robot and asks it for help. It runs in one
process with the HRC communication runtime (`Interface/server.py`): the watch, the
decision layer and the voice (microphone and speech) share one event queue, so a
tap on the watch and a spoken "yes" do exactly the same thing.

![Watch screens](./watch-preview.png)

The backend dependencies (fastapi, uvicorn) are in the project's `pyproject.toml`;
`uv sync` from the repository root installs them.

## Try it

No robot needed: the simulator plays the robot, and the microphone and speech are
real.

```powershell
uv run python -B Interface\server.py --simulate --demo --voice
```

Then open:

| Page | Address |
|---|---|
| Watch + simulator panel | http://127.0.0.1:8765/sim |
| Flow & signals (decision view) | http://127.0.0.1:8770/ |

Robot tasks run on their own in the simulator ("auto robot" on the panel). Stop the
server with **Ctrl+C**.

A run through the opening:

| The robot says | You say |
|---|---|
| "Hello! Would you like to start the assembly?" | **"yes"** |
| "Would you like me to pull the cables?" | **"yes"**: "Okay, I will pull the cables of the left panel." (**"I'll do it"**: you pull them, and it asks "Would you like to move on?") |
| (while it pulls) | **"faster"**: "Speed increased." |
| "Would you like me to lift the panel?" (asked while it still pulls) | **"yes"**: it lifts right after the pull. **"later"**: it asks again in 30 s |
| "Panel in place. Adjust it, then say done." | **"done"** |
| "Holding the panel. Say done when it's screwed." | **"done"**: screwing is done, and it asks "Would you like me to let go of the panel?" |
| (at any time) | **"stop"** |
| (while the robot is idle) | **"hey UR, bring the connector"** |

Instead of speaking you can tap the watch, or type the same words in the terminal
the server runs in.

## Talking to the robot

- **"hey UR" only starts something new**: while the robot is idle, or to ask for
  another task while it works ("hey UR, bring the connector"). Other talk in the room
  is then ignored. You can also say "hey UR", pause, and say the command within 5 s.
- **Everything about the task under way needs no name**: answers to its questions
  ("yes", "later", "I'll do it"), "faster", "slower", "stop", "pause", "resume",
  "cancel", and the "done" it waits for. "stop", "pause" and "cancel" work while it is
  idle too.
- **Confirmations say what happened**: "Speed increased." only when the speed went up
  ("Already at top speed." when it cannot); "Paused.", "Resumed.", "Canceled.".
- **"I'll do it"** hands a task over: the robot waits, and moves on to the next task
  when you say **"done"**. The steps after it are recognition's to follow (in the
  simulator, the panel's recognition buttons). **"later"** gives you time: the robot
  waits 30 s (`config.LATER_ASK_AGAIN_S`), offers nothing else meanwhile, then asks
  again; ask for the task by name ("lift the panel") to have it sooner. **"no"** means
  you do it and it will not ask again.

Some replies mean something only where they fit (`_STATE_REPLIES` in
`3_communication/cmd_parser.py`): "I'll do it" answers a "Would you like me to...?"
question, "just hold" answers "adjust it by hand?", and "not yet" is "later" to a
question but "not ready" while the robot holds an item out. Said anywhere else, they
are not understood, so they can never trigger the wrong action. Keep to that table
when adding words.

Voice settings are in `config.py`: `VOICE_TTS_VOICE` (David, or Zira for a female
voice), `VOICE_TTS_RATE`, `VOICE_WAKE_WORD` (`False`: no name needed, and the robot
does not listen while idle), `VOICE_INPUT_DEVICE_NAME` if the microphone is not the
system default.

## With the real robot

Start the interface instead of `run_communication.py`: both would bind the same
recognition UDP port (5010).

```powershell
# Terminal 1: watch + communication + voice
uv run python -B Interface\server.py --demo

# Terminal 2: recognition (sends its events to UDP 5010)
uv run python run_recognition.py --camera
```

Open the watch at http://127.0.0.1:8765/. The opening ("Would you like to start the
assembly?") starts once recognition has reported in and warmed up
(`config.RECOGNITION_ACTIVATION_S`, 10 s). To wear the watch, start with
`--host 0.0.0.0` and open `http://<laptop-ip>:8765/` on a phone on the same Wi-Fi
(`?shape=round` for a round layout); it vibrates on new questions (Android only).

### Options

| Option | Effect |
|---|---|
| `--demo` | Open with the scripted dialogue (`2_decision_making/demo_opening.py`) |
| `--simulate` | No robot, ROS or Grasshopper: the simulator plays the robot. Voice is off unless `--voice` |
| `--voice` | Keep the real microphone and speech with `--simulate` |
| `--no-voice` | Start without microphone and speech |
| `--sim-task-seconds N` | How long a simulated robot task takes (default 16) |
| `--host`, `--port` | Where the watch is served (default 127.0.0.1:8765) |
| `--event-port N` | Recognition UDP port (default 5010) |
| `--free-port` | Stop whatever still holds the recognition port, then start |
| `--debug-trigger` | Inject one recognition trigger (Pull Cables) at startup |

If the server says the recognition port is in use, an older `server.py` or
`run_communication.py` is still running: close it, or start with `--free-port`.

## How the watch works

The backend decides what is on screen (`watch_screens.py`) and the page only renders
it: one screen per robot task state, plus the opening's questions. Every action is
checked against the decision layer's state machine, so the watch never offers a
command it would reject. Robot requests on the idle screen follow the flow (the lift
only after the cables, the next panel's cables once the robot is done with a panel).
See [UX_GUIDE.md](./UX_GUIDE.md) for the design rules and the screen map.

| Endpoint | Purpose |
|---|---|
| `GET /api/watch` | Snapshot: state, task, screen, speed, pending tasks, recent messages, voice |
| `POST /api/watch/command` | `{"command": "H_PAUSE", "task_instance_id": "..."}`; only commands shown on the current screen are accepted (409 otherwise) |
| `POST /api/sim` | Simulator only: `RECOGNITION_TRIGGER` (+`step_id`), `ROBOT_RUNNING`, `ROBOT_SUCCESS`, `ROBOT_HOMED`, `SETTINGS` |
| `GET` / `POST /api/permission` | The earlier Human Step 0 page's API (`_legacy_h0/`) |

## Tests

```powershell
uv run python -m pytest tests Interface
```

The step model tests need the recognition models in `1_recognition/best_model/`
(`S3_10fps_8s_bg05`, `3d_skeleton_05`, `3d_skeleton_02`); they are not in git.
