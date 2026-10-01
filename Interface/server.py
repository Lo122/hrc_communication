"""FastAPI entry point for the H0 watch permission interface and HRC runtime."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import functools
import queue
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn


INTERFACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTERFACE_DIR.parent

# The original project uses layer directories rather than one installable Python
# package. Adding those directories here lets this standalone entry point import
# the exact Event, EventType, and runtime classes used by the rest of the system.
for layer in (
    "0_core",
    "1_recognition",
    "2_decision_making",
    "3_communication",
    "4_execution",
):
    path = str(REPO_ROOT / layer)
    if path not in sys.path:
        sys.path.insert(0, path)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(INTERFACE_DIR))

import config
from event_transport import UDPEventReceiver
from events import Event, EventType, RobotTaskState
from watch_screens import TASK_NAMES, WATCH_COMMANDS, allowed_commands, build_screen


# Literal gives FastAPI/Pydantic an explicit allowlist. A different command is
# rejected with HTTP 422 before the route function runs.
PermissionCommand = Literal["H_ACCEPT", "H_REFUSE", "H_DEFER"]


# Classes derived from BaseModel describe the JSON contract. FastAPI parses the
# request body into PermissionRequest and serializes the response models to JSON.
class PermissionRequest(BaseModel):
    command: PermissionCommand
    task_instance_id: str


class PermissionState(BaseModel):
    available: bool
    question: str | None = None
    task_instance_id: str | None = None
    human_step_id: int | None = None
    task_id: int | None = None


class PermissionEvent(BaseModel):
    event_type: PermissionCommand
    source: str
    task_instance_id: str
    human_step_id: int
    task_id: int
    timestamp: float


class PermissionAcknowledgement(BaseModel):
    processed: bool
    event: PermissionEvent


class WatchCommand(BaseModel):
    """Any human command the state-driven watch screen can send."""

    command: str
    task_instance_id: str | None = None
    # H_REQUEST_ROBOT_TASK only: which robot task (task database name).
    task_name: str | None = None


SimEventName = Literal[
    "RECOGNITION_TRIGGER", "ROBOT_RUNNING", "ROBOT_SUCCESS", "ROBOT_HOMED", "SETTINGS"
]


class SimRequest(BaseModel):
    """Simulator panel input (only accepted when started with --simulate)."""

    event: SimEventName
    step_id: int | None = None
    auto_robot: bool | None = None
    safe_home: bool | None = None


@dataclass(frozen=True)
class RuntimeSettings:
    """Values needed to run recognition input beside the HTTP server."""

    event_host: str = config.EVENT_TRANSPORT_HOST
    event_port: int = config.EVENT_TRANSPORT_PORT
    start_cli: bool = True
    debug_trigger: bool = False
    debug_round_id: int = 0
    debug_piece_id: int = 1
    simulate: bool = False
    sim_task_seconds: float = 16.0
    demo: bool = False             # open with the scripted dialogue (demo_opening.py)
    voice: bool = True
    force_voice: bool = False      # keep the real mic/TTS even with --simulate
    free_port: bool = False


def _udp_port_owners(host: str, port: int) -> list[tuple[int, str]]:
    """PIDs (and process names) holding a UDP port, for a friendlier error."""

    import subprocess

    owners: list[tuple[int, str]] = []
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["netstat", "-ano", "-p", "UDP"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                # UDP  127.0.0.1:5010  *:*  <pid>
                if len(parts) >= 4 and parts[0].upper() == "UDP" and parts[1].endswith(f":{port}"):
                    with suppress(ValueError):
                        owners.append((int(parts[-1]), ""))
            names = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            table = {}
            for row in names.splitlines():
                cells = [c.strip('"') for c in row.split('","')]
                if len(cells) >= 2:
                    with suppress(ValueError):
                        table[int(cells[1])] = cells[0]
            owners = [(pid, table.get(pid, "")) for pid, _ in owners]
        else:
            out = subprocess.run(
                ["lsof", "-nP", f"-iUDP:{port}", "-t"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            owners = [(int(pid), "") for pid in out.split() if pid.isdigit()]
    except Exception:
        return []
    # Never offer to kill this very process.
    return [(pid, name) for pid, name in dict(owners).items() if pid != os.getpid()]


def _free_udp_port(host: str, port: int) -> bool:
    """Stop whatever is holding the recognition port (opt-in: --free-port)."""

    import subprocess

    owners = _udp_port_owners(host, port)
    if not owners:
        print(f"[port] nothing to free on UDP {port}.")
        return False
    for pid, name in owners:
        label = f"{name} (pid {pid})" if name else f"pid {pid}"
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=10)
            else:
                os.kill(pid, signal.SIGTERM)
            print(f"[port] stopped {label} holding UDP {port}.")
        except Exception as exc:
            print(f"[port] could not stop {label}: {exc}")
    time.sleep(1.0)
    return True


TICK_SECONDS = 0.1      # runtime steps per second; higher rates starve speech
# The microphone opens this long after the robot stops talking (its last word's echo),
# and never waits longer than the limit for a line that hangs.
LISTEN_AFTER_SPEECH_S = 0.1
SPEECH_WAIT_LIMIT_S = 20.0


class HRCBridge:
    """Own one HRCSystem and connect HTTP, recognition, timers, voice, and ROS."""

    def __init__(
        self,
        system: Any,
        receiver: UDPEventReceiver | None = None,
        *,
        start_cli: bool = False,
        debug_trigger: bool = False,
        debug_round_id: int = 0,
        debug_piece_id: int = 1,
        simulate: bool = False,
        sim_task_seconds: float = 16.0,
        demo: bool = False,
    ):
        self.system = system
        self.demo = demo
        self.receiver = receiver
        self.start_cli = start_cli
        self.debug_trigger = debug_trigger
        self.debug_round_id = debug_round_id
        self.debug_piece_id = debug_piece_id
        self._process_lock = asyncio.Lock()
        self._snapshot_cache: dict[str, Any] | None = None
        self._snapshot_wanted = True
        self._process_task: asyncio.Task | None = None
        self.messages: deque[dict] = deque(maxlen=6)
        self._speaking_until = 0.0
        # Without a real voice the spoken line is instant, so the simulator
        # stretches the "robot is speaking" window: the stepped signal on the
        # watch has to be readable, not a flicker.
        self._speak_rate = 1.2 if simulate else 2.6      # words per second
        self._speak_min = 3.0 if simulate else 1.2       # seconds
        self._speech_queue: queue.Queue = queue.Queue()
        self._speech_thread: threading.Thread | None = None
        # Set while no line is queued or being spoken: the microphone opens on it,
        # right when the robot actually stops talking (not on the estimate above).
        self._speech_idle = threading.Event()
        self._speech_idle.set()
        self._speech_lines = 0
        self._speech_lock = threading.Lock()
        self._pending_listen: tuple | None = None
        self._install_message_tap()
        self.sim = None
        if simulate:
            from sim_robot import SimRobot

            self.sim = SimRobot(system, task_seconds=sim_task_seconds)

    def _install_message_tap(self) -> None:
        """Mirror every CLI/voice message so the watch can show it as a toast."""

        comm = getattr(self.system, "communication", None)
        if comm is None:
            return
        for name in ("show_message", "show_permission_request"):
            original = getattr(comm, name, None)
            if original is None:
                continue

            def tapped(message, *args, _original=original, **kwargs):
                text = kwargs.get("speech") or message
                self.messages.append({"text": text, "timestamp": time.time()})
                if self._speaking_until < time.time():
                    words = len(str(text).split())
                    self._speaking_until = time.time() + max(
                        self._speak_min, words / self._speak_rate
                    )
                return _original(message, *args, **kwargs)

            setattr(comm, name, tapped)

        # The watch shows the speaking signal while the robot talks; pyttsx3
        # gives no "is speaking" flag, so estimate it from the spoken words.
        tts = getattr(self.system, "tts", None)
        speak = getattr(tts, "speak", None)
        if speak is not None:
            def speaking(text, *args, _speak=speak, **kwargs):
                words = len(str(text).split())
                self._speaking_until = time.time() + max(
                    self._speak_min, words / self._speak_rate
                )
                # Hand the sentence to the speech thread and return at once:
                # spoken inline it would block the event loop, and the watch
                # would only learn about the speech after it had finished.
                with self._speech_lock:
                    self._speech_lines += 1
                    self._speech_idle.clear()
                self._speech_queue.put((_speak, text, args, kwargs))

            tts.speak = speaking
            self._speech_thread = threading.Thread(
                target=self._speech_worker, name="hrc-speech", daemon=True
            )
            self._speech_thread.start()

        voice = getattr(self.system, "voice", None)
        start_listening = getattr(voice, "start_listening", None)
        if start_listening is not None:
            def deferred(*args, _start=start_listening, **kwargs):
                if self._speech_idle.is_set():
                    self._pending_listen = None
                    return _start(*args, **kwargs)
                # Only the newest request survives: an older one would open the
                # microphone with callbacks the runtime has already replaced.
                self._pending_listen = (args, kwargs)
                pending = self._pending_listen

                def later():
                    # Open the microphone as soon as the robot has stopped talking,
                    # just past the echo of its last word.
                    self._speech_idle.wait(timeout=SPEECH_WAIT_LIMIT_S)
                    time.sleep(LISTEN_AFTER_SPEECH_S)
                    if self._pending_listen is pending:
                        self._pending_listen = None
                        _start(*args, **kwargs)

                threading.Thread(target=later, name="hrc-listen", daemon=True).start()

            voice.start_listening = deferred

    def _speech_worker(self) -> None:
        """Speak queued lines one after another, off the event loop."""

        while True:
            item = self._speech_queue.get()
            if item is None:
                return
            speak, text, args, kwargs = item
            try:
                speak(text, *args, **kwargs)
            except Exception as exc:        # a broken voice must not stop the UI
                print(f"[voice] could not speak: {exc}")
            finally:
                with self._speech_lock:
                    self._speech_lines -= 1
                    if self._speech_lines == 0:
                        self._speech_idle.set()
                        if config.VOICE_ENABLED:
                            # Real speech has ended: so has the watch's speaking signal
                            # (the simulator's stretched one is kept without a voice).
                            self._speaking_until = min(self._speaking_until, time.time())

    def _voice_status(self) -> dict[str, bool]:
        """Who is talking right now: the robot (TTS) or the human (mic open)."""

        now = time.time()
        comm = getattr(self.system, "communication", None)
        mode = getattr(comm, "mode", None)
        listening = bool(mode is not None and getattr(mode, "name", "OFF") != "OFF")
        speaking = now < self._speaking_until
        if self.sim is not None and not config.VOICE_ENABLED and not listening and not speaking:
            # Voice is off in the simulator: fake the channel so the orb is
            # demonstrable -- the robot "speaks" right after a new message and
            # then "listens" while a question is open.
            last = self.messages[-1]["timestamp"] if self.messages else 0.0
            speaking = now - last < 5.0
            task = self.system.task_manager.active_task
            listening = not speaking and task is not None and task.state.name in {
                "R_WAITING_RESPONSE", "R_WAITING_FREE_DRIVE", "R_WAITING_HANDOVER",
                "R_WAITING_HOME_PERMISSION",
            }
        return {"speaking": speaking, "listening": listening and not speaking}

    @classmethod
    def build_live(cls, settings: RuntimeSettings) -> "HRCBridge":
        """Build the same HRC runtime and recognition receiver as the CLI entry point."""

        # Importing here keeps module import side-effect free. The microphone,
        # ROS connection, UDP sender, and other runtime resources are created
        # only when FastAPI actually starts its application lifespan.
        import communication_runtime
        from communication_runtime import build_system

        if settings.simulate:
            # No rosbridge: the SimRobot plays the robot side. The microphone
            # stays off too, unless --voice asks for the real voice channel so
            # the watch and speech can be tried together without a robot.
            if not settings.force_voice:
                config.VOICE_ENABLED = False
            communication_runtime.ROSCommunication = functools.partial(
                communication_runtime.ROSCommunication, auto_connect=False
            )
        if not settings.voice:
            config.VOICE_ENABLED = False
        if settings.simulate and settings.demo:
            # No recognition process to warm up in the simulator: open right away.
            config.RECOGNITION_ACTIVATION_S = 0.0

        # rosbridge already fails softly (console fallback), but the voice stack
        # raises when the Vosk model or the microphone is missing. That should
        # not take the whole interface down: start without voice and say so, so
        # the watch, the timers and the robot side still run. Voice is only
        # blamed when the retry without it succeeds; anything else (a broken
        # task database, say) is raised as it is.
        try:
            system = build_system(demo=settings.demo)
        except Exception as exc:
            if not config.VOICE_ENABLED:
                raise
            config.VOICE_ENABLED = False
            try:
                system = build_system(demo=settings.demo)
            except Exception:
                raise exc from None
            print(f"[voice] disabled: {exc}")
            print("[voice] starting without speech input/output "
                  "(use --no-voice to skip this attempt next time).")
        if settings.free_port:
            _free_udp_port(settings.event_host, settings.event_port)
        try:
            receiver = UDPEventReceiver(settings.event_host, settings.event_port)
        except OSError as exc:
            # Almost always a second copy of this server (or main.py) still
            # holding the recognition port. Say that instead of a raw winerror.
            system.close()
            owners = _udp_port_owners(settings.event_host, settings.event_port)
            who = ", ".join(
                f"{name or 'python'} (pid {pid})" for pid, name in owners
            ) or "another process"
            raise RuntimeError(
                f"Recognition UDP port {settings.event_host}:{settings.event_port} "
                f"is already in use by {who} ({exc}). It is almost always an older "
                f"server.py or main.py. Restart this one with --free-port to close "
                f"it automatically, or use --event-port <free port> (and point the "
                f"recognition sender at the same port)."
            ) from exc
        except Exception:
            system.close()
            raise
        return cls(
            system,
            receiver,
            start_cli=settings.start_cli,
            debug_trigger=settings.debug_trigger,
            debug_round_id=settings.debug_round_id,
            debug_piece_id=settings.debug_piece_id,
            simulate=settings.simulate,
            sim_task_seconds=settings.sim_task_seconds,
            demo=settings.demo,
        )

    async def start(self) -> None:
        """Start the existing HRC event-processing loop inside FastAPI."""

        self.system.system_running = True
        if self.start_cli:
            self.system.start_cli_thread()
        if self.demo:
            # "Shall we start the assembly?" -- once recognition is active.
            self.system.start_demo()
        if self.debug_trigger:
            self.system.event_queue.put(
                Event(
                    event_type=EventType.RECOGNITION_TRIGGER,
                    source="watch_debug_recognition",
                    payload={
                        "step_id": config.HUMAN_PULL_CABLES,
                        "piece_id": self.debug_piece_id,
                        "round_id": self.debug_round_id,
                        "progress": 1.0,
                    },
                )
            )
        self._process_task = asyncio.create_task(
            self._process_events(), name="hrc-event-loop"
        )

    async def stop(self) -> None:
        """Release HRC and recognition resources when FastAPI shuts down."""

        self.system.system_running = False
        if self._process_task is not None:
            self._process_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._process_task
            self._process_task = None
        if self.receiver is not None:
            self.receiver.close()
        if self._speech_thread is not None:
            self._speech_queue.put(None)
        self.system.close()

    async def _process_events(self) -> None:
        """Continuously feed recognition and queued events to TaskManager."""

        while self.system.system_running:
            if self.receiver is not None:
                for event in self.receiver.poll():
                    self.system.event_queue.put(event)
            async with self._process_lock:
                await asyncio.to_thread(self._tick)
            await asyncio.sleep(TICK_SECONDS)

    def _tick(self) -> None:
        """One pass of the runtime, run in a worker thread."""

        if self.sim is not None:
            self.sim.tick()
        self.system.process_events()
        # Only build the JSON the watch actually asked for (about three times a
        # second), not on every tick: that work competes with the speech.
        if self._snapshot_wanted or self._snapshot_cache is None:
            self._snapshot_wanted = False
            self._snapshot_cache = self._snapshot()

    def _active_h0_permission(self):
        task = self.system.task_manager.active_task
        if task is None:
            return None
        if task.state is not RobotTaskState.R_WAITING_RESPONSE:
            return None
        if task.step_id != config.HUMAN_PULL_CABLES:
            return None
        if task.task_id != config.TASK_LIFT_PANEL:
            return None
        return task

    async def permission_state(self) -> PermissionState:
        """Return only the H0 permission that this UI is allowed to answer."""

        async with self._process_lock:
            task = self._active_h0_permission()
            if task is None:
                return PermissionState(available=False)
            return PermissionState(
                available=True,
                question="Lift the panel?",
                task_instance_id=task.task_instance_id,
                human_step_id=task.step_id,
                task_id=task.task_id,
            )

    async def submit_permission(self, payload: PermissionRequest) -> PermissionEvent:
        """Convert an HTTP command into the shared core Event and process it."""

        async with self._process_lock:
            task = self._active_h0_permission()
            if task is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="No Human Step 0 permission is waiting for a response.",
                )
            if payload.task_instance_id != task.task_instance_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The permission belongs to an older task instance.",
                )

            # This is the key business-logic connection: the HTTP request becomes
            # the project's real Event and enters the same queue used by voice,
            # CLI, timers, recognition, and ROS callbacks.
            event = Event(
                event_type=EventType[payload.command],
                source="human_watch",
                task_instance_id=task.task_instance_id,
            )
            self.system.event_queue.put(event)
            self.system.process_events()

            return PermissionEvent(
                event_type=payload.command,
                source=event.source,
                task_instance_id=event.task_instance_id,
                human_step_id=task.step_id,
                task_id=task.task_id,
                timestamp=event.timestamp,
            )


    # ------------------------------------------------------------------ #
    # State-driven watch (all tasks, all states)                          #
    # ------------------------------------------------------------------ #
    def _pending(self) -> list:
        pool = getattr(self.system, "pending_pool", None)
        if pool is None:
            pool = getattr(self.system.task_manager, "pending_pool", None)
        return pool.list_all() if pool is not None else []

    def _state_machine(self):
        machine = getattr(self.system, "state_machine", None)
        return machine or self.system.task_manager.state_machine

    def _screen_context(self) -> dict[str, Any]:
        """What the screen needs besides the active task: the robot task asked about
        while the active one runs, and the assembly tracker for the idle screen."""

        manager = self.system.task_manager
        return {
            "advance": getattr(manager, "advance_task", None),
            "advance_answer": getattr(manager, "advance_answer", None),
            "tracker": getattr(manager, "tracker", None),
            "question": getattr(manager, "question", None),
            "opening": getattr(manager, "in_opening", False),
            "human_turn": getattr(manager, "human_turn", None),
        }

    def _snapshot(self) -> dict[str, Any]:
        now = time.time()
        task = self.system.task_manager.active_task
        pending = self._pending()
        screen = build_screen(task, pending, self._state_machine(), now,
                              **self._screen_context())
        return {
            "mode": "sim" if self.sim is not None else "live",
            "server_time": now,
            "state": task.state.name if task is not None else None,
            "task": None if task is None else {
                "task_instance_id": task.task_instance_id,
                "task_id": task.task_id,
                "name": TASK_NAMES.get(task.task_id, f"Task {task.task_id}"),
                "human_step_id": task.step_id,
                "piece_id": task.piece_id,
            },
            "speed": {
                "value": getattr(task, "speed", config.DEFAULT_SPEED),
                "min": config.MIN_SPEED,
                "max": config.MAX_SPEED,
            },
            "progress": self.sim.progress if self.sim is not None else None,
            "screen": screen.to_dict(),
            "pending": [
                {
                    "task_instance_id": p.task_instance_id,
                    "name": TASK_NAMES.get(p.task_id, f"Task {p.task_id}"),
                    "reason": p.pending_reason,
                }
                for p in pending
            ],
            "voice": self._voice_status(),
            "messages": list(self.messages),
            "sim": None if self.sim is None else {
                "auto_robot": self.sim.auto_robot,
                "safe_home": self.sim.safe_home,
                # The human steps recognition can report, in step_id order.
                "steps": list(config.STEP_NAMES),
            },
        }

    async def watch_state(self) -> dict[str, Any]:
        self._snapshot_wanted = True
        cached = self._snapshot_cache
        if cached is not None and self._process_task is not None:
            # Only while the runtime loop is the one driving the system: at
            # most one tick old (50 ms), the countdown is derived from the
            # deadline on the watch, and every command refreshes it. Tests and
            # other callers that step the system themselves read it live.
            return cached
        async with self._process_lock:
            return self._snapshot()

    async def submit_watch_command(self, payload: WatchCommand) -> dict[str, Any]:
        """Validate a watch command against the current screen, then queue it."""

        if payload.command not in WATCH_COMMANDS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown watch command {payload.command}.",
            )
        async with self._process_lock:
            task = self.system.task_manager.active_task
            pending = self._pending()
            allowed = allowed_commands(task, pending, self._state_machine(),
                                       **self._screen_context())
            task_name = payload.task_name if payload.command == "H_REQUEST_ROBOT_TASK" else None
            if (payload.command, task_name) not in allowed:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"{payload.command} is not available right now.",
                )

            instance_id = payload.task_instance_id
            if payload.command == "H_EXECUTE_PENDING_TASK":
                ids = [p.task_instance_id for p in pending]
                instance_id = instance_id or ids[0]
                if instance_id not in ids:
                    raise HTTPException(status.HTTP_409_CONFLICT, "Pending task not found.")
            elif task is None:
                # Idle-screen inputs (task done, robot request) belong to no robot task.
                instance_id = None
            elif instance_id is not None and instance_id != task.task_instance_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The command belongs to an older task instance.",
                )
            else:
                instance_id = task.task_instance_id

            self.system.event_queue.put(
                Event(
                    event_type=EventType[payload.command],
                    source="human_watch",
                    task_instance_id=instance_id,
                    payload={"task_name": task_name} if task_name else {},
                )
            )
            self.system.process_events()
            self._snapshot_cache = self._snapshot()
            return self._snapshot_cache

    async def simulate(self, payload: SimRequest) -> dict[str, Any]:
        if self.sim is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Start the server with --simulate.")
        async with self._process_lock:
            if payload.event == "SETTINGS":
                if payload.auto_robot is not None:
                    self.sim.auto_robot = payload.auto_robot
                if payload.safe_home is not None:
                    self.sim.set_safe_home(payload.safe_home)
            elif payload.event == "RECOGNITION_TRIGGER":
                if payload.step_id is None or not 0 <= payload.step_id < len(config.STEP_NAMES):
                    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unknown human step.")
                self.sim.trigger(payload.step_id)
            else:
                self.sim.emit(EventType[payload.event])
            self.system.process_events()
            self._snapshot_cache = self._snapshot()
            return self._snapshot_cache


def create_app(
    settings: RuntimeSettings | None = None,
    bridge: HRCBridge | None = None,
) -> FastAPI:
    """Create the FastAPI application, with an injectable bridge for tests."""

    runtime_settings = settings or RuntimeSettings()

    # A lifespan function is FastAPI's startup/shutdown context. The HRC runtime
    # is created once at startup, stored on app.state, and closed on shutdown.
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        active_bridge = bridge or HRCBridge.build_live(runtime_settings)
        application.state.hrc_bridge = active_bridge
        try:
            await active_bridge.start()
            yield
        finally:
            await active_bridge.stop()

    app = FastAPI(
        title="HRC Watch Permission",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # FastAPI calls dependencies before a route. This dependency retrieves the
    # single bridge stored during startup instead of constructing a new system
    # for every HTTP request.
    def get_bridge(request: Request) -> HRCBridge:
        return request.app.state.hrc_bridge

    def static_file(name: str) -> FileResponse:
        return FileResponse(
            INTERFACE_DIR / name,
            headers={"Cache-Control": "no-store"},
        )

    # Route decorators connect an HTTP method and path to a Python function.
    # FastAPI converts each returned FileResponse directly into an HTTP response.
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        # The watch alone. The whole flow -- every task, signal and decision -- is
        # on the decision view's own port (config.DECISION_VIEW_PORT); the
        # Wizard-of-Oz panel stays on /sim.
        return static_file("watch.html")

    # State-driven watch: /watch = watch only (open on a phone), /sim = watch
    # plus the Wizard-of-Oz simulator panel.
    @app.get("/watch", include_in_schema=False)
    def watch_page() -> FileResponse:
        return static_file("watch.html")

    @app.get("/sim", include_in_schema=False)
    def sim_page() -> FileResponse:
        return static_file("simulator.html")

    @app.get("/watch.css", include_in_schema=False)
    def watch_styles() -> FileResponse:
        return static_file("watch.css")

    @app.get("/watch.js", include_in_schema=False)
    def watch_javascript() -> FileResponse:
        return static_file("watch.js")

    @app.get("/api/watch")
    async def watch_state(hrc_bridge: HRCBridge = Depends(get_bridge)) -> dict:
        return await hrc_bridge.watch_state()

    @app.post("/api/watch/command")
    async def watch_command(
        payload: WatchCommand, hrc_bridge: HRCBridge = Depends(get_bridge)
    ) -> dict:
        return await hrc_bridge.submit_watch_command(payload)

    @app.post("/api/sim")
    async def sim_event(
        payload: SimRequest, hrc_bridge: HRCBridge = Depends(get_bridge)
    ) -> dict:
        return await hrc_bridge.simulate(payload)

    # This GET lets the watch discover the current real task instance. It keeps
    # a stale browser page from replying to a newer permission request.
    @app.get("/api/permission", response_model=PermissionState)
    async def current_permission(
        hrc_bridge: HRCBridge = Depends(get_bridge),
    ) -> PermissionState:
        return await hrc_bridge.permission_state()

    # FastAPI parses the JSON body as PermissionRequest before calling this route.
    # The response_model also verifies and documents the JSON returned to the UI.
    @app.post(
        "/api/permission",
        response_model=PermissionAcknowledgement,
        status_code=status.HTTP_200_OK,
    )
    async def receive_permission(
        payload: PermissionRequest,
        hrc_bridge: HRCBridge = Depends(get_bridge),
    ) -> PermissionAcknowledgement:
        event = await hrc_bridge.submit_permission(payload)
        return PermissionAcknowledgement(processed=True, event=event)

    return app


# ASGI servers can import this object as `server:app`. Running this file directly
# uses main() below so command-line options can configure the integrated runtime.
app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the watch interface with the HRC communication runtime."
    )
    parser.add_argument("--host", default="127.0.0.1", help="FastAPI bind host.")
    parser.add_argument("--port", default=8765, type=int, help="FastAPI port.")
    parser.add_argument("--event-host", default=config.EVENT_TRANSPORT_HOST)
    parser.add_argument("--event-port", default=config.EVENT_TRANSPORT_PORT, type=int)
    parser.add_argument(
        "--debug-trigger",
        action="store_true",
        help="Inject one Human Step 0 recognition trigger at startup.",
    )
    parser.add_argument("--debug-round-id", default=0, type=int)
    parser.add_argument("--debug-piece-id", default=1, type=int)
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run without robot/ROS/voice; open /sim for the watch simulator.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Open with the scripted dialogue: start the assembly?, pull the "
             "cables?, then the lift (2_decision_making/demo_opening.py).",
    )
    parser.add_argument(
        "--free-port",
        action="store_true",
        help="Stop whatever still holds the recognition UDP port, then start.",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Start without microphone/TTS (watch and ROS still run).",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Keep the real microphone and TTS even with --simulate, so the "
             "watch and speech can be answered together without a robot.",
    )
    parser.add_argument(
        "--sim-task-seconds", default=16.0, type=float,
        help="Simulated duration of one robot task at default speed.",
    )
    args = parser.parse_args()

    settings = RuntimeSettings(
        event_host=args.event_host,
        event_port=args.event_port,
        debug_trigger=args.debug_trigger,
        debug_round_id=args.debug_round_id,
        debug_piece_id=args.debug_piece_id,
        simulate=args.simulate,
        sim_task_seconds=args.sim_task_seconds,
        demo=args.demo,
        voice=not args.no_voice,
        force_voice=args.voice and not args.no_voice,
        free_port=args.free_port,
    )
    shown_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Watch:          http://{shown_host}:{args.port}/")
    if config.DECISION_VIEW_PORT:
        print(f"Flow & signals: http://{config.DECISION_VIEW_HOST}:{config.DECISION_VIEW_PORT}/")
    if args.simulate:
        print(f"Simulator:      http://{shown_host}:{args.port}/sim")
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
