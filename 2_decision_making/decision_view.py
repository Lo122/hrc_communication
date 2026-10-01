"""Live view of the decision layer, for watching how it behaves in a session.

While the communication runtime runs, http://127.0.0.1:<config.DECISION_VIEW_PORT>/
shows, refreshed twice a second:
  - the task pool: every piece's tasks as not done / pending / working / done, who
    does them, progress, detector signals, how likely a pending task is next, and
    where the robot's active, queued and pooled tasks sit;
  - the human's reference task and the robot's active task, queue and pending pool;
  - each trigger rule and why it does or does not offer its task right now;
  - what the detectors see;
  - a timeline of what the layer did: events, state transitions, log messages;
  - the robot's own state: what its active task waits for and which events that state
    accepts, and what the ROS link last heard (connection, gripper, joints, TCP force).

build_snapshot() runs on the thread that runs TaskManager (HRCSystem.process_events,
via DecisionView.update) and only while someone has the page open; the server thread
serves the latest finished snapshot as /state.json and never touches live state.

The page's manual control changes state -- for when recognition or a detector got it
wrong, as they do in a study: it POSTs a command to /command, which the server only
checks and hands on (submit); HRCSystem queues it as a MANUAL_CONTROL event and
TaskManager applies it between the other events (manual_control.py). The server is
bound to this machine, and a command needs a header only this page sends.

The page itself is decision_view.html, read on every request, so it can be edited
while the runtime runs.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# src/ holds this layer's helpers (task database, transition table, trigger policy).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config
from manual_control import EVENT_TYPES, OPS
from task_database import ROBOT

PAGE = Path(__file__).resolve().parent / "decision_view.html"
TIMELINE_SIZE = 300
# Snapshots are built only while the page asked within WATCHED_S, at most every MIN_INTERVAL_S.
WATCHED_S = 3.0
MIN_INTERVAL_S = 0.25
# Too frequent to be worth a line each: the human's position is relayed several times a second.
UNLOGGED_EVENTS = {"HUMAN_LOCATION_UPDATE"}
# Log messages the page already shows as a whole: the tracker's one-line status.
UNLOGGED_MESSAGES = {"Task status."}

ROBOT_TASK_NAMES = {task_id: name for name, task_id in config.TRACKED_TO_ROBOT_TASK.items()}

# What a robot task in each state waits for, in words (the page also lists the events
# the state machine accepts there).
WAITING_FOR = {
    "R_WAITING_RESPONSE": "the human's yes / no / later",
    "R_ACCEPTED": "the robot to report it is running",
    "R_EXECUTING": "the robot to report success",
    "R_PAUSED": "resume (or cancel)",
    "R_REDO": "the robot to report it is running again",
    "R_DEFER": "its delay to run out (then it starts)",
    "R_WAITING_FREE_DRIVE": "yes / no to free drive",
    "R_FREE_DRIVE": "the human's \"adjustment done\"",
    "R_HOLDING": "screw done, or the panel looking secured",
    "R_WAITING_HANDOVER": "the human to take the item",
    "R_HOLDING_HANDOVER": "the human to ask for the item",
    "R_RECOVERY_EVALUATING": "whether it can return home",
    "R_WAITING_HOME_PERMISSION": "home / no (manual recovery)",
    "R_MANUAL_RECOVERY": "\"done\" after the manual recovery",
}

# A command POSTed to /command must carry this header. A page on another site cannot
# add it without the browser asking first, and this server never says yes -- so only
# this page (same origin) can change anything.
CONTROL_HEADER = "X-HRC-Control"
MAX_COMMAND_BYTES = 4096


# -- timeline -------------------------------------------------------------------

class TimelineLogger:
    """An EventLogger that also keeps its latest entries for the live view.

    Wraps the real logger: everything still goes to the log file. Other threads
    (voice) log too, so entries are guarded by a lock."""

    def __init__(self, inner, size: int = TIMELINE_SIZE):
        self.inner = inner
        self._entries: deque[dict] = deque(maxlen=size)
        self._lock = threading.Lock()
        self._seq = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def log_event(self, event) -> None:
        self.inner.log_event(event)
        if event.event_type.name not in UNLOGGED_EVENTS:
            text, tag = _event_text(event)
            self._add("event", tag, text, {"source": event.source} if event.source else None)

    def log_transition(self, task, event, old_state, new_state, message=None) -> None:
        self.inner.log_transition(task, event, old_state, new_state, message)
        name = ROBOT_TASK_NAMES.get(task.task_id, f"task {task.task_id}")
        self._add("transition", "transition",
                  f"{name} (piece {task.piece_id}): {old_state.name} -> {new_state.name}",
                  {"on": event.event_type.name, "note": message} if message else {"on": event.event_type.name})

    def log_message(self, message: str, context: dict | None = None) -> None:
        self.inner.log_message(message, context)
        if message in UNLOGGED_MESSAGES:
            return
        context = context or {}
        if message == "Task status changed.":
            text = (f"{context.get('task_name')} (piece {context.get('piece_id')}): "
                    f"{context.get('old_status')} -> {context.get('new_status')}")
            self._add("log", "status", text, {key: context[key] for key in ("executor", "inferred")
                                                if context.get(key)})
            return
        self._add("log", "log", message, context or None)

    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._entries)

    def _add(self, kind: str, tag: str, text: str, detail: dict | None) -> None:
        with self._lock:
            self._seq += 1
            self._entries.append({"seq": self._seq, "t": time.time(), "kind": kind, "tag": tag,
                                  "text": text, "detail": detail})


def _event_text(event) -> tuple[str, str]:
    payload = event.payload or {}
    name = event.event_type.name
    if name in ("HUMAN_TASK_UPDATE", "RECOGNITION_TRIGGER"):
        step = payload.get("step_id")
        task = (config.STEP_NAMES[step] if isinstance(step, int) and 0 <= step < len(config.STEP_NAMES)
                else step)
        return f"recognized {task} at {float(payload.get('progress') or 0.0):.2f}", "recognition"
    if name == "TASK_SIGNAL":
        return (f"{payload.get('task_name')} (piece {payload.get('piece_id')}): "
                f"{payload.get('signal')} = {payload.get('value')}", "detector")
    task_name = payload.get("task_name")
    return name + (f" {task_name}" if task_name else ""), "event"


# -- snapshot -------------------------------------------------------------------

def _robot_task(task, state_machine=None) -> dict | None:
    """A robot task for the page; with the state machine, also what its state waits for
    ({event: next state})."""
    if task is None:
        return None
    now = time.time()
    return {"instance": task.task_instance_id, "task": ROBOT_TASK_NAMES.get(task.task_id, str(task.task_id)),
            "task_id": task.task_id, "piece_id": task.piece_id, "state": task.state.name,
            "reason": task.pending_reason, "updated_at": task.updated_at,
            "seconds_in_state": None if task.updated_at is None else round(now - task.updated_at, 1),
            "waiting_for": WAITING_FOR.get(task.state.name, ""),
            "accepts": ({} if state_machine is None
                        else state_machine.events_from(task.state, task.task_id)),
            "speed": task.speed, "free_drive": task.free_drive_active,
            "running_reported": task.robot_running_received,
            "success_reported": task.robot_success_received,
            "delay_s": task.defer_seconds}


def _ros_status(ros) -> dict:
    """What the ROS link knows of the robot itself, for the page. Read only."""
    client = getattr(ros, "client", None)
    wrench = getattr(ros, "latest_wrench", None)
    force = None
    if isinstance(wrench, tuple) and len(wrench) == 2:
        t, values = wrench
        force = {"age_s": round(time.time() - t, 1), "force_n": [round(v, 1) for v in values[:3]],
                 "torque_nm": [round(v, 2) for v in values[3:6]]}
    joints = getattr(ros, "latest_joint_positions", None)
    gripper = getattr(ros, "latest_gripper_open", None)
    return {"connected": bool(client is not None and getattr(client, "is_connected", False) is True),
            "joints_rad": [round(v, 3) for v in joints] if isinstance(joints, list) else None,
            "gripper_open": gripper if isinstance(gripper, bool) else None,
            "wrench": force}


def _choice(choice) -> dict | None:
    """The step selector's last pick (sequence_step_selector.StepChoice): the model's
    step, the one chosen, why, and {task: [p, weight, score]} of the tasks it weighed."""
    if choice is None:
        return None
    return {"model": choice.model_task, "selected": choice.task_name, "reason": choice.reason,
            "progress": round(choice.progress, 3),
            "scores": {name: [round(value, 3) for value in values]
                       for name, values in choice.scores.items()}}


def build_snapshot(manager, *, timeline=None, detectors=None) -> dict:
    """Everything the page shows, as plain JSON-ready data. Call it on the thread
    that runs TaskManager."""
    tracker, policy = manager.tracker, manager.policy
    held = manager.held_piece_id
    snapshot = {
        "time": time.time(),
        "mode": getattr(manager, "mode", "proactive"),
        "detector_mode": config.TASK_DETECTORS_MODE,
        "robot": {
            "active": _robot_task(manager.active_task, getattr(manager, "state_machine", None)),
            # Asked while the active task runs; its answer is kept until that one succeeds.
            "advance": _robot_task(manager.advance_task),
            "advance_answer": manager.advance_answer,
            "demo_opening": manager.demo_opening,
            "held_piece_id": held,
            "queue": [dict(entry) for entry in manager.waiting_triggers],
            "pending_pool": [_robot_task(task) for task in manager.pending_pool.list_all()],
            "ros": _ros_status(getattr(manager, "ros", None)),
        },
        "manual": {
            "results": list(getattr(manager, "manual_results", [])),
            "events": list(EVENT_TYPES),
            "robot_tasks": [name for name in config.TRACKED_TO_ROBOT_TASK
                            if tracker is None or tracker.database.can_execute(name, ROBOT)],
        },
        "human": None,
        "pieces": [],
        "rules": [],
        "detectors": [],
        "timeline": timeline.entries() if hasattr(timeline, "entries") else [],
    }
    if tracker is not None:
        likely = {(task.piece_id, task.task_name): probability for task, probability in tracker.pending()}
        open_ids = tracker.open_piece_ids
        snapshot["human"] = {
            "recognition_active": manager.recognition_active,
            "reference_task": tracker.reference_task,
            "reference_piece_id": tracker.reference_piece_id,
            "reference_progress": tracker.reference_progress,
            "human_piece_id": tracker.human_piece_id,
            "current_piece_id": tracker.current_piece_id,
            "reference_seconds": tracker.reference_seconds(),
            "reference_limit_s": tracker.reference_limit(),
            "ignored_task": tracker.ignored_task,
            "status_line": tracker.status_line(),
            "recognition_choice": _choice(getattr(manager, "last_recognition", None)),
        }
        for piece in tracker.database.pieces:
            tasks = []
            for name in piece.task_list:
                task = tracker.get(name, piece.piece_id)
                tasks.append({
                    "name": name, "status": task.status.name, "executor": task.executor,
                    "progress": round(task.progress, 3), "inferred": task.inferred,
                    "robot_offered": task.robot_offered, "signals": dict(task.signals),
                    "probability": likely.get((piece.piece_id, name)),
                    "recognized": tracker.is_recognized(name),
                    "robot_can": tracker.database.can_execute(name, ROBOT),
                })
            snapshot["pieces"].append({"piece_id": piece.piece_id, "location": piece.location,
                                       "open": piece.piece_id in open_ids, "tasks": tasks})
    if policy is not None:
        rules = policy.explain()
        for row in rules:
            # TaskManager's own gate on top of the rules: the robot can only leave a
            # panel it holds.
            if config.TRACKED_TO_ROBOT_TASK.get(row["task"]) == config.TASK_LEAVE and row["offers"]:
                row["offers"] = [piece for piece in row["offers"] if piece == held]
                if not row["offers"]:
                    row["verdict"] = "the robot holds no panel of that piece"
        snapshot["rules"] = rules
    if detectors is not None:
        snapshot["detectors"] = [{"name": detector.name, "status": detector.status()}
                                 for detector in detectors.detectors]
    return snapshot


# -- server -----------------------------------------------------------------------

class DecisionView:
    """Serves decision_view.html and the latest snapshot; see the module docstring."""

    def __init__(self, host: str, port: int, submit=None):
        self._lock = threading.Lock()
        self._state = b'{"waiting": true}'
        self._wanted_at = 0.0
        self._built_at = 0.0
        # submit(payload): hands a manual-control command to the thread that runs
        # TaskManager (HRCSystem queues it as a MANUAL_CONTROL event). None: read only.
        self._submit = submit
        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path.split("?", 1)[0] != "/command":
                    self.send_error(404)
                    return
                status, reply = view._command(self.headers, self.rfile)
                body = json.dumps(reply).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
                elif path == "/state.json":
                    body, kind = view._request_state(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # the communication console is also the CLI prompt: keep it clean

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://{host}:{self.server.server_address[1]}/"
        self._thread = threading.Thread(target=self.server.serve_forever, name="decision-view", daemon=True)
        self._thread.start()

    @classmethod
    def start(cls, host: str, port: int) -> "DecisionView | None":
        """The running view, or None (with a note) when the port is taken."""
        try:
            view = cls(host, port)
        except OSError as exc:
            print(f"[decision view] not started, {host}:{port} is not free: {exc}")
            return None
        print(f"[decision view] {view.url}")
        return view

    def update(self, build) -> None:
        """Call on the thread that runs TaskManager. build() makes the snapshot; it is
        only called while the page is open, and at most every MIN_INTERVAL_S."""
        now = time.time()
        with self._lock:
            watched = now - self._wanted_at < WATCHED_S
        if not watched or now - self._built_at < MIN_INTERVAL_S:
            return
        self._built_at = now
        state = json.dumps(build(), default=str).encode("utf-8")
        with self._lock:
            self._state = state

    def set_submit(self, submit) -> None:
        """Accept manual-control commands from now on, handing each to submit(payload)."""
        self._submit = submit

    def _command(self, headers, stream) -> tuple[int, dict]:
        """One POSTed manual-control command: checked here, applied by TaskManager later
        (manual_control.py), on its own thread. The outcome shows in the next snapshot."""
        if headers.get(CONTROL_HEADER) != "1":
            return 403, {"error": f"missing the {CONTROL_HEADER} header"}
        if self._submit is None:
            return 503, {"error": "manual control is not connected"}
        try:
            length = int(headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 < length <= MAX_COMMAND_BYTES:
            return 400, {"error": "empty or oversized command"}
        try:
            payload = json.loads(stream.read(length))
        except (ValueError, UnicodeDecodeError):
            return 400, {"error": "not JSON"}
        if not isinstance(payload, dict) or payload.get("op") not in OPS:
            return 400, {"error": f"unknown command; one of {', '.join(OPS)}"}
        self._submit(payload)
        return 202, {"queued": payload["op"]}

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _request_state(self) -> bytes:
        with self._lock:
            self._wanted_at = time.time()
            return self._state
