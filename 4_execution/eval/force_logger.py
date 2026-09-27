"""Record the TCP force together with a camera view, to find detector thresholds.

Shows the camera next to live plots of the TCP force, and records both into one
take folder along with a marker for every marker key you press (layout:
src/force_take.py). The plots are ScrewingMonitor's view of the force
(4_execution/force_monitors.py) with the thresholds given here, so you can see live whether
a push counts and when "screw done" would fire. Read a take back with
force_log_review.py: it plays the take with the plots in sync and suggests thresholds.

A trial: start the logger; when the robot holds the panel press h (the baseline is
taken 0.5 s later, so keep hands off the panel for a moment); press s / e at the
start / end of every screw, d when all screws are in, r when the robot lets go.
Markers can also be set or corrected afterwards in the review.

Keys:
    h hold start   s screw start   e screw end   d screw done   r release   m mark
    b re-take the baseline (no hold, e.g. a bench test)
    a plot 2: change since hold <-> absolute force
    q or Esc: stop and save

Wrench sources:
    --source rtde  (default) the controller over RTDE at --rtde-hz. Receive only: it
                   never takes the control session, so it runs next to the MAIL UR
                   bridge and the other readers.
    --source ros   a geometry_msgs/WrenchStamped topic via rosbridge; --ros-topic is a
                   config.ROS_TOPICS key or a topic name (default ur_tcp_force, from
                   read_ur_live_data.py --publish-ros, at its --ros-hz: 20 Hz by default).
    --source fake  synthetic pushes and a level shift, to try the tool without a robot.

Usage:
    uv run python 4_execution/eval/force_logger.py --ip 169.254.130.206 --video-source 0 --name screw01
    uv run python 4_execution/eval/force_logger.py --iphone --source ros
    uv run python 4_execution/eval/force_logger.py --source fake --video-source 0
    uv run python 4_execution/eval/force_logger.py --video-source none     # force only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

EXECUTION = Path(__file__).resolve().parents[1]  # 4_execution/
ROOT = EXECUTION.parent
# The repo root last, so it ends up first: multi-actor-interface ships its own
# top-level `config` package (see pyproject.toml).
for _path in (ROOT / "1_recognition", ROOT / "1_recognition" / "src", EXECUTION / "src", EXECUTION, ROOT):
    sys.path.insert(0, str(_path))

import cv2
import numpy as np

import config
import force_take as take_io
from camera_utils.frame_source import BACKEND_NAMES, FrameSource
from force_monitors import ScrewingMonitor, ScrewingThresholds
from force_take import INK, INK_2, AXIS_COLORS, clock
from logging_setup import configure_logging, get_logger
from render_utils.video_recorder import VideoRecorder
from vision_model.vision_config import CameraConfig

logger = get_logger(__name__)

WINDOW = "force logger"
IPHONE_CAPTURE_ROTATE90 = 90  # run_recognition.py's default


# -- recording ------------------------------------------------------------------

def _cell(value) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _cells(values) -> list[str]:
    return [""] * 6 if values is None else [f"{float(v):.5f}" for v in values]


class ForceLog:
    """force.csv, plus the recent samples and ScrewingMonitor state for the view.

    Sources call add() from their own thread; the UI thread calls snapshot() and
    reset_monitor(). One lock covers all of it, so the monitor sees every sample in
    order and a reset never lands in the middle of an update."""

    def __init__(self, path: Path, monitor: ScrewingMonitor, history_s: float):
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        self._csv.writerow(take_io.FORCE_COLUMNS)
        self._lock = threading.Lock()
        self.monitor = monitor
        self.history_s = history_s
        self._recent: deque = deque()  # (t, wrench, delta or None, load or None)
        self._flushed_at = 0.0
        self.count = 0

    def add(self, t_wall: float, wrench, *, t_robot=None, pose=None, raw=None) -> None:
        wrench = [float(v) for v in wrench]
        row = [f"{t_wall:.6f}", _cell(t_robot), *_cells(wrench), *_cells(pose), *_cells(raw)]
        with self._lock:
            if self._file.closed:
                return
            self._csv.writerow(row)
            if t_wall - self._flushed_at >= 1.0:
                self._file.flush()
                self._flushed_at = t_wall
            monitor = self.monitor
            monitor.update(t_wall, wrench)
            self._recent.append((
                t_wall, wrench,
                None if monitor.delta is None else monitor.delta.copy(),
                None if monitor.load is None else monitor.load.copy(),
            ))
            while self._recent[0][0] < t_wall - self.history_s:
                self._recent.popleft()
            self.count += 1

    def reset_monitor(self, t: float, settle_s: float | None) -> None:
        with self._lock:
            self.monitor.reset(t, settle_s)

    def snapshot(self):
        """(t, wrench, delta, load, state) -- arrays of the recent samples, NaN where
        there was no baseline yet, and the monitor's state."""
        nan6 = [np.nan] * 6
        with self._lock:
            recent = list(self._recent)
            m = self.monitor
            state = {
                "baseline": m.baseline is not None, "active": m.active,
                "pushes": m.push_count, "active_s": m.active_s, "quiet_s": m.quiet_for(),
                "done_at": m.done_at, "level_shifts": len(m.level_shifts),
                "load": None if m.load is None else m.load.copy(),
            }
        n = len(recent)
        t = np.array([r[0] for r in recent])
        wrench = np.array([r[1] for r in recent], dtype=float).reshape(n, 6)
        delta = np.array([nan6 if r[2] is None else r[2] for r in recent], dtype=float).reshape(n, 6)
        load = np.array([nan6 if r[3] is None else r[3] for r in recent], dtype=float).reshape(n, 6)
        return t, wrench, delta, load, state

    def close(self) -> None:
        with self._lock:
            self._file.close()


class MarkerLog:
    """markers.csv, flushed on every marker so a crash keeps them."""

    def __init__(self, path: Path):
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["t_wall", "label"])
        self.items: list[tuple[float, str]] = []

    def add(self, t: float, label: str) -> None:
        self._csv.writerow([f"{t:.6f}", label])
        self._file.flush()
        self.items.append((t, label))

    def close(self) -> None:
        self._file.close()


# -- wrench sources -------------------------------------------------------------

class RtdeSource:
    label = "rtde"

    def __init__(self, ip: str, hz: float, raw_wrench: bool):
        self.ip, self.hz, self.raw_wrench = ip, hz, raw_wrench
        self.error: str | None = None
        self._stop = threading.Event()
        self._rtde = None
        self._thread = None

    def connect(self) -> None:
        try:
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise SystemExit("--source rtde needs ur_rtde: pip install ur_rtde") from exc
        print(f"[rtde] connecting to {self.ip} at {self.hz:.0f} Hz ...")
        try:
            self._rtde = RTDEReceiveInterface(self.ip, frequency=self.hz)
        except RuntimeError as exc:
            raise SystemExit(f"[rtde] could not connect to {self.ip}: {exc}") from exc

    def start(self, log: ForceLog) -> None:
        self._thread = threading.Thread(target=self._run, args=(log,), name="rtde-reader", daemon=True)
        self._thread.start()

    def _run(self, log: ForceLog) -> None:
        rtde_r = self._rtde
        try:
            while not self._stop.is_set():
                t_start = rtde_r.initPeriod()
                wrench = rtde_r.getActualTCPForce()
                log.add(time.time(), wrench, t_robot=rtde_r.getTimestamp(),
                        pose=rtde_r.getActualTCPPose(),
                        raw=rtde_r.getFtRawWrench() if self.raw_wrench else None)
                rtde_r.waitPeriod(t_start)
        except Exception as exc:  # pragma: no cover - needs a robot
            self.error = f"RTDE read failed: {exc}"
            logger.error(self.error)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._rtde is not None:
            try:
                self._rtde.disconnect()
            except Exception:
                pass


class RosSource:
    label = "ros"

    def __init__(self, topic: str, host: str, port: int):
        self.topic_name = config.ROS_TOPICS.get(topic, topic)
        self.host, self.port = host, int(port)
        self.error: str | None = None
        self._client = None
        self._topic = None

    def connect(self) -> None:
        try:
            import roslibpy
        except ImportError as exc:
            raise SystemExit("--source ros needs roslibpy") from exc
        print(f"[ros] connecting to rosbridge at {self.host}:{self.port} ...")
        self._client = roslibpy.Ros(host=self.host, port=self.port)
        try:
            self._client.run()
        except Exception as exc:
            raise SystemExit(f"[ros] could not connect to rosbridge: {exc}") from exc

    def start(self, log: ForceLog) -> None:
        import roslibpy

        self._topic = roslibpy.Topic(self._client, self.topic_name, "geometry_msgs/WrenchStamped")
        self._topic.subscribe(lambda message: self._on_message(log, message))
        print(f"[ros] subscribed to {self.topic_name}")

    def _on_message(self, log: ForceLog, message: dict) -> None:
        try:
            force, torque = message["wrench"]["force"], message["wrench"]["torque"]
            wrench = [force["x"], force["y"], force["z"], torque["x"], torque["y"], torque["z"]]
        except (KeyError, TypeError):
            self.error = f"unexpected message on {self.topic_name}"
            return
        stamp = (message.get("header") or {}).get("stamp") or {}
        # ROS 1 stamps are {secs, nsecs}, ROS 2 ones {sec, nanosec}.
        secs = stamp.get("secs", stamp.get("sec"))
        nsecs = stamp.get("nsecs", stamp.get("nanosec", 0))
        t_robot = None if secs is None else secs + 1e-9 * nsecs
        log.add(time.time(), wrench, t_robot=t_robot)

    def stop(self) -> None:
        for close in (lambda: self._topic.unsubscribe(), lambda: self._client.terminate()):
            try:
                close()
            except Exception:
                pass


class FakeSource:
    """A synthetic take: the held panel's weight, four 2.5 s screw pushes with
    screwdriver ripple 7 s apart, then a 6 N level shift (the panel's weight going
    over to the frame) 1 s after the last push."""

    label = "fake"

    def __init__(self, hz: float):
        self.hz = hz
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = None

    def connect(self) -> None:
        pass

    def start(self, log: ForceLog) -> None:
        self._thread = threading.Thread(target=self._run, args=(log,), name="fake-wrench", daemon=True)
        self._thread.start()

    @staticmethod
    def wrench_at(s: float, rng: np.random.Generator) -> np.ndarray:
        """The wrench s seconds into the take."""
        wrench = np.array([0.4, -0.2, -25.0, 0.05, -0.03, 0.01])
        screw, phase = divmod(s - 6.0, 7.0)
        if 0 <= screw < 4 and phase < 2.5:
            wrench += [2.0, 0.0, -12.0 - 3.0 * np.sin(2 * np.pi * 6.0 * s), 0.1, 0.0, 0.8]
        if s > 6.0 + 3 * 7.0 + 2.5 + 1.0:
            wrench[2] += 6.0
        return wrench + rng.normal(0.0, [0.25, 0.25, 0.25, 0.02, 0.02, 0.02])

    def _run(self, log: ForceLog) -> None:
        rng = np.random.default_rng(0)
        start = next_at = time.time()
        while not self._stop.is_set():
            now = time.time()
            log.add(now, self.wrench_at(now - start, rng), t_robot=now - start)
            next_at += 1.0 / self.hz
            time.sleep(max(0.0, next_at - time.time()))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def make_source(args):
    if args.source == "rtde":
        return RtdeSource(args.ip, args.rtde_hz, args.raw_wrench)
    if args.source == "ros":
        return RosSource(args.ros_topic, args.ros_host, args.ros_port)
    return FakeSource(args.rtde_hz)


# -- view -------------------------------------------------------------------------

class RateMeter:
    def __init__(self):
        self._times: deque[float] = deque()

    def tick(self, t: float) -> None:
        self._times.append(t)

    def rate(self, now: float) -> float:
        while self._times and self._times[0] < now - 1.0:
            self._times.popleft()
        return float(len(self._times))


def render_view(frame, log: ForceLog, source, thresholds: ScrewingThresholds, *, now, t_start,
                markers, absolute, frame_rate, take_name, args) -> np.ndarray:
    t, wrench, delta, load, state = log.snapshot()
    width, height = args.plot_width, args.view_height
    heights = [int(height * 0.36), int(height * 0.28), int(height * 0.18)]
    heights.append(height - sum(heights))
    t_range = (now - args.plot_seconds, now)
    marks = [(mt, label) for mt, label in markers if mt >= t_range[0]]
    common = {"t_origin": t_start, "markers": marks}

    def last(values):
        return values[-1] if len(values) else None

    def now_text(values) -> str:
        value = last(values)
        return "--" if value is None or not np.isfinite(value) else f"{value:.2f}"

    push = np.linalg.norm(delta[:, :3], axis=1)
    p1 = take_io.render_panel(
        width, heights[0], t, [("", push, INK, None)], t_range,
        title=f"|dF| push signal (N)  {now_text(push)}",
        hlines=[(thresholds.active_force_n, f"active {thresholds.active_force_n:g}"),
                (thresholds.quiet_force_n, f"quiet {thresholds.quiet_force_n:g}")],
        min_span=thresholds.active_force_n * 1.2, **common)
    if absolute:
        components, title = wrench[:, :3], "F absolute (N)"
    else:
        components, title = load[:, :3], "dF since hold (N)"
    p2 = take_io.render_panel(
        width, heights[1], t,
        [(name, components[:, i], AXIS_COLORS[i], last(components[:, i]))
         for i, name in enumerate(("x", "y", "z"))],
        t_range, title=title, marker_labels=False, **common)
    torque = np.linalg.norm(delta[:, 3:], axis=1)
    p3 = take_io.render_panel(
        width, heights[2], t, [("", torque, INK, None)], t_range,
        title=f"|dT| (Nm)  {now_text(torque)}",
        min_span=0.5, marker_labels=False, **common)

    force_rate = float(np.sum(t >= now - 1.0))
    line1 = (f"REC {clock(now - t_start)}   force {force_rate:.0f} Hz ({source.label})   "
             f"video {frame_rate:.0f} fps   {take_name}")
    lines = [(line1, INK)]
    if source.error:
        lines.append((f"! {source.error}", INK))
    elif force_rate == 0:
        lines.append(("! no force samples in the last second", INK))
    elif not state["baseline"]:
        lines.append(("taking the baseline -- keep hands off the panel", INK))
    else:
        verdict = (f"DONE at {clock(state['done_at'] - t_start)}" if state["done_at"]
                   else "pushing" if state["active"] else "waiting")
        lines.append((f"pushes {state['pushes']}   pushing {state['active_s']:.1f} s   "
                      f"quiet {state['quiet_s']:.1f} s   level shifts {state['level_shifts']}   "
                      f"-> {verdict}", INK_2))
    last_marker = (f"last marker: {markers[-1][1]} at {clock(markers[-1][0] - t_start)}"
                   if markers else "no markers yet")
    lines.append((f"{last_marker}   plot 2: {'absolute' if absolute else 'change'} (a)", INK_2))
    lines.append((take_io.KEY_HELP + "  a view  q quit", INK_2))
    status = take_io.render_text(width, heights[3], lines)

    left = None if frame is None else take_io.fit_height(frame, height)
    return take_io.side_by_side(left, [p1, p2, p3, status])


def _placeholder(height: int, text: str) -> np.ndarray:
    img = take_io.render_text(height * 4 // 3, height, [])
    take_io.put_text(img, text, (16, height // 2), INK_2, 0.6)
    return img


# -- main -------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("wrench")
    source.add_argument("--source", choices=("rtde", "ros", "fake"), default="rtde")
    source.add_argument("--ip", default=config.ROBOT_IP, help="Robot controller IP (default config.ROBOT_IP).")
    source.add_argument("--rtde-hz", type=float, default=125.0, help="RTDE read rate (fake: sample rate).")
    source.add_argument("--raw-wrench", action="store_true",
                        help="Also log ft_raw_wrench (uncompensated sensor reading; needs "
                             "a controller that provides it).")
    source.add_argument("--ros-topic", default="ur_tcp_force",
                        help="config.ROS_TOPICS key or topic name, for --source ros.")
    source.add_argument("--ros-host", default=config.ROS_BRIDGE_HOST)
    source.add_argument("--ros-port", type=int, default=config.ROS_BRIDGE_PORT)

    video = parser.add_argument_group("video")
    video.add_argument("--video-source", default="0",
                       help="Webcam index, stream URL, 'iphone', or 'none' for force only.")
    video.add_argument("--iphone", action="store_true", help="Same as --video-source iphone.")
    video.add_argument("--iphone-rotate", type=int, default=IPHONE_CAPTURE_ROTATE90, choices=[0, 90, 180, 270])
    video.add_argument("--capture-width", type=int, default=None)
    video.add_argument("--capture-height", type=int, default=None)
    video.add_argument("--backend", choices=BACKEND_NAMES, default="auto")
    video.add_argument("--video-fps", type=float, default=30.0,
                       help="Nominal fps written into video.mp4; the real capture times go "
                            "to video.timestamps.csv.")
    video.add_argument("--save-overlay", action="store_true",
                       help="Also record the composed view (camera + plots) to overlay.mp4.")

    view = parser.add_argument_group("view")
    view.add_argument("--view-height", type=int, default=600)
    view.add_argument("--plot-width", type=int, default=760)
    view.add_argument("--plot-seconds", type=float, default=15.0)

    th = ScrewingThresholds()
    monitor = parser.add_argument_group("ScrewingMonitor thresholds (live preview only; "
                                        "the raw force is always recorded)")
    monitor.add_argument("--active", type=float, default=th.active_force_n, help="push starts, N")
    monitor.add_argument("--quiet", type=float, default=th.quiet_force_n, help="push ends / quiet, N")
    monitor.add_argument("--min-active", type=float, default=th.min_active_s, help="pushing needed, s")
    monitor.add_argument("--quiet-seconds", type=float, default=th.quiet_s, help="quiet before done, s")
    monitor.add_argument("--max-push", type=float, default=th.max_push_s, help="longer = level shift, s")

    output = parser.add_argument_group("output")
    output.add_argument("--name", default=None, help="Suffix for the take folder name.")
    output.add_argument("--out-dir", default=str(take_io.DEFAULT_TAKES_DIR))
    return parser.parse_args()


def _open_frames(args) -> FrameSource | None:
    source = "iphone" if args.iphone else args.video_source
    if str(source).lower() == "none":
        return None
    camera = CameraConfig(
        video_source=source,
        capture_rotate90=args.iphone_rotate if str(source).lower() == "iphone" else 0,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        capture_backend=args.backend,
        calib_dir=None,  # no posture here, so no calibration is needed
    )
    return FrameSource(camera, fallback_fps=args.video_fps)


def main() -> None:
    args = _parse_args()
    # Robot and camera first, so a wrong IP or camera index leaves no empty take behind.
    source = make_source(args)
    source.connect()
    frames = _open_frames(args)
    if frames is not None:
        try:
            frames.read()
        except RuntimeError as exc:
            source.stop()
            raise SystemExit(str(exc)) from exc

    take = take_io.new_take_dir(args.out_dir, args.name)
    configure_logging("force_logger", log_file=take / "logger.log")
    thresholds = ScrewingThresholds(
        active_force_n=args.active, quiet_force_n=args.quiet, min_active_s=args.min_active,
        quiet_s=args.quiet_seconds, max_push_s=args.max_push)
    log = ForceLog(take / take_io.FORCE_CSV, ScrewingMonitor(thresholds), history_s=args.plot_seconds + 1.0)
    markers = MarkerLog(take / take_io.MARKERS_CSV)
    raw_video = (VideoRecorder(take / take_io.VIDEO, args.video_fps, label="camera frames",
                               write_timestamps=True) if frames is not None else None)
    overlay = (VideoRecorder(take / "overlay.mp4", args.video_fps, label="overlay view")
               if args.save_overlay else None)

    t_start = time.time()
    meta = {
        "started": datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
        "t_start": t_start,
        "source": args.source,
        "ip": args.ip if args.source == "rtde" else None,
        "rtde_hz": args.rtde_hz if args.source != "ros" else None,
        "ros_topic": source.topic_name if args.source == "ros" else None,
        "video_source": None if frames is None else str(frames.camera.video_source),
        "video_fps_nominal": args.video_fps,
        "wrench": "actual_TCP_force: base frame, at the TCP, controller payload subtracted",
        "thresholds": asdict(thresholds),
        "marker_keys": take_io.MARKER_KEYS,
    }
    (take / take_io.META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.monitor.reset(t_start)
    source.start(log)

    print(f"Recording to {take}")
    print(take_io.KEY_HELP + "  a view  q quit")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    frame_rate = RateMeter()
    last_frame = None
    absolute = False
    try:
        while True:
            frame = frames.read() if frames is not None else None
            now = time.time()
            if frame is not None:
                raw_video.write(frame, now)
                frame_rate.tick(now)
                last_frame = frame
            elif frames is not None:
                time.sleep(0.01)  # a live source with no frame yet returns at once

            left = last_frame
            if frames is not None and left is None:
                left = _placeholder(args.view_height, "waiting for the camera ...")
            view = render_view(left, log, source, thresholds, now=now, t_start=t_start,
                               markers=markers.items, absolute=absolute,
                               frame_rate=frame_rate.rate(now), take_name=take.name, args=args)
            if overlay is not None:
                overlay.write(view, now)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1 if frames is not None else 30)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key < 0:
                continue
            key &= 0xFF
            if key in (ord("q"), 27):
                break
            char = chr(key).lower()
            if char in take_io.MARKER_KEYS:
                label = take_io.MARKER_KEYS[char]
                markers.add(now, label)
                print(f"[{clock(now - t_start)}] {label}")
                if label in take_io.MONITOR_RESETS:
                    log.reset_monitor(now, take_io.MONITOR_RESETS[label])
            elif char == "a":
                absolute = not absolute
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        log.close()
        markers.close()
        for recorder in (raw_video, overlay):
            if recorder is not None:
                recorder.close()
        if frames is not None:
            frames.release()
        cv2.destroyAllWindows()
        meta.update({"ended": datetime.now().isoformat(timespec="seconds"),
                     "samples": log.count, "markers": len(markers.items),
                     "source_error": source.error})
        (take / take_io.META).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {log.count} force samples and {len(markers.items)} markers to {take}")
    print(f"Review: uv run python 4_execution/eval/force_log_review.py \"{take}\"")


if __name__ == "__main__":
    main()
