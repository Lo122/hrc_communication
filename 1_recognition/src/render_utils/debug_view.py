"""Live debug windows for RecognitionManager: the skeleton preview and the
scrolling time-series plot.

Two OpenCV windows, both optional (with `show_video=False` and no `record_path`
every call here is a no-op):

  - **preview**: the 2D keypoint overlay, optionally side by side with a
    four-view orthographic 3D posture panel and a top-down world-location
    panel (same layout as eval/pose_detection_live.py), with the model's current output
    burned into the top-left corner as text. Having the numbers ON the frame
    is the point -- otherwise debugging means correlating a separate console
    stream against the video by eye.
  - **plot**: the step over time (raw and stable, labelled by step name),
    progress/confidence, the mistake head's label and world x/y/z scrolling
    against sample index. A single-frame text overlay cannot show a trend, and the trend is
    usually what is wrong.

The preview can also be written to an .mp4 (`record_path`), which is the
composited frame the preview window shows -- overlay, 3D panel and burned-in
readout -- not the raw camera feed (RecognitionManager's raw_record_path records
that). Recording is independent of display, so a headless run can capture the
same view it would have shown.

Extracted from RecognitionManager because none of this is recognition: it
reads the manager's output and owns nothing the manager needs back. Keeping
it here also means the manager no longer carries cv2, the renderer handles
and three history deques purely for a debug path that is off by default.

The renderers are attached late (`attach_renderers`) rather than passed to
the constructor, because they live under src/render_utils/ and
are only imported once the realtime pipeline spins up.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from logging_setup import get_logger
from render_utils.video_recorder import VideoRecorder

logger = get_logger(__name__)

# Plot layout: every row leaves this many pixels on the left for the step row's
# name labels, so all rows share one time axis.
PLOT_LABEL_GUTTER = 110
PLOT_LABEL_CHARS = 16
PLOT_HEADER_H = 18
# Relative heights of the plot rows: step, progress/confidence, mistake, world.
PLOT_ROW_WEIGHTS = (0.34, 0.24, 0.16, 0.26)


def fit_on_white(image, size: tuple[int, int]):
    """image scaled to fit size (width, height) with its aspect ratio kept, centred on
    white -- a 16:9 camera frame in a square panel gets white bands, not a squash."""
    import cv2

    panel_w, panel_h = size
    h, w = image.shape[:2]
    if (w, h) == (panel_w, panel_h):
        return image
    scale = min(panel_w / w, panel_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    canvas = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    x, y = (panel_w - new_w) // 2, (panel_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


class DebugView:
    """Owns the two debug windows and the history behind the plot.

    history_len: how many samples the scrolling plot keeps. Progress/
    confidence only accumulate once the LSTM window buffer is full, while
    world position accumulates on every frame with a valid 3D lift, so the
    series are independent lengths/timelines by design -- each is plotted
    against its own sample index.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        window_name: str,
        panel_size: tuple[int, int],
        plot_window_name: str,
        plot_panel_size: tuple[int, int],
        history_len: int,
        conf_threshold: float,
        render_world_skeleton: bool = False,
        record_path: str | Path | None = None,
        record_fps: float = 20.0,
        step_names: list[str] | tuple[str, ...] | None = None,
        mistake_names: list[str] | tuple[str, ...] | None = None,
        trajectory_len: int = 200,
    ):
        # Showing the windows and recording the composite are independent reasons to
        # do the compositing work, so neither implies the other: --no-display with a
        # recording path is a valid headless capture, and self.enabled -- which gates
        # whether show() does anything at all -- is the OR of the two.
        self.show_windows = enabled
        self.record_path = Path(record_path) if record_path is not None else None
        self.record_fps = float(record_fps)
        self.enabled = bool(enabled or self.record_path is not None)
        self.window_name = window_name
        self.panel_size = panel_size
        self.plot_window_name = plot_window_name
        self.plot_panel_size = plot_panel_size
        self.conf_threshold = conf_threshold
        # Which skeleton the 3D panel draws. False (default) draws
        # pipeline_out["root_relative"] -- the pelvis-centred, camera-frame posture that
        # is ACTUALLY fed to the feature extractor and hence the model, so what you see
        # is what the model sees. True draws pipeline_out["skeleton"], the world-frame
        # fusion of that posture with the depth estimate, which lives in the calibration
        # target's frame and is therefore rotated away from the video's own axes.
        # Absolute position is already reported separately (the World XYZ overlay and
        # the top-down trajectory), so the world skeleton adds nothing to a POSTURE view
        # except the extrinsics' rotation.
        self.render_world_skeleton = render_world_skeleton
        # Step id -> label for the readout (index == the step head's output column).
        # Passed in rather than imported so this module stays independent of the root
        # config; an id with no name falls back to "#<id>".
        self.step_names = tuple(step_names) if step_names is not None else ()
        # Same for the mistake head (index 0 == no mistake).
        self.mistake_names = tuple(mistake_names) if mistake_names is not None else ()

        self._cv2 = None
        self._draw_2d_skeleton = None
        self._renderer_3d = None
        self._world_renderer = None
        self._recorder = (VideoRecorder(self.record_path, self.record_fps, label="debug view")
                          if self.record_path is not None else None)

        self._progress_history: deque[float] = deque(maxlen=history_len)
        self._confidence_history: deque[float] = deque(maxlen=history_len)
        self._world_xyz_history: deque[tuple[float, float, float]] = deque(maxlen=history_len)
        # Recorded alongside progress/confidence, so the step plot shares their x axis.
        # NaN stands for "no step yet" (stable is None until the stabilizer commits).
        self._raw_step_history: deque[float] = deque(maxlen=history_len)
        self._stable_step_history: deque[float] = deque(maxlen=history_len)
        self._mistake_history: deque[float] = deque(maxlen=history_len)
        # World XY trail for the top-down location panel.
        self._trajectory: deque[tuple[float, float]] = deque(maxlen=trajectory_len)

    # -- wiring ------------------------------------------------------------

    def attach_renderers(self, draw_2d_skeleton, renderer_3d=None, world_renderer=None) -> None:
        """Hand over the drawing callables once the realtime pipeline has
        imported them. Until this is called the preview still works, just
        without the skeleton overlay or the 3D panel.

        world_renderer is a WorldTrajectoryRenderer, or None when there are no
        extrinsics -- the location panel then says so instead of drawing an empty
        map, since without extrinsics there is no world origin to place anyone in."""
        self._draw_2d_skeleton = draw_2d_skeleton
        self._renderer_3d = renderer_3d
        self._world_renderer = world_renderer

    def _step_label(self, step_id: int | None) -> str:
        if step_id is None:
            return "-"
        if 0 <= step_id < len(self.step_names):
            return self.step_names[step_id]
        return f"#{step_id}"

    def _ensure_cv2(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2

    # -- history -----------------------------------------------------------

    def record_world(self, world_xyz: tuple[float, float, float]) -> None:
        self._world_xyz_history.append(world_xyz)
        self._trajectory.append((world_xyz[0], world_xyz[1]))

    def record_prediction(self, progress: float, confidence: float,
                          raw_step_id: int | None = None,
                          stable_step_id: int | None = None,
                          mistake_id: int | None = None) -> None:
        self._progress_history.append(progress)
        self._confidence_history.append(confidence)
        self._raw_step_history.append(float(raw_step_id) if raw_step_id is not None else np.nan)
        self._stable_step_history.append(
            float(stable_step_id) if stable_step_id is not None else np.nan)
        self._mistake_history.append(float(mistake_id) if mistake_id is not None else np.nan)

    # -- drawing -----------------------------------------------------------

    def show(
        self,
        frame,
        pipeline_out: dict | None = None,
        *,
        raw_step_id: int | None = None,
        stable_step_id: int | None = None,
        progress: float | None = None,
        confidence: float | None = None,
        world_xyz: tuple[float, float, float] | None = None,
        status_line: str = "",
        mistake_id: int | None = None,
        mistake_score: float | None = None,
        idle_probability: float | None = None,
    ) -> None:
        """Draw and display one frame. Raises KeyboardInterrupt when the
        user presses q, which is how the run loop is asked to stop."""
        if not self.enabled:
            return
        cv2 = self._ensure_cv2()

        overlay = frame
        skeleton = None
        if pipeline_out is not None:
            if pipeline_out.get("keypoints_2d") is not None and self._draw_2d_skeleton is not None:
                overlay = self._draw_2d_skeleton(
                    frame, pipeline_out["keypoints_2d"], pipeline_out["keypoints_conf"],
                    conf_threshold=self.conf_threshold)
            # root_relative is MotionBERT's own output: pelvis at (0,0,0), +z up,
            # untouched by extrinsics or the depth estimate. FastSkeleton3DRenderer
            # assumes +z up, so this renders upright and matches the video's framing.
            skeleton = pipeline_out.get(
                "skeleton" if self.render_world_skeleton else "root_relative")
            if skeleton is None and not self.render_world_skeleton:
                # No lift this frame -- fall back rather than blanking the panel.
                skeleton = pipeline_out.get("skeleton")

        panel_w, panel_h = self.panel_size
        display = overlay
        if self._renderer_3d is not None:
            # Side-by-side: 2D overlay | 3D posture (oblique/front/side/top) | world
            # location (top-down), same layout as eval/pose_detection_live.py's preview.
            panel_3d = self._renderer_3d.render(skeleton)
            display = cv2.hconcat([
                fit_on_white(overlay, (panel_w, panel_h)),
                fit_on_white(panel_3d, (panel_w, panel_h)),
                fit_on_white(self._render_world_panel(world_xyz), (panel_w, panel_h)),
            ])

        display = self._draw_overlay(
            display, raw_step_id=raw_step_id, stable_step_id=stable_step_id,
            progress=progress, confidence=confidence, world_xyz=world_xyz,
            status_line=status_line, mistake_id=mistake_id, mistake_score=mistake_score,
            idle_probability=idle_probability)

        self._write_frame(display)

        # Recording-only runs stop here: no window to update, and no waitKey to poll,
        # which is what makes --no-display --record usable without a display attached.
        if not self.show_windows:
            return

        cv2.imshow(self.window_name, display)
        self._draw_plot()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def _render_world_panel(self, world_xyz):
        cv2 = self._cv2
        if self._world_renderer is None:
            panel_w, panel_h = self.panel_size
            panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
            for i, text in enumerate(("no calibrated extrinsics", "(no world location)")):
                cv2.putText(panel, text, (10, panel_h // 2 + 22 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
            return panel
        current_xy = (world_xyz[0], world_xyz[1]) if world_xyz is not None else None
        return self._world_renderer.render(list(self._trajectory), current_xy=current_xy)

    # -- recording ---------------------------------------------------------

    def _write_frame(self, display) -> None:
        """Append one composited frame to the recording -- the finished composite
        (2D overlay beside the 3D and location panels, model readout burned in),
        exactly what the preview window shows."""
        if self._recorder is not None:
            self._recorder.write(display)

    def _draw_overlay(
        self,
        display,
        *,
        raw_step_id: int | None,
        stable_step_id: int | None,
        progress: float | None,
        confidence: float | None,
        world_xyz: tuple[float, float, float] | None,
        status_line: str,
        mistake_id: int | None = None,
        mistake_score: float | None = None,
        idle_probability: float | None = None,
    ):
        """Burn live model output + absolute human position as text onto the
        top-left corner of the display frame, for debugging without needing
        to correlate a separate console/log stream against the video.

        status_line is what to show INSTEAD of the step readout when there is
        no prediction yet -- the caller owns that wording because it depends
        on why (step model off vs. window buffer still filling)."""
        cv2 = self._cv2
        display = display.copy()

        if world_xyz is not None:
            x, y, z = world_xyz
            world_line = f"World XYZ: ({x:+.2f}, {y:+.2f}, {z:+.2f}) m"
        else:
            world_line = "World XYZ: --"

        if raw_step_id is not None:
            step_lines = [
                f"Raw step: {self._step_label(raw_step_id)}",
                f"Stable step: {self._step_label(stable_step_id)}",
                f"Progress: {progress:.2f}  Confidence: {confidence:.2f}",
            ]
            # Multi-head models: every task score is P(task) x (1 - P(idle)), so a high
            # idle is why no task gets near 1 -- show it rather than leave that a puzzle.
            if idle_probability is not None:
                step_lines.append(f"Idle (bg head): {idle_probability:.2f}")
        else:
            step_lines = [status_line]

        # Only present for a model trained with a mistake head; a model without one
        # leaves these None and the line is simply absent rather than reading 0.00,
        # which would look like a confident "no mistake" the model never made.
        mistake_line = None
        if mistake_score is not None:
            verdict = "MISTAKE" if mistake_id else "ok"
            mistake_line = f"Mistake: {verdict} ({mistake_score:.2f})"

        lines = [world_line, *step_lines]
        if mistake_line is not None:
            lines.append(mistake_line)
        for i, text in enumerate(lines):
            origin = (10, 24 + i * 22)
            # Red for a flagged mistake so it reads at a glance; green otherwise.
            colour = ((0, 0, 255) if text is mistake_line and mistake_id
                      else (0, 255, 0))
            # Black outline then colored fill so the text stays legible over
            # any background (skeleton overlay, bright frame, etc.).
            cv2.putText(display, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, colour, 1, cv2.LINE_AA)
        return display

    def _draw_plot(self) -> None:
        """Scrolling time-series window (separate from the skeleton/overlay
        window) of the step, progress/confidence, mistake and world x/y/z -- the
        trend over time that a single-frame text overlay can't show.

        Every row leaves the same left gutter (the lane rows' name labels), so the
        step, progress and mistake rows -- all one sample per LSTM prediction --
        line up in time sample for sample."""
        panel_w, panel_h = self.plot_panel_size
        canvas = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
        # Row boundaries from PLOT_ROW_WEIGHTS: step, progress/confidence, mistake, world.
        edges = np.round(np.cumsum([0.0, *PLOT_ROW_WEIGHTS]) / sum(PLOT_ROW_WEIGHTS)
                         * panel_h).astype(int)
        step_rows, progress_rows, mistake_rows, world_rows = (
            (int(edges[i]), int(edges[i + 1])) for i in range(4))

        self._draw_step_series(canvas, row_range=step_rows)
        self._draw_series(
            canvas, row_range=progress_rows, y_range=(0.0, 1.0), title="Progress / Confidence",
            series=[
                (list(self._progress_history), (0, 150, 0), "progress"),
                (list(self._confidence_history), (200, 0, 0), "confidence"),
            ],
        )
        self._draw_mistake_series(canvas, row_range=mistake_rows)
        world = list(self._world_xyz_history)
        self._draw_series(
            canvas, row_range=world_rows, y_range=None, title="World position (m)",
            series=[
                ([p[0] for p in world], (255, 0, 0), "x"),
                ([p[1] for p in world], (0, 150, 150), "y"),
                ([p[2] for p in world], (0, 0, 255), "z"),
            ],
        )
        self._cv2.imshow(self.plot_window_name, canvas)

    def _draw_series(
        self,
        canvas: np.ndarray,
        *,
        row_range: tuple[int, int],
        y_range: tuple[float, float] | None,
        title: str,
        series: list[tuple[list[float], tuple[int, int, int], str]],
    ) -> None:
        """Draw one or more scrolling line series into canvas[row0:row1, :].
        y_range=None auto-scales to the min/max across all series (with a
        small margin), falling back to (-1, 1) if nothing has data yet.
        Newest sample is at the right edge, oldest scrolls off the left, in
        deque-maxlen-relative x -- so the plot doesn't jump width as history
        fills up."""
        cv2 = self._cv2
        row0, row1 = row_range
        height = row1 - row0

        if y_range is None:
            # Finite values only: one NaN sample would otherwise make lo/hi NaN and
            # poison every point in the panel, including the healthy series drawn
            # beside it.
            values = [v for values, _color, _label in series for v in values
                      if np.isfinite(v)]
            if values:
                lo, hi = min(values), max(values)
                margin = max((hi - lo) * 0.1, 0.05)
                y_range = (lo - margin, hi + margin)
            else:
                y_range = (-1.0, 1.0)
        y_lo, y_hi = y_range
        y_span = (y_hi - y_lo) or 1.0

        def to_point(index: int, value: float, n: int) -> tuple[int, int]:
            y = row0 + int(round((1.0 - (value - y_lo) / y_span) * (height - 1)))
            return self._plot_x(canvas, index, n), y

        for values, color, _label in series:
            n = len(values)
            if n < 2:
                continue
            # A non-finite sample BREAKS the line rather than being dropped from it.
            # int(round(nan)) raises ValueError ("cannot convert float NaN to
            # integer"), which is how a frame with nothing detected used to take the
            # whole run down from inside the debug window; and joining across the gap
            # would draw a straight line through frames the model never produced a
            # number for, which reads as data that does not exist.
            segment: list[tuple[int, int]] = []
            for i, value in enumerate(values):
                if not np.isfinite(value):
                    if len(segment) >= 2:
                        cv2.polylines(canvas, [np.array(segment, dtype=np.int32)],
                                      False, color, 1, cv2.LINE_AA)
                    segment = []
                    continue
                segment.append(to_point(i, value, n))
            if len(segment) >= 2:
                cv2.polylines(canvas, [np.array(segment, dtype=np.int32)], False,
                              color, 1, cv2.LINE_AA)

        legend = f"{title}  [" + ", ".join(label for _v, _c, label in series) + f"]  y:[{y_lo:.2f},{y_hi:.2f}]"
        self._draw_row_frame(canvas, row_range, legend)

    def _plot_x(self, canvas: np.ndarray, index: int, n: int) -> int:
        """x pixel of sample `index` of an n-long series: newest at the right edge,
        in deque-maxlen-relative position right of the label gutter."""
        maxlen = self._progress_history.maxlen or 1
        span = canvas.shape[1] - 1 - PLOT_LABEL_GUTTER
        return PLOT_LABEL_GUTTER + int(round(span * (maxlen - n + index) / max(maxlen - 1, 1)))

    def _draw_row_frame(self, canvas: np.ndarray, row_range: tuple[int, int], legend: str) -> None:
        cv2 = self._cv2
        row0, row1 = row_range
        cv2.putText(canvas, legend, (6, row0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.line(canvas, (PLOT_LABEL_GUTTER, row0 + PLOT_HEADER_H), (PLOT_LABEL_GUTTER, row1),
                 (220, 220, 220), 1)
        if row0 > 0:
            cv2.line(canvas, (0, row0), (canvas.shape[1], row0), (210, 210, 210), 1)

    def _draw_step_series(self, canvas: np.ndarray, *, row_range: tuple[int, int]) -> None:
        """Step over time: the stable step as a thick line and the raw argmax as
        dots -- so a flickering raw prediction the stabilizer is holding back shows
        as dots leaving the line."""
        self._draw_lanes(
            canvas, row_range=row_range, title="Step  [stable = line, raw = dots]",
            lane_names=self.step_names, line_values=list(self._stable_step_history),
            dot_values=list(self._raw_step_history))

    def _draw_mistake_series(self, canvas: np.ndarray, *, row_range: tuple[int, int]) -> None:
        """Mistake head's argmax over time, red while it says anything but class 0
        ("no mistake"). A model without a mistake head only ever records NaN, and
        the row says so rather than looking like a run with no mistakes."""
        values = list(self._mistake_history)
        title = "Mistake  [argmax of mistake head]"
        if values and not any(np.isfinite(v) for v in values):
            title = "Mistake  [model has no mistake head]"
        lane_colors = [(200, 90, 0)] + [(0, 0, 220)] * max(len(self.mistake_names) - 1, 1)
        self._draw_lanes(
            canvas, row_range=row_range, title=title, lane_names=self.mistake_names,
            line_values=values, lane_colors=lane_colors)

    def _draw_lanes(
        self,
        canvas: np.ndarray,
        *,
        row_range: tuple[int, int],
        title: str,
        lane_names: tuple[str, ...],
        line_values: list[float],
        dot_values: list[float] | None = None,
        lane_colors: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Label over time: one horizontal lane per class id, named in the left
        gutter; line_values drawn as a thick step line, dot_values as dots.
        lane_colors colours each flat stretch of the line by the lane it is in
        (default: one colour for all). NaN breaks the line (no value yet)."""
        cv2 = self._cv2
        row0, row1 = row_range
        dots = dot_values or []
        seen = [v for v in line_values + dots if np.isfinite(v)]
        num_lanes = max(len(lane_names), int(max(seen)) + 1 if seen else 1)

        top = row0 + PLOT_HEADER_H
        lane_h = (row1 - top) / num_lanes

        def lane_y(lane: float) -> int:
            return int(round(top + (lane + 0.5) * lane_h))

        def lane_color(lane: float) -> tuple[int, int, int]:
            if lane_colors and 0 <= int(lane) < len(lane_colors):
                return lane_colors[int(lane)]
            return (200, 90, 0)

        for lane in range(num_lanes):
            y = lane_y(lane)
            cv2.line(canvas, (PLOT_LABEL_GUTTER, y), (canvas.shape[1], y), (238, 238, 238), 1)
            label = lane_names[lane] if lane < len(lane_names) else f"#{lane}"
            cv2.putText(canvas, label[:PLOT_LABEL_CHARS], (6, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (80, 80, 80), 1, cv2.LINE_AA)

        n = len(dots)
        for i, value in enumerate(dots):
            if np.isfinite(value):
                cv2.circle(canvas, (self._plot_x(canvas, i, n), lane_y(value)), 2,
                           (110, 110, 110), -1)

        # Step-function line: hold the previous value until the sample where it
        # changes, then jump -- a sloped line would suggest classes in between.
        n = len(line_values)
        previous: tuple[int, int, float] | None = None  # (x, y, value)
        for i, value in enumerate(line_values):
            if not np.isfinite(value):
                previous = None
                continue
            x, y = self._plot_x(canvas, i, n), lane_y(value)
            if previous is not None:
                px, py, pvalue = previous
                cv2.line(canvas, (px, py), (x, py), lane_color(pvalue), 2, cv2.LINE_AA)
                if py != y:
                    cv2.line(canvas, (x, py), (x, y), (160, 160, 160), 1, cv2.LINE_AA)
            previous = (x, y, value)

        self._draw_row_frame(canvas, row_range, title)

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Finalise the recording and destroy both windows if they were opened."""
        # Before the early return below: an unfinalised .mp4 is unplayable.
        if self._recorder is not None:
            self._recorder.close()
        if not self.show_windows or self._cv2 is None:
            return
        for window_name in (self.window_name, self.plot_window_name):
            try:
                self._cv2.destroyWindow(window_name)
            except self._cv2.error:
                pass
