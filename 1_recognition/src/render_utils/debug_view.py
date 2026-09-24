"""Live debug windows for RecognitionManager: the skeleton preview and the
scrolling time-series plot.

Two OpenCV windows, both optional (with `show_video=False` and no `record_path`
every call here is a no-op):

  - **preview**: the 2D keypoint overlay, optionally side by side with a
    four-view orthographic 3D posture panel, with the model's current output
    burned into the top-left corner as text. Having the numbers ON the frame
    is the point -- otherwise debugging means correlating a separate console
    stream against the video by eye.
  - **plot**: progress/confidence and world x/y/z scrolling against sample
    index. A single-frame text overlay cannot show a trend, and the trend is
    usually what is wrong.

The preview can also be written to an .mp4 (`record_path`), which is the
composited frame the preview window shows -- overlay, 3D panel and burned-in
readout -- not the raw camera feed. Recording is independent of display, so a
headless run can capture the same view it would have shown.

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

logger = get_logger(__name__)


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

        self._cv2 = None
        self._draw_2d_skeleton = None
        self._renderer_3d = None
        self._writer = None
        self._record_size: tuple[int, int] | None = None
        self._record_size_warned = False

        self._progress_history: deque[float] = deque(maxlen=history_len)
        self._confidence_history: deque[float] = deque(maxlen=history_len)
        self._world_xyz_history: deque[tuple[float, float, float]] = deque(maxlen=history_len)

    # -- wiring ------------------------------------------------------------

    def attach_renderers(self, draw_2d_skeleton, renderer_3d=None) -> None:
        """Hand over the drawing callables once the realtime pipeline has
        imported them. Until this is called the preview still works, just
        without the skeleton overlay or the 3D panel."""
        self._draw_2d_skeleton = draw_2d_skeleton
        self._renderer_3d = renderer_3d

    def _ensure_cv2(self):
        if self._cv2 is None:
            import cv2
            self._cv2 = cv2
        return self._cv2

    # -- history -----------------------------------------------------------

    def record_world(self, world_xyz: tuple[float, float, float]) -> None:
        self._world_xyz_history.append(world_xyz)

    def record_prediction(self, progress: float, confidence: float) -> None:
        self._progress_history.append(progress)
        self._confidence_history.append(confidence)

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
            # Side-by-side: 2D overlay | 3D posture (oblique/front/side/top),
            # same layout as eval/pose_detection_live.py's preview window.
            panel_3d = self._renderer_3d.render(skeleton)
            display = cv2.hconcat([
                cv2.resize(overlay, (panel_w, panel_h)),
                cv2.resize(panel_3d, (panel_w, panel_h)),
            ])

        display = self._draw_overlay(
            display, raw_step_id=raw_step_id, stable_step_id=stable_step_id,
            progress=progress, confidence=confidence, world_xyz=world_xyz,
            status_line=status_line, mistake_id=mistake_id, mistake_score=mistake_score)

        self._write_frame(display)

        # Recording-only runs stop here: no window to update, and no waitKey to poll,
        # which is what makes --no-display --record usable without a display attached.
        if not self.show_windows:
            return

        cv2.imshow(self.window_name, display)
        self._draw_plot()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    # -- recording ---------------------------------------------------------

    def _write_frame(self, display) -> None:
        """Append one composited frame to the recording, opening the writer on the
        first call.

        The writer's frame size comes from that first frame rather than from
        panel_size, because what is recorded is the finished composite -- 2D overlay
        beside the 3D panel, model readout burned in -- which is wider than one panel
        and is exactly what the preview window shows.

        A writer that will not open (missing codec, unwritable path) disables
        recording and logs once, rather than raising: losing the recording should not
        take the recognition run down with it."""
        if self.record_path is None:
            return
        cv2 = self._ensure_cv2()
        height, width = display.shape[:2]

        if self._writer is None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(self.record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                self.record_fps, (width, height))
            if not writer.isOpened():
                logger.error("Could not open %s for recording (codec or path problem); "
                             "continuing without a recording.", self.record_path)
                self.record_path = None
                return
            self._writer = writer
            self._record_size = (width, height)
            logger.info("Recording the debug view to %s (%dx%d @ %.1f fps)",
                        self.record_path, width, height, self.record_fps)

        if (width, height) != self._record_size:
            # A mid-run size change (source resolution switch) would be written as
            # garbage by VideoWriter, so drop the frame and say so once.
            if not self._record_size_warned:
                logger.warning("Display size changed from %s to %s; those frames are "
                               "left out of the recording.", self._record_size, (width, height))
                self._record_size_warned = True
            return
        self._writer.write(display)

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
            stable_text = str(stable_step_id) if stable_step_id is not None else "-"
            step_lines = [
                f"Raw step: {raw_step_id}  Stable step: {stable_text}",
                f"Progress: {progress:.2f}  Confidence: {confidence:.2f}",
            ]
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
        window) of step progress/confidence and world x/y/z -- the trend
        over time that a single-frame text overlay can't show."""
        panel_w, panel_h = self.plot_panel_size
        canvas = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
        top_h = panel_h // 2

        self._draw_series(
            canvas, row_range=(0, top_h), y_range=(0.0, 1.0), title="Progress / Confidence",
            series=[
                (list(self._progress_history), (0, 150, 0), "progress"),
                (list(self._confidence_history), (200, 0, 0), "confidence"),
            ],
        )
        world = list(self._world_xyz_history)
        self._draw_series(
            canvas, row_range=(top_h, panel_h), y_range=None, title="World position (m)",
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
        width = canvas.shape[1]
        maxlen = self._progress_history.maxlen or 1

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
            x = int(round((width - 1) * (maxlen - n + index) / max(maxlen - 1, 1)))
            y = row0 + int(round((1.0 - (value - y_lo) / y_span) * (height - 1)))
            return x, y

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
        cv2.putText(canvas, legend, (6, row0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (40, 40, 40), 1, cv2.LINE_AA)
        if row0 > 0:
            cv2.line(canvas, (0, row0), (width, row0), (210, 210, 210), 1)

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Finalise the recording and destroy both windows if they were opened."""
        if self._writer is not None:
            # Without this the container is left without its index and the file is
            # unplayable, so it runs before the early return below.
            self._writer.release()
            self._writer = None
            logger.info("Recording written to %s", self.record_path)
        if not self.show_windows or self._cv2 is None:
            return
        for window_name in (self.window_name, self.plot_window_name):
            try:
                self._cv2.destroyWindow(window_name)
            except self._cv2.error:
                pass
