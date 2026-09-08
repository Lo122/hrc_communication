"""Live preview window: a rolling line graph of each task step's softmax
probability plus the progress-head output, both over the last few seconds --
so you can watch the LSTM's raw per-frame output change over time instead of
only the sparse, debounced RecognitionResult stream (see
recognition_manager.py's last_step_probabilities/last_progress).

Pure OpenCV/numpy (no matplotlib) so it stays cheap enough to redraw every
frame at recognition's loop rate -- same style as
skeleton_pipeline/render/skeleton_video.py's FastSkeleton3DRenderer.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


def _step_color(step_id: int, num_steps: int) -> tuple[int, int, int]:
    """Deterministic, evenly-spaced BGR color per step id, via the HSV
    colorwheel -- so line identity stays stable across frames/runs."""
    hue = int(180 * step_id / max(num_steps, 1))
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class StepProbabilityPlot:
    """Rolling line-graph preview of per-step probabilities (top) and
    progress (bottom) over the last `history_seconds`. Call update() once
    per frame that has a probabilities/progress reading; call release()
    when done (mirrors RecognitionManager.release())."""

    def __init__(
        self,
        num_steps: int,
        *,
        history_seconds: float = 12.0,
        size: tuple[int, int] = (760, 420),
        window_name: str = "Step Probabilities & Progress",
        step_labels: list[str] | None = None,
        window_position: tuple[int, int] = (980, 60),
    ):
        self.num_steps = num_steps
        self.history_seconds = history_seconds
        self.size = size
        self.window_name = window_name
        self.step_labels = step_labels or [f"step {i}" for i in range(num_steps)]
        self.colors = [_step_color(i, num_steps) for i in range(num_steps)]

        # cv2 places new windows at a default top-left position, which lands this one
        # directly behind/under RecognitionManager's own preview window (also opened at the
        # default position) -- create it explicitly and move it aside so it's actually
        # visible as a separate window instead of looking like it never opened.
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, *window_position)

        # (timestamp, probabilities array, progress) -- trimmed to history_seconds on
        # every update rather than a fixed maxlen, so it stays correct regardless of the
        # actual frame rate.
        self._history: deque[tuple[float, np.ndarray, float]] = deque()

        # Layout: legend strip on the right, two stacked plots (probabilities/progress)
        # sharing the remaining width.
        self._legend_w = 150
        self._margin = 40
        total_w, total_h = size
        self._plot_w = total_w - self._legend_w - self._margin
        gap = 30
        self._prob_h = int((total_h - self._margin - gap) * 0.65)
        self._progress_h = total_h - self._margin - gap - self._prob_h
        self._prob_top = 20
        self._progress_top = self._prob_top + self._prob_h + gap

    def update(self, timestamp: float, probabilities: np.ndarray, progress: float) -> np.ndarray:
        """Append one reading, redraw, and show the window. Returns the
        rendered canvas (BGR uint8) in case a caller wants to reuse it
        (e.g. compose into another panel) instead of/in addition to
        cv2.imshow. Raises KeyboardInterrupt on 'q', same convention as
        RecognitionManager._show_frame."""
        self._history.append((timestamp, np.asarray(probabilities, dtype=np.float32), float(progress)))
        cutoff = timestamp - self.history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        canvas = self._render()
        cv2.imshow(self.window_name, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
        return canvas

    def release(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def _render(self) -> np.ndarray:
        total_w, total_h = self.size
        canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

        plot_left = self._margin
        self._draw_axes(canvas, plot_left, self._prob_top, self._plot_w, self._prob_h,
                         title="task step probabilities")
        self._draw_axes(canvas, plot_left, self._progress_top, self._plot_w, self._progress_h,
                         title="progress")

        if len(self._history) >= 2:
            latest_t = self._history[-1][0]
            xs = [latest_t - t for t, _, _ in self._history]  # seconds-ago, newest -> 0

            for step_id in range(self.num_steps):
                ys = [probs[step_id] if step_id < len(probs) else 0.0 for _, probs, _ in self._history]
                self._draw_line(canvas, plot_left, self._prob_top, self._plot_w, self._prob_h,
                                 xs, ys, self.colors[step_id])

            progress_ys = [p for _, _, p in self._history]
            self._draw_line(canvas, plot_left, self._progress_top, self._plot_w, self._progress_h,
                             xs, progress_ys, (60, 60, 60))

        self._draw_legend(canvas, plot_left + self._plot_w + 15, self._prob_top)
        return canvas

    def _draw_axes(self, canvas, x0, y0, w, h, *, title: str) -> None:
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (200, 200, 200), 1, cv2.LINE_AA)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = int(y0 + h - frac * h)
            cv2.line(canvas, (x0, y), (x0 + w, y), (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{frac:.2f}", (max(x0 - 34, 0), y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, title, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"-{self.history_seconds:.0f}s", (x0, y0 + h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, "now", (x0 + w - 24, y0 + h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    def _draw_line(self, canvas, x0, y0, w, h, xs_seconds_ago, ys, color) -> None:
        # A frame's probabilities/progress can be NaN (e.g. no valid detection yet) --
        # skip those points (leaving a gap in the line) rather than letting a NaN reach
        # int() and crash.
        points = []
        for seconds_ago, y in zip(xs_seconds_ago, ys):
            if not np.isfinite(y):
                points.append(None)
                continue
            frac_x = 1.0 - min(seconds_ago / self.history_seconds, 1.0)
            px = int(x0 + frac_x * w)
            py = int(y0 + h - min(max(float(y), 0.0), 1.0) * h)
            points.append((px, py))
        for a, b in zip(points, points[1:]):
            if a is not None and b is not None:
                cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)

    def _draw_legend(self, canvas, x0, y0) -> None:
        for i, (label, color) in enumerate(zip(self.step_labels, self.colors)):
            y = y0 + i * 20
            cv2.line(canvas, (x0, y), (x0 + 20, y), color, 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (x0 + 26, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (30, 30, 30), 1, cv2.LINE_AA)
        y_progress = y0 + self.num_steps * 20 + 12
        cv2.line(canvas, (x0, y_progress), (x0 + 20, y_progress), (60, 60, 60), 3, cv2.LINE_AA)
        cv2.putText(canvas, "progress", (x0 + 26, y_progress + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (30, 30, 30), 1, cv2.LINE_AA)
