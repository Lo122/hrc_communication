"""Top-down (floor-plan) view of where the person is in the world frame.

Shared by eval/pose_detection_live.py's preview and RecognitionManager's debug
view (render_utils/debug_view.py), so both show location the same way.
"""

import cv2
import numpy as np


class WorldTrajectoryRenderer:
    """Top-down (world X-Y, i.e. floor-plan) view of the root's world-frame
    position -- where the person actually is relative to the calibration
    origin (the ChArUco board/ArUco marker's location, world +Z up), not
    their distance from the camera lens (the other panels show that). Also
    draws a fading trail of its last N frames of position.
    """

    def __init__(self, panel_size, view_range_m=3.0, camera_xy_world=None):
        self.panel_size = panel_size
        self.view_range_m = view_range_m
        self.camera_xy_world = camera_xy_world

    def _to_px(self, xy):
        w, h = self.panel_size
        scale = min(w, h) / (2.0 * self.view_range_m)
        # World +X -> screen right, world +Y -> screen up (image rows grow
        # downward, so world +Y needs the sign flip).
        px = w / 2 + xy[0] * scale
        py = h / 2 - xy[1] * scale
        return int(round(px)), int(round(py))

    def render(self, trajectory_xy, current_xy=None):
        w, h = self.panel_size
        img = np.full((h, w, 3), 255, dtype=np.uint8)

        step = max(1, int(round(self.view_range_m / 3)))
        for m in range(-int(self.view_range_m), int(self.view_range_m) + 1, step):
            gx, _ = self._to_px((m, 0))
            _, gy = self._to_px((0, m))
            cv2.line(img, (gx, 0), (gx, h), (230, 230, 230), 1, cv2.LINE_AA)
            cv2.line(img, (0, gy), (w, gy), (230, 230, 230), 1, cv2.LINE_AA)

        ox, oy = self._to_px((0, 0))
        cv2.line(img, (0, oy), (w, oy), (195, 195, 195), 1, cv2.LINE_AA)
        cv2.line(img, (ox, 0), (ox, h), (195, 195, 195), 1, cv2.LINE_AA)
        cv2.circle(img, (ox, oy), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(img, "origin", (ox + 8, oy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)

        if self.camera_xy_world is not None:
            camx, camy = self._to_px(self.camera_xy_world)
            cv2.drawMarker(img, (camx, camy), (150, 0, 0), cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
            cv2.putText(img, "camera", (camx + 8, camy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (150, 0, 0), 1, cv2.LINE_AA)

        n = len(trajectory_xy)
        for i in range(1, n):
            frac = i / max(n - 1, 1)
            color = (int(220 - 100 * frac), int(200 - 80 * frac), int(220 + 30 * frac))
            cv2.line(img, self._to_px(trajectory_xy[i - 1]), self._to_px(trajectory_xy[i]),
                     color, 2, cv2.LINE_AA)

        if current_xy is not None:
            cx_px, cy_px = self._to_px(current_xy)
            cv2.circle(img, (cx_px, cy_px), 7, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (cx_px, cy_px), 7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"({current_xy[0]:.2f},{current_xy[1]:.2f})m", (cx_px + 10, cy_px + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(img, f"world XY (top-down), +-{self.view_range_m:.1f}m, trail={n}f",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        return img
