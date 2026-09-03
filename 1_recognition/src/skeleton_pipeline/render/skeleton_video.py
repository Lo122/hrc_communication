"""Side-by-side (2D overlay | 3D skeleton) frame renderer -- app/venv-local
port of world_pose/demo/offline_lift_test.py's draw_2d_skeleton +
FastSkeleton3DRenderer (plain OpenCV, no matplotlib -- matplotlib's canvas
draw costs ~100ms/frame regardless of plot complexity, dwarfing this
project's actual model cost; see that module's docstring for the
measurement this reasoning is based on).

Includes the front/side/top mirroring fix already established in
world_pose/demo/offline_lift_test.py (explicit axis-aligned (right, up)
vectors for the three orthographic views instead of a generic elevation/
azimuth formula, which was fragile at exactly those axis-aligned angles --
see that module's history for the full reasoning) -- ported here rather
than imported, since this venv doesn't depend on world_pose/mmpose.
"""
import cv2
import numpy as np

COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

H36M_SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

H36M_LEFT_JOINTS = {4, 5, 6, 11, 12, 13}
H36M_RIGHT_JOINTS = {1, 2, 3, 14, 15, 16}


def _edge_side(i, j):
    if i in H36M_LEFT_JOINTS and j in H36M_LEFT_JOINTS:
        return "left"
    if i in H36M_RIGHT_JOINTS and j in H36M_RIGHT_JOINTS:
        return "right"
    return "center"


def draw_2d_skeleton(frame, keypoints_2d, keypoints_conf, conf_threshold=0.3):
    out = frame.copy()
    for i, j in COCO_SKELETON_EDGES:
        if keypoints_conf[i] < conf_threshold or keypoints_conf[j] < conf_threshold:
            continue
        p1 = tuple(keypoints_2d[i].astype(int))
        p2 = tuple(keypoints_2d[j].astype(int))
        cv2.line(out, p1, p2, (0, 255, 0), 2)
    for k in range(keypoints_2d.shape[0]):
        conf = float(np.clip(keypoints_conf[k], 0.0, 1.0))
        color = (0, int(255 * conf), int(255 * (1.0 - conf)))
        radius = 2 + int(round(3 * conf))
        cv2.circle(out, tuple(keypoints_2d[k].astype(int)), radius, color, -1, cv2.LINE_AA)
    return out


class FastSkeleton3DRenderer:
    """Four-view (oblique/front/side/top) orthographic H36M skeleton
    renderer -- see module docstring on the front/side/top mirroring fix.
    """

    _EDGE_COLORS = {"left": (30, 130, 230), "right": (200, 120, 30), "center": (90, 90, 90)}

    def __init__(self, panel_size, view_range=1.0):
        self.panel_size = panel_size
        self.view_range = view_range
        self._sub_w = panel_size[0] // 2
        self._sub_h = panel_size[1] // 2
        self._bases = {
            "oblique": self._camera_basis(15.0, 60.0),
            "front": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "side": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "top": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        }
        self._view_names = ["oblique", "front", "side", "top"]

    @staticmethod
    def _camera_basis(elev_deg, azim_deg):
        elev, azim = np.radians(elev_deg), np.radians(azim_deg)
        forward = np.array([np.cos(elev) * np.cos(azim),
                             np.cos(elev) * np.sin(azim),
                             np.sin(elev)])
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(forward @ world_up)) > 0.999:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up

    def _render_view(self, skeleton_3d, right, up, w, h, root, floor_z):
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        centered = skeleton_3d - root
        scale = min(w, h) / (2.0 * self.view_range)
        px = (w / 2 + centered @ right * scale).astype(int)
        py = (h / 2 - centered @ up * scale).astype(int)

        grid_half = self.view_range
        for g in np.arange(-grid_half, grid_half + 1e-6, grid_half / 2):
            for p0, p1 in [((g, -grid_half, floor_z), (g, grid_half, floor_z)),
                           ((-grid_half, g, floor_z), (grid_half, g, floor_z))]:
                c0, c1 = np.array(p0) - root, np.array(p1) - root
                x0, y0 = int(w / 2 + c0 @ right * scale), int(h / 2 - c0 @ up * scale)
                x1, y1 = int(w / 2 + c1 @ right * scale), int(h / 2 - c1 @ up * scale)
                cv2.line(img, (x0, y0), (x1, y1), (230, 230, 230), 1, cv2.LINE_AA)

        for i, j in H36M_SKELETON_EDGES:
            cv2.line(img, (px[i], py[i]), (px[j], py[j]), self._EDGE_COLORS[_edge_side(i, j)], 2, cv2.LINE_AA)
        for k in range(skeleton_3d.shape[0]):
            cv2.circle(img, (px[k], py[k]), 4, (0, 0, 0), -1, cv2.LINE_AA)
        return img

    def render(self, skeleton_3d):
        panel_w, panel_h = self.panel_size
        if skeleton_3d is None or np.any(np.isnan(skeleton_3d)):
            canvas = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
            cv2.putText(canvas, "no detection", (10, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 200), 2, cv2.LINE_AA)
            return canvas

        root = skeleton_3d[0]
        floor_z = float(np.min(skeleton_3d[:, 2]))
        sub_w, sub_h = self._sub_w, self._sub_h
        views = {}
        for name in self._view_names:
            right, up = self._bases[name]
            view_img = self._render_view(skeleton_3d, right, up, sub_w, sub_h, root, floor_z)
            cv2.putText(view_img, name.upper(), (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (100, 100, 100), 1, cv2.LINE_AA)
            views[name] = view_img

        top_row = np.hstack([views["oblique"], views["front"]])
        bottom_row = np.hstack([views["side"], views["top"]])
        grid = np.vstack([top_row, bottom_row])
        canvas = cv2.resize(grid, (panel_w, panel_h)) if grid.shape[:2] != (panel_h, panel_w) else grid
        cv2.putText(canvas, f"+-{self.view_range:.1f} m", (8, panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        return canvas


def render_combined_frame(frame_bgr, keypoints_2d, keypoints_conf, skeleton_3d,
                           renderer_3d, panel_size):
    """One (2D overlay | 3D 4-view) side-by-side frame, ready to write to a
    cv2.VideoWriter. keypoints_2d/keypoints_conf: COCO-17 (YOLO order);
    skeleton_3d: (17,3) H36M-order root-relative, or None."""
    panel_w, panel_h = panel_size
    if keypoints_2d is not None and keypoints_conf is not None:
        overlay = draw_2d_skeleton(frame_bgr, keypoints_2d, keypoints_conf)
    else:
        overlay = frame_bgr
    panel_3d = renderer_3d.render(skeleton_3d)
    return np.hstack([cv2.resize(overlay, (panel_w, panel_h)), panel_3d])
