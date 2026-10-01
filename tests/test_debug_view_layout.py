"""The recognition preview keeps each frame's aspect ratio inside its panel, and its
text readout says what the model output."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))

from render_utils.debug_view import DebugView, fit_on_white


class FitOnWhiteTests(unittest.TestCase):
    def test_wide_frame_gets_white_bands_above_and_below(self):
        frame = np.zeros((360, 640, 3), np.uint8)  # 16:9, like camera 6 at 640x360
        panel = fit_on_white(frame, (480, 480))
        self.assertEqual(panel.shape, (480, 480, 3))
        self.assertTrue((panel[:105] == 255).all() and (panel[-105:] == 255).all())
        self.assertTrue((panel[106:374] == 0).all())  # 480 x 270: the frame, not squashed

    def test_tall_frame_gets_white_bands_at_the_sides(self):
        panel = fit_on_white(np.zeros((720, 405, 3), np.uint8), (480, 480))
        self.assertTrue((panel[:, :100] == 255).all() and (panel[:, -100:] == 255).all())
        self.assertTrue((panel[:, 106:374] == 0).all())

    def test_frame_of_the_panel_size_is_untouched(self):
        frame = np.zeros((480, 480, 3), np.uint8)
        self.assertIs(fit_on_white(frame, (480, 480)), frame)


class OverlayTextTests(unittest.TestCase):
    def overlay_lines(self, **readout):
        view = DebugView(enabled=False, window_name="w", panel_size=(480, 480),
                         plot_window_name="p", plot_panel_size=(600, 640), history_len=10,
                         conf_threshold=0.3, step_names=["Pull Cables", "Lift"])
        view._ensure_cv2()
        with patch.object(view._cv2, "putText") as put_text:
            view._draw_overlay(np.zeros((100, 100, 3), np.uint8), world_xyz=None,
                               status_line="Buffering: 3/80", **readout)
        return list(dict.fromkeys(call.args[1] for call in put_text.call_args_list))

    def test_multi_head_readout_says_how_idle_the_model_thinks_it_is(self):
        lines = self.overlay_lines(raw_step_id=1, stable_step_id=1, progress=0.4,
                                   confidence=0.25, idle_probability=0.72)
        self.assertIn("Raw step: Lift", lines)
        self.assertIn("Idle (bg head): 0.72", lines)

    def test_legacy_readout_has_no_idle_line(self):
        lines = self.overlay_lines(raw_step_id=0, stable_step_id=None, progress=0.4,
                                   confidence=0.9)
        self.assertFalse(any(line.startswith("Idle") for line in lines))


if __name__ == "__main__":
    unittest.main()
