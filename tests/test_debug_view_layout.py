"""The recognition preview keeps each frame's aspect ratio inside its panel."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "1_recognition" / "src"))

from render_utils.debug_view import fit_on_white


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


if __name__ == "__main__":
    unittest.main()
