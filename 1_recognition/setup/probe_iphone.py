"""Report what an iPhone running Record3D is actually streaming.

setup/video_source.py --list enumerates DirectShow/MSMF devices and so never
sees the phone: Record3D arrives over USB through Record3DStream, not as a
webcam. This is that tool's iPhone counterpart -- connect, take one frame, and
print the sizes and intrinsics the device really emits.

Two sizes matter and they are NOT the same number:

  RAW        what the app emits, i.e. the resolution picked in Record3D's
             settings (landscape, e.g. 1920x1440).
  ROTATED    what every consumer in this project actually works on, after
             --capture-rotate90 is applied. 90 or 270 SWAPS width and height,
             so a 1920x1440 stream becomes 1440x1920.

The intrinsics the device reports are rotated to match, exactly as
IPhoneCamera does per frame -- so the fx/fy/cx/cy printed here are the ones a
calibration captured at this rotation has to agree with.

Usage (from the repo root):
    uv run python 1_recognition/setup/probe_iphone.py
    uv run python 1_recognition/setup/probe_iphone.py --capture-rotate90 270
    uv run python 1_recognition/setup/probe_iphone.py --watch 30

--watch samples for N seconds and reports how much fx moves, which is how you
find out whether the phone's autofocus is quietly rescaling your intrinsics
(MetricDepthEstimator's depth is linear in fy, so a 2% swing in fy is a 2%
error on every world position).

Uses record3d, a regular project dependency (installed by uv sync).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_RECOGNITION_DIR = Path(__file__).resolve().parents[1]
for _p in (_RECOGNITION_DIR, _RECOGNITION_DIR / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from camera_utils.iphone_connection import IPhoneCamera  # noqa: E402
from logging_setup import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-idx", type=int, default=0,
                        help="Record3D device index when more than one is connected.")
    parser.add_argument("--capture-rotate90", type=int, default=0, choices=(0, 90, 180, 270),
                        help="Report the frame as this project will see it after rotation. "
                             "Must match whatever the calibration was captured with.")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Seconds to wait for the first frame (default 30).")
    parser.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                        help="Keep sampling for this long and report how much fx/fy drift -- "
                             "refocus the phone near and far while it runs to see what "
                             "autofocus does to the intrinsics.")
    return parser.parse_args()


def main():
    configure_logging("probe_iphone")
    args = parse_args()

    devices = IPhoneCamera.list_devices()
    print(f"Record3D devices visible: {len(devices)}")
    for i, dev in enumerate(devices):
        # DeviceInfo has no __repr__ worth printing -- it would just be the object address.
        print(f"    [{i}] udid={getattr(dev, 'udid', '?')} "
              f"product_id={getattr(dev, 'product_id', '?')}")
    if not devices:
        raise SystemExit(
            "No device. Check: phone connected by USB, Record3D open, and 'USB Streaming' "
            "enabled in the app (Settings > Enable USB streaming / Live RGBD).")

    with IPhoneCamera(dev_idx=args.dev_idx, capture_rotate90=args.capture_rotate90) as cam:
        frame = cam.get_latest_frame(timeout=args.timeout)
        if frame is None:
            # The device being listed only means libusbmuxd can see the phone. Frames
            # need the app in the foreground with streaming actually started -- a
            # repeating "stream disconnected" above is what that looks like.
            raise SystemExit(
                f"Connected but no frame within {args.timeout:.0f}s.\n"
                "The phone is visible over USB but is not streaming. In Record3D: open the "
                "app, switch to LIVE RGBD / USB streaming mode and leave it in the "
                "foreground with the screen on, then re-run. If 'stream disconnected' "
                "repeats above, the app is not in streaming mode.")
        rgb, depth, K, _pose = frame

        print(f"\ndevice type      : {'LiDAR' if cam.is_lidar() else 'TrueDepth'}")
        print(f"--capture-rotate90: {args.capture_rotate90}")

        h, w = rgb.shape[:2]
        print(f"\nRGB   as this project sees it : {w} x {h}")
        if args.capture_rotate90 in (90, 270):
            print(f"      RAW from the app        : {h} x {w}   "
                  "(rotation swaps width/height)")
        else:
            print(f"      RAW from the app        : {w} x {h}   (no rotation applied)")
        if depth is not None:
            dh, dw = depth.shape[:2]
            print(f"DEPTH as this project sees it : {dw} x {dh}"
                  f"   ({w / dw:.2f}x lower resolution than RGB)"
                  if dw else "")

        print(f"\nintrinsics reported by the device, rotated to match:")
        print(f"    fx = {K[0, 0]:9.3f}    fy = {K[1, 1]:9.3f}")
        print(f"    cx = {K[0, 2]:9.3f}    cy = {K[1, 2]:9.3f}")
        print(f"    (image centre would be {w / 2:.1f}, {h / 2:.1f})")
        print(f"    horizontal FOV ~ {2 * np.degrees(np.arctan(w / (2 * K[0, 0]))):.1f} deg")

        if args.watch <= 0:
            print("\nPass --watch 30 to see whether autofocus moves fx while you refocus.")
            return

        print(f"\nsampling for {args.watch:.0f}s -- refocus the phone near and far now...")
        fxs, fys, sizes, t0 = [], [], set(), time.time()
        while time.time() - t0 < args.watch:
            got = cam.get_latest_frame(timeout=1.0)
            if got is None:
                continue
            rgb_i, _d, K_i, _p = got
            fxs.append(float(K_i[0, 0]))
            fys.append(float(K_i[1, 1]))
            sizes.add(rgb_i.shape[:2])
            time.sleep(0.05)

        if not fxs:
            print("no frames sampled.")
            return
        fx, fy = np.array(fxs), np.array(fys)
        print(f"\n{len(fx)} samples")
        print(f"    frame sizes seen : {{{', '.join(f'{w_}x{h_}' for h_, w_ in sizes)}}}"
              f"{'  <-- CHANGED MID-STREAM' if len(sizes) > 1 else ''}")
        for name, arr in (("fx", fx), ("fy", fy)):
            spread = (arr.max() - arr.min()) / arr.mean() * 100.0
            print(f"    {name}: min {arr.min():8.3f}  max {arr.max():8.3f}  "
                  f"mean {arr.mean():8.3f}  spread {spread:5.2f}%")
        worst = max((fx.max() - fx.min()) / fx.mean(), (fy.max() - fy.min()) / fy.mean()) * 100.0
        if worst < 0.5:
            print("\n-> Focal length is effectively fixed. A one-off calibration is fine.")
        else:
            print(f"\n-> Focal length moves by {worst:.2f}%. Depth scales linearly with fy, so "
                  f"that is ~{worst:.2f}% on every world position. Either lock focus in the app, "
                  "or use the per-frame intrinsics the stream already reports (see "
                  "camera_utils/iphone_connection.py's _on_new_frame -- the adapter currently "
                  "discards them).")


if __name__ == "__main__":
    main()
