"""Finding a live camera's index, and opening it at a KNOWN resolution.

Two problems that both bite before calibration and neither of which announces
itself.

WHICH index. OpenCV addresses cameras by position, not by name, and the
numbering shifts whenever a device appears or disappears -- starting the OBS
Virtual Camera renumbers everything after it. There is no API for "which index
is OBS", so list_cameras() opens each index in a range and reports what it
actually delivered, and find_camera() picks one by device name or frame size.

WHAT mode. cv2.VideoCapture does not ask a device for its best mode. On Windows
the DirectShow backend negotiates whatever format it settles on first --
routinely 640x480 -- regardless of what the device can deliver, so an OBS
Virtual Camera emitting 1920x1080 arrives as 640x480 and nothing in the capture
path reports the loss. That matters here because resolution is part of the
calibration: K is only valid for the mode it was solved in, so a calibration
that silently ran at 640x480 while the live pipeline runs at 1920x1080 is wrong
by a factor of 3 in fx, fy, cx and cy, and neither run looks abnormal.

open_camera() therefore requests a resolution and VERIFIES it against a decoded
frame. The verification is the point: cap.get(CAP_PROP_FRAME_WIDTH) frequently
echoes back the value just set even when the device ignored it, so the
properties cannot be trusted and only a real frame's shape can.

    uv run python 1_recognition/setup/video_source.py --list
    - Full HD capture
    uv run python 1_recognition/setup/video_source.py --list --capture-width 1920 --capture-height 1080
    - 4K capture
    uv run python 1_recognition/setup/video_source.py --list --capture-width 3840 --capture-height 2160
    uv run python 1_recognition/setup/video_source.py --find OBS
"""
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# resolve_backend lives with FrameSource, the library-side capture opener, so the
# live pipeline and this probe agree on what "auto" means.
from camera_utils.frame_source import BACKEND_NAMES, resolve_backend
from logging_setup import configure_logging, get_logger

logger = get_logger(__name__)

# How many indices list_cameras() walks by default. Consumer machines rarely
# expose more; raise it with --max-index on a machine with many capture devices.
DEFAULT_MAX_INDEX = 8

# Some devices -- virtual cameras especially -- hand back an empty first buffer
# while their pipeline spins up, so a single failed read is not proof of absence.
PROBE_READ_ATTEMPTS = 3


@contextmanager
def _quiet_opencv():
    """Silence OpenCV's own stderr chatter for the duration of a probe.

    Walking an index range means most indices do not exist, and the backend
    logs an error for each one. Those errors are the expected outcome here and
    would otherwise scroll the results off the screen.
    """
    try:
        previous = cv2.getLogLevel()
        cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_SILENT", 0))
    except Exception:  # older builds expose no log-level control
        yield
        return
    try:
        yield
    finally:
        cv2.setLogLevel(previous)


# -- discovery -------------------------------------------------------------


def device_names(backend="auto"):
    """DirectShow device names in OpenCV index order, or None when unavailable.

    pygrabber enumerates the same DirectShow filter graph CAP_DSHOW opens, so
    position i in the returned list is camera index i -- the only way to map a
    name onto an index, since OpenCV itself exposes no device names at all.
    Optional (uv add pygrabber) and Windows/DirectShow only; everywhere else
    listing still works, just without names.
    """
    if sys.platform != "win32" or resolve_backend(backend) != cv2.CAP_DSHOW:
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return None
    try:
        return list(FilterGraph().get_input_devices())
    except Exception as exc:
        logger.debug("pygrabber device enumeration failed: %s", exc)
        return None


def probe_camera(index, width=None, height=None, backend="auto"):
    """Open `index`, decode one frame, and report what came out.

    Returns {"index", "width", "height", "fps", "backend"} or None if the
    device will not open or delivers nothing. Never raises: probing a range
    means most indices are absent, and absence is an answer, not an error.

    The reported size is the decoded frame's, not the capture properties' --
    see the module docstring for why those cannot be trusted.
    """
    backend_id = resolve_backend(backend)
    capture = None
    try:
        with _quiet_opencv():
            capture = cv2.VideoCapture(index, backend_id)
            if not capture.isOpened():
                return None

            if width and height:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

            for _ in range(PROBE_READ_ATTEMPTS):
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
            else:
                return None

        fps = capture.get(cv2.CAP_PROP_FPS)
        return {
            "index": index,
            "width": frame.shape[1],
            "height": frame.shape[0],
            # Unlike the frame size this can only come from the property, and
            # virtual cameras often report 0. Treated as "unknown", not as 0 fps.
            "fps": round(float(fps), 2) if fps and fps > 0 else None,
            # What was asked for, so a caller can tell "delivers 640x480" from
            # "delivers 640x480 despite being asked for 1920x1080".
            "requested": (width, height) if width and height else None,
            "backend": backend,
        }
    finally:
        if capture is not None:
            capture.release()


def list_cameras(max_index=DEFAULT_MAX_INDEX, width=None, height=None, backend="auto"):
    """Probe indices 0..max_index-1 and return the ones that deliver a frame.

    Every index is probed rather than stopping at the first gap: the numbering
    is not contiguous, and the device actually wanted -- a virtual camera --
    usually sits above the built-in webcam with holes in between.

    Each result carries a "name" key when device_names() could resolve one.
    A camera already held open by another process will not appear, which is
    the usual reason a known-good index goes missing from this list.
    """
    names = device_names(backend)
    cameras = []
    for index in range(max_index):
        info = probe_camera(index, width=width, height=height, backend=backend)
        if info is None:
            continue
        info["name"] = names[index] if names and index < len(names) else None
        cameras.append(info)
    return cameras


def find_camera(name=None, size=None, width=None, height=None,
                max_index=DEFAULT_MAX_INDEX, backend="auto"):
    """Return the index of the first camera matching `name` or `size`.

    `name` is a case-insensitive substring of the device name ("obs" matches
    "OBS Virtual Camera") and needs pygrabber; `size` is an exact (width,
    height) a decoded frame must equal. Give at least one.

    width/height REQUEST a mode before matching, which is a different question
    from `size`: an OBS Virtual Camera left alone delivers 640x480 and only
    reaches 1920x1080 when asked, so matching on size alone would never find
    it. Pass both together to mean "the camera that can give me this mode".

    Raises LookupError listing what was actually found, so a failure names the
    alternatives instead of just saying no.
    """
    if name is None and size is None:
        raise ValueError("find_camera needs a name, a size, or both.")

    cameras = list_cameras(max_index=max_index, width=width, height=height,
                           backend=backend)
    for info in cameras:
        if name is not None:
            if not info["name"] or name.lower() not in info["name"].lower():
                continue
        if size is not None and (info["width"], info["height"]) != tuple(size):
            continue
        return info["index"]

    wanted = " and ".join(part for part in (
        f"name containing {name!r}" if name is not None else None,
        f"size {size[0]}x{size[1]}" if size is not None else None,
    ) if part)
    raise LookupError(
        f"No camera matched {wanted}. Found: {format_camera_table(cameras)}")


def format_camera_table(cameras):
    """One line per camera, for an error message or the --list output.

    A camera that ignored a requested resolution is marked REFUSED, since that
    is the one thing worth noticing before calibrating against it.
    """
    if not cameras:
        return "(no cameras responded)"

    lines = []
    for info in cameras:
        parts = [f"  index {info['index']}", f"{info['width']}x{info['height']}"]
        if info.get("fps"):
            parts.append(f"{info['fps']} fps")
        if info.get("name"):
            parts.append(info["name"])
        requested = info.get("requested")
        if requested and tuple(requested) != (info["width"], info["height"]):
            parts.append(f"(REFUSED {requested[0]}x{requested[1]})")
        lines.append("  ".join(parts))
    return "\n".join(lines)


# -- opening ---------------------------------------------------------------


def open_camera(index, width=None, height=None, backend="auto", quiet=False):
    """Open camera `index`, optionally requesting width x height.

    Returns (capture, (actual_width, actual_height)). The actual size is read
    off a decoded frame, not off the capture properties. Raises IOError if the
    device will not open; warns rather than raising when it opens but refuses
    the requested size, since a different-but-usable mode is still worth
    calibrating in as long as the operator knows which one they got.
    """
    capture = cv2.VideoCapture(index, resolve_backend(backend))
    if not capture.isOpened():
        capture.release()
        raise IOError(
            f"Could not open camera index {index} on the {backend} backend. "
            f"Run 'python {Path(__file__).name} --list' to see which indices exist.")

    if width and height:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise IOError(
            f"Camera index {index} opened but delivered no frame. If this is an OBS "
            f"Virtual Camera, start it in OBS (Controls > Start Virtual Camera).")
    actual = (frame.shape[1], frame.shape[0])

    if width and height and actual != (width, height):
        logger.warning(
            "Requested %dx%d but camera index %d delivered %dx%d. Calibrating anyway -- "
            "the intrinsics will be valid for %dx%d, so the live pipeline must run at "
            "that size too. For an OBS Virtual Camera, set Settings > Video > Output "
            "(Scaled) Resolution, then STOP and START the virtual camera.",
            width, height, index, actual[0], actual[1], actual[0], actual[1])
    elif not quiet:
        logger.info("Capturing at %dx%d.", actual[0], actual[1])

    return capture, actual


def resolve_capture_size(capture_width, capture_height, calibrated_size):
    """The resolution to open a camera at, given an optional explicit request
    and the resolution the intrinsics were solved at.

    Defaults to the calibrated size, because K only holds there: capturing at
    any other size scales fx, fy, cx and cy and makes every solve using that K
    confidently wrong with nothing looking abnormal. An explicit request wins
    but is warned about, since the only correct reason to differ is an
    intrinsics file converted with setup/rescale_intrinsics.py.

    Returns (width, height), or None when neither is known -- callers then
    leave the mode to the backend.
    """
    if capture_width and capture_height:
        requested = (int(capture_width), int(capture_height))
        if calibrated_size and requested != tuple(calibrated_size):
            logger.warning(
                "Capturing at %dx%d but the intrinsics were solved at %dx%d. K does not "
                "transfer between resolutions -- the result will be wrong unless this "
                "intrinsics file was produced by rescale_intrinsics.py for this size.",
                requested[0], requested[1], calibrated_size[0], calibrated_size[1])
        return requested
    return tuple(calibrated_size) if calibrated_size else None


def verify_frame_size(frame, calibrated_size, source_label=""):
    """Check a decoded frame against the calibrated resolution and warn on a
    mismatch. Returns the frame's (width, height).

    For sources whose size cannot be requested -- a recorded file plays at
    whatever it was written at -- checking is all that is available, and it is
    still worth doing: a clip at the wrong resolution silently corrupts every
    metric quantity derived from K.
    """
    actual = (int(frame.shape[1]), int(frame.shape[0]))
    if calibrated_size and actual != tuple(calibrated_size):
        logger.warning(
            "%s delivers %dx%d but the intrinsics were solved at %dx%d. K does not "
            "transfer between resolutions -- fx, fy, cx and cy are out by roughly %.2fx, "
            "so every metric result from this run will be wrong. Convert the calibration "
            "with setup/rescale_intrinsics.py, or use a source at the calibrated size.",
            source_label or "This source", actual[0], actual[1],
            calibrated_size[0], calibrated_size[1],
            actual[0] / calibrated_size[0] if calibrated_size[0] else float("nan"))
    return actual


# Preview windows are sized by the frame unless told otherwise, so a 4K capture
# opens a 4K window that does not fit on screen. This is the width they are
# shown at instead; the frames themselves are never touched.
DEFAULT_PREVIEW_WIDTH = 1280

_SIZED_WINDOWS = set()


def preview_scale(frame_width, preview_width=DEFAULT_PREVIEW_WIDTH):
    """Multiplier for cv2 font scales and line thicknesses drawn on a frame
    that will be SHOWN at preview_width.

    Overlays are drawn at full resolution, so without this a 0.8 font on a 4K
    frame shown in a 1280-wide window renders at an unreadable 0.27. Scaling
    by frame_width/preview_width makes it land at 0.8 on screen whatever the
    capture resolution. Never shrinks below 1.0 -- a frame narrower than the
    preview is shown at its own size.
    """
    return max(float(frame_width) / float(preview_width or DEFAULT_PREVIEW_WIDTH), 1.0)


def show_preview(window_name, frame, preview_width=DEFAULT_PREVIEW_WIDTH):
    """imshow `frame` in a window capped at preview_width.

    The window is scaled, not the image: cv2 does the downscaling for display
    only, so detection keeps running on the full-resolution frame that K was
    calibrated for. WINDOW_NORMAL also makes the window draggable to any size.
    Sized once per window, since resizing every frame fights the user doing it
    by hand.
    """
    if window_name not in _SIZED_WINDOWS:
        preview_width = int(preview_width or DEFAULT_PREVIEW_WIDTH)
        height, width = frame.shape[:2]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if width > preview_width:
            cv2.resizeWindow(window_name, preview_width,
                             max(int(round(height * preview_width / width)), 1))
        else:
            cv2.resizeWindow(window_name, width, height)
        _SIZED_WINDOWS.add(window_name)
    cv2.imshow(window_name, frame)


def forget_preview_windows():
    """Drop the sized-window memory, so a later run sizes its windows again.
    Call alongside cv2.destroyAllWindows()."""
    _SIZED_WINDOWS.clear()


def add_capture_args(parser):
    """Shared --capture-width/--capture-height/--backend flags, so every
    entry point that opens a live camera spells them the same way."""
    parser.add_argument("--capture-width", type=int, default=None,
                        help="Request this capture width. Without it OpenCV picks a "
                             "default that is often 640 regardless of what the device "
                             "can deliver -- see setup/video_source.py.")
    parser.add_argument("--capture-height", type=int, default=None,
                        help="Request this capture height. Use together with "
                             "--capture-width.")
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="auto",
                        help="Capture backend (default auto: DirectShow on Windows).")
    parser.add_argument("--preview-width", type=int, default=DEFAULT_PREVIEW_WIDTH,
                        help=f"Width of the preview window (default {DEFAULT_PREVIEW_WIDTH}). "
                             "Display only -- detection always uses the full-resolution "
                             "frame, since K is only valid there.")


# -- CLI -------------------------------------------------------------------


def parse_size(text):
    """Parse "1920x1080" into (1920, 1080), for argparse's type=."""
    try:
        width, height = (int(part) for part in text.lower().split("x", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Expected WIDTHxHEIGHT (e.g. 1920x1080), got {text!r}.") from None
    return width, height


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="List the live cameras OpenCV can see, or find one by name or size.")
    parser.add_argument("--list", action="store_true",
                        help="Probe every index and print what each one delivers.")
    parser.add_argument("--find", metavar="NAME", default=None,
                        help="Print the index of the first camera whose device name "
                             "contains NAME (case-insensitive), e.g. --find OBS. Needs "
                             "pygrabber; without it, use --match-size instead.")
    parser.add_argument("--match-size", metavar="WIDTHxHEIGHT", type=parse_size, default=None,
                        help="Find the camera whose frames come out at exactly this "
                             "size. This MATCHES a delivered size; --capture-width/"
                             "--capture-height REQUEST a mode. Combine them to mean "
                             "'ask for this mode, then take whichever camera honoured it'.")
    parser.add_argument("--max-index", type=int, default=DEFAULT_MAX_INDEX,
                        help=f"Highest index to probe, exclusive (default {DEFAULT_MAX_INDEX}).")
    add_capture_args(parser)
    args = parser.parse_args(argv)

    configure_logging("video_source")

    if args.find is not None or args.match_size is not None:
        try:
            print(find_camera(name=args.find, size=args.match_size,
                              width=args.capture_width, height=args.capture_height,
                              max_index=args.max_index, backend=args.backend))
        except LookupError as exc:
            logger.error("%s", exc)
            return 1
        return 0

    cameras = list_cameras(max_index=args.max_index, width=args.capture_width,
                           height=args.capture_height, backend=args.backend)
    print(format_camera_table(cameras))
    if cameras and all(info["name"] is None for info in cameras):
        # Without names the table is just sizes, which is usually still enough to
        # spot a virtual camera, but worth saying why the column is missing.
        print("\n(No device names: install pygrabber -- uv add pygrabber -- to see "
              "which index is which.)")
    return 0 if cameras else 1


if __name__ == "__main__":
    raise SystemExit(main())
