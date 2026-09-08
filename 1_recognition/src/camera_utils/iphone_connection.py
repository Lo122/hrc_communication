"""iPhone camera connection via the Record3D app (paid, USB streaming).

Requires the optional ``iphonecamera`` extra (``record3d``, see
pyproject.toml) and the Record3D iOS app (https://record3d.app/) on the
phone, connected to this machine over USB with "USB Streaming" mode enabled
in the app.

``Record3DStream`` delivers frames via an ``on_new_frame`` callback from its
own background thread; ``IPhoneCamera`` wraps that into a small thread-safe
"latest frame" store so callers can just poll ``get_latest_frame()`` from
their own loop, the same way the other calibration tools poll
``cv2.VideoCapture.read()``.

Unlike a marker/solvePnP-only setup, Record3D also reports ARKit's
fused, drift-corrected 6DOF camera pose per frame (``get_camera_pose()``),
so ``get_latest_frame()`` returns it alongside the RGB/depth/intrinsics --
useful for cross-checking the ArUco-based extrinsic from
``iphone_extrinsic_calibration.py``, though that marker-based
T_world_from_camera is still the source of truth used elsewhere in this
project (ARKit's pose is relative to its own session-start origin, not this
project's world frame).

USB streaming is prone to drops (cable jostled, phone locks, app backgrounded).
By default ``IPhoneCamera`` runs a background watchdog that notices a drop
(via Record3D's ``on_stream_stopped`` callback) and keeps retrying
``connect()`` every ``reconnect_interval_sec`` until the device reappears --
callers just keep calling ``get_latest_frame()``; it returns ``None`` while
disconnected and resumes returning frames once reconnected, no restart needed.

Record3D can deliver ``get_rgb_frame()``/``get_depth_frame()`` in the
sensor's own native orientation, independent of how the phone is actually
held/mounted -- if that doesn't match, EVERY consumer (2D pose detection,
ArUco marker detection, etc.) sees a rotated person/scene, which is a much
bigger problem than a crooked preview: YOLO/lifter models are trained on
upright people and degrade badly on a 90-degree-rotated input, corrupting
downstream depth/3D estimates, not just how things look on screen. Pass
``capture_rotate90`` (one of 0/90/180/270) to ``IPhoneCamera``/
``IPhoneVideoCaptureAdapter`` to correct this AT THE SOURCE -- every
``get_latest_frame()``/``read()`` caller then transparently gets an
upright rgb+depth+intrinsic_mat, with no per-caller math needed. This is
different from (and independent of) ``rotate_frame``/``roll_from_camera_
pose`` below, which only rotate a copy for on-screen display and leave the
actual working frame (and any K it's calibrated against) untouched --
use capture_rotate90 to fix the real problem, not just the display.

Because this changes actual pixel content and frame dimensions, K
(intrinsics) calibrated WITHOUT capture_rotate90 no longer matches frames
captured WITH it (or vice versa) -- always recalibrate both intrinsics
(iphone_intrinsic_calibration.py) and extrinsics
(iphone_extrinsic_calibration.py) with the same capture_rotate90 you intend
to use at runtime, and pass that same value to every script consistently.
"""
import threading
import time

import cv2
import numpy as np


def _intrinsic_coeffs_to_matrix(coeffs):
    """record3d.Record3DStream.get_intrinsic_mat() returns an
    IntrinsicMatrixCoeffs object (.fx, .fy, .tx, .ty), not an array --
    build the actual 3x3 pinhole matrix from it (tx/ty are the principal
    point, i.e. cx/cy)."""
    return np.array([
        [coeffs.fx, 0.0, coeffs.tx],
        [0.0, coeffs.fy, coeffs.ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# Exact (no-interpolation) quarter-turn flags for _on_new_frame's
# capture_rotate90 -- these are the only rotations that are a clean pixel
# permutation, hence the only ones with a closed-form K update below (an
# arbitrary angle, see rotate_frame, needs interpolation and isn't worth
# using for a live capture path).
_ROTATE_FLAGS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _rotate_intrinsics_90(K, image_size, angle_deg):
    """Exact K update matching cv2.rotate(img, _ROTATE_FLAGS[angle_deg])
    applied to the image K was calibrated for. image_size is the
    PRE-rotation (width, height). Returns (K_new, new_image_size)."""
    angle_deg = int(angle_deg) % 360
    w, h = image_size
    if angle_deg == 0:
        return K.copy(), (w, h)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if angle_deg == 90:
        K_new = np.array([[fy, 0.0, h - 1 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]])
        size_new = (h, w)
    elif angle_deg == 180:
        K_new = np.array([[fx, 0.0, w - 1 - cx], [0.0, fy, h - 1 - cy], [0.0, 0.0, 1.0]])
        size_new = (w, h)
    elif angle_deg == 270:
        K_new = np.array([[fy, 0.0, cy], [0.0, fx, w - 1 - cx], [0.0, 0.0, 1.0]])
        size_new = (h, w)
    else:
        raise ValueError(f"angle_deg must be one of 0/90/180/270, got {angle_deg}")
    return K_new.astype(np.float64), size_new


def rotate_camera_vector_90(v, angle_deg):
    """Re-express a vector's local-camera-frame coordinates (OpenCV
    convention: +X right, +Y down, +Z forward) for a camera whose actual
    working frame has been rotated by angle_deg (0/90/180/270, see
    capture_rotate90) -- i.e. takes the SAME physical vector's coordinates
    in the ORIGINAL (pre-rotation) camera axes and returns its coordinates
    in the NEW (post-rotation) camera's own local axes. This is the exact
    axis relabeling implied by _rotate_intrinsics_90 -- use it on any
    camera-frame vector (not just K) that was derived assuming the
    original, un-rotated axes -- e.g. camera_gravity_direction_cv's output,
    before combining it with anything computed in the capture_rotate90'd
    frame (solvePnP results, etc.). Mixing un-rotated-frame vectors with
    rotated-frame ones without this is a silent ~90deg-class error.
    """
    angle_deg = int(angle_deg) % 360
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v
    if angle_deg == 0:
        return v.copy()
    if angle_deg == 90:
        return np.array([-y, x, z])
    if angle_deg == 180:
        return np.array([-x, -y, z])
    if angle_deg == 270:
        return np.array([y, -x, z])
    raise ValueError(f"angle_deg must be one of 0/90/180/270, got {angle_deg}")


def rotate_frame(frame, angle_deg):
    """Rotate a BGR frame by angle_deg (counter-clockwise positive) about its
    center, auto-expanding the canvas so corners don't get cropped.

    Display-only helper: rotating a frame invalidates any intrinsics (K) it
    was captured with (cx/cy, and fx/fy for a 90 degree rotation no longer
    match), so only use this on a copy meant for cv2.imshow -- never on the
    frame passed to marker detection / solvePnP / depth back-projection.
    """
    if angle_deg == 0.0:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(frame, M, (new_w, new_h))


def camera_pose_rotation_matrix(pose):
    """3x3 R_world_from_camera in ARKit's OWN axis convention (local +X
    right, +Y up, +Z toward the viewer/out of the screen; world frame
    gravity-aligned, +Y opposite gravity) -- from a record3d.CameraPose
    (fields qx, qy, qz, qw, tx, ty, tz), IPhoneCamera.get_latest_frame()'s
    4th element. See _ARKIT_TO_OPENCV_CAMERA_AXES for converting a vector
    expressed in this camera's local axes into OpenCV's convention (+X
    right, +Y down, +Z forward -- what geometry/transforms.py etc. use)."""
    qx, qy, qz, qw = pose.qx, pose.qy, pose.qz, pose.qw
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ])


# ARKit's local camera axes (X right, Y up, Z toward viewer) vs. OpenCV's
# (X right, Y down, Z forward -- see geometry/transforms.py's docstring):
# same physical camera, Y and Z just point the opposite way. Self-inverse
# (applying it twice is a no-op), so the same matrix converts either
# direction between the two conventions.
_ARKIT_TO_OPENCV_CAMERA_AXES = np.diag([1.0, -1.0, -1.0])


def camera_gravity_direction_cv(pose):
    """Unit 'down' (gravity) direction, expressed in the camera's OWN local
    axes using OpenCV convention (+X right, +Y down, +Z forward) -- e.g. for
    auto-correcting a marker-based extrinsic calibration's roll/pitch using
    ARKit's true gravity sensing instead of a manually-specified/assumed
    marker orientation (see iphone_extrinsic_calibration.py's
    --auto-gravity-correct)."""
    R_world_from_camera_arkit = camera_pose_rotation_matrix(pose)
    gravity_world_arkit = np.array([0.0, -1.0, 0.0])  # ARKit world +Y is opposite gravity
    gravity_camera_arkit = R_world_from_camera_arkit.T @ gravity_world_arkit
    return _ARKIT_TO_OPENCV_CAMERA_AXES @ gravity_camera_arkit


def roll_from_camera_pose(pose):
    """Signed roll (degrees) of an ARKit camera pose relative to gravity --
    how much the phone is tilted sideways from level, i.e. the rotation that
    would make the image's horizon horizontal again if undone. Only roll
    (rotation about the camera's own forward axis) tilts the *image* --
    pitch/yaw change what's framed, not whether it looks level.
    """
    R = camera_pose_rotation_matrix(pose)  # R_world_from_camera, ARKit axes
    cam_up_world = R @ np.array([0.0, 1.0, 0.0])
    cam_fwd_world = R @ np.array([0.0, 0.0, -1.0])
    world_up = np.array([0.0, 1.0, 0.0])

    # World-up projected into the camera's image plane (perpendicular to the
    # viewing direction) -- this is "what level looks like" in-frame.
    proj = world_up - np.dot(world_up, cam_fwd_world) * cam_fwd_world
    proj_norm = np.linalg.norm(proj)
    if proj_norm < 1e-6:
        return 0.0  # looking straight up/down -- roll is undefined, don't rotate
    proj /= proj_norm

    cos_roll = np.clip(np.dot(proj, cam_up_world), -1.0, 1.0)
    sin_roll = np.dot(np.cross(cam_up_world, proj), cam_fwd_world)
    return float(np.degrees(np.arctan2(sin_roll, cos_roll)))


class IPhoneCamera:
    """Thread-safe wrapper around ``record3d.Record3DStream`` with auto-reconnect.

    Usage:
        with IPhoneCamera(dev_idx=0) as cam:
            while True:
                frame = cam.get_latest_frame(timeout=1.0)
                if frame is None:
                    if not cam.is_connected:
                        print("reconnecting...")
                    continue
                rgb, depth, intrinsic_mat, pose = frame
                ...
    """

    def __init__(self, dev_idx=0, auto_reconnect=True, reconnect_interval_sec=2.0,
                 capture_rotate90=0):
        if capture_rotate90 not in (0, 90, 180, 270):
            raise ValueError(f"capture_rotate90 must be one of 0/90/180/270, got {capture_rotate90}")
        self._dev_idx = dev_idx
        self._auto_reconnect = auto_reconnect
        self._reconnect_interval_sec = reconnect_interval_sec
        self._capture_rotate90 = capture_rotate90
        self._stream = None
        self._generation = 0  # bumped on every successful connect; guards stale callbacks
        self._lock = threading.Lock()
        self._latest = None  # (rgb, depth, intrinsic_mat, pose)
        self._new_frame_event = threading.Event()
        self._connected_event = threading.Event()
        self._stop_event = threading.Event()
        self._watchdog_thread = None

    @staticmethod
    def list_devices():
        """Return the list of connected Record3D devices (USB + WiFi)."""
        from record3d import Record3DStream
        return Record3DStream.get_connected_devices()

    # record3d.Record3DStream.get_device_type() returns a plain int rather
    # than an enum: 0 = TrueDepth, 1 = LiDAR (per its docstring).
    _DEVICE_TYPE_LIDAR = 1

    def is_lidar(self):
        """True if the connected device has a LiDAR sensor (metric depth)."""
        return self._stream.get_device_type() == self._DEVICE_TYPE_LIDAR

    @property
    def is_connected(self):
        return self._connected_event.is_set()

    def connect(self):
        self._stop_event.clear()
        self._connect_once(raise_if_absent=True)
        if self._auto_reconnect:
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, daemon=True, name="IPhoneCameraWatchdog")
            self._watchdog_thread.start()
        return self

    def _connect_once(self, raise_if_absent):
        """Try to (re)connect once. Returns True on success, False if no device
        was found (and raise_if_absent is False)."""
        from record3d import Record3DStream

        devices = self.list_devices()
        if not devices or self._dev_idx >= len(devices):
            if raise_if_absent:
                raise IOError(
                    "No Record3D devices found. Make sure the Record3D app is open "
                    "on the iPhone, the phone is connected via USB, and 'USB "
                    "Streaming' is enabled in the app."
                )
            return False

        generation = self._generation + 1

        stream = Record3DStream()
        stream.on_new_frame = lambda: self._on_new_frame(stream, generation)
        stream.on_stream_stopped = lambda: self._on_stream_stopped(generation)
        stream.connect(devices[self._dev_idx])

        self._stream = stream
        self._generation = generation
        self._connected_event.set()
        return True

    def disconnect(self):
        self._stop_event.set()
        self._connected_event.clear()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=self._reconnect_interval_sec + 1.0)
            self._watchdog_thread = None
        if self._stream is not None:
            self._stream.disconnect()
            self._stream = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

    def _on_new_frame(self, stream, generation):
        if generation != self._generation:
            return  # stale callback from a stream a reconnect has already replaced
        rgb = stream.get_rgb_frame()
        depth = stream.get_depth_frame()  # metric depth if LiDAR device
        intrinsic_mat = _intrinsic_coeffs_to_matrix(stream.get_intrinsic_mat())  # per-frame, focus-corrected
        pose = stream.get_camera_pose()  # ARKit-tracked position + quaternion
        if self._capture_rotate90:
            # Rotate rgb+depth+intrinsic_mat together here, once, so EVERY
            # caller downstream (calibration scripts, metric_depth_test.py,
            # ...) transparently sees an already-upright, mutually
            # consistent (frame, K) pair -- see module docstring on why this
            # matters far more than the display-only rotate_frame below.
            h, w = rgb.shape[:2]
            flag = _ROTATE_FLAGS[self._capture_rotate90]
            rgb = cv2.rotate(rgb, flag)
            depth = cv2.rotate(depth, flag)
            intrinsic_mat, _ = _rotate_intrinsics_90(intrinsic_mat, (w, h), self._capture_rotate90)
        with self._lock:
            self._latest = (rgb, depth, intrinsic_mat, pose)
        self._new_frame_event.set()

    def _on_stream_stopped(self, generation):
        if generation != self._generation:
            return  # already superseded by a newer (re)connection
        with self._lock:
            self._latest = None
        self._connected_event.clear()
        print("IPhoneCamera: stream disconnected.")

    def _watchdog_loop(self):
        while not self._stop_event.is_set():
            if not self._connected_event.is_set():
                if self._connect_once(raise_if_absent=False):
                    print("IPhoneCamera: reconnected.")
            self._stop_event.wait(self._reconnect_interval_sec)

    def get_latest_frame(self, timeout=None):
        """Return the most recently received (rgb, depth, intrinsic_mat, pose), or None
        (including while disconnected and waiting to reconnect -- see is_connected)."""
        if timeout is not None:
            got_frame = self._new_frame_event.wait(timeout=timeout)
            self._new_frame_event.clear()
            if not got_frame:
                return None
        with self._lock:
            return self._latest


class IPhoneVideoCaptureAdapter:
    """Makes an ``IPhoneCamera`` look like a ``cv2.VideoCapture`` (``read()``
    returning ``(ok, frame_bgr)``, ``release()``), so anything built around
    that interface -- e.g. ``pose.pose_worker.PoseWorker`` -- can use an
    iPhone/Record3D stream as a drop-in camera source instead of a local
    webcam index.

    ``IPhoneCamera`` already auto-reconnects through brief USB drops (see its
    docstring), so ``read()`` rides those out transparently by waiting up to
    ``max_wait_sec`` for a frame before giving up and returning ``(False,
    None)`` -- mirroring cv2.VideoCapture's "stream ended" signal, which
    callers like PoseWorker treat as a reason to stop.

    ``cv2.VideoCapture`` has no notion of an IMU/ARKit pose, so ``read()``'s
    return signature stays a plain ``(ok, frame_bgr)`` drop-in match -- the
    record3d.CameraPose for the most recently read frame is instead cached
    on ``last_pose`` after each ``read()`` call (``None`` if that read
    failed), for callers that want it (e.g. auto-leveling a preview via
    ``roll_from_camera_pose``) without breaking the VideoCapture-alike API.
    """

    def __init__(self, dev_idx=0, max_wait_sec=30.0, capture_rotate90=0):
        self._cam = IPhoneCamera(dev_idx=dev_idx, capture_rotate90=capture_rotate90)
        self._max_wait_sec = max_wait_sec
        self._cam.connect()
        self.last_pose = None

    def isOpened(self):
        return True

    def read(self):
        import cv2

        deadline = time.time() + self._max_wait_sec
        while time.time() < deadline:
            result = self._cam.get_latest_frame(timeout=1.0)
            if result is not None:
                rgb, _depth, _intrinsic_mat, pose = result
                self.last_pose = pose
                return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.last_pose = None
        return False, None

    def release(self):
        self._cam.disconnect()


if __name__ == "__main__":
    # Quick manual smoke test: connect, print a few frames' shapes/intrinsics.
    # Try unplugging/replugging the phone mid-run to see the watchdog kick in.
    with IPhoneCamera(dev_idx=0) as cam:
        print(f"Connected. LiDAR device: {cam.is_lidar()}")
        for _ in range(1000):
            frame = cam.get_latest_frame(timeout=2.0)
            if frame is None:
                status = "reconnecting..." if not cam.is_connected else "no frame (timeout)"
                print(f"No frame received ({status}).")
                continue
            rgb, depth, intrinsic_mat, pose = frame
            print(f"rgb={rgb.shape} depth={depth.shape} K=\n{intrinsic_mat}")
            time.sleep(0.1)
