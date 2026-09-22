"""Per-subject body calibration from a held T-pose plus a slow 180deg turn:
bone-length targets for BoneLengthConstraintFilter, and (given calibrated
extrinsics) a metric stature for MetricDepthEstimator.

WHY SEED THE BONE FILTER AT ALL. BoneLengthConstraintFilter converges on its
own, but look at how: the target for a bone is seeded from whatever the FIRST
frame happened to measure, refined by a running mean for `warmup_frames`, and
from then on only accepts updates within [min_length_ratio, max_length_ratio]
of the current target. A bad first frame therefore does not just cost a slow
convergence -- it sets a wrong target that the ratio gate then actively
DEFENDS, because the correct lengths now look like outliers. Seeding from a
validated, multi-view median removes that failure mode, and removes the
convergence transient too (length_alpha=0.02 is a ~50-frame time constant,
i.e. ~5 s at the 10 Hz this pipeline actually runs at live).

WHY A T-POSE. Limbs extended, maximally separated in the image, minimal
self-occlusion and minimal foreshortening -- the configuration where a
monocular lifter's bone-length estimates are most trustworthy. It helps the
arms and shoulders a great deal and the legs and spine barely at all, since
standing is standing either way; do not expect the leg chain to improve.

WHY THE TURN, AND WHY IT IS SAMPLED THROUGHOUT. Monocular depth ambiguity is
worst along the camera ray, so a frontal T-pose measures the in-image arm
span well and the depth axis poorly. The fix is a view where the previously
ambiguous axis is in-plane, which is the 90deg view -- NOT the 180deg one:
facing away puts the arms back in the image plane, so 0deg and 180deg are
nearly the same measurement for this purpose. The information is in the
SWEEP, so collect every valid frame through the turn and take a per-bone
MEDIAN over all of them. Median, not mean, for the same reason
skeleton3d_pipeline.py's running shoulder width uses one: a handful of bad
lifts mid-rotation must not move the target.

UNITS. Bone lengths here are in MotionBERT's own root-relative output units
(crop_scale-normalized image space, NOT meters -- see depth_anchor.py's
docstring). That is fine and needs no conversion: they are only ever compared
against, and used to rescale, other quantities in that same space. The
stature below is the one genuinely metric number, and it does NOT come from
these bone lengths -- it cannot, for exactly this reason.

HOW STATURE IS RECOVERED. Not by summing bone lengths (wrong units, see
above) but from the calibrated ground plane, which load_extrinsics already
returns as `ground_z` and which both the live and offline pipelines currently
discard. For an upright standing subject:

  1. Intersect the camera ray through the ANKLE pixel with the ground plane.
     This gives a metric 3D point WITHOUT assuming a height -- which is the
     whole point, since MetricDepthEstimator's usual depth estimate needs a
     height and would make this circular.
  2. Take that point's camera-frame depth. A standing person's head is
     directly above their feet, so the head sits at essentially the same
     depth.
  3. Back-project the NOSE pixel at that depth, convert to world, and
     subtract ground_z.

Step 3 yields nose height, which is exact given the calibration. Converting
nose height to full stature is the one approximate step in the chain -- see
NOSE_HEIGHT_RATIO.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from camera_utils import transforms as tf
from skeleton_utils.bone_length_filter import H36M_BONE_TREE

# H36M-17 (see coco_h36m.py). Vertical axis is index 2 -- motionbert_lifter's
# _postprocess_axes remaps to [x, z, -y], which is also what
# skeleton3d_pipeline.estimate_pitch_from_torso_vector assumes.
PELVIS, THORAX = 0, 8
L_SHOULDER, L_ELBOW, L_WRIST = 11, 12, 13
R_SHOULDER, R_ELBOW, R_WRIST = 14, 15, 16
UP = 2

# COCO-17 (ultralytics YOLO-pose order), for the 2D pixel work only.
COCO_NOSE = 0
COCO_LEFT_ANKLE, COCO_RIGHT_ANKLE = 15, 16

# Nose height as a fraction of stature. THE one approximate constant in the
# height chain: everything before it is exact given the calibration. ~0.93 is
# a standard adult anthropometric figure (the nose sits just below eye height,
# itself ~0.936 of stature), but it varies by a couple of percent across
# individuals and populations, so it is worth checking against one person of
# known height and overriding per deployment. Same spirit as
# metric_depth_estimator.TORSO_HEIGHT_RATIO, which is likewise a population
# average standing in for a measurement.
NOSE_HEIGHT_RATIO = 0.93


# A stature outside this range is almost certainly a units mistake -- 182 for
# 1.82 is the obvious one, and depth scales LINEARLY with this number, so a
# 100x error would make every world position meaningless.
MIN_PLAUSIBLE_STATURE_M = 0.5
MAX_PLAUSIBLE_STATURE_M = 2.5


def validate_stature_m(stature_m, source="stature"):
    """Return float(stature_m), or raise ValueError if it cannot be a human
    height in METRES."""
    value = float(stature_m)
    if not (MIN_PLAUSIBLE_STATURE_M <= value <= MAX_PLAUSIBLE_STATURE_M):
        raise ValueError(
            f"{source} of {value} is outside {MIN_PLAUSIBLE_STATURE_M}-"
            f"{MAX_PLAUSIBLE_STATURE_M} m and cannot be a human height in metres. "
            f"Give metres, not centimetres (1.82, not 182).")
    return value


@dataclass
class BodyCalibration:
    """One subject's calibration.

    bone_lengths is in MotionBERT units and is what
    BoneLengthConstraintFilter.seed() consumes. The two stature fields are
    metric and feed MetricDepthEstimator's user_height_meters:

      stature_m           measured by this pipeline from the ground plane
                          (needs extrinsics; see the module docstring). Carries
                          NOSE_HEIGHT_RATIO's couple-of-percent error.
      measured_stature_m  the subject's real height, measured with a tape.
                          Deliberately a SEPARATE field rather than overwriting
                          stature_m: keeping both is what lets you check the
                          ground-plane estimate against ground truth and retune
                          NOSE_HEIGHT_RATIO per deployment, exactly as that
                          constant's comment asks for.

    effective_stature_m picks between them -- the tape wins, being both more
    accurate and free of the ratio assumption.
    """

    bone_lengths: dict[tuple[int, int], float] = field(default_factory=dict)
    stature_m: float | None = None
    measured_stature_m: float | None = None
    nose_height_m: float | None = None
    n_pose_frames: int = 0
    n_height_frames: int = 0
    subject_id: str = ""
    notes: str = ""

    @property
    def effective_stature_m(self) -> float | None:
        """The height to actually use: the tape measurement if there is one,
        otherwise the ground-plane estimate, otherwise None (leaving
        VisionConfig.user_height_m's assumption in place)."""
        return self.measured_stature_m or self.stature_m or None

    @property
    def stature_source(self) -> str:
        """Which field effective_stature_m came from, for logging."""
        if self.measured_stature_m:
            return "measured (tape)"
        if self.stature_m:
            return "ground plane"
        return "none"

    @property
    def implied_nose_height_ratio(self) -> float | None:
        """nose_height_m / measured_stature_m -- this subject's true ratio,
        available only when both a tape measurement and a ground-plane nose
        height exist. Compare against NOSE_HEIGHT_RATIO to see the bias."""
        if not self.measured_stature_m or not self.nose_height_m:
            return None
        return float(self.nose_height_m) / float(self.measured_stature_m)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            # JSON has no tuple keys -- "parent-child", parsed back in load().
            "bone_lengths": {f"{p}-{c}": v for (p, c), v in self.bone_lengths.items()},
            "stature_m": self.stature_m,
            "measured_stature_m": self.measured_stature_m,
            "nose_height_m": self.nose_height_m,
            "n_pose_frames": self.n_pose_frames,
            "n_height_frames": self.n_height_frames,
            "subject_id": self.subject_id,
            "notes": self.notes,
            "units": ("bone_lengths are MotionBERT root-relative units (NOT meters, see "
                      "body_calibration.py's docstring); stature_m/measured_stature_m/"
                      "nose_height_m are meters."),
            "stature_fields": ("stature_m is measured by the pipeline from the ground plane; "
                               "measured_stature_m is a tape measurement and WINS when both "
                               "are present. Set measured_stature_m by hand, or with "
                               "setup/calibrate_body.py --measured-height."),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "BodyCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        bones = {}
        for key, value in data.get("bone_lengths", {}).items():
            parent, child = key.split("-")
            bones[(int(parent), int(child))] = float(value)

        # Hand-edited files are the expected way to add this, so a typo here is
        # likely -- check it at load rather than letting 182 m silently scale
        # every world position by 100.
        measured = data.get("measured_stature_m")
        if measured is not None:
            measured = validate_stature_m(measured, source=f"measured_stature_m in {path}")

        return cls(
            bone_lengths=bones,
            stature_m=data.get("stature_m"),
            measured_stature_m=measured,
            nose_height_m=data.get("nose_height_m"),
            n_pose_frames=int(data.get("n_pose_frames", 0)),
            n_height_frames=int(data.get("n_height_frames", 0)),
            subject_id=data.get("subject_id", ""),
            notes=data.get("notes", ""),
        )


def validate_tpose(skeleton_3d, arm_straightness_min=0.90, arm_horizontality_max=0.30,
                   symmetry_max=0.15, torso_upright_min=0.80) -> tuple[bool, str]:
    """skeleton_3d: (17, 3) H36M root-relative. Returns (ok, reason).

    Checked against the 3D skeleton rather than the 2D keypoints on purpose:
    an arm pointing at the camera is heavily foreshortened in 2D but still
    straight and horizontal in 3D, and those are exactly the mid-turn frames
    worth keeping (see module docstring on why the sweep matters).

      - straightness: |wrist-shoulder| / (|elbow-shoulder| + |wrist-elbow|).
        1.0 is a perfectly straight arm; the default 0.90 tolerates a slight
        bend without accepting a folded one.
      - horizontality: |dz| / |wrist-shoulder| along the vertical axis. The
        default 0.30 is roughly within 17deg of horizontal.
      - symmetry: |left straightness - right straightness|, to reject a pose
        where only one arm is actually out.
      - torso upright: the pelvis->thorax vector's vertical component, to
        reject a subject who is bending or crouching.
    """
    skeleton_3d = np.asarray(skeleton_3d, dtype=np.float64)
    if skeleton_3d.shape != (17, 3) or not np.isfinite(skeleton_3d).all():
        return False, "no valid skeleton"

    def straightness(shoulder, elbow, wrist):
        upper = np.linalg.norm(skeleton_3d[elbow] - skeleton_3d[shoulder])
        fore = np.linalg.norm(skeleton_3d[wrist] - skeleton_3d[elbow])
        span = np.linalg.norm(skeleton_3d[wrist] - skeleton_3d[shoulder])
        if upper + fore < 1e-8:
            return 0.0, 0.0
        return span / (upper + fore), span

    left_ratio, left_span = straightness(L_SHOULDER, L_ELBOW, L_WRIST)
    right_ratio, right_span = straightness(R_SHOULDER, R_ELBOW, R_WRIST)

    if min(left_ratio, right_ratio) < arm_straightness_min:
        return False, f"arms not straight (L={left_ratio:.2f} R={right_ratio:.2f})"
    if abs(left_ratio - right_ratio) > symmetry_max:
        return False, f"arms asymmetric (L={left_ratio:.2f} R={right_ratio:.2f})"

    for side, shoulder, wrist, span in (("L", L_SHOULDER, L_WRIST, left_span),
                                        ("R", R_SHOULDER, R_WRIST, right_span)):
        if span < 1e-8:
            return False, f"{side} arm collapsed"
        drop = abs(skeleton_3d[wrist][UP] - skeleton_3d[shoulder][UP]) / span
        if drop > arm_horizontality_max:
            return False, f"{side} arm not horizontal (|dz|/span={drop:.2f})"

    torso = skeleton_3d[THORAX] - skeleton_3d[PELVIS]
    torso_norm = float(np.linalg.norm(torso))
    if torso_norm < 1e-8:
        return False, "degenerate torso"
    if abs(torso[UP]) / torso_norm < torso_upright_min:
        return False, "torso not upright"

    return True, "ok"


def _ray_plane_intersection(K, T_world_from_camera, pixel_xy, plane_z):
    """World-frame point where the camera ray through pixel_xy meets the
    horizontal plane z == plane_z, or None if the ray is parallel to it or
    meets it behind the camera."""
    direction_camera = tf.pixel_depth_to_camera_point(K, pixel_xy, 1.0)
    origin = np.asarray(T_world_from_camera, dtype=np.float64)[:3, 3]
    direction = tf.transform_directions(
        T_world_from_camera, direction_camera.reshape(1, 3))[0]
    if abs(direction[2]) < 1e-9:
        return None
    t = (plane_z - origin[2]) / direction[2]
    if t <= 0:
        return None
    return origin + t * direction


def stature_from_ground_plane(K, T_world_from_camera, ground_z, keypoints_2d,
                              keypoints_conf=None, conf_threshold=0.5,
                              nose_height_ratio=NOSE_HEIGHT_RATIO):
    """keypoints_2d: (17, 2) COCO pixels. Returns (stature_m, nose_height_m),
    or (None, None) if the nose or both ankles are missing/low-confidence.

    See the module docstring for the three steps, and for why this does not
    need -- and must not use -- an assumed height."""
    if keypoints_2d is None:
        return None, None
    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float64)
    if keypoints_2d.shape[0] <= COCO_RIGHT_ANKLE:
        return None, None
    if keypoints_conf is None:
        keypoints_conf = np.ones(keypoints_2d.shape[0], dtype=np.float64)

    def usable(index):
        return (keypoints_conf[index] >= conf_threshold
                and np.all(np.isfinite(keypoints_2d[index]))
                and np.any(keypoints_2d[index]))

    ankles = [i for i in (COCO_LEFT_ANKLE, COCO_RIGHT_ANKLE) if usable(i)]
    if not ankles or not usable(COCO_NOSE):
        return None, None

    # Midpoint of whichever ankles are usable: the subject stands between
    # their feet, and the head is above THAT, not above either ankle.
    ankle_px = keypoints_2d[ankles].mean(axis=0)
    ankle_world = _ray_plane_intersection(K, T_world_from_camera, ankle_px, ground_z)
    if ankle_world is None:
        return None, None

    # Depth of the feet; an upright subject's head shares it (module docstring).
    depth = float(tf.world_point_to_camera(T_world_from_camera, ankle_world)[2])
    if not np.isfinite(depth) or depth <= 0:
        return None, None

    nose_camera = tf.pixel_depth_to_camera_point(K, keypoints_2d[COCO_NOSE], depth)
    nose_world = tf.camera_point_to_world(T_world_from_camera, nose_camera)
    nose_height = float(nose_world[2] - ground_z)
    if not np.isfinite(nose_height) or nose_height <= 0:
        return None, None
    return nose_height / nose_height_ratio, nose_height


class TPoseCalibrator:
    """Collects valid T-pose frames across a hold-and-turn capture and
    reduces them to a BodyCalibration.

    Feed every frame to update(); it validates the pose, and only accepted
    frames contribute. Read progress() to drive a countdown/prompt, and
    result() once done. The caller decides when "done" is -- min_frames is
    the floor below which result() refuses, not a stopping rule, because
    more of the turn is strictly better.
    """

    def __init__(self, K=None, T_world_from_camera=None, ground_z=0.0,
                 min_frames=30, subject_id="", **validate_kwargs):
        self.K = K
        self.T_world_from_camera = T_world_from_camera
        self.ground_z = float(ground_z)
        self.min_frames = int(min_frames)
        self.subject_id = subject_id
        self._validate_kwargs = validate_kwargs

        self._bone_samples: dict[tuple[int, int], list[float]] = {}
        self._stature_samples: list[float] = []
        self._nose_samples: list[float] = []
        self._n_accepted = 0
        self._n_rejected = 0
        self._last_reason = "no frames yet"

    def reset(self) -> None:
        self._bone_samples.clear()
        self._stature_samples.clear()
        self._nose_samples.clear()
        self._n_accepted = 0
        self._n_rejected = 0
        self._last_reason = "no frames yet"

    def update(self, skeleton_3d, keypoints_2d=None, keypoints_conf=None) -> dict:
        """skeleton_3d: (17, 3) H36M root-relative, the SAME array the rest
        of the pipeline consumes (post bone-filter is fine -- an unseeded
        filter only rescales toward its own running estimate, it does not
        change bone DIRECTIONS). Returns a status dict for the UI."""
        ok, reason = validate_tpose(skeleton_3d, **self._validate_kwargs)
        self._last_reason = reason
        if not ok:
            self._n_rejected += 1
            return self.progress()

        skeleton_3d = np.asarray(skeleton_3d, dtype=np.float64)
        for parent, child in H36M_BONE_TREE:
            length = float(np.linalg.norm(skeleton_3d[child] - skeleton_3d[parent]))
            if np.isfinite(length) and length > 1e-8:
                self._bone_samples.setdefault((parent, child), []).append(length)

        # Height is opportunistic: it needs calibrated extrinsics AND both a
        # nose and an ankle in view, so it is collected when available rather
        # than being a precondition for the bone-length half to succeed.
        if self.K is not None and self.T_world_from_camera is not None:
            stature, nose_height = stature_from_ground_plane(
                self.K, self.T_world_from_camera, self.ground_z,
                keypoints_2d, keypoints_conf)
            if stature is not None:
                self._stature_samples.append(stature)
                self._nose_samples.append(nose_height)

        self._n_accepted += 1
        return self.progress()

    def progress(self) -> dict:
        return {
            "accepted": self._n_accepted,
            "rejected": self._n_rejected,
            "needed": self.min_frames,
            "fraction": min(1.0, self._n_accepted / max(self.min_frames, 1)),
            "ready": self._n_accepted >= self.min_frames,
            "height_samples": len(self._stature_samples),
            "last_reason": self._last_reason,
        }

    def result(self) -> BodyCalibration | None:
        """Median-reduces the accepted frames. Returns None if too few were
        accepted -- a calibration from 3 frames is worse than no calibration,
        because BoneLengthConstraintFilter's ratio gate will then defend it."""
        if self._n_accepted < self.min_frames:
            return None

        bone_lengths = {edge: float(np.median(samples))
                        for edge, samples in self._bone_samples.items() if samples}
        stature = float(np.median(self._stature_samples)) if self._stature_samples else None
        nose_height = float(np.median(self._nose_samples)) if self._nose_samples else None

        notes = (f"{self._n_accepted} accepted / {self._n_rejected} rejected frames; "
                 f"{len(self._stature_samples)} with a usable ground-plane height")
        if stature is None:
            notes += " (no metric height -- extrinsics missing, or nose/ankles never both visible)"

        return BodyCalibration(
            bone_lengths=bone_lengths, stature_m=stature, nose_height_m=nose_height,
            n_pose_frames=self._n_accepted, n_height_frames=len(self._stature_samples),
            subject_id=self.subject_id, notes=notes)
