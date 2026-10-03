"""Typed, YAML-backed configuration for the pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from football_tracker.homography.pitch import PitchSpec


class IOConfig(BaseModel):
    """Video input/output and frame-sampling options."""

    input_video: Path | None = None
    output_dir: Path = Path("outputs")

    # Letterbox handling: auto-detect black bars, or crop manually when disabled.
    auto_letterbox: bool = True
    letterbox_threshold: float = 20.0
    crop_top: int = Field(0, ge=0)
    crop_bottom: int = Field(0, ge=0)
    crop_left: int = Field(0, ge=0)
    crop_right: int = Field(0, ge=0)

    # Frame range, applied after decoding: [start, stop) with a stride.
    frame_start: int = 0
    frame_stop: int | None = None
    frame_stride: int = 1


class DetectionConfig(BaseModel):
    """Object-detection model + inference thresholds."""

    weights: Path | None = None
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    ball_conf: float | None = None
    # Off, class-aware NMS keeps a `player` box and a `referee` box on one person.
    agnostic_nms: bool = True


class HomographyConfig(BaseModel):
    """Pitch calibration: anchor smoothing and the pitch mask.

    The solve gates and propagation budget are deliberately absent. They were carried
    over from the removed keypoint path, nothing ever read them, and the values here
    disagreed with the ones actually in force (15.0 vs 10.0 RANSAC, 8.0 vs 12.0 reproj,
    60 vs 240 propagated frames) -- so they read as tuning knobs while doing nothing.
    Those thresholds live as constants in `homography.fuse`, which is the single
    place they apply.
    """

    mask_margin_m: float = 1.0


class TrackingConfig(BaseModel):
    """Multi-object player tracking: backend, camera-motion source, gates."""

    # Set per clip; the calibrate stage writes it. No default, because a path left over
    # from another run loads silently and measures fine -- a wrong one is invisible.
    calibration_path: Path | None = None
    mask_margin_m: float = 1.0
    mask_margin_referee_m: float = 3.0
    min_conf: float = 0.1
    # BoT-SORT association gates (BoxMOT defaults, surfaced for tuning).
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer: int = 30


class TeamIDConfig(BaseModel):
    """Team + role assignment from torso colour.

    Role is a colour-outlier test taken before team assignment. The outlier radius is
    fitted (median + k*MAD) rather than hardcoded, so it survives a clip whose lighting
    differs; the ``gk_zone_*`` fields supply the geometry that separates goalkeepers
    from referees, which colour alone does not.
    """

    enabled: bool = True
    palette_path: Path | None = None  # None = fit in-run from this clip's detections
    # Torso band: below the head, above the shorts, inside the box's horizontal centre
    # so a neighbouring player's shoulder is less likely to intrude.
    torso_top: float = 0.20  # band start, fraction of box height
    torso_bottom: float = 0.50
    torso_width: float = 0.60  # central fraction of box width
    # Per-sample quality gates. A contaminated sample is rejected, not down-weighted:
    # the per-track median is robust to a few bad samples but the palette fit is not.
    min_kept_fraction: float = 0.35
    min_box_h_px: float = 24.0  # below this the band is <8 rows and the median is noise
    grass_hue_lo: float = 30.0  # OpenCV HSV hue (0-179)
    grass_hue_hi: float = 90.0
    grass_sat_min: float = 50.0
    shadow_l_min: float = 20.0  # Lab L gates (OpenCV 0-255); cast shadows here are deep
    highlight_l_max: float = 250.0
    # Palette fit. The separability floor makes the fit fail loudly: a clip whose two
    # kits are genuinely similar would otherwise fit a confident *wrong* palette that
    # every downstream consumer inherits silently.
    min_silhouette: float = 0.5
    reject_k_mad: float = 3.5  # reject_radius = median + k * MAD of within-cluster dist
    min_reject_radius: float = 20.0  # floor, for when a cluster is tiny
    min_samples_fit: int = 5  # accepted samples before a track can inform the fit
    # Lifetime gate, keeping off-pitch people out of the fit: a high-vis steward on the
    # touchline would pollute the goalkeeper centroid silently.
    min_track_frames_fit: int = 15
    # Assignment.
    min_samples_assign: int = 3  # below this a track is `unknown`, never guessed
    min_margin: float = 1.25  # nearest centroid must beat runner-up by this ratio
    # Role resolution (see reid/roles.py).
    gk_zone_m: float = 20.0  # goal-zone depth measured from each goal line
    gk_zone_frac: float = 0.6  # fraction of a track's life inside it to qualify as GK
    max_goalkeepers: int = 2


class StitchConfig(BaseModel):
    """Offline tracklet stitching.

    The online tracker cannot look forward, so it fragments. This pass merges, using
    constraints the online tracker structurally cannot have: temporal-overlap exclusion,
    team/role equality, and roster cardinality.
    """

    max_gap_s: float = 3.0  # beyond this the reachability circle exceeds the pitch
    v_max_ms: float = 9.0  # sprint speed; the reachability radius is v_max * dt + sigma
    pos_sigma_m: float = 1.6  # position tolerance: calibration plus foot-point error
    w_pos: float = 1.0  # geometry dominates by construction
    w_time: float = 0.2  # a shorter gap is better evidence, but only mildly
    max_link_cost: float = 0.6  # above this, mint a new id rather than merge wrongly
    roster_outfield: int = 11  # reported, never enforced: merging cannot violate it
    # Durations, not frame counts: a frame count silently means a different duration on
    # 25 fps footage, and that failure would change behaviour without raising.
    min_tracklet_s: float = 0.08  # below this a fragment has no reliable velocity
    velocity_window_s: float = 0.08  # averaging window for the tail velocity estimate
    # A team label may veto a merge only when both sides are confident; vote confidence
    # is bimodal, so this threshold separates a trusted label from a weak one.
    role_trust_conf: float = 0.90
    w_role_mismatch: float = 0.2  # linking across a weak label is allowed, not free
    short_fragment_max_gap_s: float = 0.25  # a fragment too short to have a velocity
    overlap_tolerance_s: float = 0.1  # a brief handover is not "two places at once"


class BallConfig(BaseModel):
    """Ball sub-system: ROI re-detection, Kalman, global path, gap handling.

    Runs in two stages. Per-frame: candidates from the full-frame detector merged with
    a native-resolution ROI re-detection around a GMC-compensated Kalman prediction.
    Offline: a min-cost path over all candidates, then interpolation and airborne
    flagging. Shares ``tracking.calibration_path``.
    """

    enabled: bool = True
    # None = share `detection.weights` (one model load, two threshold configs). Set it
    # when the ball needs its own model: a ball-only fine-tune detects no players, so it
    # can never be `detection.weights`, and leaving this unset then runs the ball off the
    # player detector at a large recall cost with nothing raised.
    weights: Path | None = None
    # ROI re-detection around the Kalman prediction, at native resolution.
    roi_size: int = 640
    roi_imgsz: int = 640
    roi_conf: float = 0.10
    roi_max_coast: int = 45
    merge_dist_px: float = 20.0
    # Candidate gates.
    mask_margin_m: float = 6.0
    min_box_px: float = 4.0
    max_box_px: float = 40.0
    # Kalman (constant-velocity, dt = 1 frame).
    q_pos: float = 1.0
    q_vel: float = 0.5
    r_px: float = 5.0
    gate_chi2: float = 9.21
    # Global path (offline solve).
    max_gap: int = 60
    max_speed_px: float = 40.0
    miss_cost: float = 0.3
    conf_weight: float = 0.2
    speed_weight: float = 0.1
    # Gap interpolation / airborne heuristic.
    max_interp_gap: int = 30
    airborne_window: int = 31
    airborne_min_frames: int = 18
    airborne_curvature_min: float = 0.05
    airborne_residual_px: float = 3.0
    airborne_speed_ms: float = 25.0


_PITCH_LIMITS = {"length": (100.0, 110.0), "width": (64.0, 75.0)}


class PitchConfig(BaseModel):
    """Pitch dimensions in metres.

    Validated, because a wrong value here does not error: it silently rescales every
    projected position, speed, possession radius, and the radar.
    """

    length: float = 105.0
    width: float = 68.0

    @field_validator("length", "width")
    @classmethod
    def _within_the_laws(cls, v: float, info: ValidationInfo) -> float:
        lo, hi = _PITCH_LIMITS[info.field_name]
        if not lo <= v <= hi:
            default = cls.model_fields[info.field_name].default
            raise ValueError(
                f"pitch.{info.field_name} {v} m is outside the Laws of the Game range for "
                f"competitive matches ({lo:g}-{hi:g} m). Set it from the venue, or leave "
                f"the {default} default."
            )
        return v

    def to_spec(self) -> PitchSpec:
        """The geometry object every consumer of pitch coordinates takes."""
        return PitchSpec(length=self.length, width=self.width)


class PossessionConfig(BaseModel):
    """Team-possession attribution. Frames, not seconds, except where noted."""

    radius_m: float = 2.0
    acquire_frames: int = 12
    max_ball_speed_mps: float = 10.0
    max_unknown_s: float = 2.0


class MatchConfig(BaseModel):
    """Display names for the console summary; team_a is the kit the players stage names."""

    team_a_name: str = "Team A"
    team_b_name: str = "Team B"


class ProjectionConfig(BaseModel):
    """Pitch-coordinate projection, smoothing, and radar geometry."""

    smooth_window: int = 15
    fps: float = 60.0
    scale_px_per_m: float = 10.0
    margin_px: int = 40
    possession: PossessionConfig = Field(default_factory=PossessionConfig)


class Config(BaseModel):
    """Top-level run configuration."""

    device: str = "auto"  # auto | cpu | mps | cuda
    io: IOConfig = Field(default_factory=IOConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    homography: HomographyConfig = Field(default_factory=HomographyConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    teamid: TeamIDConfig = Field(default_factory=TeamIDConfig)
    stitch: StitchConfig = Field(default_factory=StitchConfig)
    ball: BallConfig = Field(default_factory=BallConfig)
    pitch: PitchConfig = Field(default_factory=PitchConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    match: MatchConfig | None = None  # required by possession; never defaulted

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a Config from a YAML file (missing keys fall back to defaults)."""
        with Path(path).open() as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)
