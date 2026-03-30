from __future__ import annotations

"""
IEEE P370 data models for the RF de-embedding tool.

This module defines the strongly typed configuration, input, intermediate, and
output containers used by a future P370 2x-thru de-embedding backend.

Design goals
------------
1. Keep the P370 workflow separate from the existing TRL and manual time-gating
   workflow.
2. Make each processing stage explicit and serializable:
   - raw inputs
   - preprocessing outputs
   - raw quality checks
   - fixture split result
   - self-de-embedding checks
   - final DUT result
3. Provide a stable contract for backend engines, validation logic, plotting,
   project persistence, and future UI integration.
4. Support both current single-ended work and future mixed-mode extensions.

Notes
-----
These classes intentionally avoid implementation logic. They are lightweight
containers used by the forthcoming P370 engine and reporting layers.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from .sparameter_data import SParameterData


P370Mode = Literal[
    "SE_NZC_2XTHRU",
    "SE_ZC_2XTHRU",
    "MM_NZC_2XTHRU",
    "MM_ZC_2XTHRU",
]

MidpointDetectionMode = Literal[
    "t12_50pct",
    "tdr_peak",
    "hybrid",
]

DCRectificationMode = Literal[
    "linear",
    "constant",
    "rational",
]

QualityGrade = Literal[
    "good",
    "acceptable",
    "inconclusive",
    "poor",
    "unknown",
]


@dataclass(slots=True)
class P370Config:
    """
    Configuration for IEEE P370 2x-thru de-embedding.

    Parameters
    ----------
    mode:
        Algorithm family to run.

        - ``SE_NZC_2XTHRU``: single-ended, non-impedance-corrected
        - ``SE_ZC_2XTHRU``: single-ended, impedance-corrected
        - ``MM_NZC_2XTHRU``: mixed-mode, non-impedance-corrected
        - ``MM_ZC_2XTHRU``: mixed-mode, impedance-corrected

        For the first implementation, ``SE_ZC_2XTHRU`` is the recommended
        default target because it is the most generally useful path for
        real-world PCB fixtures.

    z_ref_ohm:
        Reference impedance used for quality metrics, TDR conversion, and any
        optional post-de-embedding normalization.

    auto_extrapolate_dc:
        If ``True``, preprocessing may extend the measured frequency response to
        DC before time-domain or step-response calculations.

    dc_extrapolation_mode:
        Strategy used when DC extrapolation is requested.

    enforce_uniform_grid:
        If ``True``, preprocessing may resample inputs onto a common uniform
        frequency grid before any FFT/step/TDR processing.

    midpoint_detection:
        Strategy for locating the fixture midpoint in the 2x-thru structure.
        The IEEE-P370-style default is based on the T12 step-response 50%
        crossing, but an implementation may also add hybrid heuristics.

    reference_plane_offset_left_s / reference_plane_offset_right_s:
        Optional timing offsets applied to move the left and right fixture
        reference planes when the DUT is not physically centered.

    use_impedance_correction:
        Enables impedance-corrected splitting logic when supported by the
        chosen algorithm mode.

    renormalize_after_deembed:
        If ``True``, the resulting DUT may be renormalized to ``z_ref_ohm``
        after de-embedding.

    max_self_residual_db / max_self_phase_deg:
        Acceptance thresholds for the self-de-embedded 2x-thru transparency
        check.

    max_tdr_impedance_error_pct:
        Acceptance threshold for fixture-vs-FIX-DUT-FIX impedance-profile
        similarity checks in the fixture region.

    enable_frequency_domain_qm / enable_time_domain_qm / enable_fer_checks:
        Feature flags to control which validation layers are executed.
    """

    mode: P370Mode = "SE_NZC_2XTHRU"
    z_ref_ohm: float = 50.0
    auto_extrapolate_dc: bool = True
    dc_extrapolation_mode: DCRectificationMode = "linear"
    auto_trim_near_cutoff: bool = False
    cutoff_trim_s21_db: float = -45.0
    cutoff_trim_consecutive_points: int = 3
    min_points_after_trim: int = 64
    enforce_uniform_grid: bool = True
    midpoint_detection: MidpointDetectionMode = "hybrid"
    reference_plane_offset_left_s: float = 0.0
    reference_plane_offset_right_s: float = 0.0
    use_impedance_correction: bool = True
    renormalize_after_deembed: bool = False
    max_self_residual_db: float = 0.1
    max_self_phase_deg: float = 1.0
    max_tdr_impedance_error_pct: float = 2.5
    enable_frequency_domain_qm: bool = True
    enable_time_domain_qm: bool = True
    enable_fer_checks: bool = True


@dataclass(slots=True)
class P370Inputs:
    """
    Raw measurement inputs for a P370 2x-thru de-embedding run.

    Parameters
    ----------
    fix_fix_2xthru:
        Measured 2-port or 4-port network representing the fixture connected to
        itself, i.e. fixture-left + fixture-right.

    fix_dut_fix:
        Measured network containing fixture-left + DUT + fixture-right.

    dut_name:
        Friendly output name assigned to the de-embedded DUT network.

    metadata:
        Freeform storage for run identifiers, operator notes, fixture IDs,
        sample IDs, acquisition conditions, or project-level bookkeeping.
    """

    fix_fix_2xthru: SParameterData
    fix_dut_fix: SParameterData
    dut_name: str = "dut"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class P370PreprocessResult:
    """
    Preprocessed measurement data used by the main algorithm.

    This object captures the output of the preprocessing stage so that later
    checks, reports, and plots can distinguish between:

    - raw measured data
    - uniformized data
    - DC-extrapolated data
    - data actually used by the split/de-embed engine

    Parameters
    ----------
    fix_fix_2xthru:
        Preprocessed 2x-thru network.

    fix_dut_fix:
        Preprocessed FIX-DUT-FIX network.

    dc_added:
        ``True`` if a synthetic or extrapolated DC point was inserted.

    original_start_hz:
        Original measurement start frequency before preprocessing.

    notes:
        Human-readable description of what preprocessing steps were applied.
    """

    fix_fix_2xthru: SParameterData
    fix_dut_fix: SParameterData
    dc_added: bool = False
    trimmed_leading_points: int = 0
    original_start_hz: Optional[float] = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class P370QualityReport:
    """
    Structured quality report for a network under IEEE P370-style checks.

    The fields are intentionally broad enough to support a first native
    implementation as well as future cross-validation against external
    reference libraries.

    Frequency-domain quantities are generally reported as percentage scores and
    optional grades. Time-domain quantities are typically reported as estimated
    distortion in millivolts or another physical scale depending on the chosen
    method.
    """

    subject_name: str
    fd_passivity_pct: Optional[float] = None
    fd_reciprocity_pct: Optional[float] = None
    fd_causality_pct: Optional[float] = None
    fd_passivity_grade: QualityGrade = "unknown"
    fd_reciprocity_grade: QualityGrade = "unknown"
    fd_causality_grade: QualityGrade = "unknown"
    td_passivity_mv: Optional[float] = None
    td_reciprocity_mv: Optional[float] = None
    td_causality_mv: Optional[float] = None
    fer_class: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class P370MidpointResult:
    """
    Result of midpoint detection on the 2x-thru fixture structure.

    Parameters
    ----------
    midpoint_time_s:
        Estimated electrical midpoint of the 2x-thru in seconds.

    left_reference_time_s / right_reference_time_s:
        Derived left and right reference plane positions after optional offsets.

    time_axis_s:
        Time axis used for midpoint determination.

    t12_step_response:
        Step response derived from the transmission path used for midpoint
        detection.

    notes:
        Explanatory notes about which strategy was used and any ambiguity that
        was detected.
    """

    midpoint_time_s: float
    left_reference_time_s: float
    right_reference_time_s: float
    time_axis_s: np.ndarray
    t12_step_response: np.ndarray
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class P370SplitResult:
    """
    Derived left and right fixture error boxes from the 2x-thru measurement.

    Parameters
    ----------
    left_fixture / right_fixture:
        S-parameter models of the fixture halves used for later de-cascade.

    midpoint:
        Midpoint-detection result used to define the split.

    tdr_left_ohm / tdr_right_ohm:
        Optional TDR-style impedance profiles for engineering review and
        similarity checks.

    algorithm_name:
        Human-readable identifier for the splitting algorithm variant that was
        used.

    notes:
        Additional details such as assumptions, warnings, or fallback choices.
    """

    left_fixture: SParameterData
    right_fixture: SParameterData
    midpoint: P370MidpointResult
    tdr_left_ohm: Optional[np.ndarray] = None
    tdr_right_ohm: Optional[np.ndarray] = None
    algorithm_name: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class P370SelfCheck:
    """
    Self-de-embedding verification for the derived fixture halves.

    The 2x-thru should de-embed to an almost transparent network if the split is
    correct. This is a core acceptance check for a trustworthy P370 run.
    """

    self_deembedded_2xthru: SParameterData
    residual_mag_db: np.ndarray
    residual_phase_deg: np.ndarray
    max_abs_mag_db: float
    max_abs_phase_deg: float
    tdr_match_error_pct: Optional[np.ndarray] = None
    passed: bool = False
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class P370Result:
    """
    Full output bundle from a P370 2x-thru de-embedding run.

    This object is intended to be the primary return value from
    :class:`rfdeembed.p370_2xthru.P3702xThruDeembedder`.
    """

    config: P370Config
    inputs: P370Inputs
    preprocess: Optional[P370PreprocessResult] = None
    raw_quality_2xthru: Optional[P370QualityReport] = None
    raw_quality_fix_dut_fix: Optional[P370QualityReport] = None
    split: Optional[P370SplitResult] = None
    self_check: Optional[P370SelfCheck] = None
    deembedded_dut: Optional[SParameterData] = None
    deembedded_quality: Optional[P370QualityReport] = None
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
