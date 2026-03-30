from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .p370_models import (
    P370Config,
    P370Inputs,
    P370MidpointResult,
    P370PreprocessResult,
    P370Result,
    P370SelfCheck,
    P370SplitResult,
)
from .p370_quality import P370QualityChecks
from .sparameter_data import SParameterData

try:  # pragma: no cover - optional dependency path
    from skrf import Frequency, Network
    from skrf.calibration.deembedding import IEEEP370_SE_NZC_2xThru

    SKRF_AVAILABLE = True
except Exception:  # pragma: no cover - keep native fallback import-safe
    Frequency = None
    Network = None
    IEEEP370_SE_NZC_2xThru = None
    SKRF_AVAILABLE = False


class P370Provider(Protocol):
    def deembed(self, inputs: P370Inputs, cfg: Optional[P370Config] = None) -> P370Result:
        ...


@dataclass(slots=True)
class P370DebugArtifacts:
    freq_hz: Optional[np.ndarray] = None
    time_axis_s: Optional[np.ndarray] = None
    t11_step: Optional[np.ndarray] = None
    t12_step: Optional[np.ndarray] = None
    midpoint_gate: Optional[np.ndarray] = None
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


class P3702xThruDeembedder:
    """IEEE P370 2x-thru de-embedding engine with waveguide-safe guards.

    Preferred path:
    - use scikit-rf's IEEE P370 SE_NZC_2xThru reference implementation for the
      actual split and de-embedding;
    - add preprocessing and validation guards for high-frequency/non-TEM data,
      especially D-band waveguide where near-cutoff samples and synthetic DC can
      destabilize the split.

    Fallback path:
    - when scikit-rf is unavailable, keep the deterministic matrix-square-root
      split as a non-reference approximation so the app remains usable.
    """

    def __init__(self, quality_checker: Optional[P370QualityChecks] = None):
        self.quality_checker = quality_checker or P370QualityChecks()

    # ------------------------------------------------------------------
    # Public workflow
    # ------------------------------------------------------------------
    def deembed(self, inputs: P370Inputs, cfg: Optional[P370Config] = None) -> P370Result:
        cfg = cfg or P370Config()
        self._validate_supported_mode(inputs, cfg)

        pre = self.preprocess(inputs, cfg)
        raw_quality_2xthru, raw_quality_fix_dut_fix = self.quality_check_inputs(pre, cfg)
        split = self.split_fixtures(pre, cfg)
        self_check = self.self_deembed_check(split, pre, cfg)
        dut = self.deembed_fix_dut_fix(
            pre.fix_dut_fix,
            split.left_fixture,
            split.right_fixture,
            inputs.dut_name,
        )
        dut = self._postprocess_dut(dut, cfg)
        dut_quality = self.quality_checker.build_report(dut, cfg)

        left_t_axis, _ = self._tdr_impedance_from_sii(split.left_fixture, 0)
        right_t_axis, _ = self._tdr_impedance_from_sii(split.right_fixture, 1)
        artifacts = {
            "midpoint_time_s": split.midpoint.midpoint_time_s,
            "midpoint_time_axis_s": split.midpoint.time_axis_s,
            "midpoint_step_response": split.midpoint.t12_step_response,
            "left_tdr_ohm": split.tdr_left_ohm,
            "right_tdr_ohm": split.tdr_right_ohm,
            "left_tdr_time_axis_s": left_t_axis,
            "right_tdr_time_axis_s": right_t_axis,
            "tdr_match_error_pct": self_check.tdr_match_error_pct,
            "trimmed_leading_points": pre.trimmed_leading_points,
        }

        notes = [
            "IEEE P370 workflow completed",
            split.algorithm_name,
            *pre.notes,
            *split.notes,
            *self_check.warnings,
        ]

        return P370Result(
            config=cfg,
            inputs=inputs,
            preprocess=pre,
            raw_quality_2xthru=raw_quality_2xthru,
            raw_quality_fix_dut_fix=raw_quality_fix_dut_fix,
            split=split,
            self_check=self_check,
            deembedded_dut=dut,
            deembedded_quality=dut_quality,
            notes=notes,
            artifacts=artifacts,
        )

    def preprocess(self, inputs: P370Inputs, cfg: Optional[P370Config] = None) -> P370PreprocessResult:
        cfg = cfg or P370Config()
        self._validate_supported_mode(inputs, cfg)
        self._assert_finite_network(inputs.fix_fix_2xthru, stage="Input validation", role="2x-thru")
        self._assert_finite_network(inputs.fix_dut_fix, stage="Input validation", role="FIX-DUT-FIX")

        fix_fix, fix_dut = self._ensure_same_grid(inputs.fix_fix_2xthru, inputs.fix_dut_fix)
        self._assert_finite_network(fix_fix, stage="After grid reconciliation", role="2x-thru")
        self._assert_finite_network(fix_dut, stage="After grid reconciliation", role="FIX-DUT-FIX")

        notes: list[str] = []
        original_start_hz = float(min(fix_fix.freq_hz[0], fix_dut.freq_hz[0]))
        dc_added = False
        trimmed_points = 0

        if cfg.auto_trim_near_cutoff:
            fix_fix, fix_dut, trimmed_points, trim_note = self._trim_near_cutoff_pair(fix_fix, fix_dut, cfg)
            if trim_note:
                notes.append(trim_note)
            self._assert_finite_network(fix_fix, stage="After cutoff trimming", role="2x-thru")
            self._assert_finite_network(fix_dut, stage="After cutoff trimming", role="FIX-DUT-FIX")

        if cfg.auto_extrapolate_dc:
            if fix_fix.freq_hz[0] > 0.0:
                fix_fix = self._extrapolate_dc(fix_fix, cfg)
                dc_added = True
                notes.append(f"Added DC point to 2x-thru using {cfg.dc_extrapolation_mode} extrapolation")
                self._assert_finite_network(fix_fix, stage="After DC extrapolation", role="2x-thru")
            if fix_dut.freq_hz[0] > 0.0:
                fix_dut = self._extrapolate_dc(fix_dut, cfg)
                dc_added = True
                notes.append(f"Added DC point to FIX-DUT-FIX using {cfg.dc_extrapolation_mode} extrapolation")
                self._assert_finite_network(fix_dut, stage="After DC extrapolation", role="FIX-DUT-FIX")
        else:
            notes.append("DC extrapolation disabled for this P370 run")

        if cfg.enforce_uniform_grid:
            df = np.diff(fix_fix.freq_hz)
            spread = float(np.max(np.abs(df - np.mean(df)))) if len(df) else 0.0
            if spread > max(1e-6, 1e-6 * np.mean(df)):
                freq = np.linspace(fix_fix.freq_hz[0], fix_fix.freq_hz[-1], len(fix_fix.freq_hz))
                fix_fix = fix_fix.interpolate_to(freq, name=fix_fix.name)
                fix_dut = fix_dut.interpolate_to(freq, name=fix_dut.name)
                notes.append("Resampled both networks onto a uniform frequency grid")
                self._assert_finite_network(fix_fix, stage="After uniform-grid resampling", role="2x-thru")
                self._assert_finite_network(fix_dut, stage="After uniform-grid resampling", role="FIX-DUT-FIX")

        return P370PreprocessResult(
            fix_fix_2xthru=fix_fix,
            fix_dut_fix=fix_dut,
            dc_added=dc_added,
            trimmed_leading_points=trimmed_points,
            original_start_hz=original_start_hz,
            notes=notes,
        )

    def quality_check_inputs(
        self,
        pre: P370PreprocessResult,
        cfg: Optional[P370Config] = None,
    ) -> tuple:
        cfg = cfg or P370Config()
        return (
            self.quality_checker.build_report(pre.fix_fix_2xthru, cfg),
            self.quality_checker.build_report(pre.fix_dut_fix, cfg),
        )

    def split_fixtures(
        self,
        pre: P370PreprocessResult,
        cfg: Optional[P370Config] = None,
    ) -> P370SplitResult:
        cfg = cfg or P370Config()
        zc_modes = {"SE_ZC_2XTHRU", "MM_ZC_2XTHRU"}
        if cfg.mode in zc_modes:
            return self.split_fixtures_zc(pre.fix_fix_2xthru, cfg)
        return self.split_fixtures_nzc(pre.fix_fix_2xthru, cfg)

    def split_fixtures_nzc(
        self,
        fix_fix_2xthru: SParameterData,
        cfg: Optional[P370Config] = None,
    ) -> P370SplitResult:
        cfg = cfg or P370Config()
        fix_fix_2xthru.check_2port()
        self._assert_finite_network(fix_fix_2xthru, stage="Split input", role="2x-thru")

        midpoint = self.detect_midpoint(fix_fix_2xthru, cfg)
        notes: list[str] = []

        use_skrf, skrf_reason = self._should_use_skrf_split(fix_fix_2xthru)
        if use_skrf:
            left_fixture, right_fixture, skrf_notes = self._build_fixture_error_boxes_skrf(fix_fix_2xthru, cfg)
            notes.extend(skrf_notes)
            algorithm_name = "SE_NZC_2XTHRU (scikit-rf IEEE P370 split)"
        else:
            left_fixture, right_fixture = self._build_fixture_error_boxes_fallback(fix_fix_2xthru, midpoint, cfg)
            notes.append(skrf_reason)
            algorithm_name = "SE_NZC_2XTHRU (waveguide-safe fallback split)"

        _, z_tdr_left = self._tdr_impedance_from_sii(left_fixture, 0)
        _, z_tdr_right = self._tdr_impedance_from_sii(right_fixture, 1)
        self._assert_finite_array(z_tdr_left, stage="TDR extraction", detail="left fixture impedance profile")
        self._assert_finite_array(z_tdr_right, stage="TDR extraction", detail="right fixture impedance profile")

        notes.append(f"Midpoint estimate for validation window: {midpoint.midpoint_time_s:.3e} s")
        if abs(cfg.reference_plane_offset_left_s) > 0.0 or abs(cfg.reference_plane_offset_right_s) > 0.0:
            notes.append("Reference-plane offsets applied as ideal matched delay sections on split fixtures")

        return P370SplitResult(
            left_fixture=left_fixture,
            right_fixture=right_fixture,
            midpoint=midpoint,
            tdr_left_ohm=z_tdr_left,
            tdr_right_ohm=z_tdr_right,
            algorithm_name=algorithm_name,
            notes=notes,
        )

    def split_fixtures_zc(
        self,
        fix_fix_2xthru: SParameterData,
        cfg: Optional[P370Config] = None,
    ) -> P370SplitResult:
        cfg = cfg or P370Config()
        split = self.split_fixtures_nzc(fix_fix_2xthru, cfg)
        split.algorithm_name = "SE_ZC_2XTHRU (delegated to NZC split in this build)"
        split.notes.append(
            "This build implements the full IEEE P370 NZC path; ZC mode currently reuses the NZC engine."
        )
        return split

    def detect_midpoint(
        self,
        fix_fix_2xthru: SParameterData,
        cfg: Optional[P370Config] = None,
    ) -> P370MidpointResult:
        cfg = cfg or P370Config()
        fix_fix_2xthru.check_2port()

        time_axis_s, step, impulse, time_axis_centered_s = self._time_domain_step_and_impulse(fix_fix_2xthru, cfg)
        self._assert_finite_array(step, stage="Midpoint detection", detail="transmission step response")
        self._assert_finite_array(impulse, stage="Midpoint detection", detail="transmission impulse response")

        step_real = np.real(step)
        impulse_abs = np.abs(impulse)
        notes: list[str] = []

        peak_idx = int(np.argmax(impulse_abs))
        total_delay_peak_s = float(max(0.0, time_axis_centered_s[peak_idx]))
        midpoint_from_peak_s = 0.5 * total_delay_peak_s
        notes.append("Midpoint candidate from T21 impulse peak (IEEE P370 NZC timing heuristic)")
        if not cfg.auto_extrapolate_dc and fix_fix_2xthru.freq_hz[0] > 0.0:
            notes.append("No DC extrapolation: midpoint preview uses a zero-DC placeholder for local timing estimates")

        s21 = fix_fix_2xthru.s[:, 1, 0]
        w = 2.0 * np.pi * fix_fix_2xthru.freq_hz
        gd = -np.gradient(np.unwrap(np.angle(s21)), w)
        gd_total_s = float(np.median(gd[np.isfinite(gd)])) if np.any(np.isfinite(gd)) else total_delay_peak_s
        gd_mid_s = 0.5 * gd_total_s

        if len(step_real) >= 4:
            low = float(np.mean(step_real[: max(3, len(step_real) // 50)]))
            high = float(np.mean(step_real[-max(3, len(step_real) // 50) :]))
            if abs(high - low) < 1e-15:
                midpoint_from_step_s = midpoint_from_peak_s
                notes.append("Transmission step response had very small swing; impulse-peak midpoint retained")
            else:
                target = low + 0.5 * (high - low)
                idx = int(np.argmin(np.abs(step_real - target)))
                midpoint_from_step_s = 0.5 * float(time_axis_s[idx])
                notes.append("Midpoint candidate from 50% crossing of transmission step response")
        else:
            midpoint_from_step_s = midpoint_from_peak_s
            notes.append("Insufficient step samples; impulse-peak midpoint retained")

        mode = cfg.midpoint_detection
        if mode == "t12_50pct":
            midpoint_time_s = midpoint_from_step_s
        elif mode == "tdr_peak":
            midpoint_time_s = midpoint_from_peak_s
        else:
            midpoint_time_s = float(np.mean([midpoint_from_peak_s, midpoint_from_step_s, gd_mid_s]))
            notes.append("Hybrid midpoint used average of impulse-peak, step-50%, and group-delay estimates")

        left_reference_time_s = float(midpoint_time_s + cfg.reference_plane_offset_left_s)
        right_reference_time_s = float(midpoint_time_s + cfg.reference_plane_offset_right_s)

        return P370MidpointResult(
            midpoint_time_s=float(midpoint_time_s),
            left_reference_time_s=left_reference_time_s,
            right_reference_time_s=right_reference_time_s,
            time_axis_s=time_axis_s,
            t12_step_response=step_real,
            notes=notes,
        )

    def self_deembed_check(
        self,
        split: P370SplitResult,
        pre: P370PreprocessResult,
        cfg: Optional[P370Config] = None,
    ) -> P370SelfCheck:
        cfg = cfg or P370Config()
        recovered = self.deembed_fix_dut_fix(
            pre.fix_fix_2xthru,
            split.left_fixture,
            split.right_fixture,
            dut_name="self_deembedded_2xthru",
        )
        self._assert_finite_network(recovered, stage="Self de-embedding", role="self-deembedded 2x-thru")

        s21 = recovered.s[:, 1, 0]
        s11 = recovered.s[:, 0, 0]
        s22 = recovered.s[:, 1, 1]
        s12 = recovered.s[:, 0, 1]

        residual_mag_db = 20.0 * np.log10(np.abs(s21) + 1e-15)
        residual_phase_deg = np.rad2deg(np.unwrap(np.angle(s21)))
        self._assert_finite_array(residual_mag_db, stage="Self de-embedding", detail="residual magnitude trace")
        self._assert_finite_array(residual_phase_deg, stage="Self de-embedding", detail="residual phase trace")
        max_abs_mag_db = float(np.max(np.abs(residual_mag_db)))
        max_abs_phase_deg = float(np.max(np.abs(residual_phase_deg)))

        tdr_match_error_pct = self.compare_fixture_tdr_similarity(split, pre, cfg)
        self._assert_finite_array(tdr_match_error_pct, stage="Self de-embedding", detail="fixture TDR mismatch trace")
        max_tdr_error_pct = float(np.max(tdr_match_error_pct)) if tdr_match_error_pct.size else 0.0

        reflection_error_db = 20.0 * np.log10(np.maximum(np.abs(s11), np.abs(s22)) + 1e-15)
        reverse_mag_db = 20.0 * np.log10(np.abs(s12) + 1e-15)

        passed = (
            (max_abs_mag_db <= cfg.max_self_residual_db)
            and (max_abs_phase_deg <= cfg.max_self_phase_deg)
            and (max_tdr_error_pct <= cfg.max_tdr_impedance_error_pct)
        )
        warnings: list[str] = []
        if max_abs_mag_db > cfg.max_self_residual_db:
            warnings.append(
                f"Self-check magnitude residual {max_abs_mag_db:.3f} dB exceeds limit {cfg.max_self_residual_db:.3f} dB"
            )
        if max_abs_phase_deg > cfg.max_self_phase_deg:
            warnings.append(
                f"Self-check phase residual {max_abs_phase_deg:.3f}° exceeds limit {cfg.max_self_phase_deg:.3f}°"
            )
        if max_tdr_error_pct > cfg.max_tdr_impedance_error_pct:
            warnings.append(
                f"Fixture TDR mismatch {max_tdr_error_pct:.2f}% exceeds limit {cfg.max_tdr_impedance_error_pct:.2f}%"
            )

        return P370SelfCheck(
            self_deembedded_2xthru=recovered,
            residual_mag_db=residual_mag_db,
            residual_phase_deg=residual_phase_deg,
            max_abs_mag_db=max_abs_mag_db,
            max_abs_phase_deg=max_abs_phase_deg,
            tdr_match_error_pct=tdr_match_error_pct,
            passed=passed,
            warnings=warnings,
            details={
                "max_reflection_residual_db": float(np.max(reflection_error_db)),
                "max_reverse_transmission_residual_db": float(np.max(np.abs(reverse_mag_db))),
                "max_tdr_match_error_pct": max_tdr_error_pct,
            },
        )

    def compare_fixture_tdr_similarity(
        self,
        split: P370SplitResult,
        pre: P370PreprocessResult,
        cfg: Optional[P370Config] = None,
    ) -> np.ndarray:
        cfg = cfg or P370Config()
        left_t, left_tdr = self._tdr_impedance_from_sii(split.left_fixture, 0)
        full_left_t, full_left_tdr = self._tdr_impedance_from_sii(pre.fix_dut_fix, 0)
        right_t, right_tdr = self._tdr_impedance_from_sii(split.right_fixture, 1)
        full_right_t, full_right_tdr = self._tdr_impedance_from_sii(pre.fix_dut_fix, 1)

        left_count = self._window_sample_count(left_t, full_left_t, split.midpoint.left_reference_time_s)
        right_count = self._window_sample_count(right_t, full_right_t, split.midpoint.right_reference_time_s)
        ref = max(float(cfg.z_ref_ohm), 1e-12)

        left_n = min(left_count, len(left_tdr), len(full_left_tdr))
        right_n = min(right_count, len(right_tdr), len(full_right_tdr))
        left_err = 100.0 * np.abs(full_left_tdr[:left_n] - left_tdr[:left_n]) / ref if left_n else np.array([], dtype=float)
        right_err = 100.0 * np.abs(full_right_tdr[:right_n] - right_tdr[:right_n]) / ref if right_n else np.array([], dtype=float)
        if left_err.size and right_err.size:
            return np.concatenate([left_err, right_err])
        if left_err.size:
            return left_err
        if right_err.size:
            return right_err
        return np.array([], dtype=float)

    def deembed_fix_dut_fix(
        self,
        fix_dut_fix: SParameterData,
        left_fixture: SParameterData,
        right_fixture: SParameterData,
        dut_name: str,
    ) -> SParameterData:
        fix_dut_fix.check_2port()
        left_fixture.check_2port()
        right_fixture.check_2port()
        self._assert_finite_network(fix_dut_fix, stage="De-embedding input", role="FIX-DUT-FIX")
        self._assert_finite_network(left_fixture, stage="De-embedding input", role="left fixture")
        self._assert_finite_network(right_fixture, stage="De-embedding input", role="right fixture")

        left_fixture, fix_dut_fix = self._ensure_same_grid(left_fixture, fix_dut_fix)
        right_fixture, fix_dut_fix = self._ensure_same_grid(right_fixture, fix_dut_fix)

        if SKRF_AVAILABLE:
            try:
                total = self._to_skrf_network(fix_dut_fix, name=fix_dut_fix.name)
                left = self._to_skrf_network(left_fixture, name=left_fixture.name)
                right = self._to_skrf_network(right_fixture, name=right_fixture.name)
                dut = left.inv ** total ** right.inv
                out = self._from_skrf_network(dut, name=dut_name)
                self._assert_finite_network(out, stage="De-embedding output", role=dut_name)
                return out
            except Exception as exc:
                raise ValueError(
                    "IEEE P370 de-embedding failed while de-cascading the extracted fixtures. "
                    f"Underlying error: {exc}"
                ) from exc

        total_abcd = fix_dut_fix.s_to_abcd()
        left_abcd = left_fixture.s_to_abcd()
        right_abcd = right_fixture.s_to_abcd()
        dut_abcd = SParameterData.decascade_abcd(total_abcd, left_abcd, right_abcd)
        out = SParameterData.from_abcd(
            fix_dut_fix.freq_hz,
            dut_abcd,
            z0=fix_dut_fix.z0,
            name=dut_name,
        )
        self._assert_finite_network(out, stage="De-embedding output", role=dut_name)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_supported_mode(self, inputs: P370Inputs, cfg: P370Config) -> None:
        if "SE_" in cfg.mode:
            if inputs.fix_fix_2xthru.n_ports != 2 or inputs.fix_dut_fix.n_ports != 2:
                raise ValueError("Single-ended P370 modes require 2-port networks")
        elif "MM_" in cfg.mode:
            raise NotImplementedError("Mixed-mode P370 is not implemented in this backend yet")
        else:
            raise ValueError(f"Unsupported P370 mode: {cfg.mode}")

    def _ensure_same_grid(self, a: SParameterData, b: SParameterData) -> tuple[SParameterData, SParameterData]:
        if a.same_grid_as(b):
            return a, b

        start = max(float(a.freq_hz[0]), float(b.freq_hz[0]))
        stop = min(float(a.freq_hz[-1]), float(b.freq_hz[-1]))
        if stop <= start:
            raise ValueError(f"Frequency ranges for '{a.name}' and '{b.name}' do not overlap")

        n = min(a.n_freq, b.n_freq)
        if n < 2:
            raise ValueError("Need at least two common frequency samples")
        common_freq = np.linspace(start, stop, n)
        out_a = a.interpolate_to(common_freq, name=a.name)
        out_b = b.interpolate_to(common_freq, name=b.name)
        return out_a, out_b

    def _trim_near_cutoff_pair(
        self,
        fix_fix: SParameterData,
        fix_dut: SParameterData,
        cfg: P370Config,
    ) -> tuple[SParameterData, SParameterData, int, str]:
        if fix_fix.n_freq <= cfg.min_points_after_trim:
            return fix_fix, fix_dut, 0, ""

        thr_db = float(cfg.cutoff_trim_s21_db)
        consec = max(1, int(cfg.cutoff_trim_consecutive_points))
        ff_s21_db = 20.0 * np.log10(np.abs(fix_fix.s[:, 1, 0]) + 1e-30)
        ff_s12_db = 20.0 * np.log10(np.abs(fix_fix.s[:, 0, 1]) + 1e-30)
        fd_s21_db = 20.0 * np.log10(np.abs(fix_dut.s[:, 1, 0]) + 1e-30)

        good = (
            np.isfinite(ff_s21_db)
            & np.isfinite(ff_s12_db)
            & np.isfinite(fd_s21_db)
            & (ff_s21_db >= thr_db)
            & (ff_s12_db >= thr_db)
            & (fd_s21_db >= thr_db)
        )
        start_idx = 0
        for i in range(max(1, len(good) - consec + 1)):
            if np.all(good[i : i + consec]):
                start_idx = i
                break

        if start_idx <= 0:
            return fix_fix, fix_dut, 0, ""
        if fix_fix.n_freq - start_idx < int(cfg.min_points_after_trim):
            raise ValueError(
                "Near-cutoff trimming would leave too few points for IEEE P370. "
                f"Required at least {cfg.min_points_after_trim}, but only {fix_fix.n_freq - start_idx} would remain."
            )

        trim_freq_ghz = fix_fix.freq_hz[start_idx] / 1e9
        note = (
            f"Trimmed {start_idx} leading frequency points below the near-cutoff threshold "
            f"({thr_db:.1f} dB); new start frequency is {trim_freq_ghz:.3f} GHz"
        )
        trim_fix = SParameterData(
            freq_hz=fix_fix.freq_hz[start_idx:].copy(),
            s=fix_fix.s[start_idx:].copy(),
            z0=fix_fix.z0,
            name=fix_fix.name,
            file_path=fix_fix.file_path,
            metadata=dict(fix_fix.metadata),
        )
        trim_dut = SParameterData(
            freq_hz=fix_dut.freq_hz[start_idx:].copy(),
            s=fix_dut.s[start_idx:].copy(),
            z0=fix_dut.z0,
            name=fix_dut.name,
            file_path=fix_dut.file_path,
            metadata=dict(fix_dut.metadata),
        )
        return trim_fix, trim_dut, start_idx, note

    def _extrapolate_dc(self, ntwk: SParameterData, cfg: P370Config) -> SParameterData:
        if ntwk.freq_hz[0] <= 0.0:
            return ntwk
        if ntwk.n_freq < 2:
            raise ValueError("Need at least two points to extrapolate a DC sample")

        f0 = ntwk.freq_hz[0]
        f1 = ntwk.freq_hz[1]
        s0 = ntwk.s[0]
        s1 = ntwk.s[1]

        if cfg.dc_extrapolation_mode == "constant":
            s_dc = s0.copy()
        else:
            slope = (s1 - s0) / (f1 - f0)
            s_dc = s0 - slope * f0

        self._assert_finite_array(s_dc, stage="DC extrapolation", detail=f"synthetic DC estimate for '{ntwk.name}'")
        s_new = np.concatenate([s_dc[None, :, :], ntwk.s], axis=0)
        f_new = np.concatenate([[0.0], ntwk.freq_hz])
        meta = dict(ntwk.metadata)
        meta["synthetic_dc"] = True
        out = SParameterData(
            freq_hz=f_new,
            s=s_new,
            z0=ntwk.z0,
            name=ntwk.name,
            file_path=ntwk.file_path,
            metadata=meta,
        )
        self._assert_finite_network(out, stage="DC extrapolation", role=ntwk.name or "network")
        return out

    def _time_domain_step_and_impulse(
        self,
        ntwk: SParameterData,
        cfg: P370Config,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ntwk.check_2port()
        freq = ntwk.freq_hz.copy()
        h = ntwk.s[:, 1, 0].copy()
        if freq[0] > 1e-12:
            if cfg.auto_extrapolate_dc:
                dc_h = h[0] if len(h) < 2 else h[0] - (h[1] - h[0]) * freq[0] / max(freq[1] - freq[0], 1e-30)
            else:
                dc_h = 0.0 + 0.0j
            h = np.concatenate([[dc_h], h])
            freq = np.concatenate([[0.0], freq])
        self._assert_finite_array(h, stage="Time-domain transform", detail="frequency-domain transmission trace")
        df = float(np.mean(np.diff(freq)))
        n_time = max(2 * (len(h) - 1), 2)
        impulse_causal = np.fft.irfft(h, n=n_time)
        dt = 1.0 / max(n_time * df, 1e-30)
        step_causal = np.cumsum(impulse_causal) * dt
        time_axis_causal = np.arange(n_time, dtype=float) * dt
        impulse_centered = np.fft.fftshift(impulse_causal)
        time_axis_centered = (np.arange(n_time, dtype=float) - n_time // 2) * dt
        return time_axis_causal, step_causal, impulse_centered, time_axis_centered

    def _should_use_skrf_split(self, fix_fix_2xthru: SParameterData) -> tuple[bool, str]:
        if not SKRF_AVAILABLE:
            return False, "scikit-rf unavailable; using non-reference matrix-square-root fallback split"
        freq = fix_fix_2xthru.freq_hz
        if len(freq) < 3:
            return False, "Too few points for scikit-rf P370 split; using fallback split"
        if freq[0] == 0.0:
            freq = freq[1:]
        if len(freq) < 3:
            return False, "Too few non-DC points for scikit-rf P370 split; using fallback split"
        df = float(np.mean(np.diff(freq)))
        if df <= 0.0:
            return False, "Invalid frequency spacing for scikit-rf P370 split; using fallback split"
        start_ratio = float(freq[0] / df)
        if start_ratio > 4.0:
            return (
                False,
                "Waveguide-safe fallback activated because the frequency grid starts far above its spacing; "
                "the scikit-rf IEEE P370 split expects a low-pass-compatible grid and can become unstable for high-pass bands.",
            )
        return True, ""

    def _build_fixture_error_boxes_skrf(
        self,
        fix_fix_2xthru: SParameterData,
        cfg: P370Config,
    ) -> tuple[SParameterData, SParameterData, list[str]]:
        s2x_input = fix_fix_2xthru
        if fix_fix_2xthru.freq_hz[0] == 0.0:
            s2x_input = SParameterData(
                freq_hz=fix_fix_2xthru.freq_hz[1:],
                s=fix_fix_2xthru.s[1:],
                z0=fix_fix_2xthru.z0,
                name=fix_fix_2xthru.name,
                file_path=fix_fix_2xthru.file_path,
                metadata=dict(fix_fix_2xthru.metadata),
            )
        self._assert_finite_network(s2x_input, stage="scikit-rf split input", role="2x-thru")

        try:
            s2x = self._to_skrf_network(s2x_input, name=s2x_input.name or "2xthru")
            dm = IEEEP370_SE_NZC_2xThru(dummy_2xthru=s2x, name=s2x.name or "2xthru", z0=float(np.real(cfg.z_ref_ohm)))
            left = self._from_skrf_network(dm.s_side1, name=f"{fix_fix_2xthru.name}_left_fixture")
            right_unflipped = dm.s_side2.flipped()
            right = self._from_skrf_network(right_unflipped, name=f"{fix_fix_2xthru.name}_right_fixture")
        except Exception as exc:
            err = str(exc)
            if "infs or Nans" in err or "infs or NaNs" in err:
                raise ValueError(
                    "IEEE P370 NZC split produced non-finite internal values during fixture extraction. "
                    "This commonly happens with waveguide or near-cutoff data when the lowest-frequency points are unstable, "
                    "or when synthetic DC extrapolation is not appropriate. Try enabling near-cutoff trimming, disabling DC extrapolation, "
                    "or manually removing the lowest-frequency points from the 2x-thru and FIX-DUT-FIX files."
                ) from exc
            raise ValueError(f"IEEE P370 NZC split failed during fixture extraction: {err}") from exc

        self._assert_finite_network(left, stage="scikit-rf split output", role="left fixture")
        self._assert_finite_network(right, stage="scikit-rf split output", role="right fixture")

        if abs(cfg.reference_plane_offset_left_s) > 0.0:
            left_abcd = left.s_to_abcd()
            delay_left = self._matched_delay_abcd(left.freq_hz, cfg.reference_plane_offset_left_s, left.z0)
            left = SParameterData.from_abcd(
                left.freq_hz,
                SParameterData.cascade_abcd(left_abcd, delay_left),
                z0=left.z0,
                name=left.name,
            )
        if abs(cfg.reference_plane_offset_right_s) > 0.0:
            right_abcd = right.s_to_abcd()
            delay_right = self._matched_delay_abcd(right.freq_hz, cfg.reference_plane_offset_right_s, right.z0)
            right = SParameterData.from_abcd(
                right.freq_hz,
                SParameterData.cascade_abcd(delay_right, right_abcd),
                z0=right.z0,
                name=right.name,
            )

        notes = [
            "Fixture halves extracted with scikit-rf IEEE P370 SE_NZC_2xThru",
            "Return-loss-corrected square-root branch tracking handled by the reference NZC implementation",
        ]
        if fix_fix_2xthru.freq_hz[0] == 0.0:
            notes.append("Synthetic DC point was excluded from the split stage and retained only for local time-domain checks")
        return left, right, notes

    def _build_fixture_error_boxes_fallback(
        self,
        fix_fix_2xthru: SParameterData,
        midpoint: P370MidpointResult,
        cfg: P370Config,
    ) -> tuple[SParameterData, SParameterData]:
        total_abcd = fix_fix_2xthru.s_to_abcd()
        left_abcd = self._matrix_sqrt_stack(total_abcd)
        right_abcd = np.einsum("fij,fjk->fik", np.linalg.inv(left_abcd), total_abcd)

        if abs(cfg.reference_plane_offset_left_s) > 0.0:
            delay_left = self._matched_delay_abcd(fix_fix_2xthru.freq_hz, cfg.reference_plane_offset_left_s, fix_fix_2xthru.z0)
            left_abcd = SParameterData.cascade_abcd(left_abcd, delay_left)
        if abs(cfg.reference_plane_offset_right_s) > 0.0:
            delay_right = self._matched_delay_abcd(fix_fix_2xthru.freq_hz, cfg.reference_plane_offset_right_s, fix_fix_2xthru.z0)
            right_abcd = SParameterData.cascade_abcd(delay_right, right_abcd)

        left_fixture = SParameterData.from_abcd(
            fix_fix_2xthru.freq_hz,
            left_abcd,
            z0=fix_fix_2xthru.z0,
            name=f"{fix_fix_2xthru.name}_left_fixture",
        )
        right_fixture = SParameterData.from_abcd(
            fix_fix_2xthru.freq_hz,
            right_abcd,
            z0=fix_fix_2xthru.z0,
            name=f"{fix_fix_2xthru.name}_right_fixture",
        )
        _ = midpoint
        self._assert_finite_network(left_fixture, stage="fallback split output", role="left fixture")
        self._assert_finite_network(right_fixture, stage="fallback split output", role="right fixture")
        return left_fixture, right_fixture

    def _postprocess_dut(self, dut: SParameterData, cfg: P370Config) -> SParameterData:
        out = dut.copy(name=dut.name)
        out.metadata["p370_mode"] = cfg.mode
        out.metadata["p370_renormalize_after_deembed_requested"] = cfg.renormalize_after_deembed
        out.metadata["p370_auto_trim_near_cutoff"] = cfg.auto_trim_near_cutoff
        out.metadata["p370_auto_extrapolate_dc"] = cfg.auto_extrapolate_dc
        return out

    @staticmethod
    def _matrix_sqrt_stack(mats: np.ndarray) -> np.ndarray:
        out = np.zeros_like(mats, dtype=complex)
        for k in range(mats.shape[0]):
            vals, vecs = np.linalg.eig(mats[k])
            sqrt_vals = np.sqrt(vals)
            out[k] = vecs @ np.diag(sqrt_vals) @ np.linalg.inv(vecs)
        return out

    @staticmethod
    def _matched_delay_abcd(freq_hz: np.ndarray, delay_s: float, z0: complex | float) -> np.ndarray:
        phase = np.exp(-1j * 2.0 * np.pi * np.asarray(freq_hz, dtype=float) * float(delay_s))
        s = np.zeros((len(freq_hz), 2, 2), dtype=complex)
        s[:, 1, 0] = phase
        s[:, 0, 1] = phase
        return SParameterData(freq_hz=np.asarray(freq_hz, dtype=float), s=s, z0=z0, name="delay").s_to_abcd()

    @staticmethod
    def _window_sample_count(time_a: np.ndarray, time_b: np.ndarray, window_time_s: float) -> int:
        if window_time_s <= 0.0:
            return 0
        dt_candidates = []
        if len(time_a) > 1:
            dt_candidates.append(float(np.mean(np.diff(time_a))))
        if len(time_b) > 1:
            dt_candidates.append(float(np.mean(np.diff(time_b))))
        dt = min([x for x in dt_candidates if x > 0.0], default=0.0)
        if dt <= 0.0:
            return 0
        return max(1, int(np.floor(window_time_s / dt)))

    @staticmethod
    def _tdr_impedance_from_sii(ntwk: SParameterData, port_index: int) -> tuple[np.ndarray, np.ndarray]:
        ntwk.check_2port()
        freq = ntwk.freq_hz.copy()
        gamma_f = ntwk.s[:, port_index, port_index].copy()
        if len(freq) < 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        if freq[0] > 1e-12:
            gamma_f = np.concatenate([[gamma_f[0]], gamma_f])
            freq = np.concatenate([[0.0], freq])
        df = float(np.mean(np.diff(freq)))
        n_time = max(2 * (len(gamma_f) - 1), 2)
        impulse = np.fft.irfft(gamma_f, n=n_time)
        dt = 1.0 / max(n_time * df, 1e-30)
        gamma_t = np.cumsum(impulse)
        gamma_t = np.real(gamma_t)
        z0 = float(np.real(ntwk.z0))
        gamma_t = np.clip(gamma_t, -0.999999, 0.999999)
        z_tdr = z0 * (1.0 + gamma_t) / (1.0 - gamma_t)
        t_axis = np.arange(n_time, dtype=float) * dt
        return t_axis, z_tdr

    def _to_skrf_network(self, ntwk: SParameterData, name: Optional[str] = None):
        if not SKRF_AVAILABLE:
            raise RuntimeError("scikit-rf is not available")
        self._assert_finite_network(ntwk, stage="scikit-rf conversion", role=name or ntwk.name or "network")
        freq = Frequency.from_f(ntwk.freq_hz, unit="Hz")
        return Network(frequency=freq, s=np.asarray(ntwk.s, dtype=complex), z0=complex(ntwk.z0), name=name or ntwk.name)

    @staticmethod
    def _from_skrf_network(ntwk, name: Optional[str] = None) -> SParameterData:
        return SParameterData(
            freq_hz=np.asarray(ntwk.frequency.f, dtype=float),
            s=np.asarray(ntwk.s, dtype=complex),
            z0=complex(np.asarray(ntwk.z0).reshape(-1)[0]),
            name=name or getattr(ntwk, "name", ""),
            metadata={},
        )

    @staticmethod
    def _assert_finite_array(arr: np.ndarray, stage: str, detail: str) -> None:
        arr = np.asarray(arr)
        if np.iscomplexobj(arr):
            mask = np.isfinite(arr.real) & np.isfinite(arr.imag)
        else:
            mask = np.isfinite(arr)
        if np.all(mask):
            return
        bad_idx = np.argwhere(~mask)
        first = tuple(int(x) for x in bad_idx[0]) if bad_idx.size else ()
        raise ValueError(
            f"{stage}: non-finite values detected in {detail}. First bad index: {first}. "
            "This often indicates unstable near-cutoff data, a problematic extrapolation, or a singular intermediate transform."
        )

    def _assert_finite_network(self, ntwk: SParameterData, stage: str, role: str) -> None:
        self._assert_finite_array(ntwk.freq_hz, stage=stage, detail=f"{role} frequency axis")
        self._assert_finite_array(ntwk.s, stage=stage, detail=f"{role} S-parameters")
