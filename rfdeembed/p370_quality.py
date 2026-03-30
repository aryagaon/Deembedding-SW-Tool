from __future__ import annotations

"""
IEEE P370 quality helpers for the RF de-embedding tool.

The goal of this module is not to reproduce every formal IEEE P370 metric in
full detail. Instead, it provides a practical, deterministic quality layer that
is lightweight, dependency-free, and good enough to support the new native
2x-thru workflow end-to-end.

Design notes
------------
- The implementation is intentionally conservative and transparent.
- Metrics are derived from directly inspectable quantities such as singular
  value norms, reciprocity mismatch, and simple time-domain energy tests.
- The API remains compatible with the earlier skeleton so a stricter standards
  implementation can replace these routines later without changing callers.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .p370_models import P370Config, P370QualityReport, QualityGrade
from .sparameter_data import SParameterData


@dataclass(slots=True)
class P370FrequencyDomainMetrics:
    """Container for practical frequency-domain quality metrics."""

    passivity_score_pct: Optional[float] = None
    reciprocity_score_pct: Optional[float] = None
    causality_score_pct: Optional[float] = None
    max_s_norm: Optional[float] = None
    reciprocity_error: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class P370TimeDomainMetrics:
    """Container for practical time-domain quality metrics."""

    passivity_mv: Optional[float] = None
    reciprocity_mv: Optional[float] = None
    causality_mv: Optional[float] = None
    impulse_peak: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class P370FERMetrics:
    """Container for lightweight fixture electrical requirement checks."""

    fer_class: Optional[str] = None
    passed: bool = False
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class P370QualityChecks:
    """Practical IEEE-P370-style quality-check façade."""

    @classmethod
    def build_report(cls, ntwk: SParameterData, cfg: Optional[P370Config] = None) -> P370QualityReport:
        """Build a combined quality report for one network."""
        cfg = cfg or P370Config()
        fd = cls.frequency_domain_qm(ntwk, cfg) if cfg.enable_frequency_domain_qm else None
        td = cls.time_domain_qm(ntwk, cfg) if cfg.enable_time_domain_qm else None
        fer = cls.fixture_electrical_requirements(ntwk, cfg) if cfg.enable_fer_checks else None

        warnings = cls.summarize_warnings(
            fd.warnings if fd else [],
            td.warnings if td else [],
            fer.warnings if fer else [],
        )

        details: dict[str, Any] = {}
        if fd:
            details["frequency_domain"] = fd.details
        if td:
            details["time_domain"] = td.details
        if fer:
            details["fixture_electrical_requirements"] = fer.metrics

        return P370QualityReport(
            subject_name=ntwk.name or "network",
            fd_passivity_pct=fd.passivity_score_pct if fd else None,
            fd_reciprocity_pct=fd.reciprocity_score_pct if fd else None,
            fd_causality_pct=fd.causality_score_pct if fd else None,
            fd_passivity_grade=cls.grade_percentage(fd.passivity_score_pct if fd else None),
            fd_reciprocity_grade=cls.grade_percentage(fd.reciprocity_score_pct if fd else None),
            fd_causality_grade=cls.grade_percentage(fd.causality_score_pct if fd else None),
            td_passivity_mv=td.passivity_mv if td else None,
            td_reciprocity_mv=td.reciprocity_mv if td else None,
            td_causality_mv=td.causality_mv if td else None,
            fer_class=fer.fer_class if fer else None,
            warnings=warnings,
            details=details,
        )

    @classmethod
    def frequency_domain_qm(
        cls,
        ntwk: SParameterData,
        cfg: Optional[P370Config] = None,
    ) -> P370FrequencyDomainMetrics:
        """Compute pragmatic passivity, reciprocity, and causality proxies."""
        _ = cfg or P370Config()
        max_s_norm = cls.compute_max_s_norm(ntwk)
        excess = max(0.0, max_s_norm - 1.0)
        passivity_score = float(np.clip(100.0 - 100.0 * excess, 0.0, 100.0))

        rec_trace = cls.reciprocity_error_trace(ntwk)
        ref_mag = np.mean(np.linalg.norm(ntwk.s, axis=(1, 2))) + 1e-15
        rec_error = float(np.mean(rec_trace))
        reciprocity_score = float(np.clip(100.0 * (1.0 - rec_error / ref_mag), 0.0, 100.0))

        causality_score, causality_details, causality_warnings = cls._causality_score_from_impulse(ntwk)

        warnings: list[str] = []
        if max_s_norm > 1.02:
            warnings.append(f"Passivity concern: max singular-value norm is {max_s_norm:.3f} (> 1)")
        if reciprocity_score < 80.0:
            warnings.append(f"Reciprocity concern: score is {reciprocity_score:.1f}%")
        warnings.extend(causality_warnings)

        return P370FrequencyDomainMetrics(
            passivity_score_pct=passivity_score,
            reciprocity_score_pct=reciprocity_score,
            causality_score_pct=causality_score,
            max_s_norm=max_s_norm,
            reciprocity_error=rec_error,
            warnings=warnings,
            details={
                "max_s_norm": max_s_norm,
                "passivity_excess": excess,
                "mean_reciprocity_error": rec_error,
                **causality_details,
            },
        )

    @classmethod
    def time_domain_qm(
        cls,
        ntwk: SParameterData,
        cfg: Optional[P370Config] = None,
        data_rate_hz: Optional[float] = None,
        rise_time_s: Optional[float] = None,
    ) -> P370TimeDomainMetrics:
        """Compute lightweight time-domain distortion indicators."""
        _ = data_rate_hz, rise_time_s
        cfg = cfg or P370Config()
        impulse, time_axis = cls._band_limited_impulse(ntwk, cfg)
        impulse_peak = float(np.max(np.abs(impulse))) if impulse.size else None

        neg_mask = time_axis < 0.0
        total_energy = float(np.sum(np.abs(impulse) ** 2)) + 1e-30
        neg_energy = float(np.sum(np.abs(impulse[neg_mask]) ** 2)) if np.any(neg_mask) else 0.0
        causality_mv = 1e3 * neg_energy / total_energy

        rec_trace = cls.reciprocity_error_trace(ntwk)
        reciprocity_mv = 1e3 * float(np.mean(rec_trace))

        max_s_norm = cls.compute_max_s_norm(ntwk)
        passivity_mv = 1e3 * max(0.0, max_s_norm - 1.0)

        warnings: list[str] = []
        if causality_mv > 5.0:
            warnings.append(f"Pre-cursor energy is elevated ({causality_mv:.2f} mV-equivalent)")
        if passivity_mv > 10.0:
            warnings.append(f"Passivity excess is elevated ({passivity_mv:.2f} mV-equivalent)")

        return P370TimeDomainMetrics(
            passivity_mv=float(passivity_mv),
            reciprocity_mv=float(reciprocity_mv),
            causality_mv=float(causality_mv),
            impulse_peak=impulse_peak,
            warnings=warnings,
            details={
                "impulse_peak": impulse_peak,
                "negative_time_energy_ratio": neg_energy / total_energy,
                "time_samples": int(len(time_axis)),
            },
        )

    @classmethod
    def fixture_electrical_requirements(
        cls,
        fix_fix_2xthru: SParameterData,
        cfg: Optional[P370Config] = None,
    ) -> P370FERMetrics:
        """Run simple fixture sanity checks suitable for UI reporting."""
        _ = cfg or P370Config()
        fix_fix_2xthru.check_2port()

        s11_db = 20.0 * np.log10(np.abs(fix_fix_2xthru.s[:, 0, 0]) + 1e-15)
        s22_db = 20.0 * np.log10(np.abs(fix_fix_2xthru.s[:, 1, 1]) + 1e-15)
        s21_db = 20.0 * np.log10(np.abs(fix_fix_2xthru.s[:, 1, 0]) + 1e-15)

        median_rl_db = float(np.median(np.minimum(-s11_db, -s22_db)))
        insertion_flatness_db = float(np.max(s21_db) - np.min(s21_db))
        reciprocity_err = float(np.mean(np.abs(fix_fix_2xthru.s[:, 1, 0] - fix_fix_2xthru.s[:, 0, 1])))

        warnings: list[str] = []
        passed = True
        if median_rl_db < 6.0:
            warnings.append(f"Return loss is weak (median RL {median_rl_db:.2f} dB)")
            passed = False
        if insertion_flatness_db > 12.0:
            warnings.append(f"Insertion loss varies strongly across band ({insertion_flatness_db:.2f} dB span)")
            passed = False
        if reciprocity_err > 0.1:
            warnings.append(f"Reciprocity mismatch is elevated ({reciprocity_err:.3f})")
            passed = False

        fer_class = "good" if passed else "review"
        return P370FERMetrics(
            fer_class=fer_class,
            passed=passed,
            warnings=warnings,
            metrics={
                "median_return_loss_db": median_rl_db,
                "insertion_flatness_db": insertion_flatness_db,
                "mean_reciprocity_error": reciprocity_err,
            },
        )

    @staticmethod
    def grade_percentage(score_pct: Optional[float]) -> QualityGrade:
        """Convert a 0-100 score into a coarse engineering grade."""
        if score_pct is None:
            return "unknown"
        if score_pct >= 95.0:
            return "good"
        if score_pct >= 80.0:
            return "acceptable"
        if score_pct >= 60.0:
            return "inconclusive"
        return "poor"

    @staticmethod
    def compute_max_s_norm(ntwk: SParameterData) -> float:
        """Return the maximum singular-value norm of S across frequency."""
        s = np.asarray(ntwk.s)
        max_norm = 0.0
        for k in range(s.shape[0]):
            sigma_max = np.linalg.svd(s[k], compute_uv=False)[0]
            max_norm = max(max_norm, float(np.real(sigma_max)))
        return max_norm

    @staticmethod
    def reciprocity_error_trace(ntwk: SParameterData) -> np.ndarray:
        """Return the per-frequency Frobenius norm of S - S^T."""
        return np.linalg.norm(ntwk.s - np.transpose(ntwk.s, (0, 2, 1)), axis=(1, 2))

    @staticmethod
    def summarize_warnings(*warning_lists: list[str]) -> list[str]:
        """Merge warning lists while preserving order and uniqueness."""
        seen: set[str] = set()
        merged: list[str] = []
        for warnings in warning_lists:
            for msg in warnings:
                if msg not in seen:
                    merged.append(msg)
                    seen.add(msg)
        return merged

    @staticmethod
    def _uniform_df(freq_hz: np.ndarray) -> float:
        if len(freq_hz) < 2:
            return 0.0
        return float(np.mean(np.diff(freq_hz)))

    @classmethod
    def _band_limited_impulse(cls, ntwk: SParameterData, cfg: P370Config) -> tuple[np.ndarray, np.ndarray]:
        ntwk.check_2port()
        freq = ntwk.freq_hz
        if len(freq) < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        df = cls._uniform_df(freq)
        h = ntwk.s[:, 1, 0].copy()
        if freq[0] > 1e-12 and cfg.auto_extrapolate_dc:
            h0 = h[0]
            h = np.concatenate([[h0], h])
            freq = np.concatenate([[0.0], freq])
            df = cls._uniform_df(freq)

        n_time = max(2 * (len(h) - 1), 2)
        impulse = np.fft.irfft(h, n=n_time)
        dt = 1.0 / (n_time * df + 1e-30)
        time_axis = np.arange(n_time, dtype=float) * dt
        time_axis = time_axis - 0.5 * time_axis[-1]
        impulse = np.fft.fftshift(impulse)
        return impulse, time_axis

    @classmethod
    def _causality_score_from_impulse(cls, ntwk: SParameterData) -> tuple[float, dict[str, Any], list[str]]:
        dummy_cfg = P370Config(auto_extrapolate_dc=True)
        impulse, time_axis = cls._band_limited_impulse(ntwk, dummy_cfg)
        if impulse.size == 0:
            return 0.0, {"negative_time_energy_ratio": None}, ["Not enough data for causality estimate"]

        neg_mask = time_axis < 0.0
        total_energy = float(np.sum(np.abs(impulse) ** 2)) + 1e-30
        neg_energy = float(np.sum(np.abs(impulse[neg_mask]) ** 2)) if np.any(neg_mask) else 0.0
        ratio = neg_energy / total_energy
        score = float(np.clip(100.0 * (1.0 - ratio), 0.0, 100.0))
        warnings: list[str] = []
        if ratio > 0.05:
            warnings.append(f"Causality concern: negative-time energy ratio is {ratio:.3%}")
        return score, {"negative_time_energy_ratio": ratio}, warnings
