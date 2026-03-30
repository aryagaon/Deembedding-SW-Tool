from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .sparameter_data import SParameterData
from .trl_deembedder import TRLResult
from .time_gating import TimeDomainResult


class PlotGenerator:
    """
    Plot-only module for v1 backend.
    Returns matplotlib Figure objects for GUI embedding or file export.
    """

    def plot_sparameters(
        self,
        raw: SParameterData,
        processed: SParameterData | None = None,
        title: str = "S-Parameters",
        db_floor: float = -120.0,
    ):
        raw.check_2port()
        if processed is not None:
            processed.check_2port()
            self._ensure_same_grid(raw, processed)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        fig.suptitle(title)
        freq_ghz = raw.freq_hz / 1e9
        traces = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]

        for ax, (i, j, label) in zip(axes.ravel(), traces):
            mag_raw = np.maximum(raw.magnitude_db(i, j), db_floor)
            ax.plot(freq_ghz, mag_raw, label="raw", lw=1.6, color="0.35")
            if processed is not None:
                mag_proc = np.maximum(processed.magnitude_db(i, j), db_floor)
                ax.plot(freq_ghz, mag_proc, label="processed", lw=1.3, color="tab:blue")
            ax.set_title(label)
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[1, 0].set_xlabel("Frequency (GHz)")
        axes[1, 1].set_xlabel("Frequency (GHz)")
        fig.tight_layout()
        return fig

    def plot_phase(
        self,
        raw: SParameterData,
        processed: SParameterData | None = None,
        unwrap: bool = True,
        title: str = "Phase",
    ):
        raw.check_2port()
        if processed is not None:
            processed.check_2port()
            self._ensure_same_grid(raw, processed)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        fig.suptitle(title)
        freq_ghz = raw.freq_hz / 1e9
        traces = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]

        for ax, (i, j, label) in zip(axes.ravel(), traces):
            ax.plot(freq_ghz, raw.phase_deg(i, j, unwrap=unwrap), label="raw", lw=1.6, color="0.35")
            if processed is not None:
                ax.plot(freq_ghz, processed.phase_deg(i, j, unwrap=unwrap), label="processed", lw=1.3, color="tab:blue")
            ax.set_title(label)
            ax.set_ylabel("Phase (deg)")
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[1, 0].set_xlabel("Frequency (GHz)")
        axes[1, 1].set_xlabel("Frequency (GHz)")
        fig.tight_layout()
        return fig

    def plot_group_delay(self, raw: SParameterData, processed: SParameterData | None = None, trace=(1, 0), title="Group Delay"):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        freq_ghz = raw.freq_hz / 1e9
        ax.plot(freq_ghz, raw.group_delay_s(*trace) * 1e12, label="raw", lw=1.6, color="0.35")
        if processed is not None:
            self._ensure_same_grid(raw, processed)
            ax.plot(freq_ghz, processed.group_delay_s(*trace) * 1e12, label="processed", lw=1.3, color="tab:blue")
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Group Delay (ps)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_time_domain(self, td: TimeDomainResult, gate_start_s: float | None = None, gate_stop_s: float | None = None, title: str | None = None):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        t_ns = td.time_s * 1e9
        ax.plot(t_ns, np.abs(td.response), lw=1.6, color="tab:purple", label=f"|{td.trace_name}|")
        if gate_start_s is not None and gate_stop_s is not None:
            ax.axvspan(gate_start_s * 1e9, gate_stop_s * 1e9, color="orange", alpha=0.22, label="gate")
        ax.set_title(title or f"Time Domain - {td.source_name} - {td.trace_name} ({td.transform_mode})")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_trl_diagnostics(self, result: TRLResult, freq_hz: np.ndarray, title: str = "TRL Diagnostics"):
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
        fig.suptitle(title)
        f_ghz = np.asarray(freq_hz) / 1e9

        if result.alpha_np_per_m is not None:
            axes[0].plot(f_ghz, np.real(result.alpha_np_per_m), color="tab:red", lw=1.5, label="alpha")
        axes[0].set_ylabel("Alpha (Np/m)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        if result.beta_rad_per_m is not None:
            axes[1].plot(f_ghz, np.real(result.beta_rad_per_m), color="tab:blue", lw=1.5, label="beta")
            if result.validity_mask is not None:
                invalid = ~np.asarray(result.validity_mask, dtype=bool)
                if np.any(invalid):
                    axes[1].scatter(f_ghz[invalid], np.real(result.beta_rad_per_m)[invalid], s=10, color="black", label="invalid band")
        axes[1].set_ylabel("Beta (rad/m)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        if result.residual_db is not None:
            axes[2].plot(f_ghz, np.real(result.residual_db), color="tab:green", lw=1.5, label="fixture recon residual")
        axes[2].set_ylabel("Residual (dB)")
        axes[2].set_xlabel("Frequency (GHz)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        fig.tight_layout()
        return fig

    def plot_fixture_comparison(
        self,
        measured: SParameterData,
        reconstructed: SParameterData,
        title: str = "Measured vs Reconstructed",
    ):
        self._ensure_same_grid(measured, reconstructed)
        measured.check_2port()
        reconstructed.check_2port()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
        f_ghz = measured.freq_hz / 1e9

        axes[0].plot(f_ghz, measured.magnitude_db(1, 0), color="0.35", lw=1.6, label="measured S21")
        axes[0].plot(f_ghz, reconstructed.magnitude_db(1, 0), color="tab:orange", lw=1.3, label="reconstructed S21")
        axes[0].set_title(title + " - S21")
        axes[0].set_ylabel("Magnitude (dB)")
        axes[0].set_xlabel("Frequency (GHz)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(f_ghz, measured.magnitude_db(0, 0), color="0.35", lw=1.6, label="measured S11")
        axes[1].plot(f_ghz, reconstructed.magnitude_db(0, 0), color="tab:orange", lw=1.3, label="reconstructed S11")
        axes[1].set_title(title + " - S11")
        axes[1].set_ylabel("Magnitude (dB)")
        axes[1].set_xlabel("Frequency (GHz)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        return fig

    def plot_validation_overlay(self, raw: SParameterData, deembedded: SParameterData, title: str = "Validation Overlay"):
        self._ensure_same_grid(raw, deembedded)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
        f_ghz = raw.freq_hz / 1e9

        axes[0].plot(f_ghz, raw.magnitude_db(1, 0), color="0.45", lw=1.6, label="raw S21")
        axes[0].plot(f_ghz, deembedded.magnitude_db(1, 0), color="tab:green", lw=1.4, label="de-embedded S21")
        axes[0].set_title(title + " - Insertion")
        axes[0].set_ylabel("Magnitude (dB)")
        axes[0].set_xlabel("Frequency (GHz)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(f_ghz, raw.magnitude_db(0, 0), color="0.45", lw=1.6, label="raw S11")
        axes[1].plot(f_ghz, deembedded.magnitude_db(0, 0), color="tab:green", lw=1.4, label="de-embedded S11")
        axes[1].set_title(title + " - Return")
        axes[1].set_ylabel("Magnitude (dB)")
        axes[1].set_xlabel("Frequency (GHz)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        return fig

    def plot_sparameter_overlay(
        self,
        networks: list[SParameterData],
        title: str = "S-Parameter Overlay",
        db_floor: float = -120.0,
    ):
        if not networks:
            raise ValueError("No networks were provided for overlay plotting")
        for ntwk in networks:
            ntwk.check_2port()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
        fig.suptitle(title)
        traces = [(0, 0, "S11"), (1, 0, "S21"), (0, 1, "S12"), (1, 1, "S22")]

        for ax, (i, j, label) in zip(axes.ravel(), traces):
            for ntwk in networks:
                freq_ghz = ntwk.freq_hz / 1e9
                mag_db = np.maximum(ntwk.magnitude_db(i, j), db_floor)
                ax.plot(freq_ghz, mag_db, lw=1.25, label=ntwk.name)
            ax.set_title(label)
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        axes[1, 0].set_xlabel("Frequency (GHz)")
        axes[1, 1].set_xlabel("Frequency (GHz)")
        fig.tight_layout()
        return fig

    def plot_time_domain_overlay(
        self,
        td_results: list[TimeDomainResult],
        gate_start_s: float | None = None,
        gate_stop_s: float | None = None,
        title: str = "Time-Domain Overlay",
    ):
        if not td_results:
            raise ValueError("No time-domain traces were provided for overlay plotting")
        fig, ax = plt.subplots(figsize=(11, 4.8))
        for td in td_results:
            ax.plot(td.time_s * 1e9, np.abs(td.response), lw=1.3, label=td.source_name)
        if gate_start_s is not None and gate_stop_s is not None:
            ax.axvspan(gate_start_s * 1e9, gate_stop_s * 1e9, color="orange", alpha=0.18, label="gate")
        ax.set_title(title)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        return fig

    def plot_p370_self_check(self, p370_result, title: str = "P370 Self-Check and Residuals"):
        if p370_result.split is None or p370_result.self_check is None:
            raise ValueError("P370 result does not contain split/self-check data")

        midpoint = p370_result.split.midpoint
        self_check = p370_result.self_check
        cfg = p370_result.config
        freq_hz = self_check.self_deembedded_2xthru.freq_hz
        f_ghz = freq_hz / 1e9

        fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), sharex=False)
        fig.suptitle(title)
        ax_mid, ax_mag, ax_phase, ax_tdr = axes.ravel()

        ax_mid.plot(midpoint.time_axis_s * 1e9, np.real(midpoint.t12_step_response), color="tab:purple", lw=1.5, label="T12 step")
        ax_mid.axvline(midpoint.midpoint_time_s * 1e9, color="tab:red", ls="--", lw=1.2, label="midpoint")
        ax_mid.set_ylabel("Step amplitude")
        ax_mid.set_xlabel("Time (ns)")
        ax_mid.set_title("Midpoint Detection")
        ax_mid.grid(True, alpha=0.3)
        ax_mid.legend(loc="best")

        ax_mag.plot(f_ghz, self_check.residual_mag_db, color="tab:green", lw=1.5, label="self residual |S21| (dB)")
        ax_mag.axhline(0.0, color="0.5", lw=0.8)
        ax_mag.axhline(cfg.max_self_residual_db, color="tab:red", lw=1.0, ls="--", label="limit")
        ax_mag.axhline(-cfg.max_self_residual_db, color="tab:red", lw=1.0, ls="--")
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_xlabel("Frequency (GHz)")
        ax_mag.set_title(f"Residual Magnitude · max={self_check.max_abs_mag_db:.4f} dB")
        ax_mag.grid(True, alpha=0.3)
        ax_mag.legend(loc="best")

        ax_phase.plot(f_ghz, self_check.residual_phase_deg, color="tab:blue", lw=1.5, label="self residual phase")
        ax_phase.axhline(0.0, color="0.5", lw=0.8)
        ax_phase.axhline(cfg.max_self_phase_deg, color="tab:red", lw=1.0, ls="--", label="limit")
        ax_phase.axhline(-cfg.max_self_phase_deg, color="tab:red", lw=1.0, ls="--")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_xlabel("Frequency (GHz)")
        ax_phase.set_title(f"Residual Phase · max={self_check.max_abs_phase_deg:.4f}°")
        ax_phase.grid(True, alpha=0.3)
        ax_phase.legend(loc="best")

        tdr_err = self_check.tdr_match_error_pct
        if tdr_err is not None and len(tdr_err):
            ax_tdr.plot(np.arange(len(tdr_err)), tdr_err, color="tab:orange", lw=1.4, label="fixture TDR mismatch")
            ax_tdr.axhline(cfg.max_tdr_impedance_error_pct, color="tab:red", lw=1.0, ls="--", label="limit")
            ax_tdr.set_ylabel("Mismatch (%)")
            ax_tdr.set_xlabel("Validation sample")
            ax_tdr.set_title(
                f"Fixture TDR Similarity · max={float(self_check.details.get('max_tdr_match_error_pct', 0.0)):.3f}%"
            )
            ax_tdr.grid(True, alpha=0.3)
            ax_tdr.legend(loc="best")
        else:
            ax_tdr.axis("off")
            summary = [
                f"Self-check: {'PASS' if self_check.passed else 'FAIL'}",
                f"Max |mag| residual: {self_check.max_abs_mag_db:.4f} dB",
                f"Max |phase| residual: {self_check.max_abs_phase_deg:.4f}°",
            ]
            if self_check.warnings:
                summary.extend(self_check.warnings[:3])
            ax_tdr.text(0.02, 0.95, "\n".join(summary), va="top", ha="left", fontsize=10)

        fig.tight_layout()
        return fig

    def plot_p370_deembed_overlay(
        self,
        fix_dut_fix: SParameterData,
        deembedded_dut: SParameterData,
        title: str = "P370 De-embedding Result",
    ):
        fix_dut_fix, deembedded_dut = self._overlap_networks(fix_dut_fix, deembedded_dut)
        fix_dut_fix.check_2port()
        deembedded_dut.check_2port()

        f_ghz = fix_dut_fix.freq_hz / 1e9
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        fig.suptitle(title)

        axes[0, 0].plot(f_ghz, fix_dut_fix.magnitude_db(1, 0), color="0.4", lw=1.5, label="FIX-DUT-FIX S21")
        axes[0, 0].plot(f_ghz, deembedded_dut.magnitude_db(1, 0), color="tab:green", lw=1.4, label="de-embedded DUT S21")
        axes[0, 0].set_title("Insertion")
        axes[0, 0].set_ylabel("Magnitude (dB)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(loc="best")

        axes[0, 1].plot(f_ghz, fix_dut_fix.magnitude_db(0, 0), color="0.4", lw=1.5, label="FIX-DUT-FIX S11")
        axes[0, 1].plot(f_ghz, deembedded_dut.magnitude_db(0, 0), color="tab:orange", lw=1.4, label="de-embedded DUT S11")
        axes[0, 1].set_title("Return")
        axes[0, 1].set_ylabel("Magnitude (dB)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(loc="best")

        axes[1, 0].plot(f_ghz, deembedded_dut.phase_deg(1, 0, unwrap=True), color="tab:blue", lw=1.4, label="de-embedded DUT S21 phase")
        axes[1, 0].set_title("De-embedded S21 Phase")
        axes[1, 0].set_ylabel("Phase (deg)")
        axes[1, 0].set_xlabel("Frequency (GHz)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(loc="best")

        axes[1, 1].plot(f_ghz, deembedded_dut.magnitude_db(0, 1), color="tab:red", lw=1.4, label="de-embedded DUT S12")
        axes[1, 1].set_title("De-embedded S12 Magnitude")
        axes[1, 1].set_ylabel("Magnitude (dB)")
        axes[1, 1].set_xlabel("Frequency (GHz)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(loc="best")

        fig.tight_layout()
        return fig

    @staticmethod
    def save(fig, file_path: str, dpi: int = 140):
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")

    @staticmethod
    def _ensure_same_grid(a: SParameterData, b: SParameterData):
        if not a.same_grid_as(b):
            raise ValueError(f"Frequency grids differ: '{a.name}' vs '{b.name}'")

    @staticmethod
    def _overlap_networks(a: SParameterData, b: SParameterData) -> tuple[SParameterData, SParameterData]:
        if a.same_grid_as(b):
            return a, b
        start = max(float(a.freq_hz[0]), float(b.freq_hz[0]))
        stop = min(float(a.freq_hz[-1]), float(b.freq_hz[-1]))
        if stop <= start:
            raise ValueError(f"Frequency ranges do not overlap: '{a.name}' vs '{b.name}'")
        n = min(a.n_freq, b.n_freq)
        freq = np.linspace(start, stop, n)
        return a.interpolate_to(freq, name=a.name), b.interpolate_to(freq, name=b.name)
