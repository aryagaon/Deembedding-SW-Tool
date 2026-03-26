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

    @staticmethod
    def save(fig, file_path: str, dpi: int = 140):
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")

    @staticmethod
    def _ensure_same_grid(a: SParameterData, b: SParameterData):
        if not a.same_grid_as(b):
            raise ValueError(f"Frequency grids differ: '{a.name}' vs '{b.name}'")
