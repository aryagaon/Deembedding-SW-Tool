from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.signal import get_window, find_peaks

from .sparameter_data import SParameterData


@dataclass
class GateConfig:
    transform_mode: str = "bandpass_impulse"   # bandpass_impulse | lowpass_impulse | lowpass_step
    gate_mode: str = "bandstop"                # bandstop | bandpass
    center_s: Optional[float] = None
    span_s: Optional[float] = None
    start_s: Optional[float] = None
    stop_s: Optional[float] = None
    window: str = "rectangular"
    fft_window: str = "hann"
    fft_pad_factor: int = 4
    synthetic_dc: bool = False
    per_trace: bool = False


@dataclass
class TimeDomainResult:
    time_s: np.ndarray
    response: np.ndarray
    transform_mode: str
    source_name: str
    trace_name: str


class TimeGating:
    """
    Runnable v1 time-domain gating engine.

    Features:
      - bandpass impulse transform (safe default for waveguide / band-limited DUTs)
      - low-pass impulse / step transform with optional synthetic DC
      - zero-padding for denser time axis
      - bandpass or bandstop gate
      - apply to one trace or all traces
      - simple auto-gate helper
    """

    VALID_TRANSFORMS = {"bandpass_impulse", "lowpass_impulse", "lowpass_step"}
    VALID_GATE_MODES = {"bandstop", "bandpass"}
    WINDOW_ALIASES = {"rectangular": "boxcar"}

    def to_time_domain(self, ntwk: SParameterData, i: int = 1, j: int = 0, cfg: Optional[GateConfig] = None) -> TimeDomainResult:
        cfg = cfg or GateConfig()
        self._validate_cfg(cfg)
        trace = ntwk.get_trace(i, j)

        if cfg.transform_mode == "bandpass_impulse":
            t, td = self._bandpass_transform(ntwk.freq_hz, trace, cfg)
        elif cfg.transform_mode in {"lowpass_impulse", "lowpass_step"}:
            t, td = self._lowpass_transform(ntwk.freq_hz, trace, cfg)
            if cfg.transform_mode == "lowpass_step":
                td = np.cumsum(np.real(td)) * (t[1] - t[0])
        else:
            raise ValueError(f"Unsupported transform mode: {cfg.transform_mode}")

        return TimeDomainResult(
            time_s=t,
            response=td,
            transform_mode=cfg.transform_mode,
            source_name=ntwk.name,
            trace_name=f"S{i+1}{j+1}",
        )

    def apply_gate(self, ntwk: SParameterData, i: int = 1, j: int = 0, cfg: Optional[GateConfig] = None) -> SParameterData:
        cfg = cfg or GateConfig()
        td = self.to_time_domain(ntwk, i, j, cfg)
        gate, used_start, used_stop = self._make_gate(td.time_s, cfg)

        if cfg.gate_mode == "bandstop":
            gated_td = td.response * (1.0 - gate)
        else:
            gated_td = td.response * gate

        trace_back = self._back_transform(gated_td, ntwk.freq_hz, cfg)
        out = ntwk.copy(name=f"{ntwk.name}_gated")
        out.s[:, i, j] = trace_back[: ntwk.n_freq]
        out.metadata["last_gate"] = {
            "trace": f"S{i+1}{j+1}",
            "transform_mode": cfg.transform_mode,
            "gate_mode": cfg.gate_mode,
            "start_s": used_start,
            "stop_s": used_stop,
            "fft_pad_factor": cfg.fft_pad_factor,
        }
        return out

    def apply_gate_all(self, ntwk: SParameterData, cfg: Optional[GateConfig] = None) -> SParameterData:
        cfg = cfg or GateConfig()
        out = ntwk.copy(name=f"{ntwk.name}_gated")
        for i in range(ntwk.n_ports):
            for j in range(ntwk.n_ports):
                tr_out = self.apply_gate(out, i, j, cfg)
                out.s[:, i, j] = tr_out.s[:, i, j]
        return out

    def auto_gate_from_peaks(
        self,
        ntwk: SParameterData,
        i: int = 0,
        j: int = 0,
        cfg: Optional[GateConfig] = None,
        mode: str = "largest_peak_notch",
    ) -> GateConfig:
        cfg = cfg or GateConfig()
        td = self.to_time_domain(ntwk, i, j, cfg)
        mag = np.abs(td.response)
        peaks, _ = find_peaks(mag)
        if len(peaks) == 0:
            raise ValueError("No peaks found for auto-gating")
        peaks = peaks[np.argsort(mag[peaks])[::-1]]
        peak_idx = peaks[0]
        peak_t = td.time_s[peak_idx]

        if len(peaks) > 1:
            second_t = td.time_s[peaks[1]]
            span = abs(second_t - peak_t) * 0.5
        else:
            span = max(abs(td.time_s[1] - td.time_s[0]) * 8, 1e-12)

        out = GateConfig(**cfg.__dict__)
        out.center_s = float(peak_t)
        out.span_s = float(max(span, abs(td.time_s[1] - td.time_s[0]) * 4))
        if mode == "largest_peak_notch":
            out.gate_mode = "bandstop"
        elif mode == "largest_peak_pass":
            out.gate_mode = "bandpass"
        else:
            raise ValueError("mode must be 'largest_peak_notch' or 'largest_peak_pass'")
        return out

    def estimate_time_resolution(self, ntwk: SParameterData, cfg: Optional[GateConfig] = None) -> float:
        cfg = cfg or GateConfig()
        span = ntwk.freq_hz[-1] - ntwk.freq_hz[0]
        if span <= 0:
            raise ValueError("Invalid frequency span")
        # Approximate impulse width / resolution ~ 1 / bandwidth
        return 1.0 / span

    def estimate_distance_resolution(self, ntwk: SParameterData, velocity_m_per_s: float, cfg: Optional[GateConfig] = None) -> float:
        return self.estimate_time_resolution(ntwk, cfg) * velocity_m_per_s

    # -------------------- internals --------------------
    def _validate_cfg(self, cfg: GateConfig) -> None:
        if cfg.transform_mode not in self.VALID_TRANSFORMS:
            raise ValueError(f"Invalid transform_mode: {cfg.transform_mode}")
        if cfg.gate_mode not in self.VALID_GATE_MODES:
            raise ValueError(f"Invalid gate_mode: {cfg.gate_mode}")
        if int(cfg.fft_pad_factor) < 1:
            raise ValueError("fft_pad_factor must be >= 1")

    def _bandpass_transform(self, freq_hz: np.ndarray, trace: np.ndarray, cfg: GateConfig) -> Tuple[np.ndarray, np.ndarray]:
        tr = self._apply_fft_window(trace, cfg.fft_window)
        n = self._next_pow2(len(tr) * int(cfg.fft_pad_factor))
        td = np.fft.ifft(tr, n=n)
        df = float(np.mean(np.diff(freq_hz)))
        t = np.fft.fftfreq(n, d=df)
        return t, td

    def _lowpass_transform(self, freq_hz: np.ndarray, trace: np.ndarray, cfg: GateConfig) -> Tuple[np.ndarray, np.ndarray]:
        freq = freq_hz.copy()
        tr = trace.copy()
        if not np.isclose(freq[0], 0.0):
            if not cfg.synthetic_dc:
                raise ValueError(
                    "Low-pass transform requires DC sample or synthetic_dc=True. "
                    "For waveguide/band-limited DUTs, use bandpass_impulse."
                )
            freq = np.concatenate([[0.0], freq])
            tr = np.concatenate([[tr[0]], tr])

        tr = self._apply_fft_window(tr, cfg.fft_window)
        hermitian = np.concatenate([tr, np.conj(tr[-2:0:-1])])
        n = self._next_pow2(len(hermitian) * int(cfg.fft_pad_factor))
        td = np.fft.ifft(hermitian, n=n)
        df = float(np.mean(np.diff(freq)))
        t = np.arange(n) / (n * df)
        return t, td

    def _apply_fft_window(self, trace: np.ndarray, window_name: str) -> np.ndarray:
        win = get_window(self._normalize_window_name(window_name), len(trace))
        return trace * win

    def _make_gate(self, time_s: np.ndarray, cfg: GateConfig) -> tuple[np.ndarray, float, float]:
        if cfg.start_s is not None and cfg.stop_s is not None:
            start = float(cfg.start_s)
            stop = float(cfg.stop_s)
        elif cfg.center_s is not None and cfg.span_s is not None:
            start = float(cfg.center_s - cfg.span_s / 2)
            stop = float(cfg.center_s + cfg.span_s / 2)
        else:
            raise ValueError("GateConfig must define start/stop or center/span")
        if stop <= start:
            raise ValueError("Gate stop must be greater than start")

        mask = (time_s >= start) & (time_s <= stop)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError("Gate does not overlap time axis")

        gate = np.zeros_like(time_s, dtype=float)
        win = get_window(self._normalize_window_name(cfg.window), len(idx))
        gate[idx] = win
        return gate, start, stop

    def _normalize_window_name(self, window_name: str) -> str:
        return self.WINDOW_ALIASES.get(window_name, window_name)

    def _back_transform(self, gated_td: np.ndarray, original_freq_hz: np.ndarray, cfg: GateConfig) -> np.ndarray:
        spec = np.fft.fft(gated_td)
        return spec[: len(original_freq_hz)]

    @staticmethod
    def _next_pow2(n: int) -> int:
        n = max(2, int(n))
        return int(2 ** np.ceil(np.log2(n)))
