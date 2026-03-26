from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


class TouchstoneError(Exception):
    pass


@dataclass
class SParameterData:
    """
    Canonical network container for v1 backend.

    freq_hz: shape (N,)
    s: shape (N, P, P), complex
    """

    freq_hz: np.ndarray
    s: np.ndarray
    z0: complex | float = 50.0
    name: str = ""
    file_path: Optional[Path] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.freq_hz = np.asarray(self.freq_hz, dtype=float)
        self.s = np.asarray(self.s, dtype=complex)
        if self.freq_hz.ndim != 1:
            raise ValueError("freq_hz must be 1D")
        if self.s.ndim != 3:
            raise ValueError("s must have shape (N, P, P)")
        if self.s.shape[0] != self.freq_hz.shape[0]:
            raise ValueError("Frequency and S-parameter lengths do not match")
        if self.s.shape[1] != self.s.shape[2]:
            raise ValueError("S-parameter matrices must be square")
        if len(self.freq_hz) < 2:
            raise ValueError("Need at least two frequency points")
        if not np.all(np.diff(self.freq_hz) > 0):
            raise ValueError("Frequency points must be strictly increasing")

    @property
    def n_freq(self) -> int:
        return self.freq_hz.shape[0]

    @property
    def n_ports(self) -> int:
        return self.s.shape[1]

    @property
    def df_hz(self) -> float:
        return float(np.mean(np.diff(self.freq_hz)))

    def copy(self, name: Optional[str] = None) -> "SParameterData":
        return SParameterData(
            freq_hz=self.freq_hz.copy(),
            s=self.s.copy(),
            z0=self.z0,
            name=name or self.name,
            file_path=self.file_path,
            metadata=dict(self.metadata),
        )

    def check_2port(self) -> None:
        if self.n_ports != 2:
            raise ValueError("This operation requires a 2-port network")

    def get_trace(self, i: int, j: int) -> np.ndarray:
        return self.s[:, i, j].copy()

    def with_trace(self, i: int, j: int, trace: np.ndarray, name: Optional[str] = None) -> "SParameterData":
        trace = np.asarray(trace, dtype=complex)
        if trace.shape[0] != self.n_freq:
            raise ValueError("Trace length mismatch")
        out = self.copy(name=name)
        out.s[:, i, j] = trace
        return out

    def apply_to_all_traces(self, func) -> "SParameterData":
        out = self.copy()
        for i in range(self.n_ports):
            for j in range(self.n_ports):
                out.s[:, i, j] = np.asarray(func(self.s[:, i, j], i, j), dtype=complex)
        return out

    def same_grid_as(self, other: "SParameterData", atol: float = 1e-6, rtol: float = 1e-9) -> bool:
        return self.n_freq == other.n_freq and np.allclose(self.freq_hz, other.freq_hz, atol=atol, rtol=rtol)

    def interpolate_to(self, target_freq_hz: np.ndarray, name: Optional[str] = None) -> "SParameterData":
        target_freq_hz = np.asarray(target_freq_hz, dtype=float)
        s_new = np.zeros((len(target_freq_hz), self.n_ports, self.n_ports), dtype=complex)
        for i in range(self.n_ports):
            for j in range(self.n_ports):
                tr = self.s[:, i, j]
                s_new[:, i, j] = np.interp(target_freq_hz, self.freq_hz, tr.real) + 1j * np.interp(
                    target_freq_hz, self.freq_hz, tr.imag
                )
        return SParameterData(
            freq_hz=target_freq_hz,
            s=s_new,
            z0=self.z0,
            name=name or self.name,
            metadata=dict(self.metadata),
        )

    # --------------------
    # Touchstone I/O
    # --------------------
    @classmethod
    def from_touchstone(cls, file_path: str | Path) -> "SParameterData":
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in {".s1p", ".s2p"}:
            raise TouchstoneError(f"Unsupported Touchstone extension: {ext}")
        n_ports = int(ext[2])

        freq_unit = None
        parameter = None
        data_format = None
        z0 = 50.0
        numeric_rows = []

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("!"):
                    continue
                if line.startswith("#"):
                    tokens = line[1:].strip().lower().split()
                    if len(tokens) < 5:
                        raise TouchstoneError(f"Malformed Touchstone header: {line}")
                    freq_unit = tokens[0]
                    parameter = tokens[1]
                    data_format = tokens[2]
                    if "r" not in tokens:
                        raise TouchstoneError("Missing reference impedance in Touchstone header")
                    z0 = float(tokens[tokens.index("r") + 1])
                    continue
                if "!" in line:
                    line = line.split("!", 1)[0].strip()
                if not line:
                    continue
                numeric_rows.append([float(x) for x in line.split()])

        if freq_unit is None:
            raise TouchstoneError("Missing Touchstone header")
        if parameter != "s":
            raise TouchstoneError("Only S-parameter Touchstone files are supported")

        scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(freq_unit)
        if scale is None:
            raise TouchstoneError(f"Unsupported frequency unit: {freq_unit}")

        data = np.asarray(numeric_rows, dtype=float)
        if data.ndim != 2 or data.shape[0] == 0:
            raise TouchstoneError("No numeric data found")

        freq_hz = data[:, 0] * scale

        def pair_to_complex(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            fmt = data_format.upper()
            if fmt == "RI":
                return a + 1j * b
            if fmt == "MA":
                return a * np.exp(1j * np.deg2rad(b))
            if fmt == "DB":
                return (10 ** (a / 20.0)) * np.exp(1j * np.deg2rad(b))
            raise TouchstoneError(f"Unsupported data format: {fmt}")

        if n_ports == 1:
            if data.shape[1] != 3:
                raise TouchstoneError("S1P must contain 3 columns per row")
            s = np.zeros((len(freq_hz), 1, 1), dtype=complex)
            s[:, 0, 0] = pair_to_complex(data[:, 1], data[:, 2])
        else:
            if data.shape[1] != 9:
                raise TouchstoneError("S2P must contain 9 columns per row")
            s11 = pair_to_complex(data[:, 1], data[:, 2])
            s21 = pair_to_complex(data[:, 3], data[:, 4])
            s12 = pair_to_complex(data[:, 5], data[:, 6])
            s22 = pair_to_complex(data[:, 7], data[:, 8])
            s = np.zeros((len(freq_hz), 2, 2), dtype=complex)
            s[:, 0, 0] = s11
            s[:, 1, 0] = s21
            s[:, 0, 1] = s12
            s[:, 1, 1] = s22

        return cls(
            freq_hz=freq_hz,
            s=s,
            z0=z0,
            name=path.stem,
            file_path=path,
            metadata={
                "freq_unit": freq_unit,
                "parameter": parameter,
                "data_format": data_format.upper(),
            },
        )

    def to_touchstone(self, file_path: str | Path, fmt: str = "RI", freq_unit: str = "GHz") -> None:
        path = Path(file_path)
        fmt = fmt.upper()
        freq_unit = freq_unit.lower()
        if fmt not in {"RI", "MA", "DB"}:
            raise ValueError("fmt must be RI, MA, or DB")
        if freq_unit not in {"hz", "khz", "mhz", "ghz"}:
            raise ValueError("freq_unit must be Hz/kHz/MHz/GHz")

        scale = {"hz": 1.0, "khz": 1e-3, "mhz": 1e-6, "ghz": 1e-9}[freq_unit]
        ext = f".s{self.n_ports}p"
        if path.suffix.lower() != ext:
            path = path.with_suffix(ext)

        def encode(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if fmt == "RI":
                return x.real, x.imag
            if fmt == "MA":
                return np.abs(x), np.rad2deg(np.angle(x))
            mag_db = 20 * np.log10(np.abs(x) + 1e-15)
            ang_deg = np.rad2deg(np.angle(x))
            return mag_db, ang_deg

        with path.open("w", encoding="utf-8") as f:
            f.write(f"# {freq_unit} S {fmt} R {float(np.real(self.z0))}\n")
            if self.n_ports == 1:
                a, b = encode(self.s[:, 0, 0])
                for fr, x1, x2 in zip(self.freq_hz * scale, a, b):
                    f.write(f"{fr:.12g} {x1:.12g} {x2:.12g}\n")
            elif self.n_ports == 2:
                s11a, s11b = encode(self.s[:, 0, 0])
                s21a, s21b = encode(self.s[:, 1, 0])
                s12a, s12b = encode(self.s[:, 0, 1])
                s22a, s22b = encode(self.s[:, 1, 1])
                for row in zip(self.freq_hz * scale, s11a, s11b, s21a, s21b, s12a, s12b, s22a, s22b):
                    f.write(" ".join(f"{v:.12g}" for v in row) + "\n")
            else:
                raise ValueError("Touchstone export currently supports only 1-port and 2-port")

    # --------------------
    # Network math
    # --------------------
    def s_to_abcd(self) -> np.ndarray:
        self.check_2port()
        z0 = complex(self.z0)
        s11 = self.s[:, 0, 0]
        s12 = self.s[:, 0, 1]
        s21 = self.s[:, 1, 0]
        s22 = self.s[:, 1, 1]
        eps = 1e-15
        denom = 2 * s21 + eps
        A = ((1 + s11) * (1 - s22) + s12 * s21) / denom
        B = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / denom
        C = ((1 - s11) * (1 - s22) - s12 * s21) / (z0 * denom)
        D = ((1 - s11) * (1 + s22) + s12 * s21) / denom
        abcd = np.zeros((self.n_freq, 2, 2), dtype=complex)
        abcd[:, 0, 0] = A
        abcd[:, 0, 1] = B
        abcd[:, 1, 0] = C
        abcd[:, 1, 1] = D
        return abcd

    @classmethod
    def from_abcd(cls, freq_hz: np.ndarray, abcd: np.ndarray, z0: complex | float = 50.0, name: str = "") -> "SParameterData":
        z0 = complex(z0)
        A = abcd[:, 0, 0]
        B = abcd[:, 0, 1]
        C = abcd[:, 1, 0]
        D = abcd[:, 1, 1]
        denom = A + B / z0 + C * z0 + D + 1e-15
        s11 = (A + B / z0 - C * z0 - D) / denom
        s21 = 2 / denom
        s12 = 2 * (A * D - B * C) / denom
        s22 = (-A + B / z0 - C * z0 + D) / denom
        s = np.zeros((len(freq_hz), 2, 2), dtype=complex)
        s[:, 0, 0] = s11
        s[:, 1, 0] = s21
        s[:, 0, 1] = s12
        s[:, 1, 1] = s22
        return cls(freq_hz=np.asarray(freq_hz), s=s, z0=z0, name=name)

    def s_to_z(self) -> np.ndarray:
        I = np.eye(self.n_ports, dtype=complex)[None, :, :]
        s = self.s
        out = np.zeros_like(s)
        for k in range(self.n_freq):
            out[k] = self.z0 * (I[k] + s[k]) @ np.linalg.inv(I[k] - s[k])
        return out

    def s_to_y(self) -> np.ndarray:
        I = np.eye(self.n_ports, dtype=complex)[None, :, :]
        s = self.s
        out = np.zeros_like(s)
        for k in range(self.n_freq):
            out[k] = (1 / self.z0) * (I[k] - s[k]) @ np.linalg.inv(I[k] + s[k])
        return out

    @staticmethod
    def cascade_abcd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.einsum("fij,fjk->fik", left, right)

    @staticmethod
    def invert_abcd(mats: np.ndarray) -> np.ndarray:
        return np.linalg.inv(mats)

    @staticmethod
    def decascade_abcd(total: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        inv_left = np.linalg.inv(left)
        inv_right = np.linalg.inv(right)
        return np.einsum("fij,fjk,fkl->fil", inv_left, total, inv_right)

    def magnitude_db(self, i: int, j: int) -> np.ndarray:
        return 20 * np.log10(np.abs(self.s[:, i, j]) + 1e-15)

    def phase_deg(self, i: int, j: int, unwrap: bool = False) -> np.ndarray:
        ph = np.angle(self.s[:, i, j])
        if unwrap:
            ph = np.unwrap(ph)
        return np.rad2deg(ph)

    def group_delay_s(self, i: int = 1, j: int = 0) -> np.ndarray:
        phi = np.unwrap(np.angle(self.s[:, i, j]))
        w = 2 * np.pi * self.freq_hz
        gd = -np.gradient(phi, w)
        return gd
