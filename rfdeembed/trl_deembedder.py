from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .sparameter_data import SParameterData


@dataclass
class TRLConfig:
    line_lengths_m: List[float] = field(default_factory=list)
    thru_length_m: float = 0.0
    reflect_type: str = "unknown"
    reference_impedance: float = 50.0
    mirror_symmetric_fixture: bool = True


@dataclass
class TRLResult:
    left_fixture: Optional[SParameterData] = None
    right_fixture: Optional[SParameterData] = None
    deembedded_dut: Optional[SParameterData] = None
    alpha_np_per_m: Optional[np.ndarray] = None
    beta_rad_per_m: Optional[np.ndarray] = None
    gamma_per_m: Optional[np.ndarray] = None
    zc_ohm: Optional[np.ndarray] = None
    validity_mask: Optional[np.ndarray] = None
    residual_db: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)


class TRLDeembedder:
    """
    Runnable v1 de-embedding backend.

    Supported workflows:
      1) single_line_trl_fit(): extract line propagation and approximate fixture halves
      2) deembed_with_fixtures(): de-cascade known/extracted fixture halves
      3) short_long_extract_line(): estimate differential line propagation from short/long structures

    Notes:
    - v1 emphasizes runnable engineering utility rather than a fully metrology-grade TRL solver.
    - Fixture split is exact only under the mirror-symmetric assumption used here.
    """

    def single_line_trl_fit(
        self,
        thru: SParameterData,
        line: SParameterData,
        config: TRLConfig,
        dut: Optional[SParameterData] = None,
    ) -> TRLResult:
        thru.check_2port()
        line.check_2port()
        self._ensure_same_grid(thru, line)

        if len(config.line_lengths_m) < 1:
            raise ValueError("TRLConfig.line_lengths_m must contain at least one LINE length")

        line_length_m = float(config.line_lengths_m[0])
        delta_l = line_length_m - float(config.thru_length_m)
        if abs(delta_l) < 1e-15:
            raise ValueError("LINE length must differ from THRU length")

        thru_abcd = thru.s_to_abcd()
        line_abcd = line.s_to_abcd()

        diff = np.einsum("fij,fjk->fik", np.linalg.inv(thru_abcd), line_abcd)
        gamma, zc = self._extract_gamma_zc_from_differential(diff, delta_l, reference_impedance=float(config.reference_impedance))
        alpha = np.real(gamma)
        beta = np.unwrap(np.imag(gamma))
        validity = self._validity_mask(beta, delta_l)

        left_abcd, right_abcd = self._split_fixture_from_thru(
            thru_abcd,
            mirror_symmetric=config.mirror_symmetric_fixture,
        )

        left_fixture = SParameterData.from_abcd(thru.freq_hz, left_abcd, z0=thru.z0, name="left_fixture")
        right_fixture = SParameterData.from_abcd(thru.freq_hz, right_abcd, z0=thru.z0, name="right_fixture")

        result = TRLResult(
            left_fixture=left_fixture,
            right_fixture=right_fixture,
            alpha_np_per_m=alpha,
            beta_rad_per_m=beta,
            gamma_per_m=gamma,
            zc_ohm=zc,
            validity_mask=validity,
            notes=[
                "v1 single-line TRL fit completed",
                "fixture splitting uses mirror-symmetric assumption" if config.mirror_symmetric_fixture else "fixture split fallback used",
            ],
        )

        if dut is not None:
            result.deembedded_dut = self.deembed_with_fixtures(dut, left_fixture, right_fixture)
            result.residual_db = self._fixture_reconstruction_residual_db(thru, left_fixture, right_fixture)
        else:
            result.residual_db = self._fixture_reconstruction_residual_db(thru, left_fixture, right_fixture)

        return result

    def deembed_with_fixtures(
        self,
        measured_dut: SParameterData,
        left_fixture: SParameterData,
        right_fixture: SParameterData,
        name: Optional[str] = None,
    ) -> SParameterData:
        measured_dut.check_2port()
        left_fixture.check_2port()
        right_fixture.check_2port()
        self._ensure_same_grid(measured_dut, left_fixture)
        self._ensure_same_grid(measured_dut, right_fixture)

        total = measured_dut.s_to_abcd()
        left = left_fixture.s_to_abcd()
        right = right_fixture.s_to_abcd()
        intrinsic = SParameterData.decascade_abcd(total, left, right)
        return SParameterData.from_abcd(
            measured_dut.freq_hz,
            intrinsic,
            z0=measured_dut.z0,
            name=name or f"{measured_dut.name}_deembedded",
        )

    def short_long_extract_line(
        self,
        short_line: SParameterData,
        long_line: SParameterData,
        short_length_m: float,
        long_length_m: float,
        reference_impedance: float = 50.0,
    ) -> TRLResult:
        short_line.check_2port()
        long_line.check_2port()
        self._ensure_same_grid(short_line, long_line)

        delta_l = float(long_length_m) - float(short_length_m)
        if abs(delta_l) < 1e-15:
            raise ValueError("Long and short lengths must differ")

        short_abcd = short_line.s_to_abcd()
        long_abcd = long_line.s_to_abcd()
        diff = np.einsum("fij,fjk->fik", np.linalg.inv(short_abcd), long_abcd)
        gamma, zc = self._extract_gamma_zc_from_differential(diff, delta_l, reference_impedance)
        alpha = np.real(gamma)
        beta = np.unwrap(np.imag(gamma))
        validity = self._validity_mask(beta, delta_l)

        return TRLResult(
            alpha_np_per_m=alpha,
            beta_rad_per_m=beta,
            gamma_per_m=gamma,
            zc_ohm=zc,
            validity_mask=validity,
            notes=["v1 short/long differential line extraction completed"],
        )

    def multiline_trl_fit(
        self,
        thru: SParameterData,
        lines: List[SParameterData],
        config: TRLConfig,
        dut: Optional[SParameterData] = None,
    ) -> TRLResult:
        if not lines:
            raise ValueError("At least one line is required for multiline TRL")
        if len(config.line_lengths_m) != len(lines):
            raise ValueError("Number of line lengths must match number of LINE standards")

        # Solve each line independently, then choose the line with best phase validity per frequency.
        candidates = []
        for line, length_m in zip(lines, config.line_lengths_m):
            cfg = TRLConfig(
                line_lengths_m=[length_m],
                thru_length_m=config.thru_length_m,
                reflect_type=config.reflect_type,
                reference_impedance=config.reference_impedance,
                mirror_symmetric_fixture=config.mirror_symmetric_fixture,
            )
            candidates.append(self.single_line_trl_fit(thru, line, cfg, dut=None))

        freq = thru.freq_hz
        n = len(freq)
        alpha = np.full(n, np.nan)
        beta = np.full(n, np.nan)
        zc = np.full(n, np.nan)
        validity = np.zeros(n, dtype=bool)

        for k in range(n):
            picked = None
            for cand in candidates:
                if cand.validity_mask is not None and cand.validity_mask[k]:
                    picked = cand
                    break
            if picked is None:
                # fallback to the first candidate
                picked = candidates[0]
            alpha[k] = picked.alpha_np_per_m[k]
            beta[k] = picked.beta_rad_per_m[k]
            zc[k] = picked.zc_ohm[k]
            validity[k] = bool(picked.validity_mask[k]) if picked.validity_mask is not None else False

        thru_abcd = thru.s_to_abcd()
        left_abcd, right_abcd = self._split_fixture_from_thru(
            thru_abcd,
            mirror_symmetric=config.mirror_symmetric_fixture,
        )
        left_fixture = SParameterData.from_abcd(thru.freq_hz, left_abcd, z0=thru.z0, name="left_fixture")
        right_fixture = SParameterData.from_abcd(thru.freq_hz, right_abcd, z0=thru.z0, name="right_fixture")

        result = TRLResult(
            left_fixture=left_fixture,
            right_fixture=right_fixture,
            alpha_np_per_m=alpha,
            beta_rad_per_m=beta,
            gamma_per_m=alpha + 1j * beta,
            zc_ohm=zc,
            validity_mask=validity,
            residual_db=self._fixture_reconstruction_residual_db(thru, left_fixture, right_fixture),
            notes=["v1 multiline TRL fit by per-frequency candidate selection"],
        )
        if dut is not None:
            result.deembedded_dut = self.deembed_with_fixtures(dut, left_fixture, right_fixture)
        return result

    # -------------------- Internal helpers --------------------
    def _ensure_same_grid(self, a: SParameterData, b: SParameterData) -> None:
        if not a.same_grid_as(b):
            raise ValueError(f"Frequency grid mismatch between '{a.name}' and '{b.name}'")

    def _extract_gamma_zc_from_differential(
        self,
        diff_abcd: np.ndarray,
        delta_length_m: float,
        reference_impedance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Differential transmission matrix M ~ expm(line * delta_l).
        For a reciprocal uniform line, eigenvalues are exp(±gamma*L).
        We choose the branch with smaller median attenuation magnitude as the forward mode.
        """
        eigvals = np.linalg.eigvals(diff_abcd)
        # Two eigenvalues per frequency. Choose the one whose median |lambda| is <= 1 if available.
        cand0 = eigvals[:, 0]
        cand1 = eigvals[:, 1]
        score0 = np.nanmedian(np.abs(cand0))
        score1 = np.nanmedian(np.abs(cand1))
        lam = cand0 if abs(score0 - 1) <= abs(score1 - 1) else cand1
        # Prefer stable branch if one median magnitude is <= 1 and the other >= 1
        if np.nanmedian(np.abs(cand0)) <= 1.0 < np.nanmedian(np.abs(cand1)):
            lam = cand0
        elif np.nanmedian(np.abs(cand1)) <= 1.0 < np.nanmedian(np.abs(cand0)):
            lam = cand1

        gamma = np.log(lam + 1e-30) / delta_length_m

        # Estimate Zc from differential ABCD under line assumption:
        # for uniform line B = Zc sinh(gamma l), C = sinh(gamma l)/Zc -> Zc = sqrt(B/C)
        B = diff_abcd[:, 0, 1]
        C = diff_abcd[:, 1, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            zc = np.sqrt(B / (C + 1e-30))
        zc = np.where(np.isfinite(zc), zc, reference_impedance + 0j)
        zc = np.real_if_close(zc)
        return gamma, np.asarray(zc, dtype=complex)

    def _validity_mask(self, beta_rad_per_m: np.ndarray, delta_length_m: float) -> np.ndarray:
        dphi_deg = np.rad2deg(np.abs(beta_rad_per_m * delta_length_m))
        return (dphi_deg >= 20.0) & (dphi_deg <= 160.0)

    def _split_fixture_from_thru(self, thru_abcd: np.ndarray, mirror_symmetric: bool = True) -> tuple[np.ndarray, np.ndarray]:
        n = thru_abcd.shape[0]
        left = np.zeros_like(thru_abcd)
        right = np.zeros_like(thru_abcd)

        if mirror_symmetric:
            for i in range(n):
                m = thru_abcd[i]
                eigvals, eigvecs = np.linalg.eig(m)
                sqrt_diag = np.diag(np.sqrt(eigvals + 0j))
                root = eigvecs @ sqrt_diag @ np.linalg.inv(eigvecs)
                left[i] = root
                right[i] = root
            return left, right

        ident = np.repeat(np.eye(2, dtype=complex)[None, :, :], n, axis=0)
        return thru_abcd.copy(), ident

    def _fixture_reconstruction_residual_db(
        self,
        thru: SParameterData,
        left_fixture: SParameterData,
        right_fixture: SParameterData,
    ) -> np.ndarray:
        total = thru.s_to_abcd()
        left = left_fixture.s_to_abcd()
        right = right_fixture.s_to_abcd()
        recon = SParameterData.cascade_abcd(left, right)
        recon_ntwk = SParameterData.from_abcd(thru.freq_hz, recon, z0=thru.z0, name="reconstructed_thru")
        err = np.abs(thru.s[:, 1, 0] - recon_ntwk.s[:, 1, 0])
        return 20 * np.log10(err + 1e-15)
