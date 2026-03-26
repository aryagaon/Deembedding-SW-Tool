from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

from .sparameter_data import SParameterData


@dataclass
class ValidationReport:
    passivity_ok: bool
    reciprocity_ok: bool
    causality_warning: bool
    smoothness_warning: bool
    max_passivity_excess_db: float
    reciprocity_error_db: float
    group_delay_std_ps: float
    warnings: List[str] = field(default_factory=list)


class ValidationChecks:
    """
    v1 engineering sanity checks.

    Included:
      - passivity check using eigenvalues of S^H S
      - reciprocity check for 2-port networks
      - simple group-delay smoothness heuristic on S21
      - placeholder causality warning flag
    """

    @staticmethod
    def check_passivity(ntwk: SParameterData, tol_db: float = 0.1) -> tuple[bool, float]:
        worst_excess_db = -1e9
        for k in range(ntwk.n_freq):
            s = ntwk.s[k]
            eigvals = np.linalg.eigvals(s.conj().T @ s)
            max_sv2 = np.max(np.real(eigvals))
            excess_db = 10.0 * np.log10(max(max_sv2, 1e-15))
            worst_excess_db = max(worst_excess_db, excess_db)
        return worst_excess_db <= tol_db, float(worst_excess_db)

    @staticmethod
    def check_reciprocity(ntwk: SParameterData, tol_db: float = -60.0) -> tuple[bool, float]:
        if ntwk.n_ports != 2:
            return True, -300.0
        err = np.max(np.abs(ntwk.s[:, 1, 0] - ntwk.s[:, 0, 1]))
        err_db = 20 * np.log10(max(err, 1e-15))
        return err_db <= tol_db, float(err_db)

    @staticmethod
    def check_group_delay_smoothness(ntwk: SParameterData, trace=(1, 0), threshold_ps: float = 250.0) -> tuple[bool, float]:
        try:
            gd_ps = ntwk.group_delay_s(*trace) * 1e12
            # robust smoothness metric using median-centered std
            centered = gd_ps - np.median(gd_ps)
            std_ps = float(np.std(centered))
            return std_ps <= threshold_ps, std_ps
        except Exception:
            return False, float("inf")

    @classmethod
    def build_report(cls, ntwk: SParameterData) -> ValidationReport:
        pass_ok, pass_excess_db = cls.check_passivity(ntwk)
        rec_ok, rec_err_db = cls.check_reciprocity(ntwk)
        smooth_ok, gd_std_ps = cls.check_group_delay_smoothness(ntwk)

        warnings: List[str] = []
        if not pass_ok:
            warnings.append(f"Passivity exceeded by {pass_excess_db:.2f} dB")
        if not rec_ok:
            warnings.append(f"Reciprocity mismatch = {rec_err_db:.2f} dB")
        if not smooth_ok:
            warnings.append(f"Group delay smoothness warning: std = {gd_std_ps:.1f} ps")

        # Placeholder: full causality check can be added later.
        causality_warning = False

        return ValidationReport(
            passivity_ok=pass_ok,
            reciprocity_ok=rec_ok,
            causality_warning=causality_warning,
            smoothness_warning=not smooth_ok,
            max_passivity_excess_db=pass_excess_db,
            reciprocity_error_db=rec_err_db,
            group_delay_std_ps=gd_std_ps,
            warnings=warnings,
        )
