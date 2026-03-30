from __future__ import annotations

import numpy as np

from rfdeembed import P3702xThruDeembedder, P370Config, P370Inputs, SParameterData


def line_abcd(freq_hz, z0_ref, zc, gamma_per_m, length_m):
    gl = gamma_per_m * length_m
    A = np.cosh(gl)
    B = zc * np.sinh(gl)
    C = np.sinh(gl) / zc
    D = A.copy()
    abcd = np.zeros((len(freq_hz), 2, 2), dtype=complex)
    abcd[:, 0, 0] = A
    abcd[:, 0, 1] = B
    abcd[:, 1, 0] = C
    abcd[:, 1, 1] = D
    return abcd


freq = np.linspace(1e6, 40e9, 1201)
zref = 50.0
w = 2 * np.pi * freq

alpha_fix = 0.02 + 2e-13 * freq
beta_fix = w / 1.75e8
zc_fix = 54.0
fix_abcd = line_abcd(freq, zref, zc_fix, alpha_fix + 1j * beta_fix, 0.012)

alpha_dut = 0.05 + 4e-13 * freq
beta_dut = w / 1.6e8
dut_abcd_true = line_abcd(freq, zref, 48.0, alpha_dut + 1j * beta_dut, 0.030)

fixfix_abcd = SParameterData.cascade_abcd(fix_abcd, fix_abcd)
fixdutfix_abcd = SParameterData.cascade_abcd(SParameterData.cascade_abcd(fix_abcd, dut_abcd_true), fix_abcd)

fixfix = SParameterData.from_abcd(freq, fixfix_abcd, z0=zref, name="fixfix")
fixdutfix = SParameterData.from_abcd(freq, fixdutfix_abcd, z0=zref, name="fixdutfix")
dut_true = SParameterData.from_abcd(freq, dut_abcd_true, z0=zref, name="dut_true")

engine = P3702xThruDeembedder()
result = engine.deembed(
    P370Inputs(fix_fix_2xthru=fixfix, fix_dut_fix=fixdutfix, dut_name="dut_rec"),
    P370Config(mode="SE_NZC_2XTHRU", auto_extrapolate_dc=True, use_impedance_correction=False),
)

dut_rec = result.deembedded_dut
assert dut_rec is not None

dut_true_i = dut_true.interpolate_to(dut_rec.freq_hz)
err_s21_db = np.max(np.abs(20 * np.log10(np.abs(dut_rec.s[:, 1, 0] / (dut_true_i.s[:, 1, 0] + 1e-30)) + 1e-30)))
err_s11 = np.max(np.abs(dut_rec.s[:, 0, 0] - dut_true_i.s[:, 0, 0]))

print("self_check_passed", result.self_check.passed)
print("self_check_max_mag_db", result.self_check.max_abs_mag_db)
print("self_check_max_phase_deg", result.self_check.max_abs_phase_deg)
print("self_check_max_tdr_error_pct", result.self_check.details.get("max_tdr_match_error_pct"))
print("dut_max_s21_error_db", float(err_s21_db))
print("dut_max_s11_abs_error", float(err_s11))

assert result.self_check.passed
assert result.split.algorithm_name.startswith("SE_NZC_2XTHRU")
assert result.self_check.max_abs_mag_db < 5e-4
assert result.self_check.max_abs_phase_deg < 1e-3
assert float(result.self_check.details.get("max_tdr_match_error_pct", 0.0)) < 0.1
assert float(err_s21_db) < 5e-3
assert float(err_s11) < 5e-3
