from __future__ import annotations

import numpy as np

from rfdeembed import P3702xThruDeembedder, P370Config, P370Inputs, SParameterData


def line_abcd(freq_hz, zc, gamma_per_m, length_m):
    gl = gamma_per_m * length_m
    abcd = np.zeros((len(freq_hz), 2, 2), dtype=complex)
    abcd[:, 0, 0] = np.cosh(gl)
    abcd[:, 0, 1] = zc * np.sinh(gl)
    abcd[:, 1, 0] = np.sinh(gl) / zc
    abcd[:, 1, 1] = np.cosh(gl)
    return abcd


freq = np.linspace(110e9, 170e9, 601)
zref = 50.0
w = 2 * np.pi * freq
fix_abcd = line_abcd(freq, 70.0, (0.03 + 1e-13 * freq) + 1j * (w / 2.2e8), 0.004)
dut_abcd = line_abcd(freq, 62.0, (0.05 + 2e-13 * freq) + 1j * (w / 2.0e8), 0.006)

fixfix = SParameterData.from_abcd(freq, SParameterData.cascade_abcd(fix_abcd, fix_abcd), z0=zref, name="wg_fixfix")
fixdutfix = SParameterData.from_abcd(
    freq,
    SParameterData.cascade_abcd(SParameterData.cascade_abcd(fix_abcd, dut_abcd), fix_abcd),
    z0=zref,
    name="wg_fixdutfix",
)

# Emulate near-cutoff instability in the lowest-frequency samples.
for ntwk in (fixfix, fixdutfix):
    ntwk.s[:8, 1, 0] *= 1e-4
    ntwk.s[:8, 0, 1] *= 1e-4

engine = P3702xThruDeembedder()
result = engine.deembed(
    P370Inputs(fix_fix_2xthru=fixfix, fix_dut_fix=fixdutfix, dut_name="wg_dut_rec"),
    P370Config(
        mode="SE_NZC_2XTHRU",
        auto_extrapolate_dc=False,
        auto_trim_near_cutoff=True,
        cutoff_trim_s21_db=-35.0,
        use_impedance_correction=False,
    ),
)

print("trimmed_leading_points", result.preprocess.trimmed_leading_points)
print("dc_added", result.preprocess.dc_added)
print("self_check_present", result.self_check is not None)
print("dut_present", result.deembedded_dut is not None)
print("notes", result.preprocess.notes)

assert result.preprocess.trimmed_leading_points >= 1
assert result.preprocess.dc_added is False
assert result.self_check is not None
assert result.deembedded_dut is not None
