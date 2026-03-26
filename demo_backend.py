from __future__ import annotations

import numpy as np
from pathlib import Path

from rfdeembed import SParameterData, TRLDeembedder, TRLConfig, TimeGating, GateConfig, PlotGenerator


def line_abcd(freq_hz, gamma, zc=50.0):
    gl = gamma
    A = np.cosh(gl)
    B = zc * np.sinh(gl)
    C = (1 / zc) * np.sinh(gl)
    D = np.cosh(gl)
    out = np.zeros((len(freq_hz), 2, 2), dtype=complex)
    out[:, 0, 0] = A
    out[:, 0, 1] = B
    out[:, 1, 0] = C
    out[:, 1, 1] = D
    return out


def make_fixture(freq_hz, loss_np=0.03, delay_s=25e-12, refl=0.04):
    beta = 2 * np.pi * freq_hz * delay_s / 0.01
    gamma = loss_np + 1j * beta
    base = line_abcd(freq_hz, gamma, zc=50.0)
    # Small mismatch block
    mismatch = np.zeros_like(base)
    mismatch[:, 0, 0] = 1.0
    mismatch[:, 1, 1] = 1.0
    mismatch[:, 0, 1] = refl * 50.0
    mismatch[:, 1, 0] = refl / 50.0
    return np.einsum("fij,fjk->fik", mismatch, base)


def synthesize():
    freq = np.linspace(1e9, 20e9, 2001)
    z0 = 50.0
    er_eff = 2.8
    vp = 299792458 / np.sqrt(er_eff)
    alpha = 0.08 * np.sqrt(freq / 1e9)  # Np/m

    def gamma_for_len(length_m):
        beta = 2 * np.pi * freq / vp
        return (alpha + 1j * beta) * length_m

    thru_len = 0.0
    line_len = 0.010
    dut_len = 0.017

    left_fix = make_fixture(freq, loss_np=0.015, delay_s=18e-12, refl=0.03)
    right_fix = make_fixture(freq, loss_np=0.015, delay_s=18e-12, refl=0.03)

    thru_abcd = np.einsum("fij,fjk->fik", left_fix, right_fix)
    line_ab = np.einsum("fij,fjk,fkl->fil", left_fix, line_abcd(freq, gamma_for_len(line_len), z0), right_fix)
    dut_ab = np.einsum("fij,fjk,fkl->fil", left_fix, line_abcd(freq, gamma_for_len(dut_len), z0), right_fix)

    thru = SParameterData.from_abcd(freq, thru_abcd, z0=z0, name="thru")
    line = SParameterData.from_abcd(freq, line_ab, z0=z0, name="line")
    dut = SParameterData.from_abcd(freq, dut_ab, z0=z0, name="dut")
    return thru, line, dut


def main():
    out_dir = Path("/mnt/user-data/outputs/rfdeembed_v1/demo_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    thru, line, dut = synthesize()

    # Time gating demo (bandpass mode works without DC)
    gater = TimeGating()
    gate_cfg = GateConfig(transform_mode="bandpass_impulse", gate_mode="bandstop", center_s=0.0, span_s=0.15e-9)
    td = gater.to_time_domain(dut, 0, 0, gate_cfg)

    de = TRLDeembedder()
    cfg = TRLConfig(line_lengths_m=[0.010], thru_length_m=0.0, reference_impedance=50.0, mirror_symmetric_fixture=True)
    result = de.single_line_trl_fit(thru, line, cfg, dut=dut)

    plotter = PlotGenerator()
    fig1 = plotter.plot_sparameters(dut, result.deembedded_dut, title="Synthetic DUT Raw vs De-embedded")
    fig2 = plotter.plot_trl_diagnostics(result, thru.freq_hz, title="Synthetic TRL Diagnostics")
    fig3 = plotter.plot_time_domain(td, title="Synthetic Time-Domain Response")

    plotter.save(fig1, str(out_dir / "raw_vs_deembedded.png"))
    plotter.save(fig2, str(out_dir / "trl_diagnostics.png"))
    plotter.save(fig3, str(out_dir / "time_domain.png"))

    thru.to_touchstone(out_dir / "thru.s2p")
    line.to_touchstone(out_dir / "line.s2p")
    dut.to_touchstone(out_dir / "dut.s2p")
    if result.deembedded_dut is not None:
        result.deembedded_dut.to_touchstone(out_dir / "dut_deembedded.s2p")

    print("Demo complete")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
