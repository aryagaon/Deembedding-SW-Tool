import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QListWidgetItem

from rfdeembed import SParameterData
from ui_app import DeembedMainWindow


def line_abcd(freq_hz, zc, gamma_per_m, length_m):
    gl = gamma_per_m * length_m
    abcd = np.zeros((len(freq_hz), 2, 2), dtype=complex)
    abcd[:, 0, 0] = np.cosh(gl)
    abcd[:, 0, 1] = zc * np.sinh(gl)
    abcd[:, 1, 0] = np.sinh(gl) / zc
    abcd[:, 1, 1] = np.cosh(gl)
    return abcd


def main():
    app = QApplication(sys.argv)
    window = DeembedMainWindow()

    freq = np.linspace(1e9, 18e9, 401)
    w = 2 * np.pi * freq

    left_abcd = line_abcd(freq, 53.0, (0.018 + 1.3e-12 * freq) + 1j * (w / 1.8e8), 0.007)
    right_abcd = line_abcd(freq, 47.0, (0.030 + 1.8e-12 * freq) + 1j * (w / 1.55e8), 0.011)
    dut_abcd = line_abcd(freq, 50.5, (0.045 + 2.2e-12 * freq) + 1j * (w / 1.62e8), 0.016)

    left = SParameterData.from_abcd(freq, left_abcd, z0=50.0, name="left_fixture_model")
    right = SParameterData.from_abcd(freq, right_abcd, z0=50.0, name="right_fixture_model")
    dut_true = SParameterData.from_abcd(freq, dut_abcd, z0=50.0, name="dut_true")
    measured = SParameterData.from_abcd(
        freq,
        SParameterData.cascade_abcd(SParameterData.cascade_abcd(left_abcd, dut_abcd), right_abcd),
        z0=50.0,
        name="fix_dut_fix_measured",
    )

    for ntwk in [left, right, measured]:
        window.networks[ntwk.name] = ntwk
        window.file_list.addItem(QListWidgetItem(ntwk.name))
    window._refresh_network_combos()

    window.cmb_method.setCurrentText("Known Fixture De-cascade")
    window.cmb_line1.setCurrentText(left.name)
    window.cmb_line2.setCurrentText(right.name)
    window.cmb_dut.setCurrentText(measured.name)
    window.run_trl()

    assert window.latest_trl_result is not None
    assert window.latest_trl_result.deembedded_dut is not None
    recovered = window.latest_trl_result.deembedded_dut
    assert recovered.name == "fix_dut_fix_measured_decascaded"
    err = np.max(np.abs(recovered.s - dut_true.s))

    print("known_fixture_method_present", window.cmb_method.findText("Known Fixture De-cascade") >= 0)
    print("left_label", window.lbl_line1.text())
    print("right_label", window.lbl_line2.text())
    print("dut_label", window.lbl_dut.text())
    print("result_name", recovered.name)
    print("max_abs_s_error", f"{err:.3e}")

    assert window.lbl_line1.text() == "Left fixture"
    assert window.lbl_line2.text() == "Right fixture"
    assert window.lbl_dut.text() == "Measured FIX-DUT-FIX"
    assert err < 1e-9

    window._mark_dirty(False)
    window.close()
    app.quit()


if __name__ == "__main__":
    main()
