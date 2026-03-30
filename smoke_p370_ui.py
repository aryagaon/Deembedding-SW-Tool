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

    freq = np.linspace(1e6, 20e9, 801)
    zref = 50.0
    w = 2 * np.pi * freq

    fix = line_abcd(freq, 54.0, (0.02 + 2e-13 * freq) + 1j * (w / 1.75e8), 0.012)
    dut = line_abcd(freq, 48.0, (0.05 + 4e-13 * freq) + 1j * (w / 1.6e8), 0.025)

    fixfix = SParameterData.from_abcd(freq, SParameterData.cascade_abcd(fix, fix), z0=zref, name="fixfix")
    fixdutfix = SParameterData.from_abcd(
        freq,
        SParameterData.cascade_abcd(SParameterData.cascade_abcd(fix, dut), fix),
        z0=zref,
        name="fixdutfix",
    )

    window.networks[fixfix.name] = fixfix
    window.networks[fixdutfix.name] = fixdutfix
    window.file_list.addItem(QListWidgetItem(fixfix.name))
    window.file_list.addItem(QListWidgetItem(fixdutfix.name))
    window._refresh_network_combos()

    window.cmb_method.setCurrentText("IEEE P370 2x-thru (NZC)")
    window.cmb_p370_2xthru.setCurrentText("fixfix")
    window.cmb_p370_fix_dut_fix.setCurrentText("fixdutfix")
    window.run_trl()

    print("p370_method_present", window.cmb_method.findText("IEEE P370 2x-thru (NZC)") >= 0)
    print("p370_2xthru_visible", not window.cmb_p370_2xthru.isHidden())
    print("p370_fix_dut_fix_visible", not window.cmb_p370_fix_dut_fix.isHidden())
    print("p370_result", window.latest_p370_result is not None)
    print("self_check_passed", bool(window.latest_p370_result.self_check.passed))
    print("validation_plot", window.panel_validation.canvas is not None)
    print("diag_plot", window.panel_trl.canvas is not None)
    print("latest_deembedded_name", window.latest_deembedded_name)

    window._mark_dirty(False)
    window.close()
    app.quit()


if __name__ == "__main__":
    main()
