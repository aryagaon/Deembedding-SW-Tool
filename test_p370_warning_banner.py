import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
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


def add_networks(window, fixfix, fixdutfix):
    for ntwk in (fixfix, fixdutfix):
        window.networks[ntwk.name] = ntwk
        if not window.file_list.findItems(ntwk.name, Qt.MatchExactly):
            window.file_list.addItem(QListWidgetItem(ntwk.name))
    window._refresh_network_combos()


def main():
    app = QApplication(sys.argv)
    window = DeembedMainWindow()
    window.cmb_method.setCurrentText("IEEE P370 2x-thru (NZC)")

    freq_cpwg = np.linspace(10e6, 40e9, 801)
    w_cpwg = 2 * np.pi * freq_cpwg
    fix_cpwg = line_abcd(freq_cpwg, 54.0, (0.02 + 2e-13 * freq_cpwg) + 1j * (w_cpwg / 1.75e8), 0.012)
    dut_cpwg = line_abcd(freq_cpwg, 48.0, (0.05 + 4e-13 * freq_cpwg) + 1j * (w_cpwg / 1.6e8), 0.025)
    cpwg_fixfix = SParameterData.from_abcd(freq_cpwg, SParameterData.cascade_abcd(fix_cpwg, fix_cpwg), z0=50.0, name="cpwg_fixfix")
    cpwg_fixdutfix = SParameterData.from_abcd(
        freq_cpwg,
        SParameterData.cascade_abcd(SParameterData.cascade_abcd(fix_cpwg, dut_cpwg), fix_cpwg),
        z0=50.0,
        name="cpwg_fixdutfix",
    )
    add_networks(window, cpwg_fixfix, cpwg_fixdutfix)
    window.cmb_p370_2xthru.setCurrentText("cpwg_fixfix")
    window.cmb_p370_fix_dut_fix.setCurrentText("cpwg_fixdutfix")
    window._update_p370_warning_banner()
    print("cpwg_banner_hidden", window.lbl_p370_warning.isHidden())
    assert window.lbl_p370_warning.isHidden()

    freq_wg = np.linspace(110e9, 170e9, 601)
    w_wg = 2 * np.pi * freq_wg
    fix_wg = line_abcd(freq_wg, 70.0, (0.03 + 1e-13 * freq_wg) + 1j * (w_wg / 2.2e8), 0.004)
    dut_wg = line_abcd(freq_wg, 62.0, (0.05 + 2e-13 * freq_wg) + 1j * (w_wg / 2.0e8), 0.006)
    wg_fixfix = SParameterData.from_abcd(freq_wg, SParameterData.cascade_abcd(fix_wg, fix_wg), z0=50.0, name="wg_fixfix")
    wg_fixdutfix = SParameterData.from_abcd(
        freq_wg,
        SParameterData.cascade_abcd(SParameterData.cascade_abcd(fix_wg, dut_wg), fix_wg),
        z0=50.0,
        name="wg_fixdutfix",
    )
    wg_fixfix.s[:8, 1, 0] *= 1e-4
    wg_fixfix.s[:8, 0, 1] *= 1e-4
    wg_fixdutfix.s[:8, 1, 0] *= 1e-4
    wg_fixdutfix.s[:8, 0, 1] *= 1e-4
    add_networks(window, wg_fixfix, wg_fixdutfix)
    window.cmb_p370_2xthru.setCurrentText("wg_fixfix")
    window.cmb_p370_fix_dut_fix.setCurrentText("wg_fixdutfix")
    window.chk_p370_dc.setChecked(True)
    window.chk_p370_trim.setChecked(False)
    window.edit_p370_trim_db.setText("-35")
    window._update_p370_warning_banner()
    text_enabled = window.lbl_p370_warning.text()
    print("waveguide_banner_hidden", window.lbl_p370_warning.isHidden())
    print("waveguide_banner_text_enabled", text_enabled)
    assert not window.lbl_p370_warning.isHidden()
    assert "discouraged" in text_enabled.lower()
    assert "currently enabled" in text_enabled.lower()

    window.chk_p370_dc.setChecked(False)
    window.chk_p370_trim.setChecked(True)
    window._update_p370_warning_banner()
    text_disabled = window.lbl_p370_warning.text()
    print("waveguide_banner_text_disabled", text_disabled)
    assert "currently disabled" in text_disabled.lower()
    assert "enabled" in text_disabled.lower()

    window._mark_dirty(False)
    window.close()
    app.quit()


if __name__ == "__main__":
    main()
