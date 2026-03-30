import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog, QListWidgetItem

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

    print("toolbar_export_enabled_before", window.btn_export_deembedded.isEnabled())
    print("panel_export_enabled_before", window.btn_export_deembedded_panel.isEnabled())
    assert not window.btn_export_deembedded.isEnabled()
    assert not window.btn_export_deembedded_panel.isEnabled()

    window.cmb_method.setCurrentText("Known Fixture De-cascade")
    window.cmb_line1.setCurrentText(left.name)
    window.cmb_line2.setCurrentText(right.name)
    window.cmb_dut.setCurrentText(measured.name)
    window.run_trl()

    assert window.latest_deembedded_name is not None
    exported_name = window.latest_deembedded_name
    exported_ntwk = window.networks[exported_name]

    print("toolbar_export_enabled_after", window.btn_export_deembedded.isEnabled())
    print("panel_export_enabled_after", window.btn_export_deembedded_panel.isEnabled())
    assert window.btn_export_deembedded.isEnabled()
    assert window.btn_export_deembedded_panel.isEnabled()

    export_dir = Path("test_exports")
    export_dir.mkdir(exist_ok=True)
    export_path = export_dir / "direct_export_check.s2p"

    original_get_save = QFileDialog.getSaveFileName
    try:
        QFileDialog.getSaveFileName = staticmethod(lambda *args, **kwargs: (str(export_path), "Touchstone S2P (*.s2p)"))
        assert window.export_deembedded_s2p()
    finally:
        QFileDialog.getSaveFileName = original_get_save

    assert export_path.exists()
    reloaded = SParameterData.from_touchstone(export_path)
    err = np.max(np.abs(reloaded.s - exported_ntwk.s))

    print("export_file_exists", export_path.exists())
    print("exported_file_name", export_path.name)
    print("exported_network_name", exported_name)
    print("max_abs_export_error", f"{err:.3e}")
    print("status_text", window.status_label.text())

    assert err < 1e-12
    assert "Exported de-embedded S2P" in window.status_label.text()

    window._mark_dirty(False)
    window.close()
    app.quit()


if __name__ == "__main__":
    main()
