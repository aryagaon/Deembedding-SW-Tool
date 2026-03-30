import os
import sys
from pathlib import Path

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
    out_dir = Path('assets')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'rfdeembed_main_window.png'

    window = DeembedMainWindow()
    window.resize(1480, 920)

    freq = np.linspace(110e9, 170e9, 601)
    w = 2 * np.pi * freq
    fix = line_abcd(freq, 70.0, (0.03 + 1e-13 * freq) + 1j * (w / 2.2e8), 0.004)
    dut = line_abcd(freq, 62.0, (0.05 + 2e-13 * freq) + 1j * (w / 2.0e8), 0.006)

    fixfix_abcd = SParameterData.cascade_abcd(fix, fix)
    fixdutfix_abcd = SParameterData.cascade_abcd(SParameterData.cascade_abcd(fix, dut), fix)
    fixfix = SParameterData.from_abcd(freq, fixfix_abcd, z0=50.0, name='fixfix_dband')
    fixdutfix = SParameterData.from_abcd(freq, fixdutfix_abcd, z0=50.0, name='fixdutfix_dband')
    fixfix.s[:8] *= 1e-4
    fixdutfix.s[:8] *= 1e-4

    for ntwk in [fixfix, fixdutfix]:
        window.networks[ntwk.name] = ntwk
        window.file_list.addItem(QListWidgetItem(ntwk.name))
    window._refresh_network_combos()
    window.file_list.setCurrentRow(0)

    window.cmb_method.setCurrentText('IEEE P370 2x-thru (NZC)')
    window.cmb_p370_2xthru.setCurrentText('fixfix_dband')
    window.cmb_p370_fix_dut_fix.setCurrentText('fixdutfix_dband')
    window.chk_p370_dc.setChecked(False)
    window.chk_p370_trim.setChecked(True)
    window.edit_p370_trim_db.setText('-35')
    window._update_p370_warning_banner()
    window.run_trl()

    window.show()
    app.processEvents()
    pixmap = window.grab()
    pixmap.save(str(out_path))
    print(out_path)

    window.close()
    app.quit()


if __name__ == '__main__':
    main()
