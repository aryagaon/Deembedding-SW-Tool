import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from ui_app import DeembedMainWindow


def main():
    app = QApplication(sys.argv)
    window = DeembedMainWindow()
    window.resize(1200, 800)
    window.cmb_method.setCurrentText("Short/Long Differential")
    print("ui_import_ok")
    print(window.windowTitle())
    print("thru_enabled", window.cmb_thru.isEnabled())
    print("selection_mode", window.file_list.selectionMode())
    print("default_gate_window", window.cmb_gate_window.currentText())
    print("default_fft_window", window.cmb_fft_window.currentText())
    print("plot_limits_button", window.btn_plot_limits.text())
    window.close()
    app.quit()


if __name__ == "__main__":
    main()
