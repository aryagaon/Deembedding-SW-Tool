from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from rfdeembed import (
    SParameterData,
    TRLDeembedder,
    TRLConfig,
    TimeGating,
    GateConfig,
    PlotGenerator,
    ProjectStateManager,
    ValidationChecks,
)


class MatplotlibPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.canvas: Optional[FigureCanvas] = None
        self.toolbar: Optional[NavigationToolbar] = None

    def set_figure(self, fig):
        self.clear()
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._layout.addWidget(self.toolbar)
        self._layout.addWidget(self.canvas, 1)
        self.canvas.draw_idle()

    def clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.canvas = None
        self.toolbar = None


class PlotLimitsDialog(QDialog):
    def __init__(self, main_window: "DeembedMainWindow"):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Plot Limits")
        self.resize(320, 180)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.edit_xmin = QLineEdit("")
        self.edit_xmax = QLineEdit("")
        self.edit_ymin = QLineEdit("")
        self.edit_ymax = QLineEdit("")

        form.addRow("X min", self.edit_xmin)
        form.addRow("X max", self.edit_xmax)
        form.addRow("Y min", self.edit_ymin)
        form.addRow("Y max", self.edit_ymax)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_reset = QPushButton("Reset")
        self.btn_close = QPushButton("Close")
        button_row.addWidget(self.btn_apply)
        button_row.addWidget(self.btn_reset)
        button_row.addWidget(self.btn_close)
        layout.addLayout(button_row)

        self.btn_apply.clicked.connect(self.apply_limits)
        self.btn_reset.clicked.connect(self.reset_limits)
        self.btn_close.clicked.connect(self.close)

    def apply_limits(self):
        self.main_window.apply_plot_limits_from_dialog(
            self.edit_xmin.text().strip(),
            self.edit_xmax.text().strip(),
            self.edit_ymin.text().strip(),
            self.edit_ymax.text().strip(),
        )

    def reset_limits(self):
        self.edit_xmin.clear()
        self.edit_xmax.clear()
        self.edit_ymin.clear()
        self.edit_ymax.clear()
        self.main_window.reset_plot_limits_active_panel()


class DeembedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_name = "Untitled Project"
        self.project_file: Optional[Path] = None
        self.dirty = False
        self.last_browse_dir = str(Path.home())

        self.networks: Dict[str, SParameterData] = {}
        self.latest_trl_result = None
        self.latest_deembedded_name: Optional[str] = None
        self.latest_validation_report = None
        self.plot_limits_dialog: Optional[PlotLimitsDialog] = None

        self.trl_engine = TRLDeembedder()
        self.time_gating = TimeGating()
        self.plotter = PlotGenerator()
        self.project_manager = ProjectStateManager()

        self.resize(1680, 1020)
        self._update_window_title()

        self._build_ui()
        self._connect_signals()
        self._update_method_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.addWidget(self._build_file_import_group())
        left_layout.addWidget(self._build_method_group())
        left_layout.addWidget(self._build_gating_group())
        left_layout.addWidget(self._build_validation_group())
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.addWidget(self._build_plot_tabs(), 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([460, 1220])

        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        self.action_new = QAction("New Project", self)
        self.action_load = QAction("Load Project", self)
        self.action_save = QAction("Save Project", self)
        self.action_save_as = QAction("Save Project As", self)

        toolbar.addAction(self.action_new)
        toolbar.addAction(self.action_load)
        toolbar.addAction(self.action_save)
        toolbar.addAction(self.action_save_as)
        toolbar.addSeparator()

        self.btn_import_toolbar = QPushButton("Import Files")
        self.btn_refresh_plots = QPushButton("Refresh Plots")
        self.btn_solve_toolbar = QPushButton("Run TRL")
        self.btn_validate_toolbar = QPushButton("Run Validation")
        self.btn_plot_limits = QPushButton("Plot Limits")
        toolbar.addWidget(self.btn_import_toolbar)
        toolbar.addSeparator()
        toolbar.addWidget(self.btn_refresh_plots)
        toolbar.addWidget(self.btn_solve_toolbar)
        toolbar.addWidget(self.btn_validate_toolbar)
        toolbar.addWidget(self.btn_plot_limits)

    def _build_file_import_group(self) -> QGroupBox:
        group = QGroupBox("1) File Import")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.btn_import_files = QPushButton("Import Touchstone...")
        self.btn_remove_file = QPushButton("Remove Selected")
        row.addWidget(self.btn_import_files)
        row.addWidget(self.btn_remove_file)
        layout.addLayout(row)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.file_list, 1)

        meta_group = QGroupBox("Selected File Summary")
        meta_form = QFormLayout(meta_group)
        self.lbl_file_name = QLabel("-")
        self.lbl_file_points = QLabel("-")
        self.lbl_file_ports = QLabel("-")
        self.lbl_file_freq = QLabel("-")
        self.lbl_file_z0 = QLabel("-")
        meta_form.addRow("Name", self.lbl_file_name)
        meta_form.addRow("Points", self.lbl_file_points)
        meta_form.addRow("Ports", self.lbl_file_ports)
        meta_form.addRow("Freq Range", self.lbl_file_freq)
        meta_form.addRow("Z0", self.lbl_file_z0)
        layout.addWidget(meta_group)
        return group

    def _build_method_group(self) -> QGroupBox:
        group = QGroupBox("2) Method Selection / TRL Setup")
        layout = QFormLayout(group)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems([
            "Single-line TRL",
            "Multiline TRL",
            "Short/Long Differential",
            "Known Fixture De-cascade",
        ])

        self.cmb_thru = QComboBox()
        self.cmb_line1 = QComboBox()
        self.cmb_line2 = QComboBox()
        self.cmb_dut = QComboBox()

        self.edit_line1_len = QLineEdit("0.010")
        self.edit_line2_len = QLineEdit("0.020")
        self.edit_thru_len = QLineEdit("0.0")
        self.chk_mirror_sym = QCheckBox("Assume mirror-symmetric fixture")
        self.chk_mirror_sym.setChecked(True)

        self.btn_run_trl = QPushButton("Solve / De-embed")

        layout.addRow("Method", self.cmb_method)
        layout.addRow("THRU", self.cmb_thru)
        layout.addRow("LINE 1", self.cmb_line1)
        layout.addRow("LINE 1 length (m)", self.edit_line1_len)
        layout.addRow("LINE 2", self.cmb_line2)
        layout.addRow("LINE 2 length (m)", self.edit_line2_len)
        layout.addRow("THRU length (m)", self.edit_thru_len)
        layout.addRow("DUT", self.cmb_dut)
        layout.addRow("Symmetry", self.chk_mirror_sym)
        layout.addRow(self.btn_run_trl)
        return group

    def _build_gating_group(self) -> QGroupBox:
        group = QGroupBox("3) Gating Controls")
        layout = QFormLayout(group)

        self.lbl_gating_hint = QLabel("Tip: use S11 or S22 to place the time gate visually. You do not need line lengths just to preview or adjust the gate.")
        self.lbl_gating_hint.setWordWrap(True)

        self.cmb_gate_source = QComboBox()
        self.cmb_gate_trace = QComboBox()
        self.cmb_gate_trace.addItems(["S11", "S21", "S12", "S22"])

        self.cmb_transform = QComboBox()
        self.cmb_transform.addItems(["bandpass_impulse", "lowpass_impulse", "lowpass_step"])
        self.cmb_gate_mode = QComboBox()
        self.cmb_gate_mode.addItems(["bandstop", "bandpass"])
        self.cmb_gate_window = QComboBox()
        self.cmb_gate_window.addItems(["rectangular", "hann", "hamming", "kaiser", "boxcar"])
        self.cmb_gate_window.setCurrentText("rectangular")
        self.cmb_fft_window = QComboBox()
        self.cmb_fft_window.addItems(["rectangular", "hann", "hamming", "kaiser", "boxcar"])
        self.cmb_fft_window.setCurrentText("rectangular")

        self.edit_gate_center = QLineEdit("0.0")
        self.edit_gate_span = QLineEdit("0.15e-9")
        self.edit_gate_start = QLineEdit("")
        self.edit_gate_stop = QLineEdit("")
        self.edit_fft_pad = QLineEdit("4")
        self.chk_gate_all = QCheckBox("Apply to all S-parameter traces")
        self.chk_gate_all.setChecked(True)
        self.chk_synth_dc = QCheckBox("Synthetic DC for low-pass transforms")
        self.chk_synth_dc.setChecked(False)

        gate_btn_row = QHBoxLayout()
        self.btn_auto_gate = QPushButton("Auto Gate")
        self.btn_apply_gate = QPushButton("Apply Gate")
        gate_btn_row.addWidget(self.btn_auto_gate)
        gate_btn_row.addWidget(self.btn_apply_gate)

        layout.addRow(self.lbl_gating_hint)
        layout.addRow("Source network", self.cmb_gate_source)
        layout.addRow("Preview trace", self.cmb_gate_trace)
        layout.addRow("Transform mode", self.cmb_transform)
        layout.addRow("Gate mode", self.cmb_gate_mode)
        layout.addRow("Gate window", self.cmb_gate_window)
        layout.addRow("FFT window", self.cmb_fft_window)
        layout.addRow("Center (s)", self.edit_gate_center)
        layout.addRow("Span (s)", self.edit_gate_span)
        layout.addRow("Start (s, optional)", self.edit_gate_start)
        layout.addRow("Stop (s, optional)", self.edit_gate_stop)
        layout.addRow("FFT pad factor", self.edit_fft_pad)
        layout.addRow("Low-pass option", self.chk_synth_dc)
        layout.addRow("Scope", self.chk_gate_all)
        layout.addRow(gate_btn_row)
        return group

    def _build_validation_group(self) -> QGroupBox:
        group = QGroupBox("4) Validation Summary")
        layout = QVBoxLayout(group)

        form = QFormLayout()
        self.lbl_validation_target = QLabel("-")
        self.lbl_passivity = QLabel("-")
        self.lbl_reciprocity = QLabel("-")
        self.lbl_causality = QLabel("-")
        self.lbl_smoothness = QLabel("-")
        self.lbl_passivity_value = QLabel("-")
        self.lbl_reciprocity_value = QLabel("-")
        self.lbl_gd_std = QLabel("-")
        form.addRow("Target", self.lbl_validation_target)
        form.addRow("Passivity", self.lbl_passivity)
        form.addRow("Reciprocity", self.lbl_reciprocity)
        form.addRow("Causality", self.lbl_causality)
        form.addRow("Smoothness", self.lbl_smoothness)
        form.addRow("Max passivity excess", self.lbl_passivity_value)
        form.addRow("Reciprocity error", self.lbl_reciprocity_value)
        form.addRow("Group delay std", self.lbl_gd_std)
        layout.addLayout(form)

        self.warning_list = QListWidget()
        self.warning_list.setMinimumHeight(110)
        layout.addWidget(QLabel("Warnings"))
        layout.addWidget(self.warning_list)

        row = QHBoxLayout()
        self.btn_run_validation = QPushButton("Run Validation")
        self.btn_clear_validation = QPushButton("Clear")
        row.addWidget(self.btn_run_validation)
        row.addWidget(self.btn_clear_validation)
        layout.addLayout(row)
        return group

    def _build_plot_tabs(self) -> QTabWidget:
        self.plot_tabs = QTabWidget()
        self.panel_sparams = MatplotlibPanel()
        self.panel_time = MatplotlibPanel()
        self.panel_trl = MatplotlibPanel()
        self.panel_validation = MatplotlibPanel()
        self.plot_tabs.addTab(self.panel_sparams, "S-Parameters")
        self.plot_tabs.addTab(self.panel_time, "Time Domain")
        self.plot_tabs.addTab(self.panel_trl, "TRL Diagnostics")
        self.plot_tabs.addTab(self.panel_validation, "Validation")
        return self.plot_tabs

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.action_new.triggered.connect(self.new_project)
        self.action_load.triggered.connect(self.load_project)
        self.action_save.triggered.connect(self.save_project)
        self.action_save_as.triggered.connect(self.save_project_as)

        self.btn_import_files.clicked.connect(self.import_files)
        self.btn_import_toolbar.clicked.connect(self.import_files)
        self.btn_remove_file.clicked.connect(self.remove_selected_file)
        self.file_list.currentItemChanged.connect(self.on_file_selection_changed)
        self.file_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.btn_refresh_plots.clicked.connect(self.refresh_live_plots)
        self.btn_run_trl.clicked.connect(self.run_trl)
        self.btn_solve_toolbar.clicked.connect(self.run_trl)
        self.btn_validate_toolbar.clicked.connect(self.run_validation)
        self.btn_run_validation.clicked.connect(self.run_validation)
        self.btn_clear_validation.clicked.connect(self.clear_validation_widgets)
        self.btn_apply_gate.clicked.connect(self.apply_gate)
        self.btn_plot_limits.clicked.connect(self.open_plot_limits_dialog)
        self.btn_auto_gate.clicked.connect(self.auto_gate)
        self.cmb_gate_source.currentIndexChanged.connect(self.refresh_live_plots)
        self.cmb_gate_trace.currentIndexChanged.connect(self.refresh_live_plots)
        self.cmb_method.currentIndexChanged.connect(self._update_method_ui)

    # ------------------------------------------------------------------
    # Project state
    # ------------------------------------------------------------------
    def _update_window_title(self):
        suffix = " *" if self.dirty else ""
        self.setWindowTitle(f"RF De-Embedding Lab - {self.project_name}{suffix}")

    def _mark_dirty(self, dirty: bool = True):
        self.dirty = dirty
        self._update_window_title()

    def collect_ui_state(self) -> dict:
        return {
            "method": self.cmb_method.currentText(),
            "thru": self.cmb_thru.currentText(),
            "line1": self.cmb_line1.currentText(),
            "line2": self.cmb_line2.currentText(),
            "dut": self.cmb_dut.currentText(),
            "line1_len": self.edit_line1_len.text(),
            "line2_len": self.edit_line2_len.text(),
            "thru_len": self.edit_thru_len.text(),
            "mirror_sym": self.chk_mirror_sym.isChecked(),
            "gate_source": self.cmb_gate_source.currentText(),
            "gate_trace": self.cmb_gate_trace.currentText(),
            "transform_mode": self.cmb_transform.currentText(),
            "gate_mode": self.cmb_gate_mode.currentText(),
            "gate_window": self.cmb_gate_window.currentText(),
            "plot_xmin": self.plot_limits_dialog.edit_xmin.text() if self.plot_limits_dialog else "",
            "plot_xmax": self.plot_limits_dialog.edit_xmax.text() if self.plot_limits_dialog else "",
            "plot_ymin": self.plot_limits_dialog.edit_ymin.text() if self.plot_limits_dialog else "",
            "plot_ymax": self.plot_limits_dialog.edit_ymax.text() if self.plot_limits_dialog else "",
            "fft_window": self.cmb_fft_window.currentText(),
            "gate_center": self.edit_gate_center.text(),
            "gate_span": self.edit_gate_span.text(),
            "gate_start": self.edit_gate_start.text(),
            "gate_stop": self.edit_gate_stop.text(),
            "fft_pad": self.edit_fft_pad.text(),
            "gate_all": self.chk_gate_all.isChecked(),
            "synthetic_dc": self.chk_synth_dc.isChecked(),
            "selected_file": self.file_list.currentItem().text() if self.file_list.currentItem() else "",
            "last_browse_dir": self.last_browse_dir,
        }

    def apply_ui_state(self, state: dict):
        def set_combo(combo: QComboBox, value: str):
            if value and combo.findText(value) >= 0:
                combo.setCurrentText(value)

        set_combo(self.cmb_method, state.get("method", ""))
        set_combo(self.cmb_thru, state.get("thru", ""))
        set_combo(self.cmb_line1, state.get("line1", ""))
        set_combo(self.cmb_line2, state.get("line2", ""))
        set_combo(self.cmb_dut, state.get("dut", ""))
        set_combo(self.cmb_gate_source, state.get("gate_source", ""))
        set_combo(self.cmb_gate_trace, state.get("gate_trace", ""))
        set_combo(self.cmb_transform, state.get("transform_mode", ""))
        set_combo(self.cmb_gate_mode, state.get("gate_mode", ""))
        set_combo(self.cmb_gate_window, state.get("gate_window", ""))
        set_combo(self.cmb_fft_window, state.get("fft_window", ""))

        self.edit_line1_len.setText(state.get("line1_len", "0.010"))
        self.edit_line2_len.setText(state.get("line2_len", "0.020"))
        self.edit_thru_len.setText(state.get("thru_len", "0.0"))
        self.chk_mirror_sym.setChecked(bool(state.get("mirror_sym", True)))
        self.edit_gate_center.setText(state.get("gate_center", "0.0"))
        self.edit_gate_span.setText(state.get("gate_span", "0.15e-9"))
        self.edit_gate_start.setText(state.get("gate_start", ""))
        self.edit_gate_stop.setText(state.get("gate_stop", ""))
        self.edit_fft_pad.setText(state.get("fft_pad", "4"))
        self.chk_gate_all.setChecked(bool(state.get("gate_all", True)))
        self.chk_synth_dc.setChecked(bool(state.get("synthetic_dc", False)))
        self.last_browse_dir = state.get("last_browse_dir", self.last_browse_dir)
        self._update_method_ui()

        if any(state.get(key, "") for key in ("plot_xmin", "plot_xmax", "plot_ymin", "plot_ymax")):
            dialog = self._ensure_plot_limits_dialog()
            dialog.edit_xmin.setText(state.get("plot_xmin", ""))
            dialog.edit_xmax.setText(state.get("plot_xmax", ""))
            dialog.edit_ymin.setText(state.get("plot_ymin", ""))
            dialog.edit_ymax.setText(state.get("plot_ymax", ""))

        selected_file = state.get("selected_file", "")
        if selected_file:
            for idx in range(self.file_list.count()):
                item = self.file_list.item(idx)
                if item.text() == selected_file:
                    self.file_list.setCurrentRow(idx)
                    break

    def maybe_save_before_destructive_action(self) -> bool:
        if not self.dirty:
            return True
        resp = QMessageBox.question(
            self,
            "Unsaved changes",
            "This project has unsaved changes. Save before continuing?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if resp == QMessageBox.Cancel:
            return False
        if resp == QMessageBox.Yes:
            return self.save_project()
        return True

    def new_project(self):
        if not self.maybe_save_before_destructive_action():
            return
        self.networks.clear()
        self.latest_trl_result = None
        self.latest_deembedded_name = None
        self.latest_validation_report = None
        self.project_name = "Untitled Project"
        self.project_file = None
        self.file_list.clear()
        self._refresh_network_combos()
        self.clear_metadata_labels()
        self.clear_validation_widgets()
        self.panel_sparams.clear()
        self.panel_time.clear()
        self.panel_trl.clear()
        self.panel_validation.clear()
        self._mark_dirty(False)
        self._set_status("New project created")

    def save_project(self) -> bool:
        if self.project_file is None:
            return self.save_project_as()
        try:
            self.project_manager.save_project(
                self.project_file,
                self.networks,
                ui_state=self.collect_ui_state(),
                latest_deembedded_name=self.latest_deembedded_name,
                project_name=self.project_name,
            )
            self._mark_dirty(False)
            self._set_status(f"Saved project to {self.project_file}")
            return True
        except Exception as exc:
            self.show_error("Save Project Failed", exc)
            return False

    def save_project_as(self) -> bool:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(Path(self.last_browse_dir) / "rf_deembed_project.json"),
            "Project JSON (*.json)",
        )
        if not file_path:
            return False
        self.project_file = Path(file_path)
        self.last_browse_dir = str(self.project_file.parent)
        self.project_name = self.project_file.stem
        self._update_window_title()
        return self.save_project()

    def load_project(self):
        if not self.maybe_save_before_destructive_action():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            self.last_browse_dir,
            "Project JSON (*.json)",
        )
        if not file_path:
            return
        try:
            networks, ui_state, latest_deembedded_name, project_name = self.project_manager.load_project(file_path)
            self.networks = networks
            self.latest_deembedded_name = latest_deembedded_name
            self.latest_trl_result = None
            self.latest_validation_report = None
            self.project_file = Path(file_path)
            self.last_browse_dir = str(self.project_file.parent)
            self.project_name = project_name

            self.file_list.clear()
            for name in self.networks:
                self.file_list.addItem(QListWidgetItem(name))
            self._refresh_network_combos()
            self.apply_ui_state(ui_state)
            if self.file_list.count() > 0 and self.file_list.currentItem() is None:
                self.file_list.setCurrentRow(0)
            self._mark_dirty(False)
            self.refresh_live_plots()
            self._set_status(f"Loaded project from {self.project_file}")
        except Exception as exc:
            self.show_error("Load Project Failed", exc)

    def closeEvent(self, event):
        if self.maybe_save_before_destructive_action():
            event.accept()
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # File import / selection
    # ------------------------------------------------------------------
    def import_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Touchstone Files",
            self.last_browse_dir,
            "Touchstone (*.s1p *.s2p)",
        )
        if not files:
            return

        self.last_browse_dir = str(Path(files[0]).resolve().parent)

        errors = []
        added = 0
        for file_path in files:
            try:
                ntwk = SParameterData.from_touchstone(file_path)
                base_name = ntwk.name
                unique_name = self._ensure_unique_name(base_name)
                ntwk.name = unique_name
                self.networks[unique_name] = ntwk
                item = QListWidgetItem(unique_name)
                item.setToolTip(file_path)
                self.file_list.addItem(item)
                added += 1
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")

        self._refresh_network_combos()
        if self.file_list.count() > 0 and self.file_list.currentItem() is None:
            self.file_list.setCurrentRow(0)
        if added:
            self._mark_dirty(True)
        self._set_status(f"Imported {added} file(s)")

        if errors:
            QMessageBox.warning(self, "Import warnings", "\n".join(errors))

    def remove_selected_file(self):
        item = self.file_list.currentItem()
        if item is None:
            return
        name = item.text()
        self.networks.pop(name, None)
        row = self.file_list.row(item)
        self.file_list.takeItem(row)
        if self.latest_deembedded_name == name:
            self.latest_deembedded_name = None
        self._refresh_network_combos()
        self.clear_metadata_labels()
        self.clear_validation_widgets()
        self.refresh_live_plots()
        self._mark_dirty(True)
        self._set_status(f"Removed {name}")

    def on_file_selection_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]):
        _ = current
        _ = previous
        self._update_selection_summary()

    def on_selection_changed(self):
        self._update_selection_summary()
        self.refresh_live_plots()

    def _selected_file_names(self) -> List[str]:
        names = [item.text() for item in self.file_list.selectedItems() if item.text() in self.networks]
        if names:
            return names
        current_item = self.file_list.currentItem()
        if current_item is not None and current_item.text() in self.networks:
            return [current_item.text()]
        return []

    def _update_selection_summary(self):
        names = self._selected_file_names()
        if not names:
            self.clear_metadata_labels()
            return
        if len(names) == 1:
            ntwk = self.networks.get(names[0])
            if ntwk is None:
                self.clear_metadata_labels()
                return
            self.lbl_file_name.setText(ntwk.name)
            self.lbl_file_points.setText(str(ntwk.n_freq))
            self.lbl_file_ports.setText(str(ntwk.n_ports))
            self.lbl_file_freq.setText(f"{ntwk.freq_hz[0]/1e9:.3f} – {ntwk.freq_hz[-1]/1e9:.3f} GHz")
            self.lbl_file_z0.setText(str(ntwk.z0))
            return

        selected = [self.networks[name] for name in names if name in self.networks]
        if not selected:
            self.clear_metadata_labels()
            return
        first = selected[0]
        same_grid = all(first.same_grid_as(other) for other in selected[1:])
        same_ports = len({ntwk.n_ports for ntwk in selected}) == 1
        same_z0 = len({str(ntwk.z0) for ntwk in selected}) == 1
        self.lbl_file_name.setText(f"{len(selected)} files selected")
        self.lbl_file_points.setText(", ".join(str(ntwk.n_freq) for ntwk in selected[:3]) + ("..." if len(selected) > 3 else ""))
        self.lbl_file_ports.setText(str(first.n_ports) if same_ports else "mixed")
        self.lbl_file_freq.setText(
            f"{min(ntwk.freq_hz[0] for ntwk in selected)/1e9:.3f} – {max(ntwk.freq_hz[-1] for ntwk in selected)/1e9:.3f} GHz"
            + ("" if same_grid else " (mixed grids)")
        )
        self.lbl_file_z0.setText(str(first.z0) if same_z0 else "mixed")

    def clear_metadata_labels(self):
        for lbl in [self.lbl_file_name, self.lbl_file_points, self.lbl_file_ports, self.lbl_file_freq, self.lbl_file_z0]:
            lbl.setText("-")

    def _refresh_network_combos(self):
        combo_list = [self.cmb_thru, self.cmb_line1, self.cmb_line2, self.cmb_dut, self.cmb_gate_source]
        names = list(self.networks.keys())
        for combo in combo_list:
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if combo is self.cmb_thru:
                combo.addItem("(None)")
            combo.addItems(names)
            if current and combo.findText(current) >= 0:
                combo.setCurrentText(current)
            elif combo is self.cmb_thru:
                combo.setCurrentText("(None)")
            combo.blockSignals(False)
        self._update_method_ui()

    def _ensure_unique_name(self, base_name: str) -> str:
        if base_name not in self.networks:
            return base_name
        idx = 1
        while f"{base_name}_{idx}" in self.networks:
            idx += 1
        return f"{base_name}_{idx}"

    # ------------------------------------------------------------------
    # Gating actions
    # ------------------------------------------------------------------
    def current_gate_config(self) -> GateConfig:
        kwargs = {
            "transform_mode": self.cmb_transform.currentText(),
            "gate_mode": self.cmb_gate_mode.currentText(),
            "window": self.cmb_gate_window.currentText(),
            "fft_window": self.cmb_fft_window.currentText(),
            "fft_pad_factor": int(float(self.edit_fft_pad.text().strip() or "4")),
            "synthetic_dc": self.chk_synth_dc.isChecked(),
        }
        start_text = self.edit_gate_start.text().strip()
        stop_text = self.edit_gate_stop.text().strip()
        if start_text and stop_text:
            kwargs["start_s"] = float(start_text)
            kwargs["stop_s"] = float(stop_text)
        else:
            kwargs["center_s"] = float(self.edit_gate_center.text().strip() or "0")
            kwargs["span_s"] = float(self.edit_gate_span.text().strip() or "0")
        return GateConfig(**kwargs)

    def trace_to_indices(self) -> tuple[int, int]:
        text = self.cmb_gate_trace.currentText()
        mapping = {"S11": (0, 0), "S21": (1, 0), "S12": (0, 1), "S22": (1, 1)}
        return mapping[text]

    def auto_gate(self):
        try:
            ntwk = self.get_selected_network(self.cmb_gate_source)
            i, j = self.trace_to_indices()
            cfg = self.current_gate_config()
            auto_cfg = self.time_gating.auto_gate_from_peaks(ntwk, i=i, j=j, cfg=cfg)
            if auto_cfg.center_s is not None:
                self.edit_gate_center.setText(f"{auto_cfg.center_s:.6e}")
            if auto_cfg.span_s is not None:
                self.edit_gate_span.setText(f"{auto_cfg.span_s:.6e}")
            self.cmb_gate_mode.setCurrentText(auto_cfg.gate_mode)
            self.refresh_live_plots()
            self._set_status("Auto gate updated from strongest peak")
        except Exception as exc:
            self.show_error("Auto Gate Failed", exc)

    def apply_gate(self):
        try:
            ntwk = self.get_selected_network(self.cmb_gate_source)
            cfg = self.current_gate_config()
            if self.chk_gate_all.isChecked():
                gated = self.time_gating.apply_gate_all(ntwk, cfg)
            else:
                i, j = self.trace_to_indices()
                gated = self.time_gating.apply_gate(ntwk, i=i, j=j, cfg=cfg)
            gated.name = self._ensure_unique_name(gated.name)
            self.networks[gated.name] = gated
            self.file_list.addItem(QListWidgetItem(gated.name))
            self._refresh_network_combos()
            self.cmb_gate_source.setCurrentText(gated.name)
            self.file_list.setCurrentRow(self.file_list.count() - 1)
            self.refresh_live_plots()
            self._mark_dirty(True)
            self._set_status(f"Applied gate and created '{gated.name}'")
        except Exception as exc:
            self.show_error("Gating Failed", exc)

    # ------------------------------------------------------------------
    # TRL solve
    # ------------------------------------------------------------------
    def run_trl(self):
        try:
            method = self.cmb_method.currentText()
            line1 = self.get_selected_network(self.cmb_line1)
            line1_len = self._parse_optional_float(self.edit_line1_len)
            line2_len = self._parse_optional_float(self.edit_line2_len)
            thru_len = self._parse_optional_float(self.edit_thru_len, default=0.0)
            mirror = self.chk_mirror_sym.isChecked()

            if method == "Single-line TRL":
                thru = self.get_selected_network(self.cmb_thru)
                dut = self.get_selected_network(self.cmb_dut, allow_none=True)
                if line1_len is None:
                    raise ValueError("Enter the LINE 1 physical length in meters for Single-line TRL.")
                cfg = TRLConfig(
                    line_lengths_m=[line1_len],
                    thru_length_m=thru_len or 0.0,
                    mirror_symmetric_fixture=mirror,
                )
                result = self.trl_engine.single_line_trl_fit(thru, line1, cfg, dut=dut)
            elif method == "Multiline TRL":
                thru = self.get_selected_network(self.cmb_thru)
                dut = self.get_selected_network(self.cmb_dut, allow_none=True)
                line2 = self.get_selected_network(self.cmb_line2)
                if line1_len is None or line2_len is None:
                    raise ValueError("Enter both LINE lengths in meters for Multiline TRL.")
                cfg = TRLConfig(
                    line_lengths_m=[line1_len, line2_len],
                    thru_length_m=thru_len or 0.0,
                    mirror_symmetric_fixture=mirror,
                )
                result = self.trl_engine.multiline_trl_fit(thru, [line1, line2], cfg, dut=dut)
            elif method == "Short/Long Differential":
                line2 = self.get_selected_network(self.cmb_line2)
                if line1_len is None or line2_len is None:
                    raise ValueError(
                        "Time-gate preview does not require line lengths, but Solve still needs the short and long physical lengths in meters.\n\n"
                        "Use S11/S22 to place the gate first, then enter the short and long lengths when you are ready to extract propagation."
                    )
                result = self.trl_engine.short_long_extract_line(
                    short_line=line1,
                    long_line=line2,
                    short_length_m=line1_len,
                    long_length_m=line2_len,
                )
            else:
                raise NotImplementedError("Known Fixture De-cascade panel wiring can be added next")

            self.latest_trl_result = result
            if result.deembedded_dut is not None:
                result.deembedded_dut.name = self._ensure_unique_name(result.deembedded_dut.name)
                self.networks[result.deembedded_dut.name] = result.deembedded_dut
                self.file_list.addItem(QListWidgetItem(result.deembedded_dut.name))
                self.latest_deembedded_name = result.deembedded_dut.name
                self._refresh_network_combos()
                self.cmb_dut.setCurrentText(result.deembedded_dut.name)

            self.refresh_live_plots()
            self._mark_dirty(True)
            self._set_status(f"{method} completed")
            self.run_validation(auto=True)
        except Exception as exc:
            self.show_error("TRL Solve Failed", exc)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_target_network(self) -> Optional[SParameterData]:
        if self.latest_deembedded_name and self.latest_deembedded_name in self.networks:
            return self.networks[self.latest_deembedded_name]
        current_item = self.file_list.currentItem()
        if current_item is not None:
            return self.networks.get(current_item.text())
        return self.get_selected_network(self.cmb_dut, allow_none=True)

    def run_validation(self, auto: bool = False):
        try:
            ntwk = self.validation_target_network()
            if ntwk is None:
                if not auto:
                    raise ValueError("No network available to validate")
                return
            report = ValidationChecks.build_report(ntwk)
            self.latest_validation_report = report
            self.update_validation_widgets(ntwk.name, report)
            if not auto:
                self._set_status(f"Validation completed for '{ntwk.name}'")
        except Exception as exc:
            if not auto:
                self.show_error("Validation Failed", exc)

    def update_validation_widgets(self, target_name: str, report):
        def status_text(ok: bool, warn: bool = False) -> str:
            if warn:
                return "WARN"
            return "OK" if ok else "FAIL"

        self.lbl_validation_target.setText(target_name)
        self.lbl_passivity.setText(status_text(report.passivity_ok))
        self.lbl_reciprocity.setText(status_text(report.reciprocity_ok))
        self.lbl_causality.setText(status_text(not report.causality_warning, warn=report.causality_warning))
        self.lbl_smoothness.setText(status_text(not report.smoothness_warning, warn=report.smoothness_warning))
        self.lbl_passivity_value.setText(f"{report.max_passivity_excess_db:.2f} dB")
        self.lbl_reciprocity_value.setText(f"{report.reciprocity_error_db:.2f} dB")
        self.lbl_gd_std.setText(f"{report.group_delay_std_ps:.1f} ps")

        self.warning_list.clear()
        if report.warnings:
            for msg in report.warnings:
                self.warning_list.addItem(QListWidgetItem(msg))
        else:
            self.warning_list.addItem(QListWidgetItem("No warnings"))

    def clear_validation_widgets(self):
        self.latest_validation_report = None
        self.lbl_validation_target.setText("-")
        self.lbl_passivity.setText("-")
        self.lbl_reciprocity.setText("-")
        self.lbl_causality.setText("-")
        self.lbl_smoothness.setText("-")
        self.lbl_passivity_value.setText("-")
        self.lbl_reciprocity_value.setText("-")
        self.lbl_gd_std.setText("-")
        self.warning_list.clear()

    # ------------------------------------------------------------------
    # Plot refresh
    # ------------------------------------------------------------------
    def refresh_live_plots(self):
        try:
            selected_names = self._selected_file_names()
            if not selected_names:
                fallback = self.cmb_gate_source.currentText().strip()
                if fallback and fallback in self.networks:
                    selected_names = [fallback]
            selected_networks = [self.networks[name] for name in selected_names if name in self.networks]
            if not selected_networks:
                self.panel_sparams.clear()
                self.panel_time.clear()
                self.panel_trl.clear()
                self.panel_validation.clear()
                return

            gate_cfg = self.current_gate_config()
            gate_start = float(gate_cfg.start_s) if gate_cfg.start_s is not None else None
            gate_stop = float(gate_cfg.stop_s) if gate_cfg.stop_s is not None else None
            if gate_start is None or gate_stop is None:
                if gate_cfg.center_s is not None and gate_cfg.span_s is not None:
                    gate_start = gate_cfg.center_s - gate_cfg.span_s / 2
                    gate_stop = gate_cfg.center_s + gate_cfg.span_s / 2

            i, j = self.trace_to_indices()
            if len(selected_networks) > 1:
                fig_s = self.plotter.plot_sparameter_overlay(selected_networks, title="S-Parameter Overlay")
                self.panel_sparams.set_figure(fig_s)
                td_results = [self.time_gating.to_time_domain(ntwk, i=i, j=j, cfg=gate_cfg) for ntwk in selected_networks]
                fig_t = self.plotter.plot_time_domain_overlay(td_results, gate_start_s=gate_start, gate_stop_s=gate_stop)
                self.panel_time.set_figure(fig_t)
                if self.latest_trl_result is not None and self.latest_trl_result.alpha_np_per_m is not None:
                    fig_trl = self.plotter.plot_trl_diagnostics(self.latest_trl_result, selected_networks[0].freq_hz)
                    self.panel_trl.set_figure(fig_trl)
                else:
                    self.panel_trl.clear()
                self.panel_validation.clear()
                self._set_status(f"Overlaying {len(selected_networks)} files")
                return

            ntwk = selected_networks[0]
            comparison = None
            if self.latest_deembedded_name and ntwk.name != self.latest_deembedded_name:
                comparison = self.networks.get(self.latest_deembedded_name)
                if comparison is not None and not ntwk.same_grid_as(comparison):
                    comparison = None
            fig_s = self.plotter.plot_sparameters(ntwk, comparison, title=f"S-Parameters: {ntwk.name}")
            self.panel_sparams.set_figure(fig_s)

            td = self.time_gating.to_time_domain(ntwk, i=i, j=j, cfg=gate_cfg)
            fig_t = self.plotter.plot_time_domain(td, gate_start_s=gate_start, gate_stop_s=gate_stop)
            self.panel_time.set_figure(fig_t)

            if self.latest_trl_result is not None and self.latest_trl_result.alpha_np_per_m is not None:
                fig_trl = self.plotter.plot_trl_diagnostics(self.latest_trl_result, ntwk.freq_hz)
                self.panel_trl.set_figure(fig_trl)
            else:
                self.panel_trl.clear()

            if self.latest_deembedded_name:
                de = self.networks.get(self.latest_deembedded_name)
                raw_for_validation = self.get_selected_network(self.cmb_dut, allow_none=True)
                if de is not None and raw_for_validation is not None and raw_for_validation.same_grid_as(de):
                    fig_v = self.plotter.plot_validation_overlay(raw_for_validation, de)
                    self.panel_validation.set_figure(fig_v)
                else:
                    self.panel_validation.clear()
            else:
                self.panel_validation.clear()

        except Exception as exc:
            self._set_status(f"Plot refresh warning: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_selected_network(self, combo: QComboBox, allow_none: bool = False) -> Optional[SParameterData]:
        name = combo.currentText().strip()
        if name == "(None)":
            if allow_none:
                return None
            raise ValueError("A required network is not selected")
        if not name:
            if allow_none:
                return None
            raise ValueError("A required network is not selected")
        ntwk = self.networks.get(name)
        if ntwk is None and not allow_none:
            raise ValueError(f"Network '{name}' not found")
        return ntwk

    def _parse_optional_float(self, edit: QLineEdit, default: Optional[float] = None) -> Optional[float]:
        text = edit.text().strip()
        if not text:
            return default
        return float(text)

    def _update_method_ui(self):
        method = self.cmb_method.currentText()
        is_short_long = method == "Short/Long Differential"
        is_multiline = method == "Multiline TRL"
        is_single = method == "Single-line TRL"

        self.cmb_thru.setEnabled(not is_short_long)
        self.edit_thru_len.setEnabled(not is_short_long)
        if is_short_long and self.cmb_thru.findText("(None)") >= 0:
            self.cmb_thru.setCurrentText("(None)")

        self.cmb_line2.setEnabled(is_multiline or is_short_long)
        self.edit_line2_len.setEnabled(is_multiline or is_short_long)
        self.cmb_dut.setEnabled(not is_short_long)

        self.edit_line1_len.setPlaceholderText("Length in meters")
        self.edit_line2_len.setPlaceholderText("Length in meters")
        self.edit_thru_len.setPlaceholderText("0 for ideal thru")

        if is_short_long:
            self.edit_line1_len.setPlaceholderText("Short physical length in meters (needed for Solve)")
            self.edit_line2_len.setPlaceholderText("Long physical length in meters (needed for Solve)")
            self._set_status("Short/Long Differential selected: THRU disabled. Use S11/S22 time-gating visually first; enter lengths only when solving.")
        elif is_single:
            self._set_status("Single-line TRL selected")
        elif is_multiline:
            self._set_status("Multiline TRL selected")

    def _ensure_plot_limits_dialog(self) -> PlotLimitsDialog:
        if self.plot_limits_dialog is None:
            self.plot_limits_dialog = PlotLimitsDialog(self)
        return self.plot_limits_dialog

    def open_plot_limits_dialog(self):
        dialog = self._ensure_plot_limits_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _active_plot_panel(self) -> Optional[MatplotlibPanel]:
        widget = self.plot_tabs.currentWidget()
        return widget if isinstance(widget, MatplotlibPanel) else None

    def apply_plot_limits_from_dialog(self, xmin_text: str, xmax_text: str, ymin_text: str, ymax_text: str):
        panel = self._active_plot_panel()
        if panel is None or panel.canvas is None or panel.canvas.figure is None:
            raise ValueError("No active plot is available")

        xmin = self._parse_optional_text_float(xmin_text)
        xmax = self._parse_optional_text_float(xmax_text)
        ymin = self._parse_optional_text_float(ymin_text)
        ymax = self._parse_optional_text_float(ymax_text)

        for ax in panel.canvas.figure.axes:
            if xmin is not None or xmax is not None:
                cur_xmin, cur_xmax = ax.get_xlim()
                ax.set_xlim(xmin if xmin is not None else cur_xmin, xmax if xmax is not None else cur_xmax)
            if ymin is not None or ymax is not None:
                cur_ymin, cur_ymax = ax.get_ylim()
                ax.set_ylim(ymin if ymin is not None else cur_ymin, ymax if ymax is not None else cur_ymax)

        panel.canvas.draw_idle()
        self._set_status("Applied manual plot limits to active tab")

    def reset_plot_limits_active_panel(self):
        panel = self._active_plot_panel()
        if panel is None or panel.canvas is None or panel.canvas.figure is None:
            return
        for ax in panel.canvas.figure.axes:
            ax.relim()
            ax.autoscale(enable=True, axis="both", tight=False)
            ax.autoscale_view()
        panel.canvas.draw_idle()
        self._set_status("Reset plot limits on active tab")

    def _parse_optional_text_float(self, text: str) -> Optional[float]:
        text = text.strip()
        if not text:
            return None
        return float(text)

    def show_error(self, title: str, exc: Exception):
        QMessageBox.critical(self, title, str(exc))
        self._set_status(f"Error: {exc}")

    def _set_status(self, text: str):
        self.status_label.setText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeembedMainWindow()
    window.show()
    sys.exit(app.exec())
