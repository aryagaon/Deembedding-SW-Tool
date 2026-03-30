import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFormLayout, QScrollArea

from ui_app import DeembedMainWindow


def main():
    app = QApplication(sys.argv)
    window = DeembedMainWindow()
    window.resize(900, 620)
    window.show()
    app.processEvents()

    scroll_areas = window.centralWidget().findChildren(QScrollArea)
    assert scroll_areas, "Expected at least one QScrollArea in the main window"
    left_scroll = scroll_areas[0]

    method_layout = window._build_method_group.__self__.findChild(type(window.cmb_method.parentWidget().layout()), None)
    # Validate the actual form layouts used by the visible groups.
    visible_form_layouts = []
    for group in [window.cmb_method.parentWidget(), window.cmb_gate_source.parentWidget(), window.lbl_validation_target.parentWidget()]:
        layout = group.layout()
        if isinstance(layout, QFormLayout):
            visible_form_layouts.append(layout)

    print("window_size", window.size().width(), window.size().height())
    print("scroll_present", bool(scroll_areas))
    print("scroll_widget_resizable", left_scroll.widgetResizable())
    print("vertical_scroll_max", left_scroll.verticalScrollBar().maximum())
    print("method_wrap_policy", window.cmb_method.parentWidget().layout().rowWrapPolicy())
    print("gating_wrap_policy", window.cmb_gate_source.parentWidget().layout().rowWrapPolicy())

    assert left_scroll.widgetResizable()
    assert left_scroll.verticalScrollBar().maximum() > 0
    assert window.cmb_method.parentWidget().layout().rowWrapPolicy() == QFormLayout.WrapLongRows
    assert window.cmb_gate_source.parentWidget().layout().rowWrapPolicy() == QFormLayout.WrapLongRows

    window._mark_dirty(False)
    window.close()
    app.quit()


if __name__ == "__main__":
    main()
