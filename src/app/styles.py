"""QSS styles for the application."""

MAIN_STYLE = """
QMainWindow {
    background-color: #1e1e2e;
}

QWidget {
    color: #cdd6f4;
    font-family: "Segoe UI", "SF Pro Display", system-ui, sans-serif;
    font-size: 14px;
}

/* Sidebar */
#sidebar {
    background-color: #181825;
    border-right: 1px solid #313244;
    min-width: 200px;
    max-width: 200px;
}

#sidebar QPushButton {
    text-align: left;
    padding: 12px 16px;
    border: none;
    border-radius: 6px;
    margin: 2px 8px;
    background-color: transparent;
    color: #bac2de;
    font-size: 14px;
}

#sidebar QPushButton:hover {
    background-color: #313244;
    color: #cdd6f4;
}

#sidebar QPushButton:checked {
    background-color: #45475a;
    color: #cdd6f4;
    font-weight: bold;
}

/* Content area */
#content_area {
    background-color: #1e1e2e;
}

/* Status bar */
#status_bar {
    background-color: #181825;
    border-top: 1px solid #313244;
    padding: 4px 12px;
    font-size: 12px;
    color: #a6adc8;
}

/* Cards */
.card {
    background-color: #313244;
    border-radius: 8px;
    padding: 16px;
    margin: 8px;
}

/* Primary button */
QPushButton#primary {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
    font-size: 14px;
}

QPushButton#primary:hover {
    background-color: #74c7ec;
}

QPushButton#primary:pressed {
    background-color: #89dceb;
}

QPushButton#primary:disabled {
    background-color: #45475a;
    color: #6c7086;
}

/* Danger button */
QPushButton#danger {
    background-color: #f38ba8;
    color: #1e1e2e;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
}

QPushButton#danger:hover {
    background-color: #eba0ac;
}

/* Secondary button */
QPushButton#secondary {
    background-color: #45475a;
    color: #cdd6f4;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
}

QPushButton#secondary:hover {
    background-color: #585b70;
}

/* Generic buttons */
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}

QPushButton:hover {
    background-color: #585b70;
}

QPushButton:disabled {
    background-color: #313244;
    color: #6c7086;
}

/* Labels */
QLabel {
    color: #cdd6f4;
}

QLabel#title {
    font-size: 22px;
    font-weight: bold;
    color: #cdd6f4;
    margin-bottom: 8px;
}

QLabel#subtitle {
    font-size: 16px;
    color: #a6adc8;
    margin-bottom: 16px;
}

QLabel#section {
    font-size: 16px;
    font-weight: bold;
    color: #cdd6f4;
    margin-top: 12px;
}

/* Input fields */
QLineEdit, QTextEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 12px;
    color: #cdd6f4;
    font-size: 14px;
}

QLineEdit:focus, QTextEdit:focus {
    border-color: #89b4fa;
}

/* Combo box */
QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 12px;
    color: #cdd6f4;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #45475a;
    border: 1px solid #45475a;
}

/* Progress bar */
QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 4px;
}

/* Scroll area */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #1e1e2e;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* Table */
QTableWidget {
    background-color: #1e1e2e;
    gridline-color: #313244;
    border: 1px solid #313244;
    border-radius: 6px;
}

QTableWidget::item {
    padding: 8px;
}

QTableWidget::item:selected {
    background-color: #45475a;
}

QHeaderView::section {
    background-color: #181825;
    color: #a6adc8;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #313244;
    font-weight: bold;
}

/* Group box */
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 24px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #89b4fa;
}

/* Spin box */
QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 10px;
    color: #cdd6f4;
}

/* Check box */
QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

/* Tab widget */
QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 6px;
    background-color: #1e1e2e;
}

QTabBar::tab {
    background-color: #181825;
    color: #a6adc8;
    padding: 10px 20px;
    border: none;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:selected {
    color: #89b4fa;
    border-bottom: 2px solid #89b4fa;
}

QTabBar::tab:hover {
    color: #cdd6f4;
}
"""
