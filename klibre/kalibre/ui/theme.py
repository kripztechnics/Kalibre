"""
Thème sombre inspiré de ton interface MATLAB FxLMS.

Centralise les couleurs pour que toute l'UI reste cohérente.
"""

# Fond général
BG_DARK = "#1e1e1e"
BG_PANEL = "#252526"
BG_INPUT = "#2d2d30"
BORDER = "#3e3e42"

# Texte
TEXT = "#e0e0e0"
TEXT_DIM = "#9e9e9e"

# Boutons d'action (comme MATLAB : rouge stop, bleu génération, vert activer)
BTN_STOP = "#c0392b"
BTN_PRIMARY = "#2980b9"
BTN_SUCCESS = "#27ae60"
BTN_NEUTRAL = "#444444"

# Courbes des graphiques
COLOR_REF = "#3498db"       # signal émis / référence
COLOR_MIC = "#e67e22"       # capté au micro
COLOR_SUM = "#2ecc71"       # somme / combiné
COLOR_TARGET = "#9b59b6"    # profil cible
COLOR_ERROR = "#e74c3c"     # erreur / écart

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT};
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 10pt;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG_PANEL};
}}
QTabBar::tab {{
    background: {BG_INPUT};
    color: {TEXT_DIM};
    padding: 6px 14px;
    border: 1px solid {BORDER};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background: {BG_PANEL};
    color: {TEXT};
}}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    color: {TEXT};
    min-height: 1.2em;
}}
QSpinBox, QDoubleSpinBox {{
    padding-right: 22px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid {BORDER};
    border-top-right-radius: 5px;
    background: {BG_PANEL};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid {BORDER};
    border-bottom-right-radius: 5px;
    background: {BG_PANEL};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: #4a4a4e;
}}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
    background: #3a3a3e;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {TEXT_DIM};
    margin-right: 6px;
}}

QPushButton {{
    background: {BTN_NEUTRAL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 14px;
    color: {TEXT};
}}
QPushButton:hover {{
    background: #4d4d52;
    border-color: #5a5a5e;
}}
QPushButton:pressed {{
    background: #38383c;
}}

QTableWidget {{
    background: {BG_INPUT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
}}
QHeaderView::section {{
    background: {BG_PANEL};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    padding: 4px;
}}

QCheckBox {{
    spacing: 6px;
}}

QStatusBar {{
    background: {BG_PANEL};
    color: {TEXT_DIM};
}}

QSplitter::handle {{
    background: {BORDER};
    width: 6px;
}}
QSplitter::handle:hover {{
    background: #5a5a5e;
}}

QScrollArea {{
    border: none;
    background: {BG_DARK};
}}
QScrollBar:vertical {{
    background: {BG_PANEL};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #555;
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar:horizontal {{
    background: {BG_PANEL};
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: #555;
    border-radius: 4px;
    min-width: 24px;
}}
"""
