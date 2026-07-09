"""
Réglages de graphique type Excel — axes, auto, courbe de tendance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from kalibre.ui.theme import BG_INPUT, BG_PANEL, BORDER, TEXT, TEXT_DIM


@dataclass
class PlotViewConfig:
    auto_x: bool = True
    auto_y: bool = True
    x_min: float = 20.0
    x_max: float = 20_000.0
    y_min: float = -80.0
    y_max: float = 10.0
    show_trend: bool = False


def compute_log_trend(
    freqs: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    x_min: float = 20.0,
    x_max: float = 20_000.0,
    degree: int = 4,
    n_points: int = 250,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Tendance polynomiale sur log10(f) — style régression Excel."""
    mask = (freqs >= x_min) & (freqs <= x_max) & np.isfinite(values)
    if np.count_nonzero(mask) < degree + 2:
        return None

    x = np.log10(freqs[mask])
    y = values[mask]
    deg = min(degree, len(x) - 1)
    coef = np.polyfit(x, y, deg)
    x_line = np.linspace(float(x.min()), float(x.max()), n_points)
    y_line = np.polyval(coef, x_line)
    return (10.0**x_line, y_line.astype(np.float64))


class PlotSettingsBar(QWidget):
    """Panneau axes en 2 lignes — sans chevauchement."""

    changed = pyqtSignal()

    def __init__(
        self,
        *,
        log_x: bool = False,
        x_default: tuple[float, float] = (20.0, 20_000.0),
        y_default: tuple[float, float] = (-80.0, 10.0),
        x_range: tuple[float, float] = (0.001, 100_000.0),
        y_range: tuple[float, float] = (-200.0, 200.0),
        allow_trend: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._log_x = log_x
        self._x_default = x_default
        self._y_default = y_default
        self.config = PlotViewConfig(
            x_min=x_default[0],
            x_max=x_default[1],
            y_min=y_default[0],
            y_max=y_default[1],
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(0)

        frame = QFrame()
        frame.setStyleSheet(
            f"""
            QFrame#plotSettings {{
                background: {BG_INPUT};
                border: 1px solid {BORDER};
                border-radius: 4px;
            }}
            QLabel#axisTitle {{
                color: {TEXT};
                font-size: 9pt;
                font-weight: 600;
                min-width: 42px;
            }}
            QLabel#fieldLabel {{
                color: {TEXT_DIM};
                font-size: 8pt;
                min-width: 28px;
            }}
            QCheckBox {{
                font-size: 9pt;
                spacing: 4px;
            }}
            QPushButton {{
                font-size: 9pt;
                padding: 3px 10px;
            }}
            """
        )
        frame.setObjectName("plotSettings")

        grid = QGridLayout(frame)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(6, 1)

        self.chk_auto_x = QCheckBox("Auto")
        self.chk_auto_x.setChecked(True)
        self.chk_auto_x.setToolTip("Ajuste automatiquement l'axe X")
        self.spin_x_min = self._spin(log_x, x_default[0], x_range[0], x_range[1])
        self.spin_x_max = self._spin(log_x, x_default[1], x_range[0], x_range[1])

        self.chk_auto_y = QCheckBox("Auto")
        self.chk_auto_y.setChecked(True)
        self.chk_auto_y.setToolTip("Ajuste automatiquement l'axe Y")
        self.spin_y_min = self._spin(False, y_default[0], y_range[0], y_range[1], step=1.0)
        self.spin_y_max = self._spin(False, y_default[1], y_range[0], y_range[1], step=1.0)

        self._add_axis_row(
            grid,
            row=0,
            title="Axe X",
            auto=self.chk_auto_x,
            spin_min=self.spin_x_min,
            spin_max=self.spin_x_max,
        )
        self._add_axis_row(
            grid,
            row=1,
            title="Axe Y",
            auto=self.chk_auto_y,
            spin_min=self.spin_y_min,
            spin_max=self.spin_y_max,
        )

        actions = QVBoxLayout()
        actions.setSpacing(4)
        actions.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.chk_trend = QCheckBox("Tendance")
        self.chk_trend.setVisible(allow_trend)
        self.chk_trend.setToolTip("Courbe de tendance (régression log fréquence)")
        actions.addWidget(self.chk_trend)

        self.btn_reset = QPushButton("Réinitialiser")
        self.btn_reset.setToolTip("Revenir aux réglages par défaut")
        actions.addWidget(self.btn_reset)

        actions_wrap = QWidget()
        actions_wrap.setLayout(actions)
        grid.addWidget(actions_wrap, 0, 7, 2, 1, Qt.AlignmentFlag.AlignTop)

        outer.addWidget(frame)

        self.chk_auto_x.toggled.connect(self._sync_enabled)
        self.chk_auto_y.toggled.connect(self._sync_enabled)
        self.btn_reset.clicked.connect(self.reset_defaults)

        for w in (
            self.chk_auto_x,
            self.chk_auto_y,
            self.chk_trend,
            self.spin_x_min,
            self.spin_x_max,
            self.spin_y_min,
            self.spin_y_max,
        ):
            if isinstance(w, QCheckBox):
                w.toggled.connect(self._emit_changed)
            else:
                w.valueChanged.connect(self._emit_changed)

        self._sync_enabled()

    def _add_axis_row(
        self,
        grid: QGridLayout,
        *,
        row: int,
        title: str,
        auto: QCheckBox,
        spin_min: QDoubleSpinBox,
        spin_max: QDoubleSpinBox,
    ) -> None:
        title_lbl = QLabel(title)
        title_lbl.setObjectName("axisTitle")

        min_lbl = QLabel("Min")
        min_lbl.setObjectName("fieldLabel")
        max_lbl = QLabel("Max")
        max_lbl.setObjectName("fieldLabel")

        grid.addWidget(title_lbl, row, 0)
        grid.addWidget(auto, row, 1)
        grid.addWidget(min_lbl, row, 2, Qt.AlignmentFlag.AlignRight)
        grid.addWidget(spin_min, row, 3)
        grid.addWidget(max_lbl, row, 4, Qt.AlignmentFlag.AlignRight)
        grid.addWidget(spin_max, row, 5)

    @staticmethod
    def _spin(
        log_x: bool,
        value: float,
        lo: float,
        hi: float,
        *,
        step: float = 1.0,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(2 if not log_x else 1)
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setMinimumWidth(72)
        spin.setMaximumWidth(96)
        spin.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        spin.setStyleSheet(
            f"background: {BG_PANEL}; border: 1px solid {BORDER}; padding: 2px 4px;"
        )
        return spin

    def _sync_enabled(self) -> None:
        self.spin_x_min.setEnabled(not self.chk_auto_x.isChecked())
        self.spin_x_max.setEnabled(not self.chk_auto_x.isChecked())
        self.spin_y_min.setEnabled(not self.chk_auto_y.isChecked())
        self.spin_y_max.setEnabled(not self.chk_auto_y.isChecked())

    def _emit_changed(self) -> None:
        self._read_into_config()
        self.changed.emit()

    def _read_into_config(self) -> None:
        self.config.auto_x = self.chk_auto_x.isChecked()
        self.config.auto_y = self.chk_auto_y.isChecked()
        self.config.x_min = float(self.spin_x_min.value())
        self.config.x_max = float(self.spin_x_max.value())
        self.config.y_min = float(self.spin_y_min.value())
        self.config.y_max = float(self.spin_y_max.value())
        self.config.show_trend = self.chk_trend.isChecked()

    def reset_defaults(self) -> None:
        self.chk_auto_x.setChecked(True)
        self.chk_auto_y.setChecked(True)
        self.spin_x_min.setValue(self._x_default[0])
        self.spin_x_max.setValue(self._x_default[1])
        self.spin_y_min.setValue(self._y_default[0])
        self.spin_y_max.setValue(self._y_default[1])
        self.chk_trend.setChecked(False)
        self._read_into_config()
        self.changed.emit()
