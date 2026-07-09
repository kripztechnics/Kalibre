"""
Widget matplotlib embarqué dans Qt + barre de navigation (zoom / pan).
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from PyQt6.QtWidgets import QPushButton

from kalibre.ui.plot_settings import PlotSettingsBar, PlotViewConfig, compute_log_trend
from kalibre.ui.theme import BG_INPUT, BG_PANEL, BORDER, TEXT, TEXT_DIM

AXES_FACE = "#2a2a2e"
TREND_COLOR = "#f1c40f"

TOOLBAR_STYLE = f"""
QToolBar {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 2px;
    spacing: 2px;
}}
QToolButton {{
    background: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px;
    margin: 1px;
}}
QToolButton:hover {{
    background: #3e3e42;
}}
"""


class CompactNavigationToolbar(NavigationToolbar2QT):
    """Toolbar zoom/pan — sans configure subplots (remplacé par nos réglages)."""

    toolitems = [
        t for t in NavigationToolbar2QT.toolitems if t and t[0] not in ("Subplots",)
    ]

    def __init__(self, canvas, parent=None) -> None:
        super().__init__(canvas, parent)
        self.setMaximumHeight(34)
        if hasattr(self, "locLabel"):
            self.locLabel.hide()


class MplPanel(FigureCanvasQTAgg):
    """Canvas matplotlib — fond sombre, marges stables."""

    def __init__(self, min_height: int = 220) -> None:
        self.figure = Figure(facecolor=AXES_FACE, figsize=(6.5, 3.4), dpi=96)
        super().__init__(self.figure)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.axes = self.figure.add_subplot(111)
        self._style_axes(self.axes)
        self._fit_layout()

    def _style_axes(self, ax) -> None:
        axes_list = [ax] if ax is not None else self.figure.axes
        for axis in axes_list:
            axis.set_facecolor(AXES_FACE)
            axis.tick_params(colors=TEXT_DIM, labelsize=9)
            axis.xaxis.label.set_color(TEXT_DIM)
            axis.yaxis.label.set_color(TEXT_DIM)
            for spine in axis.spines.values():
                spine.set_color("#666666")
                spine.set_linewidth(1.2)
            axis.grid(True, color=BORDER, alpha=0.45, linewidth=0.7)

    def _fit_layout(self) -> None:
        self.figure.subplots_adjust(left=0.13, right=0.88, top=0.96, bottom=0.17)

    def clear_axes(self) -> None:
        self.axes.clear()
        self._style_axes(self.axes)

    def draw(self) -> None:
        self._fit_layout()
        super().draw()


class PlotCard(QWidget):
    """Graphique : réglages axes + toolbar zoom/pan + canvas."""

    settings_changed = pyqtSignal()

    def __init__(
        self,
        title: str,
        *,
        log_x: bool = False,
        ylabel: str = "",
        xlabel: str = "",
        min_canvas_height: int = 220,
        x_default: tuple[float, float] = (20.0, 20_000.0),
        y_default: tuple[float, float] = (-80.0, 10.0),
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] = (-200.0, 200.0),
        allow_trend: bool = False,
    ) -> None:
        super().__init__()
        self._log_x = log_x
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._default_x = x_default
        self._default_y = y_default
        self._trend_freqs: NDArray[np.float64] | None = None
        self._trend_values: NDArray[np.float64] | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 6, 2)
        outer.setSpacing(0)

        self.frame = QGroupBox(title)
        self.frame.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 10pt;
                color: {TEXT};
                border: 2px solid {BORDER};
                border-radius: 6px;
                margin-top: 10px;
                padding: 6px 8px 8px 8px;
                background: {BG_PANEL};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
            """
        )

        inner = QVBoxLayout(self.frame)
        inner.setContentsMargins(6, 8, 8, 6)
        inner.setSpacing(6)
        self._inner = inner

        if x_range is None:
            x_range = (1.0, 100_000.0) if log_x else (0.0, 10.0)

        self.settings = PlotSettingsBar(
            log_x=log_x,
            x_default=x_default,
            y_default=y_default,
            x_range=x_range,
            y_range=y_range,
            allow_trend=allow_trend,
        )
        self.settings.changed.connect(self.settings_changed.emit)

        self.panel = MplPanel(min_height=min_canvas_height)
        self.axes = self.panel.axes
        if log_x:
            self.axes.set_xscale("log")
        if xlabel:
            self.axes.set_xlabel(xlabel, labelpad=4)
        if ylabel:
            self.axes.set_ylabel(ylabel, labelpad=6)

        self.toolbar = CompactNavigationToolbar(self.panel, self)
        self.toolbar.setStyleSheet(TOOLBAR_STYLE)

        # Small controls: zoom in/out and lock axes
        ctrl_widget = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(4)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedSize(28, 24)
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedSize(28, 24)
        self.btn_lock = QPushButton("Lock")
        self.btn_lock.setCheckable(True)
        self.btn_lock.setFixedSize(48, 24)
        ctrl_layout.addWidget(self.btn_zoom_in)
        ctrl_layout.addWidget(self.btn_zoom_out)
        ctrl_layout.addWidget(self.btn_lock)
        ctrl_layout.addStretch()

        # Connect controls
        self.btn_zoom_in.clicked.connect(lambda: self.zoom_in())
        self.btn_zoom_out.clicked.connect(lambda: self.zoom_out())
        self.btn_lock.toggled.connect(lambda v: self.toggle_lock(v))

        # Hide the settings bar (banner with Auto/Min/Max) — kept in memory so
        # the view config API remains available, but not shown to the user.
        self.settings.hide()

        # Put toolbar and controls on the same row
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)
        top_layout.addWidget(self.toolbar)
        top_layout.addWidget(ctrl_widget)
        top_layout.addStretch()
        inner.addWidget(top_row)
        inner.addWidget(self.panel, stretch=1)
        outer.addWidget(self.frame)

        # Persist interactive axis changes: when the user zooms/pans with the
        # navigation toolbar, update the hidden settings config so axes remain
        # fixed for subsequent measurements.
        self.axes.callbacks.connect("xlim_changed", self._on_xlim_changed)
        self.axes.callbacks.connect("ylim_changed", self._on_ylim_changed)
        # When True, programmatic axis changes should not be treated as user edits
        self._suppress_callbacks = False

    def _on_xlim_changed(self, ax) -> None:
        if getattr(self, "_suppress_callbacks", False):
            return
        try:
            lo, hi = ax.get_xlim()
            # store as floats; mark auto_x False to preserve manual limits
            self.settings.config.auto_x = False
            self.settings.config.x_min = float(lo)
            self.settings.config.x_max = float(hi)
            self.settings.changed.emit()
        except Exception:
            pass

    def _on_ylim_changed(self, ax) -> None:
        if getattr(self, "_suppress_callbacks", False):
            return
        try:
            lo, hi = ax.get_ylim()
            self.settings.config.auto_y = False
            self.settings.config.y_min = float(lo)
            self.settings.config.y_max = float(hi)
            self.settings.changed.emit()
        except Exception:
            pass
        outer.addWidget(self.frame)

    def set_series_row(self, layout: QHBoxLayout) -> None:
        """Ligne de cases à cocher (courbes visibles) — au-dessus des réglages."""
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                lay.addWidget(item.widget())
        lay.addStretch()
        self._inner.insertWidget(0, wrap)

    @property
    def view_config(self) -> PlotViewConfig:
        self.settings._read_into_config()
        return self.settings.config

    def _current_limits(self):
        try:
            x_lo, x_hi = self.axes.get_xlim()
            y_lo, y_hi = self.axes.get_ylim()
            return float(x_lo), float(x_hi), float(y_lo), float(y_hi)
        except Exception:
            return self._default_x[0], self._default_x[1], self._default_y[0], self._default_y[1]

    def zoom(self, factor: float) -> None:
        x_lo, x_hi, y_lo, y_hi = self._current_limits()
        x_mid = 0.5 * (x_lo + x_hi)
        y_mid = 0.5 * (y_lo + y_hi)
        x_half = 0.5 * (x_hi - x_lo) * factor
        y_half = 0.5 * (y_hi - y_lo) * factor
        new_x = (max(x_mid - x_half, 1e-6), x_mid + x_half) if self._log_x else (x_mid - x_half, x_mid + x_half)
        new_y = (y_mid - y_half, y_mid + y_half)
        self._suppress_callbacks = True
        try:
            self.axes.set_xlim(new_x)
            self.axes.set_ylim(new_y)
        finally:
            self._suppress_callbacks = False
        self.draw()

    def zoom_in(self) -> None:
        self.zoom(0.5)

    def zoom_out(self) -> None:
        self.zoom(2.0)

    def toggle_lock(self, locked: bool) -> None:
        # When locking, store current limits into config and set auto flags False.
        cfg = self.settings.config
        if locked:
            x_lo, x_hi, y_lo, y_hi = self._current_limits()
            cfg.auto_x = False
            cfg.auto_y = False
            cfg.x_min = float(x_lo)
            cfg.x_max = float(x_hi)
            cfg.y_min = float(y_lo)
            cfg.y_max = float(y_hi)
        else:
            # Unlock: revert to auto behaviour
            cfg.auto_x = True
            cfg.auto_y = True
        self.settings.changed.emit()

    def set_trend_data(
        self,
        freqs: NDArray[np.float64] | None,
        values: NDArray[np.float64] | None,
    ) -> None:
        self._trend_freqs = freqs
        self._trend_values = values

    def apply_view(self, ax) -> None:
        """Limites d'axes + courbe de tendance optionnelle."""
        cfg = self.view_config
        x_lo = cfg.x_min if not cfg.auto_x else self._default_x[0]
        x_hi = cfg.x_max if not cfg.auto_x else self._default_x[1]
        if self._log_x:
            x_lo, x_hi = sorted((max(x_lo, 1e-3), max(x_hi, 1e-3)))
            ax.set_xscale("log")
        # Suppress callbacks so programmatic changes don't mark autoscale as manual
        self._suppress_callbacks = True
        try:
            ax.set_xlim(x_lo, x_hi)
        finally:
            self._suppress_callbacks = False

        if cfg.auto_y:
            self._suppress_callbacks = True
            try:
                ax.autoscale(axis="y")
            finally:
                self._suppress_callbacks = False
            y_lo, y_hi = ax.get_ylim()
            # Guard against bogus autoscale results (extreme outliers or NaN)
        else:
            y_lo, y_hi = sorted((cfg.y_min, cfg.y_max))
            self._suppress_callbacks = True
            try:
                ax.set_ylim(y_lo, y_hi)
            finally:
                self._suppress_callbacks = False
        if cfg.auto_y:
            y_lo, y_hi = ax.get_ylim()
            # Guard against bogus autoscale results (extreme outliers or NaN)
            if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi - y_lo <= 0:
                y_lo, y_hi = self._default_y
            # Clamp range to a sensible maximum span to avoid display blowups
            max_span = 1e4
            span = y_hi - y_lo
            if span > max_span:
                mid = 0.5 * (y_hi + y_lo)
                y_lo = mid - max_span / 2.0
                y_hi = mid + max_span / 2.0
            pad = max((y_hi - y_lo) * 0.06, 0.5)
            self._suppress_callbacks = True
            try:
                ax.set_ylim(y_lo - pad, y_hi + pad)
            finally:
                self._suppress_callbacks = False
        else:
            pass

        if (
            cfg.show_trend
            and self._trend_freqs is not None
            and self._trend_values is not None
            and len(self._trend_freqs) > 0
        ):
            trend = compute_log_trend(
                self._trend_freqs,
                self._trend_values,
                x_min=x_lo,
                x_max=x_hi,
            )
            if trend is not None:
                f_line, y_line = trend
                ax.plot(
                    f_line,
                    y_line,
                    color=TREND_COLOR,
                    linewidth=2.0,
                    linestyle="--",
                    label="Tendance",
                    zorder=15,
                )

    def draw(self) -> None:
        self.panel.draw()

    def prepare_replot(self, *, log_x: bool | None = None) -> None:
        # Clear axes and set labels without triggering the x/y limit callbacks
        self._suppress_callbacks = True
        try:
            self.panel.clear_axes()
            if self._xlabel:
                self.axes.set_xlabel(self._xlabel, labelpad=4)
            if self._ylabel:
                self.axes.set_ylabel(self._ylabel, labelpad=6)
            if log_x if log_x is not None else self._log_x:
                self.axes.set_xscale("log")
        finally:
            self._suppress_callbacks = False

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return hint.expandedTo(self.panel.minimumSizeHint())
