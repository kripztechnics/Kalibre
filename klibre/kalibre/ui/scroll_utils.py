"""
Scroll fluide : les graphiques matplotlib interceptent la molette par défaut.

On redirige tous les événements Wheel vers le QScrollArea parent.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QScrollArea, QScrollBar, QWidget


class WheelToScrollFilter(QObject):
    """Capture la molette sur les enfants et fait défiler le QScrollArea."""

    def __init__(self, scroll_area: QScrollArea) -> None:
        super().__init__(scroll_area)
        self._scroll = scroll_area

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.Wheel:
            return False

        bar = self._scroll.verticalScrollBar()
        if bar is None or not bar.isVisible():
            return False

        wheel = event  # QWheelEvent
        pixel_y = wheel.pixelDelta().y()
        if pixel_y != 0:
            bar.setValue(bar.value() - pixel_y)
        else:
            angle = wheel.angleDelta().y()
            if angle == 0:
                return False
            # ~15° par cran ; division par 8 = défilement plus fluide
            bar.setValue(bar.value() - angle // 8)

        event.accept()
        return True


def configure_smooth_scroll(scroll: QScrollArea) -> WheelToScrollFilter:
    """Réglages barre de scroll + retourne le filtre à attacher aux widgets."""
    scroll.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

    bar: QScrollBar = scroll.verticalScrollBar()
    bar.setSingleStep(24)
    bar.setPageStep(280)

    filt = WheelToScrollFilter(scroll)
    scroll.viewport().installEventFilter(filt)
    return filt


def attach_wheel_to_scroll(scroll: QScrollArea, root: QWidget) -> WheelToScrollFilter:
    """
    Installe le filtre molette sur `root` et tous ses descendants.

    Conserve une référence sur le scroll area via `scroll._wheel_filter`.
    """
    filt = configure_smooth_scroll(scroll)
    scroll._wheel_filter = filt  # type: ignore[attr-defined]

    root.installEventFilter(filt)
    for widget in root.findChildren(QWidget):
        widget.installEventFilter(filt)

    return filt
