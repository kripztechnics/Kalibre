"""
Fenêtre principale Kalibre — v0.4

Mesure sweep ESS + référence loopback (IR, magnitude, cohérence, délai, EQ).
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kalibre.core.audio_engine import (
    AudioEngine,
    MeasurementAborted,
    default_input_index,
    default_output_index,
    enumerate_active_usb_devices,
    list_input_devices,
    list_output_devices,
    loopback_rms,
    rms_to_dbfs,
)
from kalibre.core.device_scan import (
    _normalize_name,
    clear_device_cache,
    filter_audio_usb_present,
    query_present_usb_devices,
)
from kalibre.core.eq_profiles import (
    PROFILE_COLORS,
    PROFILE_ORDER,
    build_all_reference_curves,
    reference_freqs,
)
from kalibre.core.polarity_analysis import (
    PolarityResult,
    analyze_all_vs_reference,
    analyze_polarity_pair,
)
from kalibre.core.signals import (
    DelayEstimate,
    compute_fft_db,
    estimate_delay,
    generate_pink_noise,
    generate_sine,
    generate_white_noise,
    simulate_loopback_capture,
    suggest_eq_from_diff,
)
from kalibre.core.sweep_analysis import (
    SWEEP_PRESETS,
    SweepAnalysis,
    analyze_acoustic_reference,
    generate_ess_sweep,
)
from kalibre.ui.plot_panel import PlotCard
from kalibre.ui.scroll_utils import attach_wheel_to_scroll
from kalibre.ui.theme import (
    BG_PANEL,
    BORDER,
    BTN_PRIMARY,
    BTN_STOP,
    BTN_SUCCESS,
    COLOR_MIC,
    COLOR_REF,
    COLOR_ERROR,
    STYLESHEET,
    TEXT_DIM,
)

COLOR_COHERENCE = "#2ecc71"
COLOR_LIVE = "#f1c40f"

IR_COMPARE_COLORS = [
    "#e67e22",
    "#3498db",
    "#2ecc71",
    "#9b59b6",
    "#e74c3c",
    "#1abc9c",
    "#f39c12",
    "#bdc3c7",
]


# ---------------------------------------------------------------------------
# Modèle de session
# ---------------------------------------------------------------------------


@dataclass
class ChannelMeasurement:
    """Mesure mémorisée (tableau delays + courbe IR comparée)."""

    name: str
    processor_out: str
    absolute_delay_ms: float
    mic_position: str = ""
    is_reference: bool = False
    ir_time_ms: np.ndarray | None = None
    ir: np.ndarray | None = None
    freqs: np.ndarray | None = None
    magnitude_db_rel: np.ndarray | None = None
    coherence: np.ndarray | None = None
    curve_color: str = COLOR_MIC


@dataclass
class SessionState:
    sample_rate: int = 48_000
    duration_s: float = 0.5
    loopback_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    mic_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    measurements: list[ChannelMeasurement] = field(default_factory=list)
    reference_channel: str = ""
    sweep_analysis: SweepAnalysis | None = None
    last_delay_ms: float | None = None

    # Compatibilité graphiques — alias loopback
    @property
    def reference_signal(self) -> np.ndarray:
        return self.loopback_signal

    @reference_signal.setter
    def reference_signal(self, value: np.ndarray) -> None:
        self.loopback_signal = value


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kalibre — Calibration multi-voies v0.5")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(STYLESHEET)

        self.state = SessionState()
        self._audio: AudioEngine | None = None
        self._stop_requested = False
        self._playback_active = False
        self._polarity_result: PolarityResult | None = None

        # Délai simulé (mode sans matériel) — une enceinte, essais de position
        self._simulated_acoustic_delay_ms = 2.5

        self._build_ui()
        self._wire_events()
        self._refresh_audio_devices()
        self._update_waveform_fields()
        self._sync_sample_rate_from_ui()
        self._sync_right_tab()

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(200)
        self._live_timer.timeout.connect(self._refresh_live_time_plot)

        self._apply_sweep_preset()
        self._draw_eq_reference_curves()
        self.statusBar().showMessage(
            f"État : prêt — {len(enumerate_active_usb_devices())} interface(s) USB active(s)"
        )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(0)

        # Splitter horizontal : glissière entre panneau gauche et graphiques
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self._build_left_sidebar())
        self.main_splitter.addWidget(self._build_right_area())
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([300, 960])

        root.addWidget(self.main_splitter)
        self.setStatusBar(QStatusBar())

    def _build_left_sidebar(self) -> QWidget:
        """Colonne gauche redimensionnable — scroll vertical + horizontal si besoin."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(480)

        panel = QWidget()
        panel.setMinimumWidth(248)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 8, 0)
        layout.setSpacing(6)

        self.lbl_state = QLabel("État : prêt")
        self.lbl_state.setStyleSheet(f"color: {TEXT_DIM}; font-weight: bold;")
        layout.addWidget(self.lbl_state)

        tabs = QTabWidget()
        tabs.addTab(self._tab_generator(), "Générateur")
        tabs.addTab(self._tab_soundcard(), "Carte son")
        tabs.addTab(self._tab_wiring(), "Câblage")
        layout.addWidget(tabs)

        layout.addWidget(self._build_delay_panel())

        scroll.setWidget(panel)
        attach_wheel_to_scroll(scroll, panel)
        return scroll

    def _tab_generator(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        self.btn_stop = QPushButton("ARRÊTER lecture / mesure")
        self.btn_stop.setStyleSheet(
            f"background: {BTN_STOP}; font-weight: bold; padding: 8px;"
        )
        layout.addWidget(self.btn_stop)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.cmb_waveform = QComboBox()
        self.cmb_waveform.addItems(
            ["Sweep log (ESS)", "Bruit rose", "Bruit blanc", "Sinusoïde"]
        )
        form.addRow("Forme d'onde :", self.cmb_waveform)

        self.cmb_sweep_preset = QComboBox()
        self.cmb_sweep_preset.addItems(list(SWEEP_PRESETS.keys()))
        form.addRow("Plage sweep :", self.cmb_sweep_preset)

        self.lbl_f_min = QLabel("Sweep min :")
        self.spin_f_min = QDoubleSpinBox()
        self.spin_f_min.setRange(10, 20_000)
        self.spin_f_min.setValue(20)
        self.spin_f_min.setSuffix(" Hz")
        form.addRow(self.lbl_f_min, self.spin_f_min)

        self.lbl_f_max = QLabel("Sweep max :")
        self.spin_f_max = QDoubleSpinBox()
        self.spin_f_max.setRange(20, 40_000)
        self.spin_f_max.setValue(20_000)
        self.spin_f_max.setSuffix(" Hz")
        form.addRow(self.lbl_f_max, self.spin_f_max)

        self.lbl_ir_window = QLabel("Fenêtre IR :")
        self.spin_ir_window = QDoubleSpinBox()
        self.spin_ir_window.setRange(2.0, 30.0)
        self.spin_ir_window.setValue(7.0)
        self.spin_ir_window.setSingleStep(0.5)
        self.spin_ir_window.setSuffix(" ms")
        self.spin_ir_window.setToolTip(
            "Coupe les réflexions de pièce après le direct (style REW Right window)"
        )
        form.addRow(self.lbl_ir_window, self.spin_ir_window)

        # f0 : visible uniquement pour la sinusoïde (pas de sens pour le bruit blanc)
        self.lbl_f0 = QLabel("f0 :")
        self.spin_f0 = QDoubleSpinBox()
        self.spin_f0.setRange(20, 20_000)
        self.spin_f0.setValue(100)
        self.spin_f0.setSuffix(" Hz")
        form.addRow(self.lbl_f0, self.spin_f0)

        self.spin_amplitude = QDoubleSpinBox()
        self.spin_amplitude.setRange(0.01, 1.0)
        self.spin_amplitude.setSingleStep(0.05)
        self.spin_amplitude.setValue(0.25)
        form.addRow("Amplitude :", self.spin_amplitude)

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.1, 5.0)
        self.spin_duration.setSingleStep(0.1)
        self.spin_duration.setValue(1.0)
        self.spin_duration.setSuffix(" s")
        form.addRow("Durée mesure :", self.spin_duration)

        self.cmb_active_voice = QComboBox()
        self.cmb_active_voice.addItems(["Enceinte (essai)", "Sub", "Bas-médium", "Aigu"])
        form.addRow("Voie / mesure :", self.cmb_active_voice)

        self.txt_mic_position = QLineEdit()
        self.txt_mic_position.setPlaceholderText("ex. 50 cm, on-axis / 30 cm off-axis")
        form.addRow("Position micro :", self.txt_mic_position)

        self.chk_set_reference = QCheckBox("Référence pour cette position (0 ms)")
        form.addRow("", self.chk_set_reference)

        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("Génération")
        self.btn_generate.setStyleSheet(f"background: {BTN_PRIMARY}; font-weight: bold;")
        self.btn_measure = QPushButton("Mesurer voie")
        self.btn_measure.setStyleSheet(f"background: {BTN_SUCCESS}; font-weight: bold;")
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_measure)
        form.addRow("", btn_row)

        sweep_hint = QLabel(
            "1. Une seule voie active dans le DSP\n"
            "2. Mesurer → ajoute une courbe · position micro = libellé distinct\n"
            "3. Génération = lecture continue · rouge ARRÊTER pour couper\n"
            "4. Cochez Réf. sur la voie la plus rapide (souvent Aigu)"
        )
        sweep_hint.setWordWrap(True)
        sweep_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        form.addRow("", sweep_hint)

        self.chk_live = QCheckBox("Défilement temps réel (affichage)")
        form.addRow("", self.chk_live)

        layout.addLayout(form)
        return w

    def _tab_soundcard(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.cmb_output = QComboBox()
        self.cmb_input = QComboBox()
        form.addRow("Sortie (vers ampli) :", self.cmb_output)
        form.addRow("Entrée (interface) :", self.cmb_input)

        self.btn_refresh_devices = QPushButton("Actualiser périphériques USB")
        form.addRow("", self.btn_refresh_devices)

        self.chk_show_all_devices = QCheckBox("Afficher aussi cartes intégrées / déconnectées")
        self.chk_show_all_devices.setToolTip(
            "Par défaut : USB branchés uniquement (comme un vrai scan matériel)."
        )
        form.addRow("", self.chk_show_all_devices)

        self.spin_out_ch = QSpinBox()
        self.spin_out_ch.setRange(1, 32)
        self.spin_out_ch.setValue(1)
        form.addRow("Canal sortie :", self.spin_out_ch)

        self.spin_in_ch = QSpinBox()
        self.spin_in_ch.setRange(1, 32)
        self.spin_in_ch.setValue(1)
        self.spin_in_ch.setToolTip("Entrée où est branché le micro (ex. IN 1)")
        form.addRow("Canal micro (IN) :", self.spin_in_ch)

        self.spin_loopback_ch = QSpinBox()
        self.spin_loopback_ch.setRange(1, 32)
        self.spin_loopback_ch.setValue(2)
        self.spin_loopback_ch.setToolTip("Entrée loopback câblée sur la sortie (ex. IN 2)")
        form.addRow("Canal loopback (IN) :", self.spin_loopback_ch)

        self.cmb_sample_rate = QComboBox()
        self.cmb_sample_rate.addItems(["44100", "48000", "96000"])
        self.cmb_sample_rate.setCurrentText("48000")
        form.addRow("Fréq. échant. :", self.cmb_sample_rate)

        self.cmb_buffer = QComboBox()
        self.cmb_buffer.addItems(["128", "256", "512", "1024"])
        self.cmb_buffer.setCurrentText("256")
        form.addRow("Buffer (samples) :", self.cmb_buffer)

        self.btn_test_output = QPushButton("Test sortie (bip 440 Hz)")
        form.addRow("", self.btn_test_output)

        self.btn_test_levels = QPushButton("Test niveaux IN1 / IN2")
        self.btn_test_levels.setToolTip(
            "Joue un bip et affiche le niveau capté sur micro et loopback (sans popup Windows)"
        )
        form.addRow("", self.btn_test_levels)

        self.chk_stereo_output = QCheckBox("Émettre sur toutes les sorties (L + R)")
        self.chk_stereo_output.setChecked(True)
        self.chk_stereo_output.setToolTip(
            "Recommandé pour le loopback : même signal sur Main Out gauche et droite"
        )
        form.addRow("", self.chk_stereo_output)

        self.lbl_channel_levels = QLabel("Niveaux : — (lancez Test niveaux ou Mesurer)")
        self.lbl_channel_levels.setWordWrap(True)
        self.lbl_channel_levels.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        form.addRow("", self.lbl_channel_levels)

        self.chk_simulation = QCheckBox("Mode simulation (sans carte son)")
        form.addRow("", self.chk_simulation)

        hint = QLabel(
            "Liste filtrée : USB branchés + testés (PortAudio).\n"
            "Les anciennes cartes débranchées n'apparaissent plus.\n"
            "OUT → ampli + loopback → IN2 | micro → IN1."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        form.addRow(hint)

        return w

    def _tab_wiring(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        hint = QLabel(
            "Associez chaque sortie DSP à une voie.\n"
            "Une seule voie active par mesure."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(hint)

        self.tbl_wiring = QTableWidget(3, 3)
        self.tbl_wiring.setHorizontalHeaderLabels(["Voie", "Sortie DSP", "Actif"])
        self.tbl_wiring.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, (voice, out, active) in enumerate(
            [("Sub", "OUT 1", True), ("Bas-médium", "OUT 2", False), ("Aigu", "OUT 3", False)]
        ):
            self.tbl_wiring.setItem(row, 0, QTableWidgetItem(voice))
            self.tbl_wiring.setItem(row, 1, QTableWidgetItem(out))
            item = QTableWidgetItem("✓" if active else "—")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tbl_wiring.setItem(row, 2, item)

        layout.addWidget(self.tbl_wiring)
        return w

    def _build_delay_panel(self) -> QGroupBox:
        """Panneau alignement — sous les onglets à gauche."""
        box = QGroupBox("Alignement — delays")
        layout = QVBoxLayout(box)

        self.tbl_delays = QTableWidget(0, 5)
        self.tbl_delays.setHorizontalHeaderLabels(
            ["Voie", "Sortie", "Delay", "Rel.", "Réf."]
        )
        self.tbl_delays.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_delays.setMaximumHeight(120)
        self.tbl_delays.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_delays.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.tbl_delays)

        btn_row = QHBoxLayout()
        self.btn_clear_meas = QPushButton("Effacer la mesure")
        self.btn_clear_meas.setToolTip("Supprime la ligne sélectionnée dans le tableau (et sa courbe).")
        self.btn_clear_all_meas = QPushButton("Tout effacer")
        self.btn_clear_all_meas.setToolTip("Supprime toutes les mesures mémorisées.")
        self.btn_save_measurement = QPushButton("Sauvegarder la mesure")
        self.btn_save_measurement.setStyleSheet(f"background: {BTN_PRIMARY}; font-weight: bold;")
        self.btn_save_measurement.setToolTip(
            "Duplique la dernière mesure dans le tableau (même données, nouveau libellé)."
        )
        btn_row.addWidget(self.btn_clear_meas)
        btn_row.addWidget(self.btn_clear_all_meas)
        btn_row.addWidget(self.btn_save_measurement)
        layout.addLayout(btn_row)

        self.lbl_delay_hint = QLabel(
            "Chaque mesure ajoute une ligne et une courbe (IR + magnitude).\n"
            "Sélectionnez une ligne puis « Effacer la mesure » pour retirer une courbe.\n"
            "Rel. = delay à entrer dans le DSP pour aligner les voies."
        )
        self.lbl_delay_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        self.lbl_delay_hint.setWordWrap(True)
        layout.addWidget(self.lbl_delay_hint)

        return box

    def _build_right_area(self) -> QWidget:
        """Onglets graphiques — une vue par type de signal."""
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self._build_sweep_tab(), "Sweep")
        self.right_tabs.addTab(self._build_noise_tab(), "Bruit aléatoire")
        self.right_tabs.addTab(self._build_sine_tab(), "Sinusoïdale")
        self.right_tabs.addTab(self._build_polarity_tab(), "Polarité")
        return self.right_tabs

    def _plot_scroll(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        attach_wheel_to_scroll(scroll, content)
        return scroll

    def _build_sweep_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 8, 12, 12)
        layout.setSpacing(10)

        self.plot_ir = PlotCard(
            "Réponse impulsionnelle — comparaison des mesures mémorisées",
            xlabel="Temps (ms)",
            ylabel="Amplitude",
            min_canvas_height=240,
            x_default=(0.0, 25.0),
            y_default=(-0.5, 1.0),
            x_range=(0.0, 100.0),
            y_range=(-1.5, 1.5),
        )
        layout.addWidget(self.plot_ir)

        ir_hint = QLabel(
            "Chaque mesure ajoute une courbe colorée — traits pointillés = pic de délai"
        )
        ir_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        ir_hint.setWordWrap(True)
        layout.addWidget(ir_hint)

        self.plot_transfer = PlotCard(
            "Magnitude & cohérence — référence loopback",
            log_x=True,
            xlabel="Fréquence (Hz)",
            ylabel="Magnitude (dB)",
            min_canvas_height=260,
            x_default=(20.0, 20_000.0),
            y_default=(-60.0, 20.0),
            allow_trend=True,
        )
        layout.addWidget(self.plot_transfer)

        transfer_hint = QLabel(
            "Orange = réponse fenêtrée · Vert = cohérence γ² (proche de 1 = mesure fiable)"
        )
        transfer_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        transfer_hint.setWordWrap(True)
        layout.addWidget(transfer_hint)

        eq_panel = self._build_eq_panel()
        layout.addWidget(eq_panel)
        layout.addStretch()

        return self._plot_scroll(content)

    def _build_noise_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 8, 12, 12)
        layout.setSpacing(10)

        self.plot_noise_time = PlotCard(
            "Temps — loopback (réf.) vs micro",
            xlabel="Temps (ms)",
            ylabel="Amplitude",
            min_canvas_height=240,
            x_default=(0.0, 500.0),
            y_default=(-1.0, 1.0),
            x_range=(0.0, 5000.0),
            y_range=(-1.5, 1.5),
        )
        layout.addWidget(self.plot_noise_time)

        self.plot_noise_fft = PlotCard(
            "Spectre FFT — diagnostic bruit rose / blanc",
            log_x=True,
            xlabel="Fréquence (Hz)",
            ylabel="Magnitude (dB)",
            min_canvas_height=260,
            x_default=(20.0, 20_000.0),
            y_default=(-80.0, 0.0),
            allow_trend=True,
        )
        layout.addWidget(self.plot_noise_fft)

        hint = QLabel(
            "Vue brute pour vérifier niveaux et spectre — le délai reste calculé en arrière-plan."
        )
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()

        return self._plot_scroll(content)

    def _build_sine_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 8, 12, 12)
        layout.setSpacing(10)

        self.plot_sine_time = PlotCard(
            "Temps — sinusoïde loopback vs micro",
            xlabel="Temps (ms)",
            ylabel="Amplitude",
            min_canvas_height=240,
            x_default=(0.0, 20.0),
            y_default=(-1.0, 1.0),
            x_range=(0.0, 5000.0),
            y_range=(-1.5, 1.5),
        )
        layout.addWidget(self.plot_sine_time)

        self.plot_sine_fft = PlotCard(
            "Spectre FFT — pic à f0",
            log_x=True,
            xlabel="Fréquence (Hz)",
            ylabel="Magnitude (dB)",
            min_canvas_height=260,
            x_default=(20.0, 20_000.0),
            y_default=(-80.0, 0.0),
            allow_trend=True,
        )
        layout.addWidget(self.plot_sine_fft)

        hint = QLabel(
            "Alignement visuel de la sinusoïde — le trait bleu pointillé marque le délai estimé."
        )
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch()

        return self._plot_scroll(content)

    def _build_polarity_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 8, 12, 12)
        layout.setSpacing(10)

        hint = QLabel(
            "Mesurez Out5 (Normal) puis Out6 (Inverse) séparément — une seule sortie active.\n"
            "Bande 40–400 Hz = ta zone LR. L'algo compare +B et −B après alignement des délais."
        )
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Référence :"))
        self.cmb_pol_ref = QComboBox()
        sel_row.addWidget(self.cmb_pol_ref, stretch=1)
        sel_row.addWidget(QLabel("Comparer :"))
        self.cmb_pol_test = QComboBox()
        sel_row.addWidget(self.cmb_pol_test, stretch=1)
        layout.addLayout(sel_row)

        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("Bande analyse :"))
        self.spin_pol_fmin = QDoubleSpinBox()
        self.spin_pol_fmin.setRange(10, 20_000)
        self.spin_pol_fmin.setValue(40)
        self.spin_pol_fmin.setSuffix(" Hz min")
        band_row.addWidget(self.spin_pol_fmin)
        self.spin_pol_fmax = QDoubleSpinBox()
        self.spin_pol_fmax.setRange(20, 20_000)
        self.spin_pol_fmax.setValue(400)
        self.spin_pol_fmax.setSuffix(" Hz max")
        band_row.addWidget(self.spin_pol_fmax)
        self.btn_analyze_polarity = QPushButton("Analyser polarité")
        self.btn_analyze_polarity.setStyleSheet(f"background: {BTN_PRIMARY}; font-weight: bold;")
        band_row.addWidget(self.btn_analyze_polarity)
        layout.addLayout(band_row)

        self.lbl_polarity_verdict = QLabel(
            "Mémorisez au moins 2 mesures sweep, puis analysez."
        )
        self.lbl_polarity_verdict.setWordWrap(True)
        self.lbl_polarity_verdict.setStyleSheet(
            f"color: {TEXT_DIM}; font-weight: bold; padding: 6px; "
            f"border: 1px solid {BORDER}; border-radius: 6px;"
        )
        layout.addWidget(self.lbl_polarity_verdict)

        self.tbl_polarity = QTableWidget(0, 7)
        self.tbl_polarity.setHorizontalHeaderLabels(
            ["Voie", "Corr. +B", "Corr. −B", "A+B vs A−B", "Phase °", "Conf.", "Statut"]
        )
        self.tbl_polarity.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tbl_polarity.setMaximumHeight(110)
        layout.addWidget(self.tbl_polarity)

        self.plot_polarity_ir = PlotCard(
            "IR bande passante — alignées (délais + signe corrigé si inversion)",
            xlabel="Temps relatif (ms)",
            ylabel="Amplitude",
            min_canvas_height=200,
            x_default=(-0.5, 5.0),
            y_default=(-1.2, 1.2),
            x_range=(-2.0, 20.0),
            y_range=(-1.5, 1.5),
        )
        layout.addWidget(self.plot_polarity_ir)

        self.plot_polarity_sum = PlotCard(
            "Somme complexe — même polarité (A+B) vs inversion (A−B)",
            log_x=True,
            xlabel="Fréquence (Hz)",
            ylabel="Magnitude (dB rel.)",
            min_canvas_height=220,
            x_default=(20.0, 500.0),
            y_default=(-40.0, 5.0),
            allow_trend=True,
        )
        layout.addWidget(self.plot_polarity_sum)

        self.plot_polarity_phase = PlotCard(
            "Différence de phase (réf. vs comparée)",
            log_x=True,
            xlabel="Fréquence (Hz)",
            ylabel="Phase (°)",
            min_canvas_height=200,
            x_default=(20.0, 500.0),
            y_default=(-200.0, 200.0),
        )
        layout.addWidget(self.plot_polarity_phase)
        layout.addStretch()

        return self._plot_scroll(content)

    def _build_eq_panel(self) -> QGroupBox:
        box = QGroupBox("Égalisation — réponse fenêtrée vs cible")
        layout = QVBoxLayout(box)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profil cible :"))
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems(PROFILE_ORDER)
        self.cmb_profile.setCurrentIndex(2)  # Bass music
        profile_row.addWidget(self.cmb_profile, stretch=1)
        self.btn_apply_profile = QPushButton("Analyser écart")
        profile_row.addWidget(self.btn_apply_profile)
        layout.addLayout(profile_row)

        self.plot_eq = PlotCard(
            "Mesuré vs courbes théoriques EQ",
            log_x=True,
            xlabel="Fréquence (Hz)",
            ylabel="dB (relatif @ 1 kHz)",
            min_canvas_height=240,
            x_default=(20.0, 20_000.0),
            y_default=(-12.0, 10.0),
            allow_trend=True,
        )
        layout.addWidget(self.plot_eq, stretch=1)

        hint = QLabel(
            "Courbe orange = magnitude sweep (pièce coupée par fenêtre IR).\n"
            "Pointillés = profils cibles · tableau = filtres suggérés."
        )
        hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9pt;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.tbl_eq_suggestions = QTableWidget(0, 4)
        self.tbl_eq_suggestions.setHorizontalHeaderLabels(
            ["Type filtre", "Fréq. (Hz)", "Gain (dB)", "Q"]
        )
        self.tbl_eq_suggestions.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tbl_eq_suggestions.setMaximumHeight(100)
        layout.addWidget(self.tbl_eq_suggestions)

        return box

    # ------------------------------------------------------------------
    # Événements
    # ------------------------------------------------------------------

    def _wire_events(self) -> None:
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_measure.clicked.connect(self._on_measure)
        self.btn_clear_meas.clicked.connect(self._on_clear_selected_measurement)
        self.btn_clear_all_meas.clicked.connect(self._on_clear_all_measurements)
        self.btn_save_measurement.clicked.connect(self._on_save_measurement)
        self.btn_apply_profile.clicked.connect(self._on_analyze_eq)
        self.btn_refresh_devices.clicked.connect(self._refresh_audio_devices)
        self.btn_test_output.clicked.connect(self._on_test_output)
        self.btn_test_levels.clicked.connect(self._on_test_levels)
        self.cmb_output.currentIndexChanged.connect(self._sync_input_to_output_device)
        self.chk_live.toggled.connect(self._on_live_toggled)
        self.cmb_waveform.currentIndexChanged.connect(self._update_waveform_fields)
        self.cmb_sweep_preset.currentIndexChanged.connect(self._apply_sweep_preset)
        self.cmb_sample_rate.currentIndexChanged.connect(self._sync_sample_rate_from_ui)
        self.chk_show_all_devices.toggled.connect(self._refresh_audio_devices)

        self.cmb_profile.currentIndexChanged.connect(self._on_analyze_eq)

        self.plot_ir.settings_changed.connect(self._draw_ir_plot)
        self.plot_transfer.settings_changed.connect(self._draw_transfer_plot)
        self.plot_noise_time.settings_changed.connect(self._draw_noise_time_plot)
        self.plot_noise_fft.settings_changed.connect(self._draw_noise_fft_plot)
        self.plot_sine_time.settings_changed.connect(self._draw_sine_time_plot)
        self.plot_sine_fft.settings_changed.connect(self._draw_sine_fft_plot)
        self.plot_eq.settings_changed.connect(self._on_analyze_eq)
        self.btn_analyze_polarity.clicked.connect(self._on_analyze_polarity)
        self.plot_polarity_ir.settings_changed.connect(self._draw_polarity_ir_plot)
        self.plot_polarity_sum.settings_changed.connect(self._draw_polarity_sum_plot)
        self.plot_polarity_phase.settings_changed.connect(self._draw_polarity_phase_plot)

    def _update_waveform_fields(self) -> None:
        kind = self._waveform_kind()
        is_sine = kind.startswith("Sinus")
        is_sweep = kind.startswith("Sweep")

        self.lbl_f0.setVisible(is_sine)
        self.spin_f0.setVisible(is_sine)

        for w in (
            self.cmb_sweep_preset,
            self.lbl_f_min,
            self.spin_f_min,
            self.lbl_f_max,
            self.spin_f_max,
            self.lbl_ir_window,
            self.spin_ir_window,
        ):
            w.setVisible(is_sweep)

        if is_sweep:
            self._apply_sweep_preset()

        self._sync_right_tab()

    def _sync_right_tab(self) -> None:
        """Affiche l'onglet graphique correspondant à la forme d'onde."""
        kind = self._waveform_kind()
        if kind.startswith("Sweep"):
            self.right_tabs.setCurrentIndex(0)
        elif kind.startswith("Bruit"):
            self.right_tabs.setCurrentIndex(1)
        elif kind.startswith("Sinus"):
            self.right_tabs.setCurrentIndex(2)

    def _apply_sweep_preset(self) -> None:
        if not self._waveform_kind().startswith("Sweep"):
            return
        key = self.cmb_sweep_preset.currentText()
        f_min, f_max, duration = SWEEP_PRESETS.get(key, (20.0, 20_000.0, 5.0))
        self.spin_f_min.setValue(f_min)
        self.spin_f_max.setValue(f_max)
        self.spin_duration.setValue(duration)

    def _waveform_kind(self) -> str:
        return self.cmb_waveform.currentText()

    def _is_sweep_mode(self) -> bool:
        return self._waveform_kind().startswith("Sweep")

    def _sweep_band(self) -> tuple[float, float, float]:
        return (
            self.spin_f_min.value(),
            self.spin_f_max.value(),
            self.spin_ir_window.value(),
        )

    def _build_test_signal(self) -> np.ndarray:
        """Construit le buffer de mesure selon la forme d'onde choisie."""
        sr = self.state.sample_rate
        duration = self.spin_duration.value()
        amp = self.spin_amplitude.value()
        kind = self._waveform_kind()

        if kind.startswith("Sweep"):
            f_min, f_max, _ = self._sweep_band()
            return generate_ess_sweep(f_min, f_max, duration, sr, amplitude=amp)
        if kind == "Bruit blanc":
            return generate_white_noise(duration, sr, amplitude=amp)
        if kind == "Bruit rose":
            return generate_pink_noise(duration, sr, amplitude=amp)
        return generate_sine(self.spin_f0.value(), duration, sr, amplitude=amp)

    def _sync_sample_rate_from_ui(self) -> None:
        self.state.sample_rate = int(self.cmb_sample_rate.currentText())

    def _refresh_audio_devices(self) -> None:
        """Scanne les USB branchés (PnP Windows + test PortAudio)."""
        clear_device_cache()
        prev_out = self.cmb_output.currentData()
        prev_in = self.cmb_input.currentData()
        usb_only = not self.chk_show_all_devices.isChecked()

        outputs = list_output_devices(active_usb_only=usb_only)
        inputs = list_input_devices(active_usb_only=usb_only)

        self.cmb_output.clear()
        self.cmb_input.clear()

        if not outputs:
            self.cmb_output.addItem("— Aucune sortie USB active —", None)
        else:
            for dev in outputs:
                self.cmb_output.addItem(dev.label(), dev.index)

        if not inputs:
            self.cmb_input.addItem("— Aucune entrée USB active —", None)
        else:
            for dev in inputs:
                self.cmb_input.addItem(dev.label(), dev.index)

        self._restore_combo_selection(self.cmb_output, prev_out, default_output_index())
        self._restore_combo_selection(self.cmb_input, prev_in, default_input_index())
        self._sync_input_to_output_device()

        present = filter_audio_usb_present(query_present_usb_devices())
        mode = "USB actifs" if usb_only else "tous"
        self.statusBar().showMessage(
            f"{mode} — {len(outputs)} sortie(s), {len(inputs)} entrée(s) | "
            f"{len(present)} périph. USB branchés (Windows)"
        )

    @staticmethod
    def _restore_combo_selection(
        combo: QComboBox,
        previous: object,
        default_index: int | None,
    ) -> None:
        if previous is not None:
            idx = combo.findData(previous)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                return
        if default_index is not None:
            idx = combo.findData(default_index)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _build_audio_engine(self) -> AudioEngine:
        out_idx = self.cmb_output.currentData()
        in_idx = self.cmb_input.currentData()
        if out_idx is None or in_idx is None:
            raise RuntimeError("Sélectionnez une entrée et une sortie audio valides.")

        return AudioEngine(
            sample_rate=int(self.cmb_sample_rate.currentText()),
            blocksize=int(self.cmb_buffer.currentText()),
            input_device=int(in_idx),
            output_device=int(out_idx),
            mic_channel=self.spin_in_ch.value(),
            loopback_channel=self.spin_loopback_ch.value(),
            output_channel=self.spin_out_ch.value(),
            stereo_output=self.chk_stereo_output.isChecked(),
        )

    def _sync_input_to_output_device(self) -> None:
        """Aligne l'entrée sur la même interface que la sortie (duplex Windows)."""
        out_idx = self.cmb_output.currentData()
        if out_idx is None:
            return
        try:
            import sounddevice as sd

            out_name = _normalize_name(str(sd.query_devices(int(out_idx))["name"]))
        except Exception:
            return

        for i in range(self.cmb_input.count()):
            in_idx = self.cmb_input.itemData(i)
            if in_idx is None:
                continue
            try:
                in_name = _normalize_name(str(sd.query_devices(int(in_idx))["name"]))
            except Exception:
                continue
            if in_name == out_name:
                self.cmb_input.setCurrentIndex(i)
                break

    def _stop_audio(self) -> None:
        """Coupe toute lecture / mesure sounddevice."""
        self._stop_requested = True
        self._playback_active = False
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def _on_stop(self) -> None:
        self._stop_audio()
        self._live_timer.stop()
        self.chk_live.setChecked(False)
        self.lbl_state.setText("État : arrêté")
        self.statusBar().showMessage("Lecture / mesure arrêtée")

    def _on_live_toggled(self, enabled: bool) -> None:
        if enabled and len(self.state.reference_signal) > 0:
            self._live_timer.start()
        else:
            self._live_timer.stop()

    def _on_generate(self) -> None:
        """Prépare le signal et lance la lecture en boucle (arrêt = bouton rouge)."""
        self._sync_sample_rate_from_ui()
        signal = self._build_test_signal()
        self.state.duration_s = self.spin_duration.value()
        self.state.loopback_signal = np.zeros_like(signal)
        self.state.mic_signal = np.zeros_like(signal)
        self.state.sweep_analysis = None

        if self.chk_simulation.isChecked():
            self.lbl_state.setText("État : signal prêt (simulation) — lancez Mesurer")
            self.statusBar().showMessage("Signal prêt — mode simulation")
            self._refresh_all_plots()
            return

        try:
            self._stop_audio()
            self._stop_requested = False
            engine = self._build_audio_engine()
            engine.play_loop(signal)
            self._playback_active = True
            self.lbl_state.setText("État : lecture en cours — ARRÊTER pour couper")
            self.statusBar().showMessage(
                f"{self._waveform_kind()} en boucle — bouton rouge pour arrêter"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Erreur audio", f"Lecture impossible :\n{exc}")
            self.lbl_state.setText("État : erreur lecture")
            return

        self._refresh_all_plots()

    def _on_measure(self) -> None:
        """
        Mesure acoustique loopback → micro :
          1. Émet le bruit blanc / sinus sur la sortie
          2. Capture IN loopback (ref. électrique) + IN micro
          3. Corrélation loopback vs micro = délai acoustique
        """
        self._sync_sample_rate_from_ui()
        self._stop_audio()
        self._stop_requested = False
        self._set_busy(True)

        try:
            emitted = self._build_test_signal()
            self.state.duration_s = self.spin_duration.value()

            if self.chk_simulation.isChecked():
                loopback, mic = simulate_loopback_capture(
                    emitted,
                    acoustic_delay_ms=self._simulated_acoustic_delay_ms,
                    sample_rate=self.state.sample_rate,
                )
                mode = "simulation"
            else:
                engine = self._build_audio_engine()
                self.lbl_state.setText("État : mesure en cours…")
                self.statusBar().showMessage(
                    "Émission + capture loopback/micro — rouge ARRÊTER pour interrompre"
                )
                QApplication.processEvents()
                capture = engine.play_and_capture(
                    emitted,
                    stop_check=lambda: self._stop_requested,
                )
                loopback, mic = capture.loopback, capture.mic
                mode = "loopback"
                self._show_channel_levels(capture)

            n = min(len(loopback), len(mic))
            loopback = loopback[:n]
            mic = mic[:n]
            self.state.loopback_signal = loopback
            self.state.mic_signal = mic
            lb_level = loopback_rms(loopback)

            f_min, f_max, ir_win = self._sweep_band()
            if not self._is_sweep_mode():
                f_min, f_max = 20.0, 20_000.0

            analysis = analyze_acoustic_reference(
                loopback,
                mic,
                self.state.sample_rate,
                f_min=f_min,
                f_max=f_max,
                ir_window_ms=ir_win,
            )
            self.state.sweep_analysis = analysis

            delay_result = estimate_delay(loopback, mic, self.state.sample_rate)
            if analysis is not None:
                measured_delay = analysis.delay_ms
                delay_result = DelayEstimate(
                    analysis.delay_ms,
                    analysis.confidence,
                    delay_result.crosstalk_ratio,
                    analysis.confidence < 0.35,
                )
            else:
                measured_delay = delay_result.delay_ms
            dist_cm = measured_delay * 34.3  # c ≈ 343 m/s
            self.state.last_delay_ms = measured_delay

            voice = self.cmb_active_voice.currentText()
            mic_pos = self.txt_mic_position.text().strip()
            label = f"{voice}" + (f" @ {mic_pos}" if mic_pos else "")

            conf_pct = int(delay_result.confidence * 100)
            coh_pct = int(analysis.mean_coherence * 100) if analysis else 0
            self._commit_measurement()
            n_curves = len(self.state.measurements)
            self.lbl_state.setText(
                f"État : mesuré — {measured_delay:.2f} ms "
                f"(conf. {conf_pct} % · coh. {coh_pct} %) · {n_curves} courbe(s)"
            )
            status = (
                f"{label} — délai {measured_delay:.2f} ms (~{dist_cm:.0f} cm) "
                f"| confiance {conf_pct} % | cohérence {coh_pct} %"
            )
            warn = self._measurement_quality_hint(delay_result, lb_level, mode=mode)
            self.statusBar().showMessage(f"{status} | {warn}" if warn else status, 20000 if warn else 0)
            self._sync_right_tab()
            self._refresh_all_plots()
            if self._is_sweep_mode():
                self._on_analyze_eq()

        except MeasurementAborted:
            self.lbl_state.setText("État : mesure interrompue")
            self.statusBar().showMessage("Mesure arrêtée (bouton rouge)")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Erreur audio",
                f"La mesure a échoué :\n{exc}\n\n{traceback.format_exc()}",
            )
            self.lbl_state.setText("État : erreur")
        finally:
            self._set_busy(False)

    def _measurement_quality_hint(
        self,
        result,
        loopback_level: float,
        *,
        mode: str = "loopback",
    ) -> str:
        """Texte d'avertissement qualité (barre de status, sans popup)."""
        if self.chk_simulation.isChecked():
            return ""

        issues: list[str] = []
        if mode == "loopback" and loopback_level < 1e-4:
            lb_db = rms_to_dbfs(loopback_level)
            issues.append(f"loopback faible ({lb_db:.0f} dBFS)")
        if result.crosstalk_ratio > 0.35:
            issues.append(f"fuite loopback ({result.crosstalk_ratio:.0%})")
        if result.confidence < 0.35:
            issues.append("corrélation faible")
        analysis = self.state.sweep_analysis
        if analysis is not None and analysis.mean_coherence < 0.45:
            issues.append(f"cohérence {analysis.mean_coherence:.0%}")
        if result.ambiguous:
            issues.append("pics multiples")
        if result.delay_ms < 0.4 and result.crosstalk_ratio > 0.2:
            issues.append("délai suspect")

        if not issues:
            return ""
        return "⚠ " + ", ".join(issues)

    def _show_channel_levels(self, capture) -> None:
        mic_db = rms_to_dbfs(capture.mic_rms)
        lb_db = rms_to_dbfs(capture.loopback_rms)
        ch_m = self.spin_in_ch.value()
        ch_lb = self.spin_loopback_ch.value()
        duplex = capture.duplex_device
        self.lbl_channel_levels.setText(
            f"Niveaux — IN{ch_m} micro : {mic_db:.1f} dBFS | "
            f"IN{ch_lb} loopback : {lb_db:.1f} dBFS | "
            f"pic loopback {capture.loopback_peak:.3f} | duplex {duplex}"
        )

    def _on_test_levels(self) -> None:
        """Bip + affichage niveaux sans popup d'erreur Windows."""
        if self.chk_simulation.isChecked():
            QMessageBox.information(self, "Kalibre", "Désactivez le mode simulation.")
            return
        try:
            self._sync_sample_rate_from_ui()
            engine = self._build_audio_engine()
            self.lbl_state.setText("État : test niveaux…")
            self.statusBar().showMessage("Bip + capture IN1/IN2…")
            QApplication.processEvents()
            capture = engine.test_output(duration_s=0.4, frequency_hz=440.0)
            self._show_channel_levels(capture)
            self.lbl_state.setText("État : prêt")
            lb_db = rms_to_dbfs(capture.loopback_rms)
            if capture.loopback_rms < 1e-4:
                self.statusBar().showMessage(
                    f"Loopback toujours faible ({lb_db:.0f} dBFS) — voir niveaux ci-dessus"
                )
            else:
                self.statusBar().showMessage(
                    f"Loopback OK ({lb_db:.1f} dBFS) — vous pouvez mesurer"
                )
        except Exception as exc:
            QMessageBox.critical(self, "Erreur audio", str(exc))

    def _on_test_output(self) -> None:
        """Bip court pour vérifier le routage sortie."""
        if self.chk_simulation.isChecked():
            QMessageBox.information(self, "Kalibre", "Désactivez le mode simulation.")
            return
        try:
            self._sync_sample_rate_from_ui()
            engine = self._build_audio_engine()
            self.lbl_state.setText("État : test sortie…")
            QApplication.processEvents()
            capture = engine.test_output()
            self._show_channel_levels(capture)
            self.lbl_state.setText("État : prêt")
            self.statusBar().showMessage("Bip envoyé — niveaux IN1/IN2 mis à jour")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur audio", str(exc))

    def _set_busy(self, busy: bool) -> None:
        self.btn_measure.setEnabled(not busy)
        self.btn_generate.setEnabled(not busy)
        self.btn_test_output.setEnabled(not busy)

    def _processor_out_for_voice(self, voice: str) -> str:
        if voice == "Enceinte (essai)":
            return "OUT 1"
        for row in range(self.tbl_wiring.rowCount()):
            if self.tbl_wiring.item(row, 0).text() == voice:
                return self.tbl_wiring.item(row, 1).text()
        return "OUT ?"

    def _unique_measurement_label(self, voice: str, mic_pos: str) -> str:
        base = f"{voice}" + (f" @ {mic_pos}" if mic_pos else "")
        used = {m.name for m in self.state.measurements}
        if base not in used:
            return base
        index = 2
        while f"{base} #{index}" in used:
            index += 1
        return f"{base} #{index}"

    def _next_ir_color(self) -> str:
        idx = len(self.state.measurements)
        return IR_COMPARE_COLORS[idx % len(IR_COMPARE_COLORS)]

    def _commit_measurement(self) -> bool:
        """Ajoute la mesure courante (nouvelle ligne + courbes IR / magnitude)."""
        if self.state.last_delay_ms is None:
            return False

        voice = self.cmb_active_voice.currentText()
        mic_pos = self.txt_mic_position.text().strip()
        label = self._unique_measurement_label(voice, mic_pos)
        is_ref = self.chk_set_reference.isChecked()

        if is_ref:
            self.state.reference_channel = label
            for m in self.state.measurements:
                m.is_reference = False

        analysis = self.state.sweep_analysis
        ir_time: np.ndarray | None = None
        ir: np.ndarray | None = None
        freqs: np.ndarray | None = None
        magnitude_db_rel: np.ndarray | None = None
        coherence: np.ndarray | None = None

        if analysis is not None and len(analysis.ir) > 0:
            ir_time = np.array(analysis.ir_time_ms, copy=True)
            ir = np.array(analysis.ir, copy=True)
        if analysis is not None and len(analysis.freqs) > 0:
            freqs = np.array(analysis.freqs, copy=True)
            magnitude_db_rel = np.array(
                analysis.magnitude_db - np.max(analysis.magnitude_db),
                copy=True,
            )
            coherence = np.array(analysis.coherence, copy=True)

        self.state.measurements.append(
            ChannelMeasurement(
                name=label,
                processor_out=self._processor_out_for_voice(voice),
                absolute_delay_ms=self.state.last_delay_ms,
                mic_position=mic_pos,
                is_reference=is_ref,
                ir_time_ms=ir_time,
                ir=ir,
                freqs=freqs,
                magnitude_db_rel=magnitude_db_rel,
                coherence=coherence,
                curve_color=self._next_ir_color(),
            )
        )
        self._refresh_delay_table()
        self._refresh_polarity_selectors()
        return True

    def _reference_absolute_delay(self) -> float:
        for m in self.state.measurements:
            if m.is_reference:
                return m.absolute_delay_ms
        return 0.0

    def _refresh_delay_table(self) -> None:
        self.tbl_delays.setRowCount(len(self.state.measurements))
        ref_abs = self._reference_absolute_delay()

        for row, m in enumerate(self.state.measurements):
            if m.is_reference:
                proc_delay = 0.0
                rel = 0.0
            else:
                rel = m.absolute_delay_ms - ref_abs
                proc_delay = rel  # consigne à entrer dans le DSP

            self.tbl_delays.setItem(row, 0, QTableWidgetItem(m.name))
            self.tbl_delays.setItem(row, 1, QTableWidgetItem(m.processor_out))
            self.tbl_delays.setItem(row, 2, QTableWidgetItem(f"{proc_delay:.2f}"))
            self.tbl_delays.setItem(row, 3, QTableWidgetItem(f"{rel:+.2f}"))
            ref_item = QTableWidgetItem("★" if m.is_reference else "")
            ref_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tbl_delays.setItem(row, 4, ref_item)

        self._refresh_polarity_selectors()

    def _on_clear_selected_measurement(self) -> None:
        if not self.state.measurements:
            self.statusBar().showMessage("Aucune mesure à effacer")
            return

        row = self.tbl_delays.currentRow()
        if row < 0 or row >= len(self.state.measurements):
            QMessageBox.information(
                self,
                "Kalibre",
                "Sélectionnez d'abord une ligne dans le tableau des delays.",
            )
            return

        removed = self.state.measurements.pop(row)
        if removed.is_reference:
            self.state.reference_channel = ""
            for m in self.state.measurements:
                if m.is_reference:
                    self.state.reference_channel = m.name
                    break

        self._refresh_delay_table()
        self._refresh_all_plots()
        self._polarity_result = None
        self._draw_polarity_plots()
        self.statusBar().showMessage(
            f"Mesure « {removed.name} » effacée ({len(self.state.measurements)} restante(s))"
        )

    def _on_clear_all_measurements(self) -> None:
        if not self.state.measurements:
            self.statusBar().showMessage("Aucune mesure à effacer")
            return

        self.state.measurements.clear()
        self.state.reference_channel = ""
        self._refresh_delay_table()
        self._refresh_all_plots()
        self._polarity_result = None
        self._draw_polarity_plots()
        self.statusBar().showMessage("Toutes les mesures mémorisées ont été effacées")

    def _on_save_measurement(self) -> None:
        if self.state.last_delay_ms is None:
            QMessageBox.information(
                self,
                "Kalibre",
                "Aucune mesure en cours.\nLancez d'abord « Mesurer voie ».",
            )
            return

        if not self._commit_measurement():
            return

        label = self.state.measurements[-1].name
        self._draw_ir_plot()
        self._draw_transfer_plot()
        self.right_tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            f"Mesure mémorisée : {label} — {self.state.last_delay_ms:.2f} ms "
            f"({len(self.state.measurements)} courbe(s))"
        )
        self.lbl_state.setText(
            f"État : mémorisé — {label} · {self.state.last_delay_ms:.2f} ms"
        )

    def _refresh_live_time_plot(self) -> None:
        """Live désactivé en v0.4 — les graphiques IR se rafraîchissent après mesure."""
        self._draw_ir_plot()

    def _refresh_all_plots(self) -> None:
        self._draw_ir_plot()
        self._draw_transfer_plot()
        self._draw_noise_time_plot()
        self._draw_noise_fft_plot()
        self._draw_sine_time_plot()
        self._draw_sine_fft_plot()
        if self.state.sweep_analysis or (
            len(self.state.mic_signal) > 0 and np.any(self.state.mic_signal)
        ):
            self._draw_eq_reference_curves()

    def _draw_ir_plot(self) -> None:
        ax = self.plot_ir.axes
        self.plot_ir.prepare_replot()
        has_saved = False
        x_max = 15.0

        for m in self.state.measurements:
            if m.ir is None or m.ir_time_ms is None or len(m.ir) == 0:
                continue
            has_saved = True
            t = m.ir_time_ms
            ax.plot(
                t,
                m.ir,
                color=m.curve_color,
                linewidth=1.3,
                alpha=0.9,
                label=f"{m.name} ({m.absolute_delay_ms:.2f} ms)",
            )
            ax.axvline(
                m.absolute_delay_ms,
                color=m.curve_color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.65,
            )
            x_max = max(x_max, float(t[-1]), m.absolute_delay_ms * 1.2)

        analysis = self.state.sweep_analysis

        if not has_saved:
            ax.text(
                0.5,
                0.5,
                "Mesurez plusieurs positions\npour comparer les réponses IR",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=TEXT_DIM,
            )
            self.plot_ir.draw()
            return

        if analysis is not None and len(analysis.ir) > 0:
            peak_t = analysis.delay_ms
            win_end = peak_t + analysis.ir_window_ms
            ax.axvspan(peak_t, win_end, color=COLOR_REF, alpha=0.1, label="Fenêtre IR")
            x_max = max(x_max, min(float(analysis.ir_time_ms[-1]), max(win_end * 1.4, 15.0)))

        self.plot_ir._default_x = (0.0, x_max)
        self.plot_ir.apply_view(ax)
        ax.legend(loc="upper right", fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER)
        self.plot_ir.draw()

    def _signal_time_ms(self) -> np.ndarray | None:
        lb = self.state.loopback_signal
        if len(lb) == 0:
            return None
        return np.arange(len(lb)) / self.state.sample_rate * 1000.0

    def _draw_time_compare(
        self,
        plot: PlotCard,
        *,
        empty_message: str,
        show_delay: bool = True,
    ) -> None:
        ax = plot.axes
        plot.prepare_replot()
        t = self._signal_time_ms()
        lb = self.state.loopback_signal
        mic = self.state.mic_signal

        if t is None or len(mic) == 0 or not np.any(mic):
            ax.text(
                0.5,
                0.5,
                empty_message,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=TEXT_DIM,
            )
            plot.draw()
            return

        n = min(len(t), len(lb), len(mic))
        ax.plot(t[:n], lb[:n], color=COLOR_REF, linewidth=0.9, alpha=0.85, label="Loopback")
        ax.plot(t[:n], mic[:n], color=COLOR_MIC, linewidth=0.9, alpha=0.85, label="Micro")

        if show_delay and self.state.last_delay_ms is not None:
            delay = self.state.last_delay_ms
            ax.axvline(
                delay,
                color=COLOR_REF,
                linestyle="--",
                linewidth=1.2,
                label=f"Délai {delay:.2f} ms",
            )

        plot._default_x = (0.0, min(float(t[n - 1]), 500.0))
        plot.apply_view(ax)
        ax.legend(loc="upper right", fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER)
        plot.draw()

    def _draw_fft_compare(self, plot: PlotCard, *, empty_message: str) -> None:
        ax = plot.axes
        plot.prepare_replot(log_x=True)
        lb = self.state.loopback_signal
        mic = self.state.mic_signal

        if len(mic) == 0 or not np.any(mic):
            ax.text(
                0.5,
                0.5,
                empty_message,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=TEXT_DIM,
            )
            plot.draw()
            return

        sr = self.state.sample_rate
        f_lb, mag_lb = compute_fft_db(lb, sr)
        f_mic, mag_mic = compute_fft_db(mic, sr)

        if len(f_lb) > 0:
            ax.plot(
                f_lb,
                mag_lb - np.max(mag_lb),
                color=COLOR_REF,
                linewidth=1.2,
                alpha=0.85,
                label="Loopback",
            )
        if len(f_mic) > 0:
            ax.plot(
                f_mic,
                mag_mic - np.max(mag_mic),
                color=COLOR_MIC,
                linewidth=1.2,
                alpha=0.85,
                label="Micro",
            )

        plot.apply_view(ax)
        ax.legend(loc="upper right", fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER)
        plot.draw()

    def _draw_noise_time_plot(self) -> None:
        self._draw_time_compare(
            self.plot_noise_time,
            empty_message="Mesurez avec bruit rose ou blanc",
        )

    def _draw_noise_fft_plot(self) -> None:
        self._draw_fft_compare(
            self.plot_noise_fft,
            empty_message="FFT après mesure bruit",
        )

    def _draw_sine_time_plot(self) -> None:
        self._draw_time_compare(
            self.plot_sine_time,
            empty_message="Mesurez avec une sinusoïde",
        )

    def _draw_sine_fft_plot(self) -> None:
        self._draw_fft_compare(
            self.plot_sine_fft,
            empty_message="FFT après mesure sinusoïde",
        )

    def _draw_transfer_plot(self) -> None:
        fig = self.plot_transfer.panel.figure
        fig.clear()
        ax = fig.add_subplot(111)
        self.plot_transfer.axes = ax
        self.plot_transfer.panel._style_axes(ax)
        ax.set_xscale("log")
        if self.plot_transfer._xlabel:
            ax.set_xlabel(self.plot_transfer._xlabel, labelpad=4)
        if self.plot_transfer._ylabel:
            ax.set_ylabel(self.plot_transfer._ylabel, labelpad=6)

        analysis = self.state.sweep_analysis
        has_saved = False
        x_min, x_max = 20.0, 20_000.0
        mag_values: list[np.ndarray] = []

        for m in self.state.measurements:
            if m.freqs is None or m.magnitude_db_rel is None or len(m.freqs) == 0:
                continue
            has_saved = True
            ax.plot(
                m.freqs,
                m.magnitude_db_rel,
                color=m.curve_color,
                linewidth=1.3,
                alpha=0.9,
                label=f"{m.name} ({m.absolute_delay_ms:.2f} ms)",
            )
            x_min = min(x_min, float(m.freqs[0]))
            x_max = max(x_max, float(m.freqs[-1]))
            mag_values.append(m.magnitude_db_rel)

        has_live = analysis is not None and len(analysis.freqs) > 0

        if not has_saved and not has_live:
            ax.text(
                0.5,
                0.5,
                "Magnitude & cohérence\naprès mesure",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=TEXT_DIM,
            )
            self.plot_transfer.draw()
            return

        if has_live:
            mag = analysis.magnitude_db - np.max(analysis.magnitude_db)
            ax.plot(
                analysis.freqs,
                mag,
                color=COLOR_LIVE,
                linewidth=1.4,
                linestyle="--",
                alpha=0.85,
                label="Dernière mesure (live)",
            )
            x_min = min(x_min, analysis.f_min)
            x_max = max(x_max, analysis.f_max)
            mag_values.append(mag)

            ax2 = ax.twinx()
            ax2.fill_between(
                analysis.freqs,
                0,
                analysis.coherence,
                color=COLOR_COHERENCE,
                alpha=0.25,
                label="Cohérence γ²",
            )
            ax2.plot(analysis.freqs, analysis.coherence, color=COLOR_COHERENCE, linewidth=1.0)
            ax2.set_ylim(0.0, 1.05)
            ax2.set_ylabel("Cohérence γ²", color=COLOR_COHERENCE, labelpad=6)
            ax2.tick_params(axis="y", colors=COLOR_COHERENCE, labelsize=9)

        self.plot_transfer._default_x = (x_min, x_max)
        cfg = self.plot_transfer.view_config
        if cfg.auto_x:
            ax.set_xlim(x_min, x_max)
        else:
            self.plot_transfer.apply_view(ax)

        if cfg.auto_y and mag_values:
            stacked = np.concatenate(mag_values)
            y_lo = float(np.percentile(stacked, 5)) - 3
            y_hi = float(np.percentile(stacked, 95)) + 6
            ax.set_ylim(y_lo, y_hi)
        elif not cfg.auto_y:
            ax.set_ylim(cfg.y_min, cfg.y_max)

        ax.legend(loc="upper right", fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER)
        self.plot_transfer.draw()

    def _measurements_with_ir(self) -> list[tuple[int, ChannelMeasurement]]:
        rows: list[tuple[int, ChannelMeasurement]] = []
        for i, m in enumerate(self.state.measurements):
            if m.ir is not None and len(m.ir) > 64:
                rows.append((i, m))
        return rows

    def _refresh_polarity_selectors(self) -> None:
        if not hasattr(self, "cmb_pol_ref"):
            return

        with_ir = self._measurements_with_ir()
        prev_ref = self.cmb_pol_ref.currentData()
        prev_test = self.cmb_pol_test.currentData()

        self.cmb_pol_ref.blockSignals(True)
        self.cmb_pol_test.blockSignals(True)
        self.cmb_pol_ref.clear()
        self.cmb_pol_test.clear()

        for idx, m in with_ir:
            label = f"{m.name} ({m.absolute_delay_ms:.2f} ms)"
            self.cmb_pol_ref.addItem(label, idx)
            self.cmb_pol_test.addItem(label, idx)

        def restore(combo: QComboBox, previous: object, offset: int = 0) -> None:
            if previous is not None:
                i = combo.findData(previous)
                if i >= 0:
                    combo.setCurrentIndex(i)
                    return
            if combo.count() > offset:
                combo.setCurrentIndex(offset)

        ref_default = 0
        for j, (idx, m) in enumerate(with_ir):
            if m.is_reference:
                ref_default = j
                break

        restore(self.cmb_pol_ref, prev_ref, ref_default)
        test_default = 1 if len(with_ir) > 1 else 0
        if test_default == ref_default and len(with_ir) > 1:
            test_default = 0 if ref_default != 0 else 1
        restore(self.cmb_pol_test, prev_test, test_default)

        self.cmb_pol_ref.blockSignals(False)
        self.cmb_pol_test.blockSignals(False)

    def _polarity_band(self) -> tuple[float, float]:
        f_min = self.spin_pol_fmin.value()
        f_max = self.spin_pol_fmax.value()
        if f_max <= f_min:
            f_max = f_min + 10.0
        return f_min, f_max

    def _on_analyze_polarity(self) -> None:
        with_ir = self._measurements_with_ir()
        if len(with_ir) < 2:
            QMessageBox.information(
                self,
                "Kalibre",
                "Il faut au moins 2 mesures sweep mémorisées (avec IR).\n"
                "Mesurez chaque voie séparément puis revenez ici.",
            )
            return

        ref_idx = self.cmb_pol_ref.currentData()
        test_idx = self.cmb_pol_test.currentData()
        if ref_idx is None or test_idx is None or ref_idx == test_idx:
            QMessageBox.information(
                self,
                "Kalibre",
                "Choisissez deux mesures différentes (référence et comparée).",
            )
            return

        ref_m = self.state.measurements[int(ref_idx)]
        test_m = self.state.measurements[int(test_idx)]
        f_min, f_max = self._polarity_band()

        result = analyze_polarity_pair(
            ref_m.name,
            test_m.name,
            ref_m.ir,  # type: ignore[arg-type]
            test_m.ir,  # type: ignore[arg-type]
            self.state.sample_rate,
            delay_ref_ms=ref_m.absolute_delay_ms,
            delay_test_ms=test_m.absolute_delay_ms,
            f_min=f_min,
            f_max=f_max,
        )
        if result is None:
            QMessageBox.warning(self, "Kalibre", "Analyse polarité impossible sur cette paire.")
            return

        self._polarity_result = result
        self._fill_polarity_summary_table(int(ref_idx), f_min, f_max)
        self._update_polarity_verdict(result)
        self._draw_polarity_plots()
        self.right_tabs.setCurrentIndex(4)
        self.statusBar().showMessage(
            f"Polarité — {result.verdict} ({result.confidence:.0%})"
        )

    def _fill_polarity_summary_table(
        self,
        reference_index: int,
        f_min: float,
        f_max: float,
    ) -> None:
        rows = analyze_all_vs_reference(
            self.state.measurements,
            reference_index,
            self.state.sample_rate,
            f_min=f_min,
            f_max=f_max,
        )
        self.tbl_polarity.setRowCount(len(rows))
        for row, (name, result, status) in enumerate(rows):
            self.tbl_polarity.setItem(row, 0, QTableWidgetItem(name))
            if result is None:
                for col in range(1, 6):
                    self.tbl_polarity.setItem(row, col, QTableWidgetItem("—"))
                self.tbl_polarity.setItem(row, 6, QTableWidgetItem(status))
                continue

            corr_plus = (
                result.ir_correlation
                if not result.inverted
                else result.ir_correlation_opposite
            )
            corr_minus = (
                result.ir_correlation_opposite
                if not result.inverted
                else result.ir_correlation
            )
            self.tbl_polarity.setItem(row, 1, QTableWidgetItem(f"{corr_plus:+.2f}"))
            self.tbl_polarity.setItem(row, 2, QTableWidgetItem(f"{corr_minus:+.2f}"))
            self.tbl_polarity.setItem(
                row, 3, QTableWidgetItem(f"{result.reinforcement_db:+.1f} dB")
            )
            self.tbl_polarity.setItem(
                row, 4, QTableWidgetItem(f"{result.median_phase_deg:+.0f}°")
            )
            self.tbl_polarity.setItem(
                row, 5, QTableWidgetItem(f"{result.confidence:.0%}")
            )
            status_item = QTableWidgetItem(status)
            if status == "Inversé":
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == "OK":
                status_item.setForeground(Qt.GlobalColor.green)
            self.tbl_polarity.setItem(row, 6, status_item)

    def _update_polarity_verdict(self, result: PolarityResult) -> None:
        if result.inverted:
            color = COLOR_ERROR if result.confidence >= 0.45 else "#e67e22"
        else:
            color = BTN_SUCCESS if result.confidence >= 0.45 else TEXT_DIM

        self.lbl_polarity_verdict.setText(
            f"{result.verdict} — {result.test_name} vs {result.reference_name}\n"
            f"{result.detail}"
        )
        self.lbl_polarity_verdict.setStyleSheet(
            f"color: {color}; font-weight: bold; padding: 6px; "
            f"border: 1px solid {BORDER}; border-radius: 6px;"
        )

    def _draw_polarity_plots(self) -> None:
        self._draw_polarity_ir_plot()
        self._draw_polarity_sum_plot()
        self._draw_polarity_phase_plot()

    def _draw_polarity_ir_plot(self) -> None:
        ax = self.plot_polarity_ir.axes
        self.plot_polarity_ir.prepare_replot()
        result = self._polarity_result

        if result is None:
            ax.text(
                0.5,
                0.5,
                "Sélectionnez deux mesures\npuis Analyser polarité",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=TEXT_DIM,
            )
            self.plot_polarity_ir.draw()
            return

        ax.plot(
            result.time_ms,
            result.ir_reference,
            color=COLOR_REF,
            linewidth=1.4,
            label=f"Réf. {result.reference_name}",
        )
        ax.plot(
            result.time_ms,
            result.ir_test,
            color=COLOR_MIC,
            linewidth=1.2,
            linestyle="--",
            label=(
                f"Test {result.test_name} (signe corrigé)"
                if result.inverted
                else f"Test {result.test_name}"
            ),
        )
        self.plot_polarity_ir._default_x = (
            float(result.time_ms[0]),
            float(result.time_ms[-1]),
        )
        self.plot_polarity_ir.apply_view(ax)
        ax.legend(loc="upper right", fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER)
        self.plot_polarity_ir.draw()

    def _draw_polarity_sum_plot(self) -> None:
        ax = self.plot_polarity_sum.axes
        self.plot_polarity_sum.prepare_replot(log_x=True)
        result = self._polarity_result

        if result is None:
            ax.text(
                0.5,
                0.5,
                "Somme A+B vs A−B\n(détecte l'inversion 180°)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=TEXT_DIM,
            )
            self.plot_polarity_sum.draw()
            return

        band = (result.freqs >= result.f_min) & (result.freqs <= result.f_max)
        ax.plot(
            result.freqs[band],
            result.mag_sum_db[band],
            color=BTN_SUCCESS,
            linewidth=1.6,
            label="A + B (même polarité)",
        )
        ax.plot(
            result.freqs[band],
            result.mag_diff_db[band],
            color=COLOR_ERROR,
            linewidth=1.4,
            linestyle="--",
            label="A − B (si B inversé)",
        )
        self.plot_polarity_sum._default_x = (result.f_min, result.f_max)
        self.plot_polarity_sum.apply_view(ax)
        ax.legend(loc="upper right", fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER)
        self.plot_polarity_sum.draw()

    def _draw_polarity_phase_plot(self) -> None:
        ax = self.plot_polarity_phase.axes
        self.plot_polarity_phase.prepare_replot(log_x=True)
        result = self._polarity_result

        if result is None:
            ax.text(
                0.5,
                0.5,
                "Phase relative\n~0° = OK · ~±180° = inversion",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=TEXT_DIM,
            )
            self.plot_polarity_phase.draw()
            return

        band = (result.freqs >= result.f_min) & (result.freqs <= result.f_max)
        phases = result.phase_diff_deg[band]
        freqs = result.freqs[band]
        ax.plot(freqs, phases, color=COLOR_REF, linewidth=1.2, label="Δ phase")
        ax.axhline(0.0, color=TEXT_DIM, linewidth=0.8, linestyle=":")
        ax.axhline(180.0, color=COLOR_ERROR, linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(-180.0, color=COLOR_ERROR, linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhspan(-35, 35, color=BTN_SUCCESS, alpha=0.08)
        ax.axhspan(145, 180, color=COLOR_ERROR, alpha=0.08)
        ax.axhspan(-180, -145, color=COLOR_ERROR, alpha=0.08)
        self.plot_polarity_phase._default_x = (result.f_min, result.f_max)
        self.plot_polarity_phase.apply_view(ax)
        ax.legend(loc="upper right", fontsize=8, facecolor=BG_PANEL, edgecolor=BORDER)
        self.plot_polarity_phase.draw()

    def _on_analyze_eq(self) -> None:
        self._draw_eq_reference_curves()

    def _draw_eq_reference_curves(self) -> None:
        """Mesuré (plein) + courbes EQ théoriques en pointillés."""
        profile = self.cmb_profile.currentText()
        ref_f = reference_freqs()
        all_curves = build_all_reference_curves(ref_f)

        measured_norm: np.ndarray | None = None
        freqs_meas = ref_f

        analysis = self.state.sweep_analysis
        if analysis is not None and len(analysis.freqs) > 0:
            freqs_meas = analysis.freqs
            measured_norm = analysis.magnitude_db - np.max(analysis.magnitude_db)
        elif len(self.state.mic_signal) > 0 and np.any(self.state.mic_signal):
            sr = self.state.sample_rate
            f_meas, measured = compute_fft_db(self.state.mic_signal, sr)
            if len(f_meas) > 0:
                freqs_meas = f_meas
                measured_norm = measured - np.max(measured)

        ax = self.plot_eq.axes
        self.plot_eq.prepare_replot(log_x=True)

        if measured_norm is not None:
            ax.plot(freqs_meas, measured_norm, color=COLOR_MIC, linewidth=2.2, label="Mesuré", zorder=10)

        for name in PROFILE_ORDER:
            sel = name == profile
            ax.plot(
                ref_f,
                all_curves[name],
                color=PROFILE_COLORS[name],
                linewidth=2.5 if sel else 1.2,
                linestyle="--" if sel else (0, (5, 4)),
                alpha=1.0 if sel else 0.85,
                label=name,
                zorder=6 if sel else 3,
            )

        self.plot_eq.set_trend_data(
            freqs_meas if measured_norm is not None else None,
            measured_norm,
        )
        self.plot_eq.apply_view(ax)
        ax.legend(
            loc="upper right",
            fontsize=7,
            ncol=2,
            framealpha=0.92,
            facecolor=BG_PANEL,
            edgecolor=BORDER,
        )
        self.plot_eq.frame.setTitle(f"Courbes EQ théoriques — actif : {profile}")
        self.plot_eq.draw()

        if measured_norm is not None:
            target_on_meas = np.interp(freqs_meas, ref_f, all_curves[profile])
            self._fill_eq_suggestions(
                suggest_eq_from_diff(freqs_meas, measured_norm, target_on_meas)
            )
        else:
            self.tbl_eq_suggestions.setRowCount(0)

    def _fill_eq_suggestions(self, suggestions: list) -> None:
        self.tbl_eq_suggestions.setRowCount(len(suggestions))
        for row, s in enumerate(suggestions):
            self.tbl_eq_suggestions.setItem(row, 0, QTableWidgetItem(str(s["type"])))
            self.tbl_eq_suggestions.setItem(row, 1, QTableWidgetItem(str(s["freq_hz"])))
            self.tbl_eq_suggestions.setItem(row, 2, QTableWidgetItem(f"{s['gain_db']:+.1f}"))
            self.tbl_eq_suggestions.setItem(row, 3, QTableWidgetItem(str(s["q"])))


def run_app() -> None:
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
