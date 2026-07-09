"""
Courbes d'égalisation théoriques de référence (live / tuning).

Sources / consensus pro :
  - Dirac : bass +4 à +8 dB vs médiums, HF −2 à −6 dB @ 20 kHz
  - ProSoundWeb (Ales Stefancic) : rock +4–6 dB LF < 70 Hz ; voix flat + rolloff HF
  - Courbe « house » : pente −1 à −3 dB/octave au-dessus de ~2 kHz (salle réelle)
  - Harman / B&K : légère pente descendante perçue « naturelle » en pièce

Les courbes sont normalisées à 0 dB @ 1 kHz pour comparaison visuelle.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Couleur matplotlib suggérée par profil (affichage pointillé)
PROFILE_COLORS: dict[str, str] = {
    "Neutre (flat)": "#95a5a6",
    "House / Salle live": "#3498db",
    "Bass music / EDM": "#e74c3c",
    "Rock live": "#e67e22",
    "Voix / Speech": "#2ecc71",
    "Jazz / Acoustique": "#9b59b6",
}

PROFILE_ORDER: list[str] = list(PROFILE_COLORS.keys())


def _norm_1khz(freqs: NDArray[np.float64], curve: NDArray[np.float64]) -> NDArray[np.float64]:
    """Référence 0 dB à 1 kHz."""
    idx = int(np.argmin(np.abs(freqs - 1000.0)))
    return curve - curve[idx]


def _low_shelf(freqs: NDArray[np.float64], fc: float, gain_db: float, order: float = 2.0) -> NDArray[np.float64]:
    ratio = np.maximum(freqs / max(fc, 1.0), 1e-6)
    return gain_db / (1.0 + ratio**order)


def _high_shelf(freqs: NDArray[np.float64], fc: float, gain_db: float, order: float = 2.0) -> NDArray[np.float64]:
    ratio = np.maximum(fc / np.maximum(freqs, 1.0), 1e-6)
    return gain_db / (1.0 + ratio**order)


def _peak(freqs: NDArray[np.float64], f0: float, gain_db: float, q: float = 1.4) -> NDArray[np.float64]:
    """Cloche gaussienne en échelle log — approximation peaking EQ."""
    log_f = np.log10(np.maximum(freqs, 1.0))
    log_f0 = np.log10(f0)
    width = 1.0 / max(q, 0.3)
    return gain_db * np.exp(-0.5 * ((log_f - log_f0) / width) ** 2)


def _octave_tilt(freqs: NDArray[np.float64], ref_hz: float, db_per_octave: float) -> NDArray[np.float64]:
    """Pente dB/octave au-dessus de ref_hz (négatif = rolloff HF)."""
    out = np.zeros_like(freqs)
    mask = freqs >= ref_hz
    out[mask] = db_per_octave * np.log2(freqs[mask] / ref_hz)
    return out


def _high_pass_slope(freqs: NDArray[np.float64], fc: float, slope_db: float = -12.0) -> NDArray[np.float64]:
    """Atténuation sous fc (voix / speech)."""
    ratio = np.maximum(freqs / max(fc, 1.0), 1e-6)
    out = np.zeros_like(freqs)
    mask = freqs < fc
    out[mask] = slope_db * (1.0 - ratio[mask])  # simplifié sous fc
    return out


def build_reference_curve(profile_name: str, freqs: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Construit la courbe EQ théorique (dB relatifs) pour un profil donné.
    """
    f = freqs.astype(np.float64)
    curve = np.zeros_like(f)

    if profile_name == "Neutre (flat)":
        # Flat anéchoïque — référence linéaire
        pass

    elif profile_name == "House / Salle live":
        # Pente salle : ~−1.5 dB/oct au-dessus de 2 kHz (HF air absorption)
        curve += _octave_tilt(f, 2000.0, -1.5)
        curve += _low_shelf(f, 80.0, +1.0)

    elif profile_name == "Bass music / EDM":
        # Dirac / live EDM : +6 dB graves 30–80 Hz, mids plats, HF doux
        curve += _low_shelf(f, 55.0, +6.0, order=1.8)
        curve += _peak(f, 45.0, +2.0, q=0.9)
        curve += _high_shelf(f, 12000.0, -3.0)
        curve += _octave_tilt(f, 8000.0, -0.8)

    elif profile_name == "Rock live":
        # Stefancic PSW : +4–6 dB < 70 Hz ; présence guitars +2–3 dB
        curve += _low_shelf(f, 70.0, +5.0, order=1.6)
        curve += _peak(f, 3000.0, +2.5, q=1.2)
        curve += _peak(f, 120.0, +1.0, q=0.8)
        curve += _high_shelf(f, 10000.0, -2.0)

    elif profile_name == "Voix / Speech":
        # Intelligibilité : coupe LF, boost 2–4 kHz (clarté consonnes)
        curve += _high_pass_slope(f, 100.0, -10.0)
        curve += _peak(f, 2500.0, +3.5, q=1.5)
        curve += _peak(f, 4500.0, +2.0, q=1.8)
        curve += _high_shelf(f, 8000.0, -2.5)

    elif profile_name == "Jazz / Acoustique":
        # Flat + léger rolloff sub + HF naturel (folk/classical PSW)
        curve += _low_shelf(f, 40.0, -1.5)
        curve += _octave_tilt(f, 3000.0, -1.0)
        curve += _peak(f, 800.0, +0.8, q=0.7)

    else:
        profile_name = "Neutre (flat)"

    return _norm_1khz(f, curve)


def build_all_reference_curves(
    freqs: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Toutes les courbes théoriques pour overlay pointillé."""
    return {name: build_reference_curve(name, freqs) for name in PROFILE_ORDER}


def reference_freqs(
    f_min: float = 20.0,
    f_max: float = 20_000.0,
    points: int = 512,
) -> NDArray[np.float64]:
    """Grille log pour tracé des courbes même sans mesure."""
    return np.logspace(np.log10(f_min), np.log10(f_max), points)


# Alias compatibilité ancien code
TARGET_PROFILES = {name: {"description": name} for name in PROFILE_ORDER}


def build_target_curve(freqs: NDArray[np.float64], profile_name: str) -> NDArray[np.float64]:
    return build_reference_curve(profile_name, freqs)
