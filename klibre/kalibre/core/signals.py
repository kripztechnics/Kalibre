"""
Génération de signaux de test et données simulées pour la maquette UI.

Plus tard, ces fonctions alimenteront la vraie capture audio ;
pour l'instant elles produisent des courbes réalistes pour tester l'affichage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DelayEstimate:
    """Résultat GCC-PHAT avec indicateurs de qualité."""

    delay_ms: float
    confidence: float  # 0…1 — netteté du pic
    crosstalk_ratio: float  # part loopback dans le micro (0 = propre)
    ambiguous: bool  # deux pics proches en intensité


def make_time_axis(duration_s: float, sample_rate: int) -> NDArray[np.float64]:
    """Axe temporel régulier en secondes."""
    n = int(duration_s * sample_rate)
    return np.linspace(0.0, duration_s, n, endpoint=False)


def generate_white_noise(
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.25,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Bruit blanc gaussien — signal typique pour mesure de délai."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration_s * sample_rate)) * amplitude).astype(np.float64)


def generate_pink_noise(
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.25,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Bruit rose (−3 dB/octave) — plus d'énergie dans les graves, classique en mesure audio.

    Généré en domaine fréquentiel : amplitude ~ 1/sqrt(f).
    """
    n = int(duration_s * sample_rate)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    # Filtre rose : évite division par zéro sur DC
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * scale, n=n)

    peak = float(np.max(np.abs(pink))) or 1.0
    return (pink / peak * amplitude).astype(np.float64)


def generate_sine(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.25,
) -> NDArray[np.float64]:
    """Sinusoïde simple (comme dans ton générateur MATLAB)."""
    t = make_time_axis(duration_s, sample_rate)
    return (amplitude * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float64)


def simulate_channel_response(
    reference: NDArray[np.float64],
    delay_ms: float,
    sample_rate: int,
    gain: float = 1.0,
    noise_level: float = 0.02,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Simule la réponse d'une voie : retard + gain + bruit de fond.

    En production, ce sera le signal capté au micro après le trajet acoustique.
    """
    delay_samples = int(round(delay_ms * 1e-3 * sample_rate))
    delayed = np.zeros_like(reference)
    if delay_samples < len(reference):
        delayed[delay_samples:] = reference[: len(reference) - delay_samples] * gain

    rng = np.random.default_rng(seed)
    return delayed + rng.standard_normal(len(reference)) * noise_level


def compute_fft_db(
    signal: NDArray[np.float64],
    sample_rate: int,
    f_min: float = 20.0,
    f_max: float = 20_000.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Magnitude en dB d'un signal (demi-spectre positif).

    Retourne (fréquences Hz, magnitude dB).
    """
    n = len(signal)
    if n == 0:
        return np.array([]), np.array([])

    window = np.hanning(n)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum) / (np.sum(window) / 2.0 + 1e-12)
    db = 20.0 * np.log10(magnitude + 1e-12)

    mask = (freqs >= f_min) & (freqs <= f_max)
    return freqs[mask], db[mask]


def _parabolic_peak_index(values: NDArray[np.float64], index: int) -> float:
    """Affine le pic de corrélation au sous-échantillon (précision ~0.1 ms)."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    y0, y1, y2 = values[index - 1], values[index], values[index + 1]
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-12:
        return float(index)
    return index + (y0 - y2) / denom


def _remove_loopback_leakage(
    reference: NDArray[np.float64],
    measured: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """
    Retire la fuite électrique loopback → micro (pic parasite près de 0 ms).

    Modèle : micro ≈ k × loopback + signal_acoustique_retardé
    """
    ref_energy = float(np.dot(reference, reference)) + 1e-12
    k = float(np.dot(measured, reference)) / ref_energy
    cleaned = measured - k * reference
    mes_std = float(np.std(measured)) + 1e-12
    crosstalk = float(np.clip(abs(k * np.std(reference)) / mes_std, 0.0, 1.0))
    return cleaned, crosstalk


def _local_maxima(values: NDArray[np.float64], threshold: float) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    for i in range(1, len(values) - 1):
        if values[i] < threshold:
            continue
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            peaks.append((i, float(values[i])))
    return peaks


def _pick_acoustic_peak(
    search: NDArray[np.float64],
    sample_rate: int,
    *,
    min_lag_ms: float = 0.35,
) -> tuple[int, bool]:
    """
    Choisit le pic acoustique plutôt que la fuite loopback résiduelle (≈ 0 ms).

    Retourne (index, ambiguous).
    """
    global_idx = int(np.argmax(search))
    global_val = float(search[global_idx])
    if global_val < 1e-12:
        return 0, False

    min_lag = int(min_lag_ms * 1e-3 * sample_rate)
    threshold = 0.45 * global_val
    peaks = _local_maxima(search, threshold)
    if not peaks:
        peaks = [(global_idx, global_val)]

    acoustic = [(i, v) for i, v in peaks if i >= min_lag]
    if acoustic:
        best_idx, best_val = max(acoustic, key=lambda p: p[1])
        near_zero = [(i, v) for i, v in peaks if i < min_lag]
        ambiguous = bool(
            near_zero
            and max(v for _, v in near_zero) > 0.65 * best_val
            and best_idx != global_idx
        )
        return best_idx, ambiguous

    return global_idx, global_val > 0.65 * float(np.median(search))


def _normalized_delay_profile(
    reference: NDArray[np.float64],
    measured: NDArray[np.float64],
    max_lag: int,
) -> NDArray[np.float64]:
    """
    Corrélation normalisée pour délais positifs (micro après loopback).

    loopback[t] s'aligne avec measured[t + D] → délai physique D.
    """
    n = len(reference)
    if max_lag <= 0:
        return np.array([1.0 if n else 0.0])

    ref_sq = np.concatenate(([0.0], np.cumsum(reference * reference)))
    mes_sq = np.concatenate(([0.0], np.cumsum(measured * measured)))

    profile = np.empty(max_lag + 1, dtype=np.float64)
    for d in range(max_lag + 1):
        overlap = n - d
        raw = float(np.dot(reference[:overlap], measured[d:n]))
        norm = np.sqrt((ref_sq[overlap] - ref_sq[0]) * (mes_sq[n] - mes_sq[d]) + 1e-12)
        profile[d] = abs(raw / norm)

    return profile


def estimate_delay(
    reference: NDArray[np.float64],
    measured: NDArray[np.float64],
    sample_rate: int,
    max_delay_ms: float = 50.0,
) -> DelayEstimate:
    """
    Délai loopback → micro : corrélation normalisée + retrait fuite loopback.

    Étapes :
      1. soustraction de la fuite électrique loopback dans le micro
      2. corrélation normalisée (micro après loopback)
      3. choix du pic acoustique (> ~0,35 ms), pas la fuite résiduelle
      4. interpolation parabolique sous-échantillon
    """
    empty = DelayEstimate(0.0, 0.0, 0.0, True)
    if len(reference) == 0 or len(measured) == 0:
        return empty

    n = min(len(reference), len(measured))
    if n < 128:
        return empty

    ref = reference[:n].astype(np.float64) - np.mean(reference[:n])
    mes = measured[:n].astype(np.float64) - np.mean(measured[:n])
    mes, crosstalk = _remove_loopback_leakage(ref, mes)

    max_lag = min(n - 1, int(max_delay_ms * 1e-3 * sample_rate))
    search = _normalized_delay_profile(ref, mes, max_lag)

    if len(search) == 0 or np.max(search) < 1e-6:
        return DelayEstimate(0.0, 0.0, crosstalk, True)

    peak_idx, ambiguous = _pick_acoustic_peak(search, sample_rate)
    peak_val = float(search[peak_idx])
    median = float(np.median(search)) + 1e-12
    confidence = float(np.clip(peak_val / median / 25.0, 0.0, 1.0))

    # Pic trop faible → mesure peu fiable
    if peak_val < 3.0 * median:
        ambiguous = True
        confidence = min(confidence, 0.25)

    refined = _parabolic_peak_index(search, peak_idx)
    delay_ms = refined * 1000.0 / sample_rate
    return DelayEstimate(delay_ms, confidence, crosstalk, ambiguous)


def estimate_delay_ms(
    reference: NDArray[np.float64],
    measured: NDArray[np.float64],
    sample_rate: int,
    max_delay_ms: float = 50.0,
) -> float:
    """Raccourci — retourne uniquement le délai en ms."""
    return estimate_delay(reference, measured, sample_rate, max_delay_ms).delay_ms


def estimate_delay_from_phase(
    freqs: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
    coherence: NDArray[np.float64] | None = None,
    *,
    f_min: float | None = None,
    f_max: float | None = None,
    weight_power: float = 2.0,
    min_points: int = 8,
) -> tuple[float, float]:
    """Estimate acoustic delay from phase slope (degrees vs Hz).

    Performs an (optionally) coherence-weighted linear regression on the
    unwrapped phase (in degrees) against frequency (Hz). Returns a tuple
    `(delay_ms, r2)` where `delay_ms` is the estimated delay in milliseconds
    (positive = mic delayed after reference) and `r2` is the weighted
    coefficient of determination (0..1) indicating fit quality.

    The slope 'a' in deg/Hz relates to delay by a = -360 * delay_s.
    """
    if len(freqs) == 0 or len(phase_deg) == 0:
        return 0.0, 0.0

    freqs = np.asarray(freqs, dtype=np.float64)
    phase_deg = np.asarray(phase_deg, dtype=np.float64)

    mask = np.isfinite(freqs) & np.isfinite(phase_deg)
    if f_min is not None:
        mask &= freqs >= float(f_min)
    if f_max is not None:
        mask &= freqs <= float(f_max)
    if coherence is not None:
        coh = np.asarray(coherence, dtype=np.float64)
        mask &= np.isfinite(coh)
    else:
        coh = None

    if np.count_nonzero(mask) < max(3, min_points):
        return 0.0, 0.0

    f = freqs[mask]
    ph = phase_deg[mask]
    ph_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(ph)))

    if coh is not None:
        w = np.clip(coh[mask], 0.0, 1.0) ** float(weight_power)
        # avoid zero weights
        w_sum = float(np.sum(w)) + 1e-12
        w = w / w_sum
    else:
        w = None

    # Weighted linear regression (degree 1)
    try:
        if w is None:
            coef = np.polyfit(f, ph_unwrapped, 1)
            slope, intercept = float(coef[0]), float(coef[1])
            yhat = slope * f + intercept
            resid = ph_unwrapped - yhat
            ss_res = float(np.sum(resid * resid))
            ss_tot = float(np.sum((ph_unwrapped - float(np.mean(ph_unwrapped))) ** 2)) + 1e-12
        else:
            # use numpy.polyfit with w as weights (it expects sigma inversely)
            coef = np.polyfit(f, ph_unwrapped, 1, w=w)
            slope, intercept = float(coef[0]), float(coef[1])
            yhat = slope * f + intercept
            ss_res = float(np.sum(w * (ph_unwrapped - yhat) ** 2))
            mean_w = float(np.sum(w * ph_unwrapped))
            ss_tot = float(np.sum(w * (ph_unwrapped - mean_w) ** 2)) + 1e-12

        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        delay_s = -slope / 360.0
        delay_ms = float(delay_s * 1000.0)

        # One robust iteration: remove large residuals and refit if many points
        if len(f) >= 12:
            sigma = float(np.sqrt(ss_res / len(f)))
            if sigma > 1e-6:
                good = np.abs(ph_unwrapped - yhat) <= (3.0 * sigma)
                if np.count_nonzero(good) >= max(3, int(0.7 * len(f))):
                    f2 = f[good]
                    ph2 = ph_unwrapped[good]
                    w2 = (w[good] if w is not None else None)
                    if w2 is None:
                        coef2 = np.polyfit(f2, ph2, 1)
                        slope2 = float(coef2[0])
                    else:
                        coef2 = np.polyfit(f2, ph2, 1, w=w2)
                        slope2 = float(coef2[0])
                    delay_ms = -slope2 / 360.0 * 1000.0
                    # recompute simple r2 on trimmed data
                    yhat2 = slope2 * f2 + float(coef2[1])
                    if w2 is None:
                        ss_res2 = float(np.sum((ph2 - yhat2) ** 2))
                        ss_tot2 = float(np.sum((ph2 - float(np.mean(ph2))) ** 2)) + 1e-12
                    else:
                        ss_res2 = float(np.sum(w2 * (ph2 - yhat2) ** 2))
                        ss_tot2 = float(np.sum(w2 * (ph2 - float(np.sum(w2 * ph2))) ** 2)) + 1e-12
                    r2 = max(0.0, 1.0 - ss_res2 / ss_tot2)

        return delay_ms, float(np.clip(r2, 0.0, 1.0))
    except Exception:
        return 0.0, 0.0


def simulate_loopback_capture(
    emitted: NDArray[np.float64],
    acoustic_delay_ms: float,
    sample_rate: int,
    mic_gain: float = 0.85,
    loopback_gain: float = 0.95,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simule loopback (quasi immédiat) + micro retardé — pour le mode simulation.

    Retourne (loopback, micro).
    """
    loopback = emitted * loopback_gain
    mic = simulate_channel_response(
        emitted,
        delay_ms=acoustic_delay_ms,
        sample_rate=sample_rate,
        gain=mic_gain,
        seed=seed,
    )
    return loopback, mic


def suggest_eq_from_diff(
    freqs: NDArray[np.float64],
    measured_db: NDArray[np.float64],
    target_db: NDArray[np.float64],
) -> list[dict[str, float | str]]:
    """
    Propositions d'EQ très grossières à partir de l'écart mesuré − cible.

    Retourne une liste de filtres suggérés (type, freq, gain) pour affichage UI.
    """
    if len(freqs) == 0:
        return []

    diff = target_db - measured_db
    suggestions: list[dict[str, float | str]] = []

    # Analyse par bandes — suffisant pour une maquette
    bands = [
        ("Low shelf", 30.0, 120.0),
        ("Peaking", 200.0, 400.0),
        ("Peaking", 800.0, 2500.0),
        ("High shelf", 4000.0, 16000.0),
    ]
    for eq_type, f_lo, f_hi in bands:
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(mask):
            continue
        avg_gap = float(np.mean(diff[mask]))
        if abs(avg_gap) < 1.5:
            continue
        center = float(np.sqrt(f_lo * f_hi))
        suggestions.append(
            {
                "type": eq_type,
                "freq_hz": round(center, 1),
                "gain_db": round(np.clip(avg_gap, -6.0, 6.0), 1),
                "q": 0.7 if "shelf" in eq_type.lower() else 1.4,
            }
        )

    return suggestions[:4]  # max 4 propositions affichées
