"""
Sweep ESS et analyse par référence loopback (style REW / Smaart).

Référence acoustique : H(f) = Micro / Loopback → IR, magnitude, cohérence, délai.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kalibre.core.signals import _parabolic_peak_index


SWEEP_PRESETS: dict[str, tuple[float, float, float]] = {
    "Complet (20 Hz – 20 kHz)": (20.0, 20_000.0, 5.0),
    "Sub (20 – 200 Hz)": (20.0, 200.0, 2.5),
    "Medium (80 Hz – 5 kHz)": (80.0, 5_000.0, 3.0),
    "Aigu (1 – 20 kHz)": (1_000.0, 20_000.0, 2.5),
}


@dataclass(frozen=True)
class SweepAnalysis:
    """Résultat analyse sweep + référence loopback."""

    delay_ms: float
    confidence: float
    mean_coherence: float
    f_min: float
    f_max: float
    ir_window_ms: float
    ir_time_ms: NDArray[np.float64]
    ir: NDArray[np.float64]
    freqs: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    phase_deg: NDArray[np.float64]
    coherence: NDArray[np.float64]
    peak_index: int


def generate_ess_sweep(
    f_min: float,
    f_max: float,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.25,
) -> NDArray[np.float64]:
    """
    Sweep sinusoïdal exponentiel (ESS) — compatible déconvolution / mesure REW.

    Fréquence instantanée : f1 → f2 sur duration_s secondes.
    """
    f1 = max(float(f_min), 1.0)
    f2 = max(float(f_max), f1 * 1.05)
    duration_s = max(float(duration_s), 0.1)

    n = int(duration_s * sample_rate)
    if n < 64:
        return np.array([], dtype=np.float64)

    t = np.arange(n, dtype=np.float64) / sample_rate
    ratio = f2 / f1
    R = np.log(ratio)
    phase = 2.0 * np.pi * f1 * duration_s / R * (np.exp(t / duration_s * R) - 1.0)
    sweep = np.sin(phase)

    peak = float(np.max(np.abs(sweep))) or 1.0
    return (sweep / peak * amplitude).astype(np.float64)


def analyze_acoustic_reference(
    loopback: NDArray[np.float64],
    mic: NDArray[np.float64],
    sample_rate: int,
    *,
    f_min: float = 20.0,
    f_max: float = 20_000.0,
    ir_window_ms: float = 7.0,
    max_delay_ms: float = 50.0,
) -> SweepAnalysis | None:
    """
    Fonction de transfert Micro/Loopback → IR, magnitude, cohérence γ², délai.

    Équivalent REW « Use acoustic reference » + fenêtre IR droite.
    """
    n = min(len(loopback), len(mic))
    if n < 256 or sample_rate <= 0:
        return None

    lb = loopback[:n].astype(np.float64) - np.mean(loopback[:n])
    mc = mic[:n].astype(np.float64) - np.mean(mic[:n])

    nfft = 1 << (n - 1).bit_length()
    L = np.fft.rfft(lb, n=nfft)
    M = np.fft.rfft(mc, n=nfft)

    power_l = np.abs(L) ** 2
    reg = 1e-9 * float(np.max(power_l) + 1e-20)
    H = M * np.conj(L) / (power_l + reg)

    Sxx = power_l
    Syy = np.abs(M) ** 2
    Sxy = M * np.conj(L)
    coherence = np.clip((np.abs(Sxy) ** 2 / (Sxx * Syy + reg)).real, 0.0, 1.0)

    ir_full = np.fft.irfft(H, n=nfft).real
    ir = ir_full[:n]

    search_len = min(n, int(max_delay_ms * 1e-3 * sample_rate) + 1)
    search = np.abs(ir[:search_len])
    min_peak = int(0.35e-3 * sample_rate)
    if min_peak < len(search):
        search[:min_peak] = 0.0
    if np.max(search) < 1e-12:
        return None

    peak_idx = int(np.argmax(search))
    refined = _parabolic_peak_index(search, peak_idx)
    delay_ms = refined * 1000.0 / sample_rate

    peak_val = float(search[peak_idx])
    median = float(np.median(search)) + 1e-12
    confidence = float(np.clip(peak_val / median / 20.0, 0.0, 1.0))

    ir_windowed = _apply_ir_right_window(ir, peak_idx, sample_rate, ir_window_ms)

    win = np.hanning(len(ir_windowed))
    spec = np.fft.rfft(ir_windowed * win)
    freqs_mag = np.fft.rfftfreq(len(ir_windowed), d=1.0 / sample_rate)
    magnitude_db = (20.0 * np.log10(np.abs(spec) + 1e-12)).astype(np.float64)
    phase_deg = np.degrees(np.angle(spec)).astype(np.float64)

    freqs_full = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    coh_interp = np.interp(
        freqs_mag,
        freqs_full[: len(coherence)],
        coherence[: len(freqs_full)],
        left=0.0,
        right=0.0,
    )

    mask = (freqs_mag >= f_min) & (freqs_mag <= f_max)
    if not np.any(mask):
        return None

    freqs_out = freqs_mag[mask]
    mean_coh = float(np.mean(coh_interp[mask]))

    ir_time_ms = np.arange(len(ir)) * 1000.0 / sample_rate

    return SweepAnalysis(
        delay_ms=delay_ms,
        confidence=confidence,
        mean_coherence=mean_coh,
        f_min=f_min,
        f_max=f_max,
        ir_window_ms=ir_window_ms,
        ir_time_ms=ir_time_ms,
        ir=ir,
        freqs=freqs_out,
        magnitude_db=magnitude_db[mask],
        phase_deg=phase_deg[mask],
        coherence=coh_interp[mask].astype(np.float64),
        peak_index=peak_idx,
    )


def _apply_ir_right_window(
    ir: NDArray[np.float64],
    peak_idx: int,
    sample_rate: int,
    window_ms: float,
) -> NDArray[np.float64]:
    """Fenêtre droite REW : conserve le direct + window_ms, coupe les réflexions."""
    out = np.zeros_like(ir)
    pre = max(0, peak_idx - int(0.3e-3 * sample_rate))
    post = min(len(ir), peak_idx + max(int(window_ms * 1e-3 * sample_rate), 32))
    out[pre:post] = ir[pre:post]

    taper = min(int(0.4e-3 * sample_rate), (post - pre) // 5)
    if taper > 2:
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper)))
        out[post - taper : post] *= ramp[::-1]
    return out
