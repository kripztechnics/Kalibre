"""
Analyse de polarité entre deux mesures sweep (IR alignées + bande passante).

Utilise les délais mesurés, un affinage par corrélation, et compare
explicitement B vs −B pour détecter une inversion DSP (bouton Inverse).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PolarityResult:
    """Résultat comparaison polarité entre mesure référence et mesure test."""

    reference_name: str
    test_name: str
    f_min: float
    f_max: float
    delay_delta_ms: float
    ir_correlation: float
    ir_correlation_opposite: float
    reinforcement_db: float
    median_phase_deg: float
    score: float
    confidence: float
    inverted: bool
    verdict: str
    detail: str
    time_ms: NDArray[np.float64]
    ir_reference: NDArray[np.float64]
    ir_test: NDArray[np.float64]
    freqs: NDArray[np.float64]
    mag_sum_db: NDArray[np.float64]
    mag_diff_db: NDArray[np.float64]
    phase_diff_deg: NDArray[np.float64]


def _measurements_with_ir(
    ir_a: NDArray[np.float64] | None,
    ir_b: NDArray[np.float64] | None,
) -> bool:
    return ir_a is not None and ir_b is not None and len(ir_a) > 64 and len(ir_b) > 64


def _shift_samples(ir: NDArray[np.float64], shift: int) -> NDArray[np.float64]:
    """Décale l'IR de `shift` échantillons (positif = retard)."""
    n = len(ir)
    out = np.zeros(n, dtype=np.float64)
    if shift == 0:
        return ir.astype(np.float64, copy=True)
    if shift > 0:
        if shift < n:
            out[shift:] = ir[: n - shift]
    else:
        s = -shift
        if s < n:
            out[: n - s] = ir[s:]
    return out


def _bandpass_ir(
    ir: NDArray[np.float64],
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> NDArray[np.float64]:
    """Filtre passe-bande ideal (FFT) pour isoler la bande d'analyse."""
    n = len(ir)
    window = np.hanning(n)
    nfft = 1 << max(10, n - 1).bit_length()
    spec = np.fft.rfft(ir * window, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    mask = (freqs >= f_min) & (freqs <= f_max)
    filtered = np.zeros_like(spec)
    filtered[mask] = spec[mask]
    out = np.fft.irfft(filtered, n=nfft)
    return out[:n].astype(np.float64)


def _fine_lag_samples(
    ref: NDArray[np.float64],
    test: NDArray[np.float64],
    max_lag: int,
) -> tuple[int, float]:
    """Lag entier maximisant la corrélation signée."""
    best_lag = 0
    best_val = 0.0
    for lag in range(-max_lag, max_lag + 1):
        shifted = _shift_samples(test, lag)
        denom = float(np.linalg.norm(ref) * np.linalg.norm(shifted))
        if denom < 1e-12:
            continue
        val = float(np.dot(ref, shifted) / denom)
        if abs(val) > abs(best_val):
            best_val = val
            best_lag = lag
    return best_lag, best_val


def align_impulse_responses(
    ir_a: NDArray[np.float64],
    ir_b: NDArray[np.float64],
    sample_rate: int,
    *,
    delay_a_ms: float = 0.0,
    delay_b_ms: float = 0.0,
    f_min: float = 40.0,
    f_max: float = 400.0,
    window_ms: float = 5.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """
    Aligne B sur A :
      1. correction des délais mesurés (tableau Kalibre)
      2. affinage par corrélation sur la bande passante
    """
    peak_a = int(np.argmax(np.abs(ir_a)))

    delay_shift = int(round((delay_b_ms - delay_a_ms) * 1e-3 * sample_rate))
    ir_b_time = _shift_samples(ir_b, -delay_shift)

    max_lag = max(1, int(0.5e-3 * sample_rate))
    win_n = max(64, int(window_ms * 1e-3 * sample_rate))
    start = max(0, min(peak_a - win_n // 8, len(ir_a) - win_n))
    end = min(len(ir_a), start + win_n)
    seg_a = ir_a[start:end]
    seg_b = ir_b_time[start:end]

    seg_a_bp = _bandpass_ir(seg_a, sample_rate, f_min, f_max)
    seg_b_bp = _bandpass_ir(seg_b, sample_rate, f_min, f_max)

    lag_same, corr_same_lag = _fine_lag_samples(seg_a_bp, seg_b_bp, max_lag)
    lag_inv, corr_inv_lag = _fine_lag_samples(seg_a_bp, -seg_b_bp, max_lag)

    if abs(corr_inv_lag) > abs(corr_same_lag):
        fine_lag = lag_inv
    else:
        fine_lag = lag_same

    ir_b_aligned = _shift_samples(ir_b_time, fine_lag)

    t_a = (np.arange(len(ir_a)) - peak_a) / sample_rate * 1000.0
    t_b = (np.arange(len(ir_b_aligned)) - peak_a) / sample_rate * 1000.0

    n = max(64, int(window_ms * 1e-3 * sample_rate))
    t_common = np.linspace(-0.5, window_ms - 0.5, n, dtype=np.float64)

    a = np.interp(t_common, t_a, ir_a, left=0.0, right=0.0)
    b = np.interp(t_common, t_b, ir_b_aligned, left=0.0, right=0.0)

    delay_delta = delay_b_ms - delay_a_ms + fine_lag * 1000.0 / sample_rate
    return t_common, a.astype(np.float64), b.astype(np.float64), float(delay_delta)


def analyze_polarity_pair(
    reference_name: str,
    test_name: str,
    ir_ref: NDArray[np.float64],
    ir_test: NDArray[np.float64],
    sample_rate: int,
    *,
    delay_ref_ms: float = 0.0,
    delay_test_ms: float = 0.0,
    f_min: float = 40.0,
    f_max: float = 400.0,
    window_ms: float = 5.0,
) -> PolarityResult | None:
    if not _measurements_with_ir(ir_ref, ir_test) or sample_rate <= 0:
        return None

    f_lo = max(1.0, float(f_min))
    f_hi = max(f_lo + 1.0, float(f_max))

    t_ms, a, b, delay_delta = align_impulse_responses(
        ir_ref,
        ir_test,
        sample_rate,
        delay_a_ms=delay_ref_ms,
        delay_b_ms=delay_test_ms,
        f_min=f_lo,
        f_max=f_hi,
        window_ms=window_ms,
    )

    a_bp = _bandpass_ir(a, sample_rate, f_lo, f_hi)
    b_bp = _bandpass_ir(b, sample_rate, f_lo, f_hi)

    peak_a = float(np.max(np.abs(a_bp))) or 1.0
    peak_b = float(np.max(np.abs(b_bp))) or 1.0
    a_n = a_bp / peak_a
    b_n = b_bp / peak_b

    if np.std(a_n) < 1e-9 or np.std(b_n) < 1e-9:
        return None

    corr_same = float(np.corrcoef(a_n, b_n)[0, 1])
    corr_inv = float(np.corrcoef(a_n, -b_n)[0, 1])

    inverted = corr_inv > corr_same
    ir_corr = corr_inv if inverted else corr_same
    ir_corr_opp = corr_same if inverted else corr_inv

    window = np.hanning(len(a))
    nfft = 1 << max(10, len(a) - 1).bit_length()
    Ha = np.fft.rfft(a * window, n=nfft)
    Hb_raw = np.fft.rfft(b * window, n=nfft)
    Hb = -Hb_raw if inverted else Hb_raw
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)

    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return None

    sum_spec = Ha + Hb
    diff_spec = Ha - Hb_raw

    e_sum = float(np.sum(np.abs(sum_spec[band]) ** 2))
    e_diff = float(np.sum(np.abs(diff_spec[band]) ** 2))
    reinforcement_db = float(10.0 * np.log10((e_sum + 1e-20) / (e_diff + 1e-20)))

    score_polarity = float(max(corr_same, corr_inv) - min(corr_same, corr_inv))
    if inverted:
        score_polarity *= -1.0
    score_energy = float((e_sum - e_diff) / (e_sum + e_diff + 1e-20))

    phase_diff = np.angle(Ha * np.conj(Hb))
    band_phases = phase_diff[band]
    median_phase = float(
        np.degrees(
            np.arctan2(
                np.median(np.sin(band_phases)),
                np.median(np.cos(band_phases)),
            )
        )
    )

    combined = 0.50 * score_polarity + 0.35 * score_energy + 0.15 * float(
        np.cos(np.radians(median_phase))
    )
    confidence = float(
        np.clip(max(abs(corr_same), abs(corr_inv)) * 0.55 + abs(combined) * 0.45, 0.0, 1.0)
    )

    b_plot = (-b_n if inverted else b_n).astype(np.float64)

    if inverted:
        if confidence >= 0.45:
            verdict = "Inversion de polarité détectée"
            detail = (
                f"Sur {f_lo:.0f}–{f_hi:.0f} Hz : −B s'aligne mieux que +B "
                f"(corr +B {corr_same:+.2f}, −B {corr_inv:+.2f}). "
                f"Δ délai résiduel {delay_delta:+.2f} ms."
            )
        else:
            verdict = "Inversion probable (faible confiance)"
            detail = (
                f"Tendance inversion (−B {corr_inv:+.2f} vs +B {corr_same:+.2f}) — "
                f"confiance {confidence:.0%}."
            )
    else:
        if confidence >= 0.45:
            verdict = "Polarité cohérente"
            detail = (
                f"Sur {f_lo:.0f}–{f_hi:.0f} Hz : +B aligné (corr {corr_same:+.2f}). "
                f"A+B domine ({reinforcement_db:+.1f} dB). Δ délai {delay_delta:+.2f} ms."
            )
        else:
            verdict = "Résultat incertain"
            detail = (
                f"+B corr {corr_same:+.2f}, −B corr {corr_inv:+.2f} — "
                f"confiance {confidence:.0%}. Une seule sortie active par mesure ? "
                f"Délais {delay_ref_ms:.2f} vs {delay_test_ms:.2f} ms."
            )

    mag_sum = 20.0 * np.log10(np.abs(Ha + Hb_raw) + 1e-12)
    mag_diff = 20.0 * np.log10(np.abs(Ha - Hb_raw) + 1e-12)
    mag_sum_db = mag_sum - float(np.max(mag_sum[band]))
    mag_diff_db = mag_diff - float(np.max(mag_diff[band]))
    phase_diff_deg = np.degrees(np.angle(Ha * np.conj(Hb_raw))).astype(np.float64)

    return PolarityResult(
        reference_name=reference_name,
        test_name=test_name,
        f_min=f_lo,
        f_max=f_hi,
        delay_delta_ms=delay_delta,
        ir_correlation=ir_corr,
        ir_correlation_opposite=ir_corr_opp,
        reinforcement_db=reinforcement_db,
        median_phase_deg=median_phase,
        score=combined,
        confidence=confidence,
        inverted=inverted,
        verdict=verdict,
        detail=detail,
        time_ms=t_ms,
        ir_reference=a_n,
        ir_test=b_plot,
        freqs=freqs.astype(np.float64),
        mag_sum_db=mag_sum_db.astype(np.float64),
        mag_diff_db=mag_diff_db.astype(np.float64),
        phase_diff_deg=phase_diff_deg,
    )


def analyze_all_vs_reference(
    measurements: list,
    reference_index: int,
    sample_rate: int,
    *,
    f_min: float = 40.0,
    f_max: float = 400.0,
) -> list[tuple[str, PolarityResult | None, str]]:
    """Compare chaque mesure (sauf la réf.) à la mesure de référence."""
    if reference_index < 0 or reference_index >= len(measurements):
        return []

    ref = measurements[reference_index]
    if ref.ir is None:
        return []

    rows: list[tuple[str, PolarityResult | None, str]] = []
    for i, m in enumerate(measurements):
        if i == reference_index:
            continue
        if m.ir is None:
            rows.append((m.name, None, "Pas d'IR sweep"))
            continue
        result = analyze_polarity_pair(
            ref.name,
            m.name,
            ref.ir,
            m.ir,
            sample_rate,
            delay_ref_ms=ref.absolute_delay_ms,
            delay_test_ms=m.absolute_delay_ms,
            f_min=f_min,
            f_max=f_max,
        )
        if result is None:
            rows.append((m.name, None, "Analyse impossible"))
        elif result.inverted and result.confidence >= 0.40:
            rows.append((m.name, result, "Inversé"))
        elif not result.inverted and result.confidence >= 0.40:
            rows.append((m.name, result, "OK"))
        else:
            rows.append((m.name, result, "Incertain"))
    return rows
