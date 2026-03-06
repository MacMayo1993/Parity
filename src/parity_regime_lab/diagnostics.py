from __future__ import annotations
import numpy as np


def ar1_fit(y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Quick AR(1) OLS fit: y_t = a y_{t-1} + e_t
    Returns (a_hat, residuals).
    """
    y0 = y[:-1]
    y1 = y[1:]
    denom = float(y0 @ y0)
    if denom == 0.0:
        return 0.0, y1.copy()
    a = float((y0 @ y1) / denom)
    e = y1 - a * y0
    return a, e


def fft_peak_over_total(y: np.ndarray) -> float:
    """
    Simple spectral concentration metric:
    peak(|FFT|) / sum(|FFT|).
    Lower => more leakage / smearing.
    """
    Y = np.fft.rfft(y)
    mag = np.abs(Y)
    total = float(mag.sum())
    if total == 0.0:
        return 0.0
    return float(mag.max() / total)


def parity_accuracy(s_true: np.ndarray, s_hat: np.ndarray) -> float:
    """
    Parity is identifiable only up to a global sign.
    Returns max(accuracy(s_hat), accuracy(-s_hat)).
    """
    same = np.mean(s_true == s_hat)
    flip = np.mean(s_true == -s_hat)
    return float(max(same, flip))
