import numpy as np
import pytest
from parity_regime_lab.diagnostics import ar1_fit, fft_peak_over_total, parity_accuracy


# --- ar1_fit ---

def test_ar1_fit_known_coefficient():
    """AR(1) OLS should recover an exact coefficient on a clean signal."""
    rng = np.random.default_rng(0)
    n = 5000
    a_true = 0.7
    y = np.zeros(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = a_true * y[t - 1] + rng.normal(0, 0.1)
    a_hat, e = ar1_fit(y)
    assert abs(a_hat - a_true) < 0.02, f"Expected a_hat ≈ {a_true}, got {a_hat:.4f}"
    assert len(e) == n - 1


def test_ar1_fit_zero_signal():
    """All-zero signal has zero-variance denominator; should return a=0 without error."""
    y = np.zeros(100)
    a_hat, e = ar1_fit(y)
    assert a_hat == 0.0
    assert np.allclose(e, 0.0)


def test_ar1_fit_constant_nonzero():
    """Constant non-zero signal gives a=1 (OLS optimal for y_t = 1*y_{t-1})."""
    y = np.ones(100) * 3.0
    a_hat, e = ar1_fit(y)
    assert abs(a_hat - 1.0) < 1e-10
    assert np.allclose(e, 0.0)


def test_ar1_fit_residuals_length():
    y = np.arange(50, dtype=float)
    a_hat, e = ar1_fit(y)
    assert len(e) == 49


# --- fft_peak_over_total ---

def test_fft_pure_sinusoid_high_concentration():
    """A pure sinusoid at an exact DFT bin should have a dominant spectral peak.

    Use freq = k/n so the sinusoid lands exactly on bin k with no spectral leakage.
    """
    n = 2048
    k = 100  # exact bin
    t = np.arange(n)
    y = np.sin(2 * np.pi * k / n * t)
    metric = fft_peak_over_total(y)
    assert metric > 0.5, f"Expected high concentration for pure sinusoid, got {metric:.3f}"


def test_fft_white_noise_low_concentration():
    """White noise should spread energy across all frequencies."""
    rng = np.random.default_rng(5)
    y = rng.normal(size=4096)
    metric = fft_peak_over_total(y)
    assert metric < 0.05, f"Expected low concentration for white noise, got {metric:.3f}"


def test_fft_zero_signal():
    """All-zero signal returns 0.0 without error."""
    y = np.zeros(256)
    assert fft_peak_over_total(y) == 0.0


# --- parity_accuracy ---

def test_parity_accuracy_perfect():
    s = np.array([1, -1, 1, 1, -1])
    assert parity_accuracy(s, s) == 1.0


def test_parity_accuracy_perfect_flipped():
    """Accuracy should be 1.0 even if the global sign is wrong (identifiable up to sign)."""
    s = np.array([1, -1, 1, 1, -1])
    assert parity_accuracy(s, -s) == 1.0


def test_parity_accuracy_half():
    """Random agreement should give ~0.5; the max(same, flipped) should still be 0.5."""
    s = np.array([1, -1, 1, -1])
    s_hat = np.array([1, 1, -1, -1])
    acc = parity_accuracy(s, s_hat)
    assert acc == 0.5


def test_parity_accuracy_symmetry():
    rng = np.random.default_rng(11)
    s = rng.choice([-1, 1], size=200)
    s_hat = rng.choice([-1, 1], size=200)
    assert parity_accuracy(s, s_hat) == parity_accuracy(s_hat, s)
