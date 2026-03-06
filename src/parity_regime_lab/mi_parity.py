from __future__ import annotations
import numpy as np


def estimate_mi_parity_bits(a: float, tau: float, rng: np.random.Generator, n: int = 300_000) -> float:
    """
    Channel: s ∈ {±1} equiprobable, y = s*a + N(0, tau^2).
    Returns MI(s;y) in bits via Monte Carlo.
    """
    s = rng.choice([-1, 1], size=n)
    y = s * a + rng.normal(0.0, tau, size=n)

    tau2 = tau * tau
    c = -0.5 * np.log(2 * np.pi * tau2)

    logp_y_given_s = c - 0.5 * ((y - s * a) ** 2) / tau2
    logN1 = c - 0.5 * ((y - a) ** 2) / tau2
    logN2 = c - 0.5 * ((y + a) ** 2) / tau2

    m = np.maximum(logN1, logN2)
    logp_y = np.log(0.5 * np.exp(logN1 - m) + 0.5 * np.exp(logN2 - m)) + m

    mi_nats = float(np.mean(logp_y_given_s - logp_y))
    return mi_nats / float(np.log(2.0))


def mi_small_snr_approx_bits(a: float, tau: float) -> float:
    """
    Small-SNR approximation:
      I(s;y) ≈ a^2 / (2 tau^2 ln 2) bits
    """
    return float((a * a) / (2.0 * tau * tau * np.log(2.0)))
