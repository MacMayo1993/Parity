from __future__ import annotations
import numpy as np


def gen_markov_parity(n: int, q: float, rng: np.random.Generator, s0: int = 1) -> np.ndarray:
    """
    Hidden parity/orientation process s_t in {+1,-1} with flip probability q.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1].")
    s = np.empty(n, dtype=int)
    s[0] = 1 if s0 >= 0 else -1
    flips = rng.random(n - 1) < q
    for t in range(1, n):
        s[t] = s[t - 1] * (-1 if flips[t - 1] else 1)
    return s


def gen_sinusoid(n: int, amp: float, freq: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n)
    return amp * np.sin(2 * np.pi * freq * t + phase)


def gen_parity_switched_signal(
    n: int,
    q: float,
    amp: float,
    freq: float,
    tau: float,
    rng: np.random.Generator,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    y_t = s_t x_t + noise, where x_t is a sinusoid and s_t flips with probability q.
    Returns (y, s, x).
    """
    s = gen_markov_parity(n=n, q=q, rng=rng)
    x = gen_sinusoid(n=n, amp=amp, freq=freq, phase=phase)
    y = s * x + rng.normal(0.0, tau, size=n)
    return y, s, x
