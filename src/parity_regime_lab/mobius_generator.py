from __future__ import annotations
import numpy as np


def gen_mobius_seam_signal(
    n: int,
    omega: float,
    amp: float,
    tau: float,
    rng: np.random.Generator,
    seam: float = np.pi,
    phase0: float = 0.0,
    noise_theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a signal whose parity flips whenever the phase crosses a seam.

    Phase evolves continuously:
        theta_{t+1} = theta_t + omega + noise

    Orientation flips whenever the trajectory crosses the seam angle (mod 2pi).
    This is the Möbius-strip analogue: a full loop brings you back with flipped
    orientation.

    Returns (y, s, x, theta).
    """
    TWO_PI = 2 * np.pi

    # --- Phase trajectory ---
    # For the deterministic case (noise_theta=0) the phase is a simple
    # arithmetic progression — fully vectorized.  For stochastic phase
    # we still need a sequential loop because each step depends on the previous.
    if noise_theta == 0.0:
        theta = phase0 + np.arange(n, dtype=float) * omega
    else:
        theta = np.empty(n, dtype=float)
        theta[0] = phase0
        for t in range(1, n):
            theta[t] = theta[t - 1] + omega + rng.normal(0.0, noise_theta)

    # --- Vectorized seam-crossing count ---
    # For each step t->t+1, shift so the seam is at 0 and count how many times
    # the linear path crosses a multiple of 2π.  Taking min/max handles both
    # positive and negative omega without a conditional swap.
    lo = np.minimum(theta[:-1], theta[1:]) - seam   # (n-1,)
    hi = np.maximum(theta[:-1], theta[1:]) - seam   # (n-1,)
    crossings = (np.floor(hi / TWO_PI) - np.floor(lo / TWO_PI)).astype(int)

    # --- Vectorized parity accumulation ---
    # s[t] = s[0] × ∏_{i=1}^{t} (-1)^crossings[i]
    #       = (-1)^(cumulative crossing count up to t)
    cumcross = np.cumsum(crossings)
    s = np.ones(n, dtype=int)
    s[1:] = np.where(cumcross % 2 == 0, 1, -1)

    x = amp * np.sin(theta)
    y = s * x + rng.normal(0.0, tau, size=n)

    return y, s, x, theta
