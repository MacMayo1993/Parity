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
    theta = np.zeros(n)
    theta[0] = phase0

    s = np.ones(n, dtype=int)

    for t in range(1, n):
        theta_prev = theta[t - 1]
        theta[t] = theta_prev + omega + rng.normal(0.0, noise_theta)

        prev_mod = theta_prev % (2 * np.pi)
        curr_mod = theta[t] % (2 * np.pi)

        # seam crossing: phase wraps past the seam angle from below
        crossed = (prev_mod < seam) and (curr_mod >= seam)

        s[t] = s[t - 1] * (-1 if crossed else 1)

    x = amp * np.sin(theta)
    y = s * x + rng.normal(0.0, tau, size=n)

    return y, s, x, theta
