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

    TWO_PI = 2 * np.pi

    for t in range(1, n):
        theta_prev = theta[t - 1]
        theta[t] = theta_prev + omega + rng.normal(0.0, noise_theta)

        prev_mod = theta_prev % TWO_PI
        curr_mod = theta[t] % TWO_PI

        # Count how many times the trajectory crosses the seam in this step.
        # We integrate the crossing count exactly: the number of times a
        # linearly-interpolated path from theta_prev to theta[t] crosses
        # (seam + 2pi*k) for integer k.  Each crossing flips orientation once.
        #
        # Equivalently: shift so the seam is at 0, count full half-turns
        # traversed (i.e. crossings of 0 mod 2pi in the shifted frame).
        lo = theta_prev - seam
        hi = theta[t] - seam
        if lo > hi:
            lo, hi = hi, lo
        n_crossings = int(np.floor(hi / TWO_PI)) - int(np.floor(lo / TWO_PI))

        s[t] = s[t - 1] * ((-1) ** n_crossings)

    x = amp * np.sin(theta)
    y = s * x + rng.normal(0.0, tau, size=n)

    return y, s, x, theta
