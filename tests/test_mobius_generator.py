import numpy as np
import pytest
from parity_regime_lab.mobius_generator import gen_mobius_seam_signal
from parity_regime_lab.hmm_parity import hmm_em_known_x
from parity_regime_lab.diagnostics import parity_accuracy


@pytest.mark.parametrize("seed", [42, 7, 13])
def test_mobius_seam_flips_occur(seed):
    """Seam crossings should produce parity flips for typical parameters."""
    rng = np.random.default_rng(seed)
    y, s, x, theta = gen_mobius_seam_signal(
        n=5000, omega=0.05, amp=1.0, tau=0.3, rng=rng, seam=np.pi
    )
    flips = int(np.sum(np.diff(s) != 0))
    assert flips > 0, f"seed={seed}: expected at least one parity flip."


def test_mobius_flip_count_matches_seam_crossings():
    """Number of parity flips should equal number of seam crossings (no noise, deterministic).

    For deterministic phase theta[t] = t*omega starting at 0, crossings of
    (seam + 2pi*k) equal:
        floor((theta_end - seam) / 2pi) - floor((theta_start - seam) / 2pi)
    """
    rng = np.random.default_rng(0)
    omega = 0.05
    n = 5000
    seam = np.pi
    phase0 = 0.0
    y, s, x, theta = gen_mobius_seam_signal(
        n=n, omega=omega, amp=1.0, tau=0.0, rng=rng, seam=seam, noise_theta=0.0
    )
    flips = int(np.sum(np.diff(s) != 0))

    theta_end = phase0 + (n - 1) * omega
    expected = int(np.floor((theta_end - seam) / (2 * np.pi))) - int(
        np.floor((phase0 - seam) / (2 * np.pi))
    )
    assert flips == expected, f"Expected {expected} flips, got {flips}"


@pytest.mark.parametrize("seed", [99, 55, 21])
def test_mobius_hmm_recovers_parity(seed):
    """HMM should recover Möbius parity above 90% accuracy at low noise."""
    rng = np.random.default_rng(seed)
    y, s, x, theta = gen_mobius_seam_signal(
        n=6000, omega=0.03, amp=1.0, tau=0.4, rng=rng, seam=np.pi
    )
    _, _, s_hat, _ = hmm_em_known_x(y=y, x=x, n_iter=25, q_init=0.01)
    acc = parity_accuracy(s, s_hat)
    assert acc > 0.90, f"seed={seed}: parity accuracy = {acc:.3f} (want > 0.90)"


def test_mobius_seam_at_zero():
    """Seam at 0 (≡ 2pi) should still trigger flips when phase completes full loops."""
    rng = np.random.default_rng(3)
    # omega > 2pi means we cross seam=0 each step; parity alternates every loop
    y, s, x, theta = gen_mobius_seam_signal(
        n=500, omega=0.5, amp=1.0, tau=0.1, rng=rng, seam=0.0, noise_theta=0.0
    )
    flips = int(np.sum(np.diff(s) != 0))
    assert flips > 0, "Seam at 0 produced no flips."
