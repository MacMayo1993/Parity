import numpy as np
from parity_regime_lab.mobius_generator import gen_mobius_seam_signal
from parity_regime_lab.hmm_parity import hmm_em_known_x
from parity_regime_lab.diagnostics import parity_accuracy


def test_mobius_seam_flips_occur():
    """Seam crossings should produce at least one parity flip for typical parameters."""
    rng = np.random.default_rng(42)
    y, s, x, theta = gen_mobius_seam_signal(
        n=5000, omega=0.05, amp=1.0, tau=0.3, rng=rng, seam=np.pi
    )
    flips = int(np.sum(np.diff(s) != 0))
    assert flips > 0, "Expected at least one seam crossing / parity flip."


def test_mobius_hmm_recovers_parity():
    """HMM should recover Möbius parity with reasonable accuracy at low noise."""
    rng = np.random.default_rng(99)
    y, s, x, theta = gen_mobius_seam_signal(
        n=6000, omega=0.03, amp=1.0, tau=0.4, rng=rng, seam=np.pi
    )
    _, _, s_hat, _ = hmm_em_known_x(y=y, x=x, n_iter=20, q_init=0.01)
    acc = parity_accuracy(s, s_hat)
    assert acc > 0.85, f"Expected parity accuracy > 0.85, got {acc:.3f}"
