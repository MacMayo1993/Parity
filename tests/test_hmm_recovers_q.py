import numpy as np
import pytest
from parity_regime_lab.generators import gen_parity_switched_signal
from parity_regime_lab.hmm_parity import hmm_em_known_x
from parity_regime_lab.diagnostics import parity_accuracy


SEEDS = [0, 1, 2, 3, 4]


@pytest.mark.parametrize("seed", SEEDS)
def test_hmm_recovers_q(seed):
    """HMM should estimate q within 0.007 and reconstruct parity above 92%."""
    rng = np.random.default_rng(seed)
    n = 10_000
    q_true = 0.02
    y, s, x = gen_parity_switched_signal(
        n=n, q=q_true, amp=1.0, freq=0.01, tau=0.5, rng=rng
    )
    q_hat, _, s_hat, _ = hmm_em_known_x(y=y, x=x, n_iter=30, q_init=0.01)
    err = abs(q_hat - q_true)
    acc = parity_accuracy(s, s_hat)
    assert err < 0.007, f"seed={seed}: |q_hat - q_true| = {err:.4f} (want < 0.007)"
    assert acc > 0.92, f"seed={seed}: parity accuracy = {acc:.3f} (want > 0.92)"


def test_hmm_q0_no_flips():
    """At q=0 the parity never flips; HMM should recover near-zero q."""
    rng = np.random.default_rng(7)
    y, s, x = gen_parity_switched_signal(
        n=8_000, q=0.0, amp=1.0, freq=0.01, tau=0.5, rng=rng
    )
    q_hat, _, s_hat, _ = hmm_em_known_x(y=y, x=x, n_iter=25, q_init=0.01)
    assert q_hat < 0.01, f"Expected near-zero q_hat, got {q_hat:.4f}"
    assert parity_accuracy(s, s_hat) > 0.99


def test_hmm_log_evidence_finite():
    """Log-evidence must be finite and negative."""
    rng = np.random.default_rng(42)
    y, s, x = gen_parity_switched_signal(
        n=5_000, q=0.01, amp=1.0, freq=0.01, tau=0.5, rng=rng
    )
    _, _, _, logZ = hmm_em_known_x(y=y, x=x)
    assert np.isfinite(logZ), f"logZ is not finite: {logZ}"
    assert logZ < 0, f"logZ should be negative, got {logZ}"


def test_hmm_mismatched_lengths_raises():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    x = rng.normal(size=99)
    with pytest.raises(ValueError, match="same length"):
        hmm_em_known_x(y=y, x=x)
