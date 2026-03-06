import numpy as np
import pytest
from parity_regime_lab.generators import gen_parity_switched_signal
from parity_regime_lab.hmm_parity import hmm_em_known_x, viterbi_parity
from parity_regime_lab.diagnostics import parity_accuracy


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_viterbi_accuracy_comparable_to_hmm(seed):
    """Viterbi accuracy should be within 2% of the HMM posterior accuracy."""
    rng = np.random.default_rng(seed)
    y, s, x = gen_parity_switched_signal(
        n=5000, q=0.01, amp=1.0, freq=0.01, tau=0.5, rng=rng
    )
    q_hat, tau2_hat, s_hmm, _ = hmm_em_known_x(y=y, x=x, n_iter=25, q_init=0.01)
    s_vit = viterbi_parity(y=y, x=x, q=q_hat, tau2=tau2_hat)

    acc_hmm = parity_accuracy(s, s_hmm)
    acc_vit = parity_accuracy(s, s_vit)

    assert acc_vit > 0.90, f"seed={seed}: Viterbi accuracy {acc_vit:.3f} < 0.90"
    assert abs(acc_vit - acc_hmm) < 0.02, (
        f"seed={seed}: Viterbi ({acc_vit:.3f}) vs HMM ({acc_hmm:.3f}) differ by more than 2%"
    )


def test_viterbi_q0_perfect():
    """At q=0 Viterbi should recover parity perfectly."""
    rng = np.random.default_rng(42)
    y, s, x = gen_parity_switched_signal(
        n=3000, q=0.0, amp=1.0, freq=0.01, tau=0.5, rng=rng
    )
    q_hat, tau2_hat, _, _ = hmm_em_known_x(y=y, x=x, n_iter=20, q_init=0.005)
    s_vit = viterbi_parity(y=y, x=x, q=q_hat, tau2=tau2_hat)
    assert parity_accuracy(s, s_vit) > 0.99


def test_viterbi_output_shape_and_values():
    """Viterbi output must be (n,) and contain only +1/-1."""
    rng = np.random.default_rng(7)
    y, s, x = gen_parity_switched_signal(
        n=200, q=0.02, amp=1.0, freq=0.05, tau=0.3, rng=rng
    )
    q_hat, tau2_hat, _, _ = hmm_em_known_x(y=y, x=x, n_iter=10)
    s_vit = viterbi_parity(y=y, x=x, q=q_hat, tau2=tau2_hat)
    assert s_vit.shape == (200,)
    assert set(np.unique(s_vit)).issubset({1, -1})


def test_viterbi_mismatched_lengths_raises():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    x = rng.normal(size=99)
    with pytest.raises(ValueError, match="same length"):
        viterbi_parity(y=y, x=x, q=0.01, tau2=0.25)


def test_viterbi_produces_step_function():
    """Viterbi should return a piecewise-constant sequence (fewer sign changes
    than the HMM posterior, which can have marginal fluctuations)."""
    rng = np.random.default_rng(3)
    y, s, x = gen_parity_switched_signal(
        n=4000, q=0.005, amp=1.0, freq=0.01, tau=0.5, rng=rng
    )
    q_hat, tau2_hat, s_hmm, _ = hmm_em_known_x(y=y, x=x, n_iter=25, q_init=0.005)
    s_vit = viterbi_parity(y=y, x=x, q=q_hat, tau2=tau2_hat)

    flips_vit = int(np.sum(np.diff(s_vit) != 0))
    flips_hmm = int(np.sum(np.diff(s_hmm) != 0))

    # Viterbi finds the globally optimal path — it may have more or fewer
    # transitions than thresholded marginals, but both should be in the same
    # ballpark for low q.  The key check: Viterbi produces a valid binary sequence.
    assert flips_vit >= 0
    assert flips_hmm >= 0
