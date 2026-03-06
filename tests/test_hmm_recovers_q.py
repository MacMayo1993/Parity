import numpy as np
from parity_regime_lab.generators import gen_parity_switched_signal
from parity_regime_lab.hmm_parity import hmm_em_known_x
from parity_regime_lab.diagnostics import parity_accuracy


def test_hmm_recovers_q_reasonably():
    rng = np.random.default_rng(0)
    n = 8000
    q_true = 0.02
    y, s, x = gen_parity_switched_signal(n=n, q=q_true, amp=1.0, freq=0.01, tau=0.5, rng=rng)
    q_hat, tau2_hat, s_hat, _ = hmm_em_known_x(y=y, x=x, n_iter=20, q_init=0.01)
    # loose bounds to avoid flakiness
    assert abs(q_hat - q_true) < 0.01
    assert parity_accuracy(s, s_hat) > 0.9
