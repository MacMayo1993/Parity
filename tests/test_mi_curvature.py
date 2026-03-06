import numpy as np
from parity_regime_lab.mi_parity import estimate_mi_parity_bits, mi_small_snr_approx_bits


def test_mi_small_snr_matches_curvature():
    rng = np.random.default_rng(1)
    tau = 1.0
    a = 0.1
    mi_est = estimate_mi_parity_bits(a=a, tau=tau, rng=rng, n=200_000)
    mi_apx = mi_small_snr_approx_bits(a=a, tau=tau)
    # should be close in small-SNR regime
    assert abs(mi_est - mi_apx) / mi_apx < 0.15


def test_mi_small_snr_approx_formula():
    """The approximation a^2/(2 tau^2 ln 2) is exact at zero amplitude."""
    assert mi_small_snr_approx_bits(a=0.0, tau=1.0) == 0.0
