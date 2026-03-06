import numpy as np
import pytest
from parity_regime_lab.mi_parity import estimate_mi_parity_bits, mi_small_snr_approx_bits


def test_mi_small_snr_matches_curvature():
    """Monte Carlo MI should be within 8% of the a^2/(2 tau^2 ln 2) approximation."""
    rng = np.random.default_rng(1)
    tau = 1.0
    a = 0.1
    mi_est = estimate_mi_parity_bits(a=a, tau=tau, rng=rng, n=500_000)
    mi_apx = mi_small_snr_approx_bits(a=a, tau=tau)
    rel_err = abs(mi_est - mi_apx) / mi_apx
    assert rel_err < 0.08, (
        f"Relative error {rel_err:.3f} exceeds 8%; est={mi_est:.5f}, approx={mi_apx:.5f}"
    )


@pytest.mark.parametrize("a", [0.05, 0.08, 0.12])
def test_mi_small_snr_multiple_amplitudes(a):
    """Approximation should hold within 10% across several small-SNR amplitudes."""
    rng = np.random.default_rng(99)
    tau = 1.0
    mi_est = estimate_mi_parity_bits(a=a, tau=tau, rng=rng, n=400_000)
    mi_apx = mi_small_snr_approx_bits(a=a, tau=tau)
    rel_err = abs(mi_est - mi_apx) / mi_apx
    assert rel_err < 0.10, (
        f"a={a}: relative error {rel_err:.3f} exceeds 10%"
    )


def test_mi_approx_zero_amplitude():
    """The approximation equals zero when amplitude is zero."""
    assert mi_small_snr_approx_bits(a=0.0, tau=1.0) == 0.0


def test_mi_approx_scales_quadratically():
    """I(s;y) ≈ a^2 / C, so doubling a should quadruple MI."""
    a1, a2 = 0.1, 0.2
    tau = 1.0
    mi1 = mi_small_snr_approx_bits(a=a1, tau=tau)
    mi2 = mi_small_snr_approx_bits(a=a2, tau=tau)
    ratio = mi2 / mi1
    assert abs(ratio - 4.0) < 1e-10, f"Expected ratio 4.0, got {ratio}"
