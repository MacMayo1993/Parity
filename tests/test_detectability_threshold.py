"""
Tests for the analytic parity detectability threshold.

From the MI analysis:
  I(s_t ; y_t) ≈ a² / (2 τ² ln 2)  bits per step

For a flip-rate of q (one flip every 1/q steps on average), parity is trackable
only when the HMM can accumulate ~1 bit of evidence within a single inter-flip
interval:

  n_critical = 1/q  steps,   detecting one flip requires I * n_critical ≥ 1 bit
  => a_critical = sqrt(2 τ² ln 2 * q)

Below a_critical: MI accumulates too slowly relative to the flip rate → accuracy
near chance.  Above a_critical: each flip is detectable → high accuracy.

We use q=0.01 (flip every ~100 steps), τ=1.0, n=5000.
  a_critical ≈ sqrt(2 * 1.0 * 0.693 * 0.01) ≈ 0.118
Empirically:
  a=0.03 → acc ≈ 0.53 (near chance)
  a=0.50 → acc ≈ 0.92 (clearly above threshold)
"""
import numpy as np
import pytest
from parity_regime_lab.generators import gen_parity_switched_signal
from parity_regime_lab.hmm_parity import hmm_em_known_x
from parity_regime_lab.mi_parity import mi_small_snr_approx_bits
from parity_regime_lab.diagnostics import parity_accuracy


Q = 0.01
TAU = 1.0
N = 5000
N_TRIALS = 8


def _a_critical() -> float:
    """Amplitude at which MI per flip interval equals 1 bit."""
    # I(s;y) * (1/q) = 1  =>  a^2/(2τ²ln2) * (1/q) = 1
    return float(np.sqrt(2.0 * TAU ** 2 * np.log(2) * Q))


def _avg_accuracy(amp: float, seed: int = 0) -> float:
    accs = []
    for si in range(N_TRIALS):
        rng = np.random.default_rng([seed, si])
        y, s, x = gen_parity_switched_signal(
            n=N, q=Q, amp=amp, freq=0.01, tau=TAU, rng=rng
        )
        _, _, s_hat, _ = hmm_em_known_x(y=y, x=x, n_iter=20, q_init=max(Q, 0.005))
        accs.append(parity_accuracy(s, s_hat))
    return float(np.mean(accs))


def test_a_critical_in_expected_range():
    """a_critical for q=0.01, tau=1.0 should be near 0.118."""
    ac = _a_critical()
    assert 0.10 < ac < 0.14, f"a_critical={ac:.4f} outside expected [0.10, 0.14]"


def test_below_threshold_accuracy_near_chance():
    """At a << a_critical, the HMM cannot track flips: mean accuracy < 0.62."""
    # a=0.03 is 4x below a_critical ≈ 0.118
    acc = _avg_accuracy(amp=0.03)
    assert acc < 0.62, (
        f"Expected near-chance accuracy below threshold (a=0.03), got {acc:.3f}"
    )


def test_above_threshold_accuracy_high():
    """At a >> a_critical, the HMM reliably tracks flips: mean accuracy > 0.88."""
    # a=0.50 is 4x above a_critical ≈ 0.118; MI per flip interval ≈ 18 bits
    acc = _avg_accuracy(amp=0.50)
    assert acc > 0.88, (
        f"Expected high accuracy above threshold (a=0.50), got {acc:.3f}"
    )


def test_accuracy_monotone_with_amplitude():
    """Accuracy should increase monotonically as amplitude grows through threshold."""
    amps = [0.03, 0.10, 0.25, 0.50]
    accs = [_avg_accuracy(amp=a, seed=2) for a in amps]
    for i in range(len(accs) - 1):
        assert accs[i] <= accs[i + 1] + 0.04, (
            f"Accuracy not monotone: amp={amps[i]} -> {accs[i]:.3f}, "
            f"amp={amps[i+1]} -> {accs[i+1]:.3f}"
        )


def test_mi_at_threshold_equals_one_bit_per_flip_interval():
    """By construction, MI(a_critical) * (1/q) should equal exactly 1 bit."""
    ac = _a_critical()
    mi_per_step = mi_small_snr_approx_bits(a=ac, tau=TAU)
    mi_per_interval = mi_per_step / Q
    assert abs(mi_per_interval - 1.0) < 1e-10, (
        f"MI per flip interval = {mi_per_interval:.6f}, expected 1.0"
    )
