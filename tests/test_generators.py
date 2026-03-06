import numpy as np
import pytest
from parity_regime_lab.generators import (
    gen_markov_parity,
    gen_sinusoid,
    gen_parity_switched_signal,
)


# --- gen_markov_parity ---

def test_markov_parity_values():
    """Output should only contain +1 and -1."""
    rng = np.random.default_rng(0)
    s = gen_markov_parity(n=1000, q=0.05, rng=rng)
    assert set(np.unique(s)).issubset({1, -1})


def test_markov_parity_q0_no_flips():
    """q=0 means never flip; all values equal s[0]."""
    rng = np.random.default_rng(0)
    s = gen_markov_parity(n=500, q=0.0, rng=rng)
    assert np.all(s == s[0])


def test_markov_parity_q1_always_flips():
    """q=1 means flip every step; alternating sequence."""
    rng = np.random.default_rng(0)
    s = gen_markov_parity(n=6, q=1.0, rng=rng)
    assert np.all(np.diff(s) != 0), f"Expected alternating, got {s}"


def test_markov_parity_flip_rate():
    """Empirical flip rate should be close to q for large n."""
    rng = np.random.default_rng(42)
    q_true = 0.1
    s = gen_markov_parity(n=50_000, q=q_true, rng=rng)
    empirical_q = float(np.mean(np.diff(s) != 0))
    assert abs(empirical_q - q_true) < 0.005, (
        f"Empirical flip rate {empirical_q:.4f} too far from q={q_true}"
    )


def test_markov_parity_invalid_q():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        gen_markov_parity(n=100, q=1.5, rng=rng)
    with pytest.raises(ValueError):
        gen_markov_parity(n=100, q=-0.1, rng=rng)


def test_markov_parity_output_shape():
    rng = np.random.default_rng(0)
    s = gen_markov_parity(n=123, q=0.05, rng=rng)
    assert s.shape == (123,)


# --- gen_sinusoid ---

def test_sinusoid_shape():
    x = gen_sinusoid(n=200, amp=2.0, freq=0.05)
    assert x.shape == (200,)


def test_sinusoid_amplitude():
    """Peak value should equal amp for a pure sinusoid."""
    x = gen_sinusoid(n=10_000, amp=3.0, freq=0.01)
    assert abs(x.max() - 3.0) < 0.001


def test_sinusoid_zero_amplitude():
    x = gen_sinusoid(n=100, amp=0.0, freq=0.1)
    assert np.allclose(x, 0.0)


# --- gen_parity_switched_signal ---

def test_parity_switched_signal_shapes():
    rng = np.random.default_rng(0)
    n = 300
    y, s, x = gen_parity_switched_signal(n=n, q=0.01, amp=1.0, freq=0.05, tau=0.5, rng=rng)
    assert y.shape == (n,)
    assert s.shape == (n,)
    assert x.shape == (n,)


def test_parity_switched_signal_parity_values():
    rng = np.random.default_rng(1)
    _, s, _ = gen_parity_switched_signal(n=500, q=0.05, amp=1.0, freq=0.05, tau=0.3, rng=rng)
    assert set(np.unique(s)).issubset({1, -1})


def test_parity_switched_signal_reproducible():
    """Same seed should give identical output."""
    kwargs = dict(n=200, q=0.02, amp=1.0, freq=0.05, tau=0.3)
    y1, s1, x1 = gen_parity_switched_signal(**kwargs, rng=np.random.default_rng(7))
    y2, s2, x2 = gen_parity_switched_signal(**kwargs, rng=np.random.default_rng(7))
    assert np.array_equal(y1, y2)
    assert np.array_equal(s1, s2)
