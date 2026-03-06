# Parity Regime Lab

This repo provides a reproducible experiment showing how a hidden **parity/orientation bit**
in a regime-switching system:

1. Makes **local** tools (FFT/AR-style fits) see "mess" (spectral leakage / residual inflation)
2. Is recoverable by an explicit **parity-aware** model (2-state HMM on sign)
3. Has a universal small-SNR information curvature:
   `I(s;y) ≈ a^2 / (2 tau^2 ln 2) bits`

A second generator replaces random flips with **topological seam crossings** (Möbius-strip
style), showing the same constant survives the transition from stochastic to geometric
non-orientability.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Run

### Standard parity experiment

```bash
python scripts/run_experiment.py --outdir out --n 12000 --tau 0.6 --freq 0.007 --amp 1.0 --seed 7
```

### Möbius seam experiment

```bash
python scripts/run_experiment.py --mobius --outdir out --n 12000 --tau 0.6 --amp 1.0 --omega 0.03 --seed 7
```

## Outputs

- `out/summary.csv` — metrics by flip rate q (standard experiment)
- `out/plot_ar_resid.png`
- `out/plot_fft_leakage.png`
- `out/plot_parity_accuracy.png`
- `out/plot_mi_curvature.png`
- `out/theta.png`, `out/true_parity.png`, `out/recovered_parity.png` (Möbius experiment)

## What "parity" means here

### Standard model (random flips)

```
s_t ∈ {+1, -1},  P(flip) = q
x_t = A sin(2π f t)
y_t = s_t x_t + N(0, tau^2)
```

Locally the system behaves like a stationary sinusoid + noise.
Globally, occasional sign flips introduce conflicting orientation.

### Möbius model (topological flips)

```
theta_{t+1} = theta_t + omega + epsilon_t
s_{t+1}     = s_t * (-1)^{1[theta crosses seam]}
y_t         = s_t A sin(theta_t) + N(0, tau^2)
```

Flips are tied to global phase geometry, not randomness. The orientation changes arise
from transport across a non-orientable seam — walking around the strip brings you back
flipped.

## What this formally demonstrates

1. **Local analysis** (AR/FFT) degrades smoothly as parity flips increase — but cannot
   identify the source as "parity".
2. A minimal **parity-aware HMM** recovers hidden orientation and estimates q.
3. The constant `1/(2 ln 2)` appears as a universal small-SNR curvature of parity
   information in bits:
   ```
   I(s;y) = Θ( a^2 / (2 tau^2 ln 2) )
   ```
4. The same curvature constant **survives** the replacement of random flips with
   deterministic topological seam crossings.

## Tests

```bash
pytest -q
```

## License

MIT
