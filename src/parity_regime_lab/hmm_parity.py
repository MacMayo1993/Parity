from __future__ import annotations
import numpy as np


def _logsum(lv: np.ndarray) -> float:
    """Numerically stable log-sum-exp over a 1-D array."""
    m = float(lv.max())
    return m + float(np.log(np.exp(lv - m).sum()))


def hmm_em_known_x(
    y: np.ndarray,
    x: np.ndarray,
    n_iter: int = 25,
    q_init: float = 0.01,
    tau2_init: float | None = None,
    tol: float = 1e-6,
) -> tuple[float, float, np.ndarray, float]:
    """
    EM for parity HMM with known x_t:
      s_t in {+1,-1}
      P(flip)=q, symmetric
      y_t ~ N(s_t * x_t, tau^2)

    Terminates early when the change in log-evidence falls below `tol`.

    Returns (q_hat, tau2_hat, s_hat, log_evidence).
    """
    n = len(y)
    if len(x) != n:
        raise ValueError("x and y must have same length.")
    q = float(np.clip(q_init, 1e-12, 1 - 1e-12))

    if tau2_init is None:
        # Use var(y) as an unbiased upper-bound initializer.
        # var(y-x) is biased at high q because it assumes s_t=+1 everywhere,
        # inflating the estimate by ~amp^2 when parity is mixed.
        # var(y) = amp^2/2 + tau^2 (for sinusoidal x, any q) — a mild overestimate
        # that EM corrects within 2-3 iterations.
        tau2 = float(np.var(y))
        if not np.isfinite(tau2) or tau2 <= 1e-12:
            tau2 = 1e-6
    else:
        tau2 = float(tau2_init)

    def log_emissions(tau2_: float) -> np.ndarray:
        c = -0.5 * np.log(2 * np.pi * tau2_)
        lp_plus = c - 0.5 * ((y - x) ** 2) / tau2_
        lp_minus = c - 0.5 * ((y + x) ** 2) / tau2_
        return np.vstack([lp_plus, lp_minus])  # shape (2, n)

    logZ = -np.inf

    for _ in range(n_iter):
        le = log_emissions(tau2)

        logA = np.log(np.array([[1 - q, q], [q, 1 - q]], dtype=float))

        # --- Vectorized forward pass ---
        logalpha = np.empty((2, n), dtype=float)
        logalpha[:, 0] = np.log(0.5) + le[:, 0]
        for t in range(1, n):
            logalpha[0, t] = le[0, t] + np.logaddexp(
                logalpha[0, t - 1] + logA[0, 0],
                logalpha[1, t - 1] + logA[1, 0],
            )
            logalpha[1, t] = le[1, t] + np.logaddexp(
                logalpha[0, t - 1] + logA[0, 1],
                logalpha[1, t - 1] + logA[1, 1],
            )

        # --- Vectorized backward pass ---
        logbeta = np.empty((2, n), dtype=float)
        logbeta[:, n - 1] = 0.0
        for t in range(n - 2, -1, -1):
            logbeta[0, t] = np.logaddexp(
                logA[0, 0] + le[0, t + 1] + logbeta[0, t + 1],
                logA[0, 1] + le[1, t + 1] + logbeta[1, t + 1],
            )
            logbeta[1, t] = np.logaddexp(
                logA[1, 0] + le[0, t + 1] + logbeta[0, t + 1],
                logA[1, 1] + le[1, t + 1] + logbeta[1, t + 1],
            )

        logZ_new = float(np.logaddexp(logalpha[0, n - 1], logalpha[1, n - 1]))
        loggamma = logalpha + logbeta - logZ_new
        gamma = np.exp(loggamma)  # shape (2, n)

        # --- Vectorized xi: work in log-space, reduce with logsumexp ---
        la = logalpha[:, :-1]  # (2, n-1)
        lb = logbeta[:, 1:]    # (2, n-1)
        le1 = le[:, 1:]        # (2, n-1)

        l00 = la[0] + logA[0, 0] + le1[0] + lb[0] - logZ_new
        l01 = la[0] + logA[0, 1] + le1[1] + lb[1] - logZ_new
        l10 = la[1] + logA[1, 0] + le1[0] + lb[0] - logZ_new
        l11 = la[1] + logA[1, 1] + le1[1] + lb[1] - logZ_new

        log_xi01 = _logsum(l01)
        log_xi10 = _logsum(l10)

        all_log = np.concatenate([l00, l01, l10, l11])
        log_denom = _logsum(all_log)

        q = float(np.exp(np.logaddexp(log_xi01, log_xi10) - log_denom))
        q = float(np.clip(q, 1e-12, 1 - 1e-12))

        # tau2 update using posterior mean of s
        s_mean = gamma[0] - gamma[1]  # E[s_t | y] = P(s=+1) - P(s=-1)
        resid = y - s_mean * x
        tau2 = float(max(np.mean(resid ** 2), 1e-12))

        # convergence check
        if abs(logZ_new - logZ) < tol:
            logZ = logZ_new
            break
        logZ = logZ_new

    s_hat = np.where(gamma[0] >= 0.5, 1, -1).astype(int)
    return float(q), float(tau2), s_hat, float(logZ)


def viterbi_parity(
    y: np.ndarray,
    x: np.ndarray,
    q: float,
    tau2: float,
) -> np.ndarray:
    """
    Viterbi (max-product in log-space) for the 2-state parity HMM.

    Unlike hmm_em_known_x (which returns soft posterior marginals), Viterbi
    returns the single most-probable *sequence* of parity states — a clean
    step-function with exact transition times and no marginal smearing.

    Parameters
    ----------
    y : observed signal (n,)
    x : known carrier signal (n,)
    q : flip probability (use the q_hat from hmm_em_known_x)
    tau2 : noise variance (use the tau2_hat from hmm_em_known_x)

    Returns
    -------
    s_hat : (n,) array of +1 / -1
    """
    n = len(y)
    if len(x) != n:
        raise ValueError("x and y must have same length.")

    logA = np.log(np.array([[1 - q, q], [q, 1 - q]], dtype=float))
    c = -0.5 * np.log(2 * np.pi * tau2)
    le = np.vstack([
        c - 0.5 * ((y - x) ** 2) / tau2,   # row 0: s = +1
        c - 0.5 * ((y + x) ** 2) / tau2,   # row 1: s = -1
    ])  # (2, n)

    delta = np.empty((2, n), dtype=float)
    psi = np.zeros((2, n), dtype=np.int8)

    delta[:, 0] = np.log(0.5) + le[:, 0]

    for t in range(1, n):
        for j in range(2):
            scores = delta[:, t - 1] + logA[:, j]
            best = int(np.argmax(scores))
            psi[j, t] = best
            delta[j, t] = scores[best] + le[j, t]

    # Backtrack
    path = np.empty(n, dtype=np.int8)
    path[n - 1] = int(np.argmax(delta[:, n - 1]))
    for t in range(n - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]

    # State 0 → +1, state 1 → -1
    return np.where(path == 0, 1, -1).astype(int)
