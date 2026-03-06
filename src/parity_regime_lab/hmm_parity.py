from __future__ import annotations
import numpy as np


def hmm_em_known_x(
    y: np.ndarray,
    x: np.ndarray,
    n_iter: int = 25,
    q_init: float = 0.01,
    tau2_init: float | None = None,
) -> tuple[float, float, np.ndarray, float]:
    """
    EM for parity HMM with known x_t:
      s_t in {+1,-1}
      P(flip)=q, symmetric
      y_t ~ N(s_t * x_t, tau^2)

    Returns (q_hat, tau2_hat, s_hat, log_evidence).
    """
    n = len(y)
    if len(x) != n:
        raise ValueError("x and y must have same length.")
    q = float(q_init)
    q = min(max(q, 1e-12), 1 - 1e-12)

    if tau2_init is None:
        tau2 = float(np.var(y - x))
        if not np.isfinite(tau2) or tau2 <= 1e-12:
            tau2 = float(np.var(y)) + 1e-6
    else:
        tau2 = float(tau2_init)

    def log_emissions(tau2_: float) -> np.ndarray:
        c = -0.5 * np.log(2 * np.pi * tau2_)
        lp_plus = c - 0.5 * ((y - x) ** 2) / tau2_
        lp_minus = c - 0.5 * ((y + x) ** 2) / tau2_
        return np.vstack([lp_plus, lp_minus])  # shape (2, n)

    for _ in range(n_iter):
        le = log_emissions(tau2)

        A = np.array([[1 - q, q], [q, 1 - q]], dtype=float)
        logA = np.log(A)

        # forward-backward in log space
        logpi = np.log(np.array([0.5, 0.5]))
        logalpha = np.empty((2, n), dtype=float)
        logbeta = np.empty((2, n), dtype=float)

        logalpha[:, 0] = logpi + le[:, 0]
        for t in range(1, n):
            logalpha[0, t] = le[0, t] + np.logaddexp(
                logalpha[0, t - 1] + logA[0, 0],
                logalpha[1, t - 1] + logA[1, 0],
            )
            logalpha[1, t] = le[1, t] + np.logaddexp(
                logalpha[0, t - 1] + logA[0, 1],
                logalpha[1, t - 1] + logA[1, 1],
            )

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

        logZ = float(np.logaddexp(logalpha[0, n - 1], logalpha[1, n - 1]))
        loggamma = logalpha + logbeta - logZ
        gamma = np.exp(loggamma)

        # expected transitions xi
        xi01 = 0.0
        xi10 = 0.0
        denom = 0.0

        for t in range(n - 1):
            l00 = logalpha[0, t] + logA[0, 0] + le[0, t + 1] + logbeta[0, t + 1] - logZ
            l01 = logalpha[0, t] + logA[0, 1] + le[1, t + 1] + logbeta[1, t + 1] - logZ
            l10 = logalpha[1, t] + logA[1, 0] + le[0, t + 1] + logbeta[0, t + 1] - logZ
            l11 = logalpha[1, t] + logA[1, 1] + le[1, t + 1] + logbeta[1, t + 1] - logZ

            p00 = np.exp(l00)
            p01 = np.exp(l01)
            p10 = np.exp(l10)
            p11 = np.exp(l11)
            xi01 += float(p01)
            xi10 += float(p10)
            denom += float(p00 + p01 + p10 + p11)

        q = (xi01 + xi10) / max(denom, 1e-12)
        q = min(max(q, 1e-12), 1 - 1e-12)

        # tau2 update using posterior mean of s
        s_mean = gamma[0, :] * 1.0 + gamma[1, :] * (-1.0)
        resid = y - s_mean * x
        tau2 = float(np.mean(resid ** 2))
        tau2 = max(tau2, 1e-12)

    s_hat = np.where(gamma[0, :] >= 0.5, 1, -1).astype(int)
    return float(q), float(tau2), s_hat, float(logZ)
