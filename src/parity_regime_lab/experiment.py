from __future__ import annotations
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from .generators import gen_parity_switched_signal
from .diagnostics import ar1_fit, fft_peak_over_total, parity_accuracy
from .hmm_parity import hmm_em_known_x
from .mi_parity import estimate_mi_parity_bits, mi_small_snr_approx_bits
from .mobius_generator import gen_mobius_seam_signal


def run_main_experiment(
    outdir: str,
    n: int,
    tau: float,
    freq: float,
    amp: float,
    seed: int,
    qs: list[float] | None = None,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    if qs is None:
        qs = [0.0, 0.002, 0.01, 0.03]

    rows = []
    for q in qs:
        y, s, x = gen_parity_switched_signal(n=n, q=q, amp=amp, freq=freq, tau=tau, rng=rng)
        a_hat, e = ar1_fit(y)
        resid_std = float(np.std(e))
        leak = fft_peak_over_total(y)
        q_hat, tau2_hat, s_hat, logZ = hmm_em_known_x(y=y, x=x, n_iter=25, q_init=max(q, 0.005))
        acc = parity_accuracy(s, s_hat)

        rows.append({
            "true_q": q,
            "AR1_a_hat": a_hat,
            "AR1_resid_std": resid_std,
            "FFT_peak_over_total": leak,
            "HMM_q_hat": q_hat,
            "HMM_tau_hat": float(np.sqrt(tau2_hat)),
            "Parity_acc": acc,
            "log_evidence": logZ,
        })

    csv_path = os.path.join(outdir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    qs_arr = np.array([r["true_q"] for r in rows])

    plt.figure()
    plt.plot(qs_arr, [r["AR1_resid_std"] for r in rows], marker="o")
    plt.xlabel("Flip probability q (true)")
    plt.ylabel("AR(1) residual std")
    plt.title("Local fit degrades as parity flips increase")
    plt.savefig(os.path.join(outdir, "plot_ar_resid.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(qs_arr, [r["FFT_peak_over_total"] for r in rows], marker="o")
    plt.xlabel("Flip probability q (true)")
    plt.ylabel("FFT peak/total magnitude")
    plt.title("Spectral concentration drops with parity flips (leakage increases)")
    plt.savefig(os.path.join(outdir, "plot_fft_leakage.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(qs_arr, [r["Parity_acc"] for r in rows], marker="o")
    plt.xlabel("Flip probability q (true)")
    plt.ylabel("Parity reconstruction accuracy")
    plt.ylim(0, 1.02)
    plt.title("Parity-aware HMM recovers hidden orientation")
    plt.savefig(os.path.join(outdir, "plot_parity_accuracy.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # MI curvature plot
    amps = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
    mi_est = []
    mi_apx = []
    for a in amps:
        mi_est.append(estimate_mi_parity_bits(a=a, tau=1.0, rng=rng, n=250_000))
        mi_apx.append(mi_small_snr_approx_bits(a=a, tau=1.0))

    plt.figure()
    plt.plot(amps, mi_est, marker="o", label="Estimated MI(s;y) [bits]")
    plt.plot(amps, mi_apx, marker="o", label="Approx: a^2/(2 tau^2 ln 2)")
    plt.xlabel("Signal amplitude a (tau=1)")
    plt.ylabel("Mutual information (bits)")
    plt.title("1/(2 ln 2) as small-SNR parity information curvature")
    plt.legend()
    plt.savefig(os.path.join(outdir, "plot_mi_curvature.png"), dpi=160, bbox_inches="tight")
    plt.close()

    return csv_path


def run_mobius_experiment(
    outdir: str,
    n: int,
    tau: float,
    amp: float,
    omega: float,
    seed: int,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    y, s, x, theta = gen_mobius_seam_signal(
        n=n,
        omega=omega,
        amp=amp,
        tau=tau,
        rng=rng,
        seam=np.pi,
    )

    a_hat, e = ar1_fit(y)
    leak = fft_peak_over_total(y)
    q_hat, tau2_hat, s_hat, logZ = hmm_em_known_x(y=y, x=x)
    acc = parity_accuracy(s, s_hat)

    print("AR1 coefficient:   ", round(a_hat, 4))
    print("Residual std:      ", round(float(np.std(e)), 4))
    print("FFT leakage metric:", round(leak, 4))
    print("Recovered flip rate:", round(q_hat, 4))
    print("Parity accuracy:   ", round(acc, 4))

    view = min(1000, n)

    plt.figure()
    plt.plot(theta[:view])
    plt.xlabel("Time")
    plt.ylabel("Phase (rad)")
    plt.title("Phase trajectory (Möbius generator)")
    plt.savefig(os.path.join(outdir, "theta.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(s[:view])
    plt.xlabel("Time")
    plt.ylabel("Parity s_t")
    plt.ylim(-1.4, 1.4)
    plt.title("True parity (flips at seam crossings)")
    plt.savefig(os.path.join(outdir, "true_parity.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(s_hat[:view])
    plt.xlabel("Time")
    plt.ylabel("Recovered parity")
    plt.ylim(-1.4, 1.4)
    plt.title(f"HMM-recovered parity (acc={acc:.3f})")
    plt.savefig(os.path.join(outdir, "recovered_parity.png"), dpi=160, bbox_inches="tight")
    plt.close()
