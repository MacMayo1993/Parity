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
    n_seeds: int = 1,
) -> str:
    """
    Run the standard parity sweep experiment.

    Each (q, seed_index) pair gets its own independent RNG derived from
    (seed, q_index, seed_index), so results are fully reproducible and
    independent across both q values and seeds.

    When n_seeds > 1, each metric is reported as mean ± std across seeds.
    """
    os.makedirs(outdir, exist_ok=True)
    if qs is None:
        qs = [0.0, 0.002, 0.01, 0.03]

    rows = []
    for qi, q in enumerate(qs):
        ar1_resids, fft_leaks, q_hats, tau_hats, accs, logZs = [], [], [], [], [], []

        for si in range(n_seeds):
            # Independent RNG per (q_index, seed_index) — no shared state across q values
            rng = np.random.default_rng([seed, qi, si])
            y, s, x = gen_parity_switched_signal(
                n=n, q=q, amp=amp, freq=freq, tau=tau, rng=rng
            )
            a_hat, e = ar1_fit(y)
            ar1_resids.append(float(np.std(e)))
            fft_leaks.append(fft_peak_over_total(y))

            q_hat, tau2_hat, s_hat, logZ = hmm_em_known_x(
                y=y, x=x, n_iter=25, q_init=max(q, 0.005)
            )
            q_hats.append(q_hat)
            tau_hats.append(float(np.sqrt(tau2_hat)))
            accs.append(parity_accuracy(s, s_hat))
            logZs.append(logZ)

        def _s(v: list[float]) -> dict:
            a = np.array(v)
            if n_seeds == 1:
                return {"mean": float(a[0]), "std": 0.0}
            return {"mean": float(a.mean()), "std": float(a.std())}

        row = {"true_q": q}
        for name, vals in [
            ("AR1_resid_std", ar1_resids),
            ("FFT_peak_over_total", fft_leaks),
            ("HMM_q_hat", q_hats),
            ("HMM_tau_hat", tau_hats),
            ("Parity_acc", accs),
            ("log_evidence", logZs),
        ]:
            s_ = _s(vals)
            row[name] = s_["mean"]
            if n_seeds > 1:
                row[f"{name}_std"] = s_["std"]

        rows.append(row)

    csv_path = os.path.join(outdir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    qs_arr = np.array([r["true_q"] for r in rows])
    ar1_vals = np.array([r["AR1_resid_std"] for r in rows])
    fft_vals = np.array([r["FFT_peak_over_total"] for r in rows])
    acc_vals = np.array([r["Parity_acc"] for r in rows])

    # Optional std arrays (zero when n_seeds == 1)
    ar1_std = np.array([r.get("AR1_resid_std_std", 0.0) for r in rows])
    fft_std = np.array([r.get("FFT_peak_over_total_std", 0.0) for r in rows])
    acc_std = np.array([r.get("Parity_acc_std", 0.0) for r in rows])

    # --- Individual metric plots ---
    plt.figure()
    plt.errorbar(qs_arr, ar1_vals, yerr=ar1_std if n_seeds > 1 else None, marker="o", capsize=4)
    plt.xlabel("Flip probability q (true)")
    plt.ylabel("AR(1) residual std")
    plt.title("Local fit degrades as parity flips increase")
    plt.savefig(os.path.join(outdir, "plot_ar_resid.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.errorbar(qs_arr, fft_vals, yerr=fft_std if n_seeds > 1 else None, marker="o", capsize=4)
    plt.xlabel("Flip probability q (true)")
    plt.ylabel("FFT peak/total magnitude")
    plt.title("Spectral concentration drops with parity flips (leakage increases)")
    plt.savefig(os.path.join(outdir, "plot_fft_leakage.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.errorbar(qs_arr, acc_vals, yerr=acc_std if n_seeds > 1 else None, marker="o", capsize=4)
    plt.xlabel("Flip probability q (true)")
    plt.ylabel("Parity reconstruction accuracy")
    plt.ylim(0, 1.02)
    plt.title("Parity-aware HMM recovers hidden orientation")
    plt.savefig(os.path.join(outdir, "plot_parity_accuracy.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # --- Comparison plot: blind vs parity-aware on one figure ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.errorbar(qs_arr, ar1_vals, yerr=ar1_std if n_seeds > 1 else None,
                marker="o", color="tab:orange", capsize=4, label="AR(1) resid std")
    ax.set_xlabel("True flip rate q")
    ax.set_ylabel("AR(1) residual std")
    ax.set_title("Blind: AR(1)\n(degrades, no diagnosis)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(qs_arr, fft_vals, yerr=fft_std if n_seeds > 1 else None,
                marker="o", color="tab:red", capsize=4, label="FFT peak/total")
    ax.set_xlabel("True flip rate q")
    ax.set_ylabel("FFT peak / total")
    ax.set_title("Blind: FFT\n(smears, no diagnosis)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.errorbar(qs_arr, acc_vals, yerr=acc_std if n_seeds > 1 else None,
                marker="o", color="tab:green", capsize=4, label="Parity accuracy")
    ax.set_xlabel("True flip rate q")
    ax.set_ylabel("Parity accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Parity-aware: HMM\n(tracks orientation, stays accurate)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Blind diagnostics vs Parity-aware HMM", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "plot_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- MI curvature plot ---
    rng_mi = np.random.default_rng([seed, 9999])
    amps = np.array([0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
    mi_est = [estimate_mi_parity_bits(a=a, tau=1.0, rng=rng_mi, n=250_000) for a in amps]
    mi_apx = [mi_small_snr_approx_bits(a=a, tau=1.0) for a in amps]

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

    resid_std = float(np.std(e))

    print("AR1 coefficient:   ", round(a_hat, 4))
    print("Residual std:      ", round(resid_std, 4))
    print("FFT leakage metric:", round(leak, 4))
    print("Recovered flip rate:", round(q_hat, 4))
    print("Parity accuracy:   ", round(acc, 4))

    # Save structured results to CSV
    csv_path = os.path.join(outdir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "omega", "n", "tau", "amp",
            "AR1_a_hat", "AR1_resid_std", "FFT_peak_over_total",
            "HMM_q_hat", "HMM_tau_hat", "Parity_acc", "log_evidence",
        ])
        w.writeheader()
        w.writerow({
            "omega": omega,
            "n": n,
            "tau": tau,
            "amp": amp,
            "AR1_a_hat": round(a_hat, 6),
            "AR1_resid_std": round(resid_std, 6),
            "FFT_peak_over_total": round(leak, 6),
            "HMM_q_hat": round(q_hat, 6),
            "HMM_tau_hat": round(float(np.sqrt(tau2_hat)), 6),
            "Parity_acc": round(acc, 6),
            "log_evidence": round(logZ, 4),
        })
    print(f"Wrote: {csv_path}")

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
