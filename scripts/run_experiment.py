from __future__ import annotations
import argparse
from parity_regime_lab.experiment import run_main_experiment, run_mobius_experiment


def main():
    ap = argparse.ArgumentParser(
        description="Run parity regime-switching experiments."
    )
    ap.add_argument("--outdir", type=str, default="out")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--freq", type=float, default=0.007)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--qs", type=float, nargs="*", default=[0.0, 0.002, 0.01, 0.03])
    ap.add_argument(
        "--mobius",
        action="store_true",
        help="Run the Möbius seam experiment instead of the standard one.",
    )
    ap.add_argument(
        "--omega",
        type=float,
        default=0.03,
        help="Phase increment per step (Möbius experiment).",
    )
    args = ap.parse_args()

    if args.mobius:
        run_mobius_experiment(
            outdir=args.outdir,
            n=args.n,
            tau=args.tau,
            amp=args.amp,
            omega=args.omega,
            seed=args.seed,
        )
        print(f"Plots in: {args.outdir}")
    else:
        csv_path = run_main_experiment(
            outdir=args.outdir,
            n=args.n,
            tau=args.tau,
            freq=args.freq,
            amp=args.amp,
            seed=args.seed,
            qs=list(args.qs),
        )
        print(f"Wrote: {csv_path}")
        print(f"Plots in: {args.outdir}")


if __name__ == "__main__":
    main()
