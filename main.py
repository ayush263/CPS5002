import argparse
from rescue_sim.simulation import Simulation
from rescue_sim.plotting import (
    plot_run_metrics,
    plot_batch_summary,
)
from rescue_sim.utils import ensure_dir
from rescue_sim.metrics import BatchAggregator


def run_single(args):
    ensure_dir(args.out)
    sim = Simulation(seed=args.seed, max_steps=args.steps, out_dir=args.out)
    outcome = sim.run(verbose=args.verbose)
    plot_run_metrics(args.out)
    print("\n=== Single Run Outcome ===")
    for k, v in outcome.items():
        print(f"{k}: {v}")


def run_batch(args):
    ensure_dir(args.out)
    agg = BatchAggregator()

    for i in range(args.runs):
        run_dir = f"{args.out}/run_{i+1:02d}"
        ensure_dir(run_dir)
        sim = Simulation(seed=args.seed + i, max_steps=args.steps, out_dir=run_dir)
        outcome = sim.run(verbose=False)
        agg.add(outcome, run_dir)

    batch_csv = agg.export_csv(args.out)
    plot_batch_summary(batch_csv, args.out)
    print("\n=== Batch Completed ===")
    print(f"Saved batch results: {batch_csv}")
    print(f"Saved plot: {args.out}/plot_batch_summary.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "batch"], default="single")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="out/run1")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.mode == "single":
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
