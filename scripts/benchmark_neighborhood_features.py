"""Run the neighbourhood feature benchmark (vtea_core.hierarchy.benchmark).

    python scripts/benchmark_neighborhood_features.py [out.csv] [--radii 15,30,60]

Prints the summary (mean and worst ARI per feature set and clustering, over
every synthetic tissue in 2D and 3D) and, with --radii, a radius sweep for
k-means; writes every row to out.csv. Takes a few minutes on one core.
"""

import argparse

import pandas as pd

from vtea_core.hierarchy.benchmark import run_benchmark, summarise


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", default="neighborhood_benchmark.csv")
    parser.add_argument("--radius", type=float, default=30.0)
    parser.add_argument("--radii", default="", help="comma-separated radii for a k-means sweep")
    args = parser.parse_args()
    pd.set_option("display.width", 200)
    results = run_benchmark(radius=args.radius)
    results["radius"] = args.radius
    print(summarise(results).round(3).to_string())
    if args.radii:
        sweep = []
        for radius in (float(r) for r in args.radii.split(",")):
            part = run_benchmark(
                feature_sets=["composition", "weighted (sigma r/2)", "composition + markers"],
                methods=("kmeans",),
                radius=radius,
            )
            part["radius"] = radius
            sweep.append(part)
        sweep = pd.concat(sweep)
        print(sweep.pivot_table(index=["features", "radius"], columns="scenario", values="ari").round(2))
        results = pd.concat([results, sweep])
    results.to_csv(args.out, index=False)
    print(f"wrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
