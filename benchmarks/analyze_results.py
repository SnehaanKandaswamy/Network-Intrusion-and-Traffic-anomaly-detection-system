#!/usr/bin/env python3
"""Turns raw benchmark CSVs into summary tables (medians, speedup, efficiency).

  S(p) = T_sequential / T_parallel(p)      E(p) = S(p) / p x 100 %

T_sequential is the median time of the SEQUENTIAL executable's kernels (built
without OpenMP), not OpenMP with one thread. Outputs in benchmarks/results/:
  summary_pipeline.csv   one row per (t, m, backend, threads), medians per stage
  summary_injection.csv  injection-study speedup/efficiency
  roofline_points.csv    FLOPs, bytes, arithmetic intensity and GFLOP/s per kernel
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
STAGES = ["means", "center", "covariance", "eigen", "select", "score", "total"]


def load_pipeline():
    frames = [pd.read_csv(p) for p in sorted(RESULTS.glob("*.csv"))
              if p.name.startswith(("scaling", "msweep_", "tsweep_")) and p.name.count(".") == 1]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    return df


def summarise(df):
    g = df.groupby(["t", "m", "backend", "threads"])
    med = g[STAGES + ["cpu_utilisation"]].median().reset_index()
    med["runs"] = g.size().values
    med["total_std"] = g["total"].std().values
    out = []
    for (t, m), block in med.groupby(["t", "m"]):
        seq = block[block.backend == "sequential"]
        if seq.empty:
            continue
        seq = seq.iloc[0]
        block = block.copy()
        for s in STAGES:
            block[f"speedup_{s}"] = seq[s] / block[s]
        block["speedup"] = block["speedup_total"]
        block["efficiency_pct"] = 100.0 * block["speedup"] / block["threads"]
        # Amdahl: serial fraction = eigensolver share of the sequential time
        block["serial_fraction_eigen"] = seq["eigen"] / seq["total"]
        block["amdahl_bound"] = 1.0 / (block["serial_fraction_eigen"]
                                       + (1 - block["serial_fraction_eigen"]) / block["threads"])
        out.append(block)
    return pd.concat(out, ignore_index=True)


def roofline_points(summary):
    """Leading-order FLOP and compulsory-byte counts (8-byte doubles).

    means      : t*m adds;              reads Y once            -> 8tm bytes
    center     : t*m subtractions;      reads + writes Y        -> 16tm
    covariance : t*m*(m+1) (upper triangle incl. diagonal, 1 FMA = 2 FLOPs);
                 compulsory traffic = read Y once + write C     -> 8tm + 8m^2
    score      : (4r+5) t*m (projection 2r, reconstruction 2r, subtract 1,
                 square-accumulate 2 for SPE and 2 for ||y||^2);
                 reads Y once                                  -> 8tm
    Measured GFLOP/s = FLOPs / median measured time.
    """
    rows = []
    for _, s in summary.iterrows():
        t, m = float(s.t), float(s.m)
        r = 6.0  # benchmark workload has 6 periodic factors; r is recorded per run
        kernels = {
            "means": (t * m, 8 * t * m),
            "center": (t * m, 16 * t * m),
            "covariance": (t * m * (m + 1), 8 * t * m + 8 * m * m),
            "score": ((4 * r + 5) * t * m, 8 * t * m),
        }
        for k, (flops, bytes_) in kernels.items():
            rows.append(dict(t=int(t), m=int(m), backend=s.backend, threads=int(s.threads), kernel=k,
                             flops=flops, bytes=bytes_, ai=flops / bytes_, seconds=s[k],
                             gflops=flops / s[k] / 1e9, gbs=bytes_ / s[k] / 1e9))
    return pd.DataFrame(rows)


def main():
    df = load_pipeline()
    if df is not None:
        # the r chosen must agree across all runs of one size (sanity)
        for (t, m), g in df.groupby(["t", "m"]):
            if g["r"].nunique() != 1:
                print(f"warning: r differs across runs for t={t}, m={m}: {sorted(g['r'].unique())}")
        summary = summarise(df)
        summary.to_csv(RESULTS / "summary_pipeline.csv", index=False, float_format="%.6g")
        roofline_points(summary).to_csv(RESULTS / "roofline_points.csv", index=False, float_format="%.6g")
        cols = ["t", "m", "backend", "threads", "total", "covariance", "eigen", "speedup", "efficiency_pct",
                "speedup_covariance", "amdahl_bound"]
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(summary[cols].round(4).to_string(index=False))
        eq = [pd.read_csv(p) for p in RESULTS.glob("*.equivalence.csv")]
        if eq:
            eq = pd.concat(eq, ignore_index=True)
            print(f"\nequivalence: {len(eq)} OpenMP configurations checked, "
                  f"bitwise-identical SPE in {int(eq.bitwise_identical_spe.sum())}, "
                  f"same r in {int(eq.same_r.sum())}, same alarms in {int(eq.same_alarms.sum())}, "
                  f"max rel. diff SPE {eq.max_rel_diff_spe.max():.2e}")

    inj = RESULTS / "injection_timing.csv"
    if inj.exists():
        d = pd.read_csv(inj)
        med = d.groupby(["backend", "threads"])["seconds"].median().reset_index()
        t_seq = float(med[med.backend == "sequential"].seconds.iloc[0])
        med["speedup"] = t_seq / med["seconds"]
        med["efficiency_pct"] = 100 * med["speedup"] / med["threads"]
        med["experiments"] = int(d["experiments"].iloc[0])
        med["experiments_per_s"] = med["experiments"] / med["seconds"]
        med.to_csv(RESULTS / "summary_injection.csv", index=False, float_format="%.6g")
        print("\ninjection study:\n" + med.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
