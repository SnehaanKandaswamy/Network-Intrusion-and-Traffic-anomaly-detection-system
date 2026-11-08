#!/usr/bin/env python3
"""Roofline plot: P = min(P_peak, B * AI).

Ceilings (all MEASURED on this machine by `bench`):
  compute : best FP64 FMA throughput over all thread counts (peak.csv)
  memory  : best STREAM-triad bandwidth over all thread counts (stream.csv,
            32 bytes/element: 2 reads + 1 write + the write-allocate read
            that the cache performs; this is the real DRAM traffic)
Reference line (THEORETICAL, an assumption, not measured): vendor
  microarchitecture figures at the frequency caps reported by the OS:
  4 P-cores x 16 FLOP/cycle x 2.6 GHz + 8 E-cores x 8 FLOP/cycle x 1.9 GHz.
Kernel points: FLOPs and compulsory bytes from analyze_results.py divided by
measured median times at the scaling size (sequential and best OpenMP).
Also writes benchmarks/results/roofline_table.csv.
"""
import json

import numpy as np
import pandas as pd

import style
from style import plt

RES = style.ROOT / "benchmarks" / "results"


def main():
    style.apply()
    stream = pd.read_csv(RES / "stream.csv")
    peak = pd.read_csv(RES / "peak.csv")
    pts = pd.read_csv(RES / "roofline_points.csv")
    env = json.loads((RES / "environment.json").read_text())
    freqs = env.get("cpuinfo_max_freq_ghz", {})
    f_p = freqs.get("cpu0", 2.6)
    f_e = freqs.get("cpu8", 1.9)
    p_theory = 4 * 16 * f_p + 8 * 8 * f_e

    bw = stream.bandwidth_gbs_32B.max()
    bw1 = stream[stream.threads == 1].bandwidth_gbs_32B.iloc[0]
    pk = peak.gflops.max()
    pk1 = peak[peak.threads == 1].gflops.iloc[0]

    counts = pts.groupby(["t", "m"]).size()
    t_sc, m_sc = counts.idxmax()
    d = pts[(pts.t == t_sc) & (pts.m == m_sc)]
    seq = d[d.backend == "sequential"]
    omp_all = d[d.backend == "openmp"]
    best_p = int(omp_all.groupby("threads").seconds.sum().idxmin())
    omp = omp_all[omp_all.threads == best_p]

    ai = np.logspace(-2, 3, 400)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ai, np.minimum(pk, bw * ai), color=style.INK, lw=1.8, label=f"measured roof, all threads ({pk:.0f} GFLOP/s, {bw:.0f} GB/s)")
    ax.plot(ai, np.minimum(pk1, bw1 * ai), color=style.MUTED, lw=1.4, ls="--",
            label=f"measured roof, 1 thread ({pk1:.0f} GFLOP/s, {bw1:.0f} GB/s)")
    ax.axhline(p_theory, color=style.MUTED, lw=1.0, ls=":", label=f"theoretical FP64 peak (assumed) {p_theory:.0f} GFLOP/s")
    for (name, grp), colour in zip(seq.groupby("kernel", sort=False), style.SERIES):
        o = omp[omp.kernel == name]
        ax.scatter(grp.ai, grp.gflops, marker="o", s=60, facecolor="none", edgecolor=colour, linewidth=1.8, zorder=3)
        ax.scatter(o.ai, o.gflops, marker="o", s=60, color=colour, edgecolor=style.SURFACE, linewidth=1.2, zorder=3,
                   label=f"{name}  (AI {grp.ai.iloc[0]:.3g})")
        ax.annotate("", xy=(o.ai.iloc[0], o.gflops.iloc[0]), xytext=(grp.ai.iloc[0], grp.gflops.iloc[0]),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte, compulsory traffic)")
    ax.set_ylabel("performance (GFLOP/s)")
    ax.set_ylim(0.1, p_theory * 2)
    ax.set_title(f"Roofline - kernels at t = {t_sc}, m = {m_sc}: hollow = sequential, filled = OpenMP {best_p} threads",
                 loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    style.save(fig, "roofline.png")

    table = []
    for _, row in pd.concat([seq, omp]).iterrows():
        bound = min(pk if row.backend == "openmp" else pk1, (bw if row.backend == "openmp" else bw1) * row.ai)
        table.append(dict(kernel=row.kernel, backend=row.backend, threads=row.threads, flops=row.flops,
                          bytes=row.bytes, ai=row.ai, measured_gflops=row.gflops, roofline_bound_gflops=bound,
                          fraction_of_bound=row.gflops / bound,
                          bound_by="memory" if bw * row.ai < pk else "compute"))
    pd.DataFrame(table).to_csv(RES / "roofline_table.csv", index=False, float_format="%.4g")
    print(pd.DataFrame(table).round(3).to_string(index=False))
    print(f"ridge point (all threads): {pk / bw:.2f} FLOP/byte; theoretical peak (assumed) {p_theory:.0f} GFLOP/s")


if __name__ == "__main__":
    main()
