#!/usr/bin/env python3
"""Performance figures from benchmarks/results (run analyze_results.py first).

  stage_times.png      sequential vs OpenMP execution time, split by stage
  speedup.png          speedup vs threads (end-to-end, covariance, scoring,
                       injection study) with ideal and Amdahl lines
  efficiency.png       parallel efficiency vs threads
  size_scaling.png     time and speedup vs matrix size (m sweep, t sweep)
"""
import json

import numpy as np
import pandas as pd

import style
from style import plt

RES = style.ROOT / "benchmarks" / "results"
STAGES = ["means", "center", "covariance", "eigen", "select", "score"]


def main():
    style.apply()
    s = pd.read_csv(RES / "summary_pipeline.csv")
    inj = pd.read_csv(RES / "summary_injection.csv") if (RES / "summary_injection.csv").exists() else None
    env = json.loads((RES / "environment.json").read_text()) if (RES / "environment.json").exists() else {}
    cpu = env.get("cpu", "")

    # scaling experiment = the (t, m) with the most thread counts
    counts = s.groupby(["t", "m"]).size()
    t_sc, m_sc = counts.idxmax()
    sc = s[(s.t == t_sc) & (s.m == m_sc)].sort_values(["backend", "threads"], ascending=[False, True])
    seq = sc[sc.backend == "sequential"].iloc[0]
    omp = sc[sc.backend == "openmp"].sort_values("threads")

    # ---- stage times
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = ["seq"] + [f"omp {p}" for p in omp.threads]
    rows = [seq] + [r for _, r in omp.iterrows()]
    bottom = np.zeros(len(rows))
    for stage, colour in zip(STAGES, style.SERIES):
        vals = np.array([r[stage] for r in rows])
        ax.bar(labels, vals, bottom=bottom, color=colour, width=0.7, label=stage, edgecolor=style.SURFACE, lw=1.0)
        bottom += vals
    for i, total in enumerate(bottom):
        ax.text(i, total, f"{total:.2f}s", ha="center", va="bottom", fontsize=8, color=style.INK2)
    ax.set_ylabel("median wall time (s)")
    ax.set_title(f"Execution time by stage - t = {t_sc}, m = {m_sc}", loc="left")
    ax.legend(ncol=6, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.text(0, -0.17, f"{cpu}; eigen = Eigen SelfAdjointEigenSolver, serial in both builds",
            transform=ax.transAxes, fontsize=8, color=style.MUTED)
    style.save(fig, "stage_times.png")

    # ---- speedup
    p = omp.threads.to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(p, p, color=style.MUTED, ls="--", lw=1.0, label="ideal S(p) = p")
    ax.plot(p, omp.amdahl_bound, color=style.MUTED, ls=":", lw=1.4,
            label=f"Amdahl bound (serial eigen = {100 * omp.serial_fraction_eigen.iloc[0]:.0f} %)")
    ax.plot(p, omp.speedup_covariance, marker="o", color=style.SERIES[0], label="covariance kernel")
    ax.plot(p, omp.speedup_score, marker="o", color=style.SERIES[1], label="SPE scoring kernel")
    ax.plot(p, omp.speedup, marker="o", color=style.SERIES[2], label="end-to-end pipeline")
    if inj is not None:
        io = inj[inj.backend == "openmp"].sort_values("threads")
        ax.plot(io.threads, io.speedup, marker="o", color=style.SERIES[3], label="injection study (outer loop)")
    ax.axvspan(4.5, 12.5, color=style.GRID, alpha=0.5, lw=0)
    ax.text(8.5, 0.3, "E-cores", ha="center", color=style.INK2, fontsize=8.5)
    ax.text(2.5, 0.3, "P-cores", ha="center", color=style.INK2, fontsize=8.5)
    ax.text(14.5, 0.3, "+SMT", ha="center", color=style.INK2, fontsize=8.5)
    ax.set_xticks(p)
    ax.set_xlabel("OpenMP threads p")
    ax.set_ylabel("speedup S(p) = T_seq / T_omp(p)")
    ax.set_ylim(0, max(p) + 0.5)
    ax.set_title(f"Speedup vs threads - t = {t_sc}, m = {m_sc}", loc="left")
    ax.legend(loc="upper left")
    style.save(fig, "speedup.png")

    # ---- efficiency
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(p, 100 * omp.speedup_covariance / p, marker="o", color=style.SERIES[0], label="covariance kernel")
    ax.plot(p, 100 * omp.speedup_score / p, marker="o", color=style.SERIES[1], label="SPE scoring kernel")
    ax.plot(p, omp.efficiency_pct, marker="o", color=style.SERIES[2], label="end-to-end pipeline")
    if inj is not None:
        ax.plot(io.threads, io.efficiency_pct, marker="o", color=style.SERIES[3], label="injection study")
    ax.axhline(100, color=style.MUTED, ls="--", lw=1.0)
    ax.set_xticks(p)
    ax.set_ylim(0, 115)
    ax.set_xlabel("OpenMP threads p")
    ax.set_ylabel("efficiency E(p) = S(p)/p  (%)")
    ax.set_title("Parallel efficiency vs threads", loc="left")
    ax.legend(loc="upper right")
    style.save(fig, "efficiency.png")

    # ---- size scaling
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    msw = s[s.t == 16384]
    ax = axes[0]
    for (backend, thr), colour in zip([("sequential", 1)] + [("openmp", q) for q in sorted(msw[msw.backend == "openmp"].threads.unique())],
                                      style.SERIES):
        d = msw[(msw.backend == backend) & (msw.threads == thr)].sort_values("m")
        ax.plot(d.m, d.total, marker="o", color=colour, label="sequential" if backend == "sequential" else f"OpenMP {thr} thr")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("links m (t = 16384)")
    ax.set_ylabel("median end-to-end time (s)")
    ax.set_title("Time vs number of links", loc="left")
    ax.legend()
    for ax, key, fixed, label in ((axes[1], "m", 16384, "links m (t = 16384)"),
                                  (axes[2], "t", 512, "time bins t (m = 512)")):
        other = "t" if key == "m" else "m"
        d = s[(s[other] == fixed) & (s.backend == "openmp")]
        for thr, colour in zip(sorted(d.threads.unique()), style.SERIES[1:]):
            dd = d[d.threads == thr].sort_values(key)
            ax.plot(dd[key], dd.speedup, marker="o", color=colour, label=f"end-to-end, {thr} thr")
        top = d[d.threads == d.threads.max()].sort_values(key)
        ax.plot(top[key], top.speedup_covariance, marker="s", ls="--", color=style.SERIES[0],
                label=f"covariance only, {d.threads.max()} thr")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(label)
        ax.set_ylabel("speedup vs sequential")
        ax.set_ylim(bottom=0)
        ax.set_title(f"Speedup vs {'links' if key == 'm' else 'time bins'}", loc="left")
        ax.legend(fontsize=8)
    fig.tight_layout()
    style.save(fig, "size_scaling.png")


if __name__ == "__main__":
    main()
