#!/usr/bin/env python3
"""Detection figures (cf. the paper's Fig. 4).

  state_vs_residual.png  ||y||^2 (all traffic) vs SPE = ||y_tilde||^2 (residual)
                         with the Q-statistic threshold, ground truth and alarms
  anomaly_scores.png     SPE / delta^2 over time, log scale
  anomaly_locations.png  which links carry the residual at each alarm

  python visualization/plot_anomalies.py [--data datasets/synthetic] [--run results/synthetic_omp]
"""
import argparse

import numpy as np
import pandas as pd

import style
from style import plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(style.ROOT / "datasets" / "synthetic"))
    ap.add_argument("--run", default=str(style.ROOT / "results" / "synthetic_omp"))
    a = ap.parse_args()
    style.apply()

    spe = pd.read_csv(f"{a.run}/spe.csv")
    labels = pd.read_csv(f"{a.data}/labels_test.csv")["label"].to_numpy().astype(bool)
    k = spe["time_index"].to_numpy()
    flagged = spe["anomaly"].to_numpy().astype(bool)
    delta2 = float(spe["threshold"].iloc[0])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax in (top, bottom):
        for t in k[labels]:
            ax.axvspan(t - 0.5, t + 0.5, color=style.WARNING_FILL, alpha=0.35, lw=0)
    top.plot(k, spe["state_norm2"], color=style.SERIES[0], lw=0.9)
    top.set_ylabel("$\\|y\\|^2$")
    top.set_title("State vector: dominated by the daily cycle, anomalies hard to see", loc="left")
    bottom.plot(k, spe["spe"], color=style.SERIES[0], lw=0.9, label="SPE $\\|\\tilde y\\|^2$")
    bottom.axhline(delta2, color=style.CRITICAL, lw=1.2, ls="--", label="Q-statistic threshold $\\delta^2_\\alpha$")
    bottom.scatter(k[flagged], spe["spe"][flagged], s=28, color=style.CRITICAL, zorder=3,
                   edgecolor=style.SURFACE, linewidth=1.2, label="alarm")
    bottom.set_ylabel("$\\|\\tilde y\\|^2$")
    bottom.set_xlabel("time bin (10 min) - held-out test week")
    bottom.set_title("Residual vector: anomalies stand out", loc="left")
    bottom.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    top.text(1.0, 1.12, "shaded = ground-truth anomaly bins", transform=top.transAxes, ha="right",
             color=style.INK2, fontsize=8.5)
    fig.tight_layout()
    style.save(fig, "state_vs_residual.png")

    fig, ax = plt.subplots(figsize=(10, 3.4))
    for t in k[labels]:
        ax.axvspan(t - 0.5, t + 0.5, color=style.WARNING_FILL, alpha=0.35, lw=0)
    ratio = spe["spe"] / delta2
    ax.plot(k, ratio, color=style.SERIES[0], lw=0.9)
    ax.axhline(1.0, color=style.CRITICAL, lw=1.2, ls="--")
    ax.text(k[-1], 1.08, "alarm threshold (SPE = $\\delta^2$)", ha="right", va="bottom", color=style.INK2, fontsize=8.5)
    ax.scatter(k[flagged], ratio[flagged], s=28, color=style.CRITICAL, zorder=3, edgecolor=style.SURFACE, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("time bin (10 min)")
    ax.set_ylabel("anomaly score  SPE / $\\delta^2$")
    ax.set_title("Anomaly score over time (red = alarm, shaded = ground truth)", loc="left")
    style.save(fig, "anomaly_scores.png")

    alarms = pd.read_csv(f"{a.run}/alarms.csv")
    if alarms.empty:
        return
    rows = []
    for _, row in alarms.iterrows():
        for item in str(row["top_links"]).split(";"):
            link, share = item.rsplit(":", 1)
            rows.append((row["time_index"], link, float(share)))
    loc = pd.DataFrame(rows, columns=["time", "link", "share"])
    order = sorted(loc["link"].unique())
    ypos = {l: i for i, l in enumerate(order)}
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.22 * len(order) + 1.2)))
    for t in k[labels]:
        ax.axvspan(t - 0.5, t + 0.5, color=style.WARNING_FILL, alpha=0.35, lw=0)
    ax.scatter(loc["time"], [ypos[l] for l in loc["link"]], s=20 + 180 * loc["share"], color=style.CRITICAL,
               alpha=0.85, edgecolor=style.SURFACE, linewidth=1.0)
    ax.set_yticks(range(len(order)), order, fontsize=7.5)
    ax.set_xlim(k[0], k[-1])
    ax.set_xlabel("time bin (10 min)")
    ax.set_title("Detected anomaly locations: top-3 residual links per alarm (dot size = share of SPE)",
                 loc="left")
    style.save(fig, "anomaly_locations.png")


if __name__ == "__main__":
    main()
