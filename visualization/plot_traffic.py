#!/usr/bin/env python3
"""Figures about the traffic itself.

  traffic_links.png      link traffic over time (4 busiest links), ground-truth
                         anomaly bins shaded
  variance_captured.png  fraction of variance per principal component (cf. the
                         paper's Fig. 2), with the chosen normal subspace marked

  python visualization/plot_traffic.py [--data datasets/synthetic] [--run results/synthetic_omp]
"""
import argparse
import json

import numpy as np
import pandas as pd

import style
from style import plt


def shade_labels(ax, labels):
    idx = np.flatnonzero(labels)
    for k in idx:
        ax.axvspan(k - 0.5, k + 0.5, color=style.WARNING_FILL, alpha=0.35, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(style.ROOT / "datasets" / "synthetic"))
    ap.add_argument("--run", default=str(style.ROOT / "results" / "synthetic_omp"))
    a = ap.parse_args()
    style.apply()

    Y = pd.read_csv(f"{a.data}/Y_test.csv")
    labels = pd.read_csv(f"{a.data}/labels_test.csv")["label"].to_numpy()
    links = Y.columns[1:]
    busiest = Y[links].mean().sort_values(ascending=False).index[:4]
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for ax, link, colour in zip(axes, busiest, style.SERIES):
        shade_labels(ax, labels)
        ax.plot(np.arange(len(Y)), Y[link] / 1e9, color=colour, lw=1.0)
        ax.set_ylabel("GB / 10 min")
        ax.set_title(f"link {link}", loc="left", fontsize=9.5)
    axes[-1].set_xlabel("time bin (10 min) - test week, Mon..Sun")
    axes[0].text(1.0, 1.25, "shaded = ground-truth anomaly bins", transform=axes[0].transAxes,
                 ha="right", color=style.INK2, fontsize=8.5)
    fig.suptitle("Synthetic link traffic (Y = X A$^T$): diurnal pattern, weekend dip, injected anomalies",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    style.save(fig, "traffic_links.png")

    model = json.load(open(f"{a.run}/model.json"))
    lam = np.clip(np.array(model["eigenvalues"]), 0, None)
    frac = lam / lam.sum()
    r = model["r"]
    n = min(len(frac), 20)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(np.arange(1, r + 1), frac[:r], color=style.SERIES[0], width=0.8, label=f"normal subspace (r = {r})")
    ax.bar(np.arange(r + 1, n + 1), frac[r:n], color=style.MUTED, width=0.8, label="anomalous (residual) subspace")
    ax.set_yscale("log")
    ax.set_xlabel("principal component")
    ax.set_ylabel("fraction of total variance (log)")
    ax.set_title(f"Variance captured per component - {model['m']} links, {model['t_train']} bins "
                 f"({model['r_method']} rule)", loc="left")
    ax.legend(loc="upper right")
    style.save(fig, "variance_captured.png")


if __name__ == "__main__":
    main()
