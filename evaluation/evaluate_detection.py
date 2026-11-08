#!/usr/bin/env python3
"""Detection metrics from a detector run (spe.csv) and ground-truth labels.

Labels are read ONLY here, after detection. They are never used to fit the
model or choose the threshold (the threshold is the analytic Q-statistic).

Per-bin metrics (one decision per time bin):
  precision = TP / (TP + FP)      recall (detection rate) = TP / (TP + FN)
  F1 = 2PR / (P + R)              false-positive rate = FP / (FP + TN)
Per-event metric: an event counts as detected if any of its bins is flagged.

  python evaluation/evaluate_detection.py --spe results/synthetic_omp/spe.csv \
      --labels datasets/synthetic/labels_test.csv --events datasets/synthetic/events_test.csv \
      --out results/synthetic_omp/metrics.json
"""
import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spe", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--events")
    ap.add_argument("--out")
    a = ap.parse_args()

    spe = pd.read_csv(a.spe)
    lab = pd.read_csv(a.labels)
    if len(spe) != len(lab):
        raise SystemExit(f"length mismatch: spe has {len(spe)} rows, labels {len(lab)}")
    y, p = lab["label"].to_numpy().astype(bool), spe["anomaly"].to_numpy().astype(bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    metrics = dict(bins=len(y), anomalous_bins=int(y.sum()), alarms=int(p.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                   precision=precision, recall=recall, f1=f1, false_positive_rate=fpr)

    if a.events:
        ev = pd.read_csv(a.events)
        flagged = set(spe.index[p])
        ev["detected"] = [any(s + k in flagged for k in range(d)) for s, d in zip(ev["start"], ev["duration"])]
        metrics["events"] = {t: dict(detected=int(g["detected"].sum()), total=len(g))
                             for t, g in ev.groupby("type")}
        metrics["event_detection_rate"] = float(ev["detected"].mean())

    print(f"bins {metrics['bins']}  anomalous {metrics['anomalous_bins']}  alarms {metrics['alarms']}")
    print(f"TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}  FPR {fpr:.4f}")
    if a.events:
        for t, v in metrics["events"].items():
            print(f"  {t:7s}: {v['detected']}/{v['total']} events detected")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
