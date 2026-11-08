#!/usr/bin/env python3
"""Synthetic network traffic generator (network mode).

Mirrors how Lakhina et al. (SIGCOMM 2004, section 3) built their link data:
OD-flow traffic X (time x flows) is mapped to link traffic through a routing
matrix A (links x flows):  Y = X A^T  (each row: y = A x).

Pipeline
  1. Topology: N PoPs on a ring plus random chords (connected, seeded).
     Links = both directions of every edge + one ingress and one egress link
     per PoP (as in the public Abilene data).
  2. Routing: hop-count shortest paths (BFS, deterministic tie-break) -> A.
  3. OD flows (N^2, including o == d): gravity-model mean
     x_od ~ total * p_o * q_d, multiplied by a diurnal profile (with a per-PoP
     time-zone shift), a weekend dip, and log-normal noise.
  4. Labelled anomalies injected into the OD flows:
       spike   : one flow, +size, 1 bin
       outage  : one flow drops to ~10 % for 1-3 bins (negative anomaly)
       ddos    : k origins -> one destination, +size each, 2-6 bins
  5. Output (weeks: 1 = training, 2 = test):
       Y_train.csv, Y_test.csv   link traffic, time x link, with anomalies
       Y_clean.csv               training week without anomalies (injection study)
       A.csv                     routing matrix, link x flow
       labels_train.csv, labels_test.csv   per-bin 0/1 ground truth
       events_train.csv, events_test.csv   one row per injected event
       meta.json                 parameters and summary statistics

Units: bytes per bin. Everything is controlled by command-line flags and a seed.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def build_topology(n_nodes: int, n_chords: int, rng: np.random.Generator):
    edges = {(i, (i + 1) % n_nodes) if i < (i + 1) % n_nodes else ((i + 1) % n_nodes, i)
             for i in range(n_nodes)}
    candidates = [(i, j) for i in range(n_nodes) for j in range(i + 2, n_nodes)
                  if (i, j) not in edges and not (i == 0 and j == n_nodes - 1)]
    order = rng.permutation(len(candidates))
    for idx in order[:n_chords]:
        edges.add(candidates[idx])
    return sorted(edges)


def shortest_paths(n_nodes: int, edges):
    """BFS from every node; neighbours visited in index order -> deterministic."""
    adj = {v: [] for v in range(n_nodes)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    for v in adj:
        adj[v].sort()
    paths = {}
    for src in range(n_nodes):
        parent = {src: None}
        queue = deque([src])
        while queue:
            v = queue.popleft()
            for w in adj[v]:
                if w not in parent:
                    parent[w] = v
                    queue.append(w)
        for dst in range(n_nodes):
            path, v = [], dst
            while v is not None:
                path.append(v)
                v = parent[v]
            paths[(src, dst)] = path[::-1]
    return paths


def routing_matrix(n_nodes: int, edges, names):
    links = []
    for a, b in edges:
        links += [(a, b), (b, a)]
    link_names = [f"{names[a]}>{names[b]}" for a, b in links]
    link_names += [f"in:{names[v]}" for v in range(n_nodes)]
    link_names += [f"out:{names[v]}" for v in range(n_nodes)]
    link_index = {l: i for i, l in enumerate(links)}
    n_internal = len(links)
    paths = shortest_paths(n_nodes, edges)
    flows = [(o, d) for o in range(n_nodes) for d in range(n_nodes)]
    flow_names = [f"{names[o]}-{names[d]}" for o, d in flows]
    A = np.zeros((len(link_names), len(flows)))
    for f, (o, d) in enumerate(flows):
        A[n_internal + o, f] = 1.0                 # enters the network at o
        A[n_internal + n_nodes + d, f] = 1.0       # leaves the network at d
        p = paths[(o, d)]
        for a, b in zip(p[:-1], p[1:]):
            A[link_index[(a, b)], f] = 1.0
    return A, link_names, flows, flow_names


def od_traffic(n_nodes, flows, t, bins_per_day, mean_flow, noise, rng):
    p = rng.lognormal(0.0, 0.8, n_nodes)
    q = rng.lognormal(0.0, 0.8, n_nodes)
    p, q = p / p.sum(), q / q.sum()
    total = mean_flow * len(flows)
    means = np.array([total * p[o] * q[d] for o, d in flows])

    k = np.arange(t)
    hour = (k % bins_per_day) * 24.0 / bins_per_day
    day = k // bins_per_day                      # day 0 = Monday
    weekend = np.where(day % 7 >= 5, 0.75, 1.0)
    tz = rng.integers(0, 4, n_nodes)              # 0-3 h shifts (time zones)

    def profile(node):
        h = hour - tz[node]
        return 1.0 + 0.45 * np.sin(2 * np.pi * (h - 9.0) / 24.0) + 0.12 * np.sin(4 * np.pi * (h - 3.0) / 24.0)

    prof = np.stack([profile(v) for v in range(n_nodes)])          # node x time
    X = np.empty((t, len(flows)))
    for f, (o, d) in enumerate(flows):
        shape = 0.5 * (prof[o] + prof[d]) * weekend
        X[:, f] = means[f] * shape * rng.lognormal(-0.5 * noise**2, noise, t)
    return X, means


def inject_events(X, flows, n_nodes, n_spikes, n_outages, n_ddos, size_range, mean_flow, rng,
                  first_bin=0):
    """Adds anomalies to X in place; returns per-bin labels and an event table."""
    t, n = X.shape
    labels = np.zeros(t, dtype=int)
    events = []
    busy = np.zeros(t, dtype=bool)

    def pick_time(duration):
        for _ in range(1000):
            s = int(rng.integers(2, t - duration - 2))
            if not busy[max(0, s - 3): s + duration + 3].any():
                busy[s: s + duration] = True
                return s
        raise RuntimeError("could not place event without overlap; reduce the number of anomalies")

    eid = 0
    for _ in range(n_spikes):
        f, s = int(rng.integers(n)), pick_time(1)
        size = float(rng.uniform(*size_range)) * mean_flow
        X[s, f] += size
        labels[s] = 1
        events.append(dict(event_id=eid, type="spike", start=first_bin + s, duration=1,
                           flows=f"{f}", n_flows=1, size_per_flow=size))
        eid += 1
    for _ in range(n_outages):
        # outage on one of the 25 % largest flows so the drop is substantial
        big = np.argsort(X.mean(axis=0))[-max(1, n // 4):]
        f, dur = int(rng.choice(big)), int(rng.integers(1, 4))
        s = pick_time(dur)
        lost = float(X[s:s + dur, f].mean() * 0.9)
        X[s:s + dur, f] *= 0.1
        labels[s:s + dur] = 1
        events.append(dict(event_id=eid, type="outage", start=first_bin + s, duration=dur,
                           flows=f"{f}", n_flows=1, size_per_flow=-lost))
        eid += 1
    for _ in range(n_ddos):
        dst = int(rng.integers(n_nodes))
        k = int(rng.integers(4, 7))
        origins = rng.choice([v for v in range(n_nodes) if v != dst], size=k, replace=False)
        fl = [o * n_nodes + dst for o in origins]
        dur = int(rng.integers(2, 7))
        s = pick_time(dur)
        size = float(rng.uniform(*size_range)) * mean_flow / 2.0   # per participating flow
        X[s:s + dur, fl] += size
        labels[s:s + dur] = 1
        events.append(dict(event_id=eid, type="ddos", start=first_bin + s, duration=dur,
                           flows=";".join(map(str, fl)), n_flows=k, size_per_flow=size))
        eid += 1
    return labels, pd.DataFrame(events).sort_values("start").reset_index(drop=True)


def write_matrix(path: Path, values: np.ndarray, times, columns, index_name="time"):
    df = pd.DataFrame(values, columns=columns)
    df.insert(0, index_name, times)
    df.to_csv(path, index=False, float_format="%.6f")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=12, help="number of PoPs (default 12)")
    ap.add_argument("--chords", type=int, default=6, help="extra edges beyond the ring (default 6)")
    ap.add_argument("--bin-minutes", type=int, default=10, help="time-bin length (default 10, as in the paper)")
    ap.add_argument("--days", type=int, default=7, help="days per week-file (default 7 -> 1008 bins)")
    ap.add_argument("--mean-flow", type=float, default=1e7, help="mean OD-flow volume, bytes/bin")
    ap.add_argument("--noise", type=float, default=0.10, help="log-normal noise sigma of OD flows")
    ap.add_argument("--spikes", type=int, default=10, help="single-flow spikes per week")
    ap.add_argument("--outages", type=int, default=3, help="single-flow outages per week")
    ap.add_argument("--ddos", type=int, default=3, help="multi-flow DDoS events per week")
    ap.add_argument("--size-min", type=float, default=6.0, help="min anomaly size, x mean flow")
    ap.add_argument("--size-max", type=float, default=15.0, help="max anomaly size, x mean flow")
    ap.add_argument("--seed", type=int, default=2004)
    ap.add_argument("--out", default="datasets/synthetic")
    a = ap.parse_args()

    if a.nodes < 3:
        ap.error("--nodes must be >= 3")
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    names = [f"N{v:02d}" for v in range(a.nodes)]
    edges = build_topology(a.nodes, a.chords, rng)
    A, link_names, flows, flow_names = routing_matrix(a.nodes, edges, names)
    bins_per_day = 24 * 60 // a.bin_minutes
    t = a.days * bins_per_day

    X, means = od_traffic(a.nodes, flows, 2 * t, bins_per_day, a.mean_flow, a.noise, rng)
    X_clean_train = X[:t].copy()
    size_range = (a.size_min, a.size_max)
    lab_tr, ev_tr = inject_events(X[:t], flows, a.nodes, a.spikes, a.outages, a.ddos, size_range,
                                  a.mean_flow, rng, first_bin=0)
    lab_te, ev_te = inject_events(X[t:], flows, a.nodes, a.spikes, a.outages, a.ddos, size_range,
                                  a.mean_flow, rng, first_bin=0)

    Y = X @ A.T                                   # y = A x for every time bin
    Y_clean = X_clean_train @ A.T
    stamps = pd.date_range("2024-01-01", periods=2 * t, freq=f"{a.bin_minutes}min").strftime("%Y-%m-%dT%H:%M")

    write_matrix(out / "Y_train.csv", Y[:t], stamps[:t], link_names)
    write_matrix(out / "Y_test.csv", Y[t:], stamps[t:], link_names)
    write_matrix(out / "Y_clean.csv", Y_clean, stamps[:t], link_names)
    write_matrix(out / "A.csv", A, link_names, flow_names, index_name="link")
    pd.DataFrame({"time_index": np.arange(t), "time": stamps[:t], "label": lab_tr}).to_csv(out / "labels_train.csv", index=False)
    pd.DataFrame({"time_index": np.arange(t), "time": stamps[t:], "label": lab_te}).to_csv(out / "labels_test.csv", index=False)
    ev_tr.to_csv(out / "events_train.csv", index=False)
    ev_te.to_csv(out / "events_test.csv", index=False)

    meta = dict(
        seed=a.seed, nodes=a.nodes, edges=len(edges), links=len(link_names), flows=len(flows),
        bins_per_week=t, bin_minutes=a.bin_minutes, mean_flow=a.mean_flow, noise=a.noise,
        median_flow_mean=float(np.median(means)), mean_link_load=float(Y[:t].mean()),
        anomaly_size_range_x_mean_flow=list(size_range),
        events_per_week=dict(spike=a.spikes, outage=a.outages, ddos=a.ddos),
        anomalous_bins=dict(train=int(lab_tr.sum()), test=int(lab_te.sum())),
        injection_sizes=dict(large=10.0 * a.mean_flow, small=2.0 * a.mean_flow),
        note="Y = X A^T as in Lakhina et al. section 3; X has gravity-model means, diurnal/weekly "
             "profiles with per-PoP time-zone shifts and log-normal noise.",
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"topology: {a.nodes} PoPs, {len(edges)} edges -> {len(link_names)} links, {len(flows)} OD flows")
    print(f"time: 2 weeks x {t} bins of {a.bin_minutes} min (week 1 = train, week 2 = test)")
    print(f"anomalous bins: train {lab_tr.sum()}, test {lab_te.sum()}")
    print(f"wrote {out}/: Y_train.csv Y_test.csv Y_clean.csv A.csv labels_*.csv events_*.csv meta.json")


if __name__ == "__main__":
    main()
