#!/usr/bin/env python3
"""Validates pca_seq and pca_omp against the NumPy reference (numpy_reference.py).

Generates small test matrices, runs both C++ executables, and compares
eigenvalues, r, delta^2, SPE and alarms. Exit code 0 = all comparisons pass.

  python tests/reference/compare_with_numpy.py --seq build/pca_seq --omp build/pca_omp
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy_reference as ref  # noqa: E402

RTOL = 1e-8


def make_case(t, m, seed, spikes):
    rng = np.random.default_rng(seed)
    k = np.arange(t)
    S = np.stack([np.sin(2 * np.pi * k / 144), np.cos(2 * np.pi * k / 144),
                  np.sin(2 * np.pi * k / 72), np.cos(2 * np.pi * k / 1008)], axis=1)
    Y = 1e6 + 1e5 * S @ rng.normal(size=(4, m)) + 1e4 * rng.normal(size=(t, m))
    for s in rng.choice(t, spikes, replace=False):
        Y[s, rng.choice(m, 4, replace=False)] += 2e5
    return Y


def write_csv(path, Y, offset=0):
    df = pd.DataFrame(Y, columns=[f"l{j}" for j in range(Y.shape[1])])
    df.insert(0, "time", np.arange(offset, offset + len(Y)))
    df.to_csv(path, index=False, float_format="%.17g")


def run(exe, args, out):
    cmd = [exe, *args, "--out", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    model = json.loads((out / "model.json").read_text())
    return model, pd.read_csv(out / "spe.csv")


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--omp", required=True)
    ap.add_argument("--workdir", default="build/numpy_reference")
    a = ap.parse_args()
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)

    cases = [  # (name, t, m, seed, spikes, extra args)
        ("sigma_rule_in_sample", 1008, 40, 1, 5, []),
        ("fixed_r_train_test", 1008, 60, 2, 5, ["--r-method", "fixed", "--r", "4"]),
        ("tall_small", 4000, 8, 3, 10, ["--alpha", "0.005"]),
        ("wide_t_lt_m", 50, 120, 4, 2, ["--r-method", "fixed", "--r", "3"]),
    ]
    failures = 0
    for name, t, m, seed, spikes, extra in cases:
        Y = make_case(t, m, seed, spikes)
        train_csv = work / f"{name}_train.csv"
        write_csv(train_csv, Y)
        args = ["--train", str(train_csv), *extra]
        Ytest = None
        if "train_test" in name:
            # train and test are consecutive halves of the same process
            both = make_case(2 * t, m, seed, 2 * spikes)
            Y, Ytest = both[:t], both[t:]
            write_csv(train_csv, Y)
            test_csv = work / f"{name}_test.csv"
            write_csv(test_csv, Ytest, offset=t)
            args += ["--test", str(test_csv)]

        r_method = "fixed" if "fixed" in extra else "sigma"
        fixed_r = int(extra[extra.index("--r") + 1]) if "--r" in extra else 4
        alpha = float(extra[extra.index("--alpha") + 1]) if "--alpha" in extra else 0.001
        model = ref.fit(Y, r_method=r_method, fixed_r=fixed_r, alpha=alpha)
        spe_ref = ref.spe(Y if Ytest is None else Ytest, model)
        lam_scale = model["eigenvalues"][0]

        for label, exe in (("seq", a.seq), ("omp", a.omp)):
            cm, cs = run(exe, args + (["--threads", "4"] if label == "omp" else []), work / f"{name}_{label}")
            lam = np.array(cm["eigenvalues"])
            checks = {
                "r": cm["r"] == model["r"],
                "eigenvalues": np.max(np.abs(lam - model["eigenvalues"])) <= RTOL * lam_scale,
                "delta2": abs(cm["delta2"] - model["delta2"]) <= 1e-6 * model["delta2"],
                "spe": rel(cs["spe"], spe_ref) <= 1e-6,
                "alarms": np.array_equal(cs["anomaly"].to_numpy(), (spe_ref > model["delta2"]).astype(int)),
            }
            ok = all(checks.values())
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} {name:22s} {label}: r={cm['r']} (ref {model['r']}), "
                  f"max|dlambda|/lambda1={np.max(np.abs(lam - model['eigenvalues'])) / lam_scale:.1e}, "
                  f"spe rel={rel(cs['spe'], spe_ref):.1e}, alarms={int(cs['anomaly'].sum())}"
                  + ("" if ok else f"  failed: {[k for k, v in checks.items() if not v]}"))
    print("all NumPy comparisons passed" if failures == 0 else f"{failures} comparison(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
