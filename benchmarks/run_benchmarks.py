#!/usr/bin/env python3
"""Runs every benchmark and writes raw CSVs to benchmarks/results/.

  python benchmarks/run_benchmarks.py [--build build] [--quick]

Experiments (all numbers are measured by the C++ binaries at run time):
  1. thread scaling   : t=32768, m=1024, threads 1..16, sequential baseline
  2. size sweep (m)   : t=16384, m in 128..2048
  3. size sweep (t)   : m=512,  t in 4096..65536
  4. roofline inputs  : STREAM triad bandwidth and FMA peak per thread count
  5. injection study  : full PCA re-fit per injected anomaly (synthetic data)
Then calls analyze_results.py to compute medians, speedup and efficiency.

Thread placement: OMP_PROC_BIND=close, OMP_PLACES=cores, so threads 1-4 run on
the four P-cores, 5-12 on the eight E-cores, and 13-16 on the P-cores' second
hardware threads (i5-13500H numbering).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def run(cmd, env):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def environment(build):
    info = dict(date=datetime.now().isoformat(timespec="seconds"), platform=platform.platform(),
                python=platform.python_version(), logical_cpus=os.cpu_count(),
                OMP_PROC_BIND="close", OMP_PLACES="cores")
    cache = read(Path(build) / "CMakeCache.txt") or ""
    for line in cache.splitlines():
        for key in ("CMAKE_CXX_COMPILER:", "CMAKE_BUILD_TYPE:", "PCA_NATIVE:"):
            if line.startswith(key):
                info[key.rstrip(":").lower()] = line.split("=", 1)[1]
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        info["cpu"] = next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name")), None)
    except OSError:
        pass
    freqs = {}
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        f = read(cpu / "cpufreq" / "cpuinfo_max_freq")
        if f:
            freqs[cpu.name] = int(f) / 1e6
    if freqs:
        info["cpuinfo_max_freq_ghz"] = freqs
        info["scaling_governor"] = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["ac_online"] = read(next(iter(Path("/sys/class/power_supply").glob("A*/online")), "/nonexistent")) \
        if Path("/sys/class/power_supply").exists() else None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(ROOT / "build"))
    ap.add_argument("--quick", action="store_true", help="small sizes, for checking the setup")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["scaling", "msweep", "tsweep", "roofline", "injection"])
    a = ap.parse_args()
    ext = ".exe" if os.name == "nt" else ""
    bench = Path(a.build) / f"bench{ext}"
    inj = Path(a.build) / f"injection_study{ext}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_PROC_BIND="close", OMP_PLACES="cores")
    env.pop("OMP_NUM_THREADS", None)

    ncpu = os.cpu_count() or 4
    threads = [p for p in (1, 2, 4, 6, 8, 12, 16) if p <= ncpu]
    tl = ",".join(map(str, threads))
    big = [p for p in (4, 8, 16) if p <= ncpu]
    q = a.quick

    (RESULTS / "environment.json").write_text(json.dumps(environment(a.build), indent=2))

    if "scaling" not in a.skip:
        run([bench, "pipeline", "--t", 4096 if q else 32768, "--m", 256 if q else 1024,
             "--threads", tl, "--reps", 2 if q else 5, "--out", RESULTS / "scaling.csv"], env)
    if "msweep" not in a.skip:
        for m in ((64, 128, 256) if q else (128, 256, 512, 1024, 2048)):
            run([bench, "pipeline", "--t", 4096 if q else 16384, "--m", m, "--threads", ",".join(map(str, big)),
                 "--reps", 2 if q else 3, "--out", RESULTS / f"msweep_m{m}.csv"], env)
    if "tsweep" not in a.skip:
        for t in ((2048, 8192) if q else (4096, 16384, 65536)):
            run([bench, "pipeline", "--t", t, "--m", 128 if q else 512, "--threads", ",".join(map(str, big)),
                 "--reps", 2 if q else 3, "--out", RESULTS / f"tsweep_t{t}.csv"], env)
    if "roofline" not in a.skip:
        run([bench, "stream", "--threads", tl, "--out", RESULTS / "stream.csv"]
            + (["--n", 1 << 22] if q else []), env)
        run([bench, "peak", "--threads", tl, "--out", RESULTS / "peak.csv"]
            + (["--iterations", 10_000_000] if q else []), env)
    if "injection" not in a.skip:
        data = ROOT / "datasets" / "synthetic"
        meta = json.loads((data / "meta.json").read_text())
        sizes = f"{meta['injection_sizes']['large']},{meta['injection_sizes']['small']}"
        # Wednesday of the clean training week (bins 288-431), as in the paper's
        # "every point in time over the period of a day", sampled every 2nd bin.
        run([inj, "--data", data / "Y_clean.csv", "--routing", data / "A.csv", "--sizes", sizes,
             "--time-start", 288, "--time-end", 300 if q else 432, "--time-step", 2,
             "--threads", tl, "--reps", 1 if q else 3, "--out", RESULTS], env)

    subprocess.run([sys.executable, str(ROOT / "benchmarks" / "analyze_results.py")], check=True)


if __name__ == "__main__":
    main()
