# Parallel PCA-Based Network-Wide Traffic Anomaly Detection Using OpenMP

A C++17 / OpenMP implementation of the **subspace method** from
A. Lakhina, M. Crovella, C. Diot, *Diagnosing Network-Wide Traffic Anomalies*, ACM SIGCOMM 2004.
HPC motif: **dense linear algebra**. Parallel framework: **OpenMP**.

Scope (as agreed for the one-day deadline): **detection only**. The paper's identification and
quantification steps are not implemented. The eigensolver is Eigen's serial solver.
One experiment is added: a parallel anomaly-injection study.

---

## 1. Quick start

Requirements: a C++17 compiler with OpenMP (GCC ≥ 9, Clang ≥ 10, or MSVC 2019+), CMake ≥ 3.18, and
Python ≥ 3.9. Eigen 3.4.0 is downloaded automatically (header-only, SHA-256-pinned) if it is not
installed on the system.

### Linux / macOS
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cmake -S . -B build -DPython3_EXECUTABLE=$PWD/.venv/bin/python
cmake --build build -j
(cd build && ctest --output-on-failure)                       # unit tests + NumPy reference

.venv/bin/python preprocessing/generate_synthetic_data.py     # -> datasets/synthetic/
./build/pca_seq --train datasets/synthetic/Y_train.csv --test datasets/synthetic/Y_test.csv --out results/synthetic_seq
./build/pca_omp --train datasets/synthetic/Y_train.csv --test datasets/synthetic/Y_test.csv --out results/synthetic_omp --threads 8
.venv/bin/python evaluation/evaluate_detection.py --spe results/synthetic_omp/spe.csv \
    --labels datasets/synthetic/labels_test.csv --events datasets/synthetic/events_test.csv \
    --out results/synthetic_omp/metrics.json

.venv/bin/python benchmarks/run_benchmarks.py                 # ~10 min; --quick for a 1-min check
cd visualization && ../.venv/bin/python plot_traffic.py && ../.venv/bin/python plot_anomalies.py \
    && ../.venv/bin/python plot_performance.py && ../.venv/bin/python plot_roofline.py
```

### Windows
Two options: MSVC (Visual Studio 2019 16.9+, OpenMP through `/openmp:llvm`), or MinGW-w64 GCC.
```bat
py -m venv .venv && .venv\Scripts\pip install -r requirements.txt
cmake -S . -B build -DPython3_EXECUTABLE=%CD%\.venv\Scripts\python.exe
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
build\Release\pca_omp.exe --train datasets\synthetic\Y_train.csv --test datasets\synthetic\Y_test.csv --out results\synthetic_omp --threads 8
```
The code uses portable APIs only. Peak memory and CPU time come from `GetProcessMemoryInfo` and
`GetProcessTimes` on Windows, and from `getrusage` elsewhere. It has been built and tested on Linux
only; Windows support has not been tested.

### Executables
| Executable | Purpose |
|---|---|
| `pca_seq` | detection, sequential kernels (compiled **without** OpenMP) |
| `pca_omp` | detection, OpenMP kernels (`--threads N`) |
| `injection_study` | parallel anomaly-injection experiment (sequential vs OpenMP) |
| `bench` | per-stage benchmarks (`pipeline`), memory bandwidth (`stream`), FMA peak (`peak`) |
| `pca_tests` | unit tests (also run by `ctest`) |

`pca_seq`/`pca_omp` options: `--train FILE` (required), `--test FILE`, `--out DIR`,
`--r-method sigma|fixed|variance`, `--sigma 3`, `--r 4`, `--variance 0.9`, `--alpha 0.001`,
`--top-links 3`, `--threads N`. Without `--test`, the model is fitted and scored on the same data,
as in the paper. Outputs: `model.json` (mean, eigenvalues, r, δ²), `spe.csv` (score per bin),
`alarms.csv` (alarm bins and the links with the largest residuals), and `timings.json` (per-stage
times, peak RSS, CPU time).

---

## 2. Repository layout
```
CMakeLists.txt            build: libraries pca_common / pca_sequential / pca_parallel + executables
include/
  matrix.hpp              Matrix type (Eigen, column-major), CSV/binary I/O, validation
  pca.hpp                 PcaModel, PcaConfig, eigendecomposition, r-selection rules
  anomaly_detection.hpp   Q-statistic, SPE result type, inverse normal CDF
  kernels.hpp             kernel interface: identical signatures in pca::seq and pca::omp
  kernels_inl.hpp         inner arithmetic (tiles / blocks) shared by both implementations
  pipeline.hpp            fit<Backend>() / score<Backend>(): the algorithm, written once
  injection.hpp           injection-study interface
src/common/               I/O, utilities, eigensolver wrapper, Q-statistic (serial, shared)
src/sequential/           sequential kernels + sequential injection loop
src/openmp/               OpenMP kernels + OpenMP injection loop
src/app/                  detect_main.cpp (built as pca_seq and pca_omp), injection_main.cpp
benchmarks/               benchmark.cpp, run_benchmarks.py, analyze_results.py, results/
preprocessing/            generate_synthetic_data.py
evaluation/               evaluate_detection.py (precision / recall / F1 / FPR)
tests/                    test_main.cpp (C++), reference/ (independent NumPy implementation)
visualization/            plot_*.py -> figures/
docs/PHASE1_DESIGN.md     paper analysis, ambiguities, design decisions
```
The detection driver and the pipeline are written **once**. They are templates over a backend
(`SeqKernels` or `OmpKernels`), so the sequential and parallel versions run the same algorithm.
The heavy kernels live in separate files and separate libraries, and the sequential library is
compiled without OpenMP.

---

## 3. Method

### 3.1 What the paper does (Lakhina et al. 2004)
- **Measurement model.** Link traffic is a superposition of origin–destination (OD) flows,
  `y = A x`, where A is the routing matrix (links × OD flows). The paper's link counts were
  themselves built from OD-flow counts through A (§3).
- **Matrix orientation.** `Y` is **t × m**: rows are time bins (10 min, t = 1008 per week) and
  columns are links (m = 41 or 49).
- **PCA.** Centre each column, then compute the eigenvectors `v_i` of `YᵀY`, the principal axes.
  Link traffic is low-dimensional: 3–4 axes capture most of the variance.
- **Subspace split (3σ rule, §4.3).** Project the data onto each axis, `u_i = Y v_i / ‖Y v_i‖`.
  The first axis whose projection deviates more than 3σ from its mean, and every later axis, form
  the *anomalous* subspace. The earlier axes form the *normal* subspace `S = span(P)`, P = [v₁…v_r].
- **Detection (§5.1).** `ỹ = (I − PPᵀ) y`, `SPE = ‖ỹ‖²`. An alarm fires if `SPE > δ²_α`, where δ²_α
  is the Jackson–Mudholkar **Q-statistic**:

  δ²_α = φ₁ [ c_α √(2φ₂h₀²)/φ₁ + 1 + φ₂h₀(h₀−1)/φ₁² ]^(1/h₀),  h₀ = 1 − 2φ₁φ₃/(3φ₂²),  φᵢ = Σ_{j>r} λⱼⁱ
- **Not implemented here:** identification of the responsible OD flow, quantification of its size,
  and the paper's validation against real Sprint/Abilene data.

### 3.2 What we implement (simplified, practical version)
| Step | Equation | Kernel | Parallel? |
|---|---|---|---|
| 1 column means | μⱼ = (1/t) Σₖ Yₖⱼ | `column_means` | OpenMP, over links |
| 2 centring | Y ← Y − 1μᵀ | `center_columns` | OpenMP, over links |
| 3 covariance | Σ = YᵀY/(t−1) | `covariance` | OpenMP, over 64×64 output tiles |
| 4 eigendecomposition | Σ = VΛVᵀ | Eigen `SelfAdjointEigenSolver` | **serial in both builds** |
| 5 choose r | 3σ rule on uᵢ = Y vᵢ (or fixed r / variance %) | `project`, `max_sigma_deviation` | OpenMP (rows; reductions) |
| 6 threshold | Q-statistic on λ_{r+1..m} | `q_statistic` | serial, O(m) |
| 7 score | SPEₖ = ‖(I − PPᵀ) yₖ‖² (fused projection + reconstruction + residual) | `spe_scores` | OpenMP, over time-row blocks |
| 8 detect | flag k if SPEₖ > δ² | `apply_threshold` | OpenMP, integer reduction |

Decisions where the paper is ambiguous (details in `docs/PHASE1_DESIGN.md` §3):
- **Scale of λ.** The paper writes λⱼ = ‖Y vⱼ‖². We use the covariance eigenvalues ‖Y vⱼ‖²/(t−1),
  which are on the same per-sample scale as the SPE. A test shows the literal reading gives a
  threshold about t times too high, so nothing is ever detected.
- **3σ rule.** Default k = 3, configurable. Axes with zero numerical variance are never tested.
  r is clamped to [1, m−1].
- **h₀ ≤ 0.** For very skewed residual spectra the Jackson–Mudholkar formula is undefined. We then
  use Box's approximation SPE ≈ g·χ²_h (g = φ₂/φ₁, h = φ₁²/φ₂, Wilson–Hilferty quantile).
  `model.json` records which formula was used.
- **Fit/apply.** Both the paper's in-sample mode and a train/test mode are supported. In train/test
  mode μ, P and δ² come from the training week only. **Labels are never used** for fitting or for
  the threshold; they are read only by `evaluate_detection.py`.

### 3.3 Matrix orientation
- The **mathematics** uses the paper's convention, Y ∈ ℝ^{t×m}.
- **Memory** uses Eigen's default column-major layout, so each link's time series is contiguous.
  Physically that is the "link × time" matrix of the project registration.
- The expensive kernels (means, centring, covariance dot products) therefore stream through memory
  with unit stride.

### 3.4 Data
`preprocessing/generate_synthetic_data.py` reproduces the paper's construction `Y = X Aᵀ`:
- 12 PoPs on a ring with 6 random chords, giving 18 edges, **60 links** (36 directed internal links
  plus 12 ingress and 12 egress links) and **144 OD flows**;
- shortest-path routing matrix A;
- gravity-model OD means, diurnal profiles shifted by per-PoP time zones, a weekend dip, and
  log-normal noise;
- labelled anomalies: single-flow **spikes**, single-flow **outages**, and multi-flow **DDoS**
  (4–6 origins sending to one destination for 2–6 bins);
- two weeks of 1008 ten-minute bins: week 1 trains, week 2 tests.
  `Y_clean.csv` is week 1 without anomalies, for the injection study.

The benchmark harness generates its own larger matrices (up to 65 536 × 2048) from a fixed seed:
low-rank daily, half-daily and weekly patterns plus noise and spikes. These are **benchmark
workloads, not a network model**.

---

## 4. OpenMP design

- **Parallelise over outputs, not over the reduction dimension.** Every column mean, covariance tile
  and SPE value is computed by exactly one thread, with the same arithmetic as the sequential code
  (`kernels_inl.hpp`). So there are no locks, atomics or shared accumulators, and **the OpenMP
  results are bitwise identical to the sequential ones** for any thread count. This is checked in
  `pca_tests` and in every benchmark run.
- **Reductions** are used where a single scalar is produced:
  - the alarm count (`reduction(+)`, exact);
  - the mean, variance and maximum deviation in the 3σ test (`reduction(+)` and `reduction(max)`).
    These can differ from the sequential results in the last bits; only the chosen r has to match,
    and the tests check that.
- **Scheduling:**
  - `static` for uniform work (means, centring, row blocks);
  - `dynamic,1` for the covariance tiles and the injection experiments. Upper-triangular tiles do
    unequal work, and this CPU mixes fast P-cores with slower E-cores.
- **Cache blocking (covariance).**
  - The kernel uses 64-column tiles × 512-row chunks, about 512 KiB of column panels, which fits in
    L2.
  - It is register-blocked: one load of `a[k]` feeds four dot products.
  - Without blocking the kernel would be memory-bound at 0.125 FLOP/byte; with it each loaded value
    is reused about 64 times.
- **Kernel fusion (scoring).** Projection, reconstruction, residual and SPE are computed in one pass
  over each row block. The reconstructed matrix Ŷ (t × m) is never written to memory.
- **No oversubscription or nesting.** `EIGEN_DONT_PARALLELIZE` stops Eigen from starting its own
  threads, and no threaded BLAS is linked. In the injection study only the *outer* loop is parallel,
  and each experiment runs the sequential kernels.
- **Thread control:** `--threads N`, or `OMP_NUM_THREADS`. Benchmarks use `OMP_PROC_BIND=close` and
  `OMP_PLACES=cores`.
- **Not parallelised:**
  - the symmetric eigendecomposition: Householder tridiagonalisation and QR iterations are chains
    of dependent steps;
  - the Q-statistic (O(m));
  - file I/O.

  The eigensolver is the **Amdahl serial fraction**, and the benchmarks report it explicitly.

---
## 5. Results (measured on this machine)

**Machine:**
- Intel Core i5-13500H: 4 P-cores (8 hardware threads) and 8 E-cores, 16 logical CPUs.
- The firmware caps P-cores at **2.6 GHz** and E-cores at **1.9 GHz** even on AC power
  (`cpuinfo_max_freq`); changing this needs root.
- 15 GiB RAM, of which about 4 GiB was free.
- Software: GCC 16.2, `-O3 -march=native`, Eigen 3.4.0, Linux 7.2.
- Thread placement: `OMP_PROC_BIND=close OMP_PLACES=cores`. Threads 1–4 run on P-cores, 5–12 on
  E-cores, and 13–16 on the P-cores' second hardware threads.
- Every timing is the **median** of 3–5 runs after one warm-up run.

Raw data: `benchmarks/results/`. Re-run with `python benchmarks/run_benchmarks.py`.

### 5.1 Correctness
| Check | Result |
|---|---|
| C++ unit tests (`pca_tests`) | 1208 checks, 0 failures |
| OpenMP vs sequential, all benchmark configurations | **31/31 bitwise-identical SPE**, same r, same alarms (max rel. diff 0) |
| NumPy SVD reference (4 cases × 2 binaries) | eigenvalues within ≤ 2.6e-15·λ₁, SPE within ≤ 2.9e-12 relative, identical r and alarms |
| Q-statistic, Gaussian data, t = 200 000 | empirical false-alarm rate 0.967 % (α = 1 %), 0.105 % (α = 0.1 %) |
| Injection study, 20 736 experiments × 7 thread counts | all outcomes identical to sequential |

### 5.2 Detection quality (synthetic network, 60 links, 1008 bins/week)
Trained on week 1, tested on held-out week 2 (`--r-method sigma`, α = 0.001). Labels used only for
scoring:

| Setting | r | Alarms | Precision | Recall | F1 | FPR | Events detected |
|---|---|---|---|---|---|---|---|
| train week 1 → test week 2 | 2 | 17 | 1.000 | 0.773 | 0.872 | 0.000 | spikes 10/10, DDoS 3/3, outages 0/3 |
| in-sample on week 2 (paper's setting) | 1 | 14 | 1.000 | 0.636 | 0.778 | 0.000 | spikes 7/10, DDoS 3/3, outages 0/3 |

- **Outages** are small negative changes (9–22 MB, compared with 65–130 MB spikes) and are not
  detected. The paper's detectability condition (§5.4) explains this: a small anomaly projects too
  little onto the residual subspace.
- **In-sample fitting is worse.** The anomalies pull the principal axes towards themselves, as the
  paper notes in §5.4.
- **False alarms are above nominal on clean data.** Fitting and scoring the anomaly-free week gives
  27/1008 alarms (2.7 %), against α = 0.1 %. The synthetic residuals are not Gaussian (multiplicative
  noise is larger at peak hours), and that breaks the Q-statistic's assumption.
- **Threshold formula.** Because the spectrum is very skewed (λ₁ holds 95 % of the variance),
  h₀ < 0, so the Box χ² fallback was used instead of Jackson–Mudholkar.

**Injection study.** In the anomaly-free week 1, each of 144 OD flows is injected in turn at every
2nd bin of Wednesday (72 bins). Each injection is followed by a full PCA re-fit, giving
20 736 experiments in total:
- large injections (10⁸ B, 10 × mean flow): **detected in 100.0 %** (10 364/10 368);
- small injections (2·10⁷ B): detected in 24.0 % (2 493/10 368).

### 5.3 Execution time, speedup, efficiency
Speedup is S(p) = T_seq / T_omp(p), where **T_seq is the separate sequential binary** (no OpenMP
runtime). Efficiency is E(p) = S(p)/p.

**Thread scaling, t = 32 768, m = 1024** (`figures/stage_times.png`, `speedup.png`, `efficiency.png`):

| Threads | Total (s) | Covariance (s) | Eigen (s, serial) | Score (s) | S(p) total | E(p) total | S(p) covariance | S(p) score |
|---|---|---|---|---|---|---|---|---|
| seq | 3.715 | 2.527 | 0.829 | 0.177 | 1.00 | 100 % | 1.00 | 1.00 |
| omp 1 | 3.742 | 2.557 | 0.828 | 0.184 | 0.99 | 99 % | 0.99 | 0.96 |
| omp 2 | 2.319 | 1.295 | 0.821 | 0.100 | 1.60 | 80 % | 1.95 | 1.77 |
| omp 4 | 1.595 | 0.645 | 0.829 | 0.053 | 2.33 | 58 % | 3.92 | 3.31 |
| omp 6 | 1.522 | 0.589 | 0.822 | 0.048 | 2.44 | 41 % | 4.29 | 3.69 |
| omp 8 | 1.564 | 0.601 | 0.828 | 0.051 | 2.38 | 30 % | 4.20 | 3.44 |
| omp 12 | 1.461 | 0.546 | 0.836 | 0.033 | 2.54 | 21 % | 4.63 | 5.41 |
| omp 16 | 1.535 | 0.555 | 0.858 | 0.036 | 2.42 | 15 % | 4.56 | 4.99 |

Amdahl bound with the serial eigensolver: f = 0.829 / 3.715 = 22 %, so S(16) ≤ 1/(0.22 + 0.78/16)
= 3.68. Peak memory: 542 MiB, which is the input matrix plus one working copy.

**Parallel injection study** (20 736 independent PCA fits, outer-loop parallelism):

| Threads | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| time (s), seq = 12.89 | 12.64 | 6.11 | 3.02 | 2.52 | 2.40 | 2.07 | 1.78 |
| speedup | 1.02 | 2.11 | 4.27 | 5.13 | 5.38 | 6.24 | **7.23** |
| efficiency | 102 % | 106 % | 107 % | 85 % | 67 % | 52 % | 45 % |

**Size sweeps** (`figures/size_scaling.png`): speedup at 16 threads.

| t × m | 16384×128 | 16384×256 | 16384×512 | 16384×1024 | 16384×2048 | 4096×512 | 65536×512 |
|---|---|---|---|---|---|---|---|
| sequential total (s) | 0.031 | 0.105 | 0.410 | 2.03 | 13.77 | 0.168 | 1.372 |
| S(16) end-to-end | 1.39 | 2.62 | 2.45 | 1.90 | 1.51 | 1.41 | **3.23** |
| S(16) covariance | 1.31 | 3.57 | 4.49 | **5.40** | 5.00 | 4.49 | 4.01 |
| eigen share of T_seq | 5 % | 10 % | 22 % | 40 % | **60 %** | 52 % | 6 % |
| peak RSS (MiB) | 38 | 72 | 139 | 285 | 614 | 43 | 529 |

**At the paper's scale** (1008 × 60), the whole fit and score takes **0.76 ms sequentially and
2.3 ms with 8 OpenMP threads**. That is slower, because thread start-up and scheduling cost more
than the work itself. Parallelism only pays off for large matrices or many re-fits.

### 5.4 Interpretation
1. **Covariance (the O(tm²) kernel) scales almost perfectly on the 4 P-cores**: 3.92× at 4 threads.
   Beyond 4 threads, gains are small, up to 4.6–5.4×. The FMA microbenchmark shows why: an E-core
   delivers about 14.3 GFLOP/s, against 36.4 GFLOP/s for a P-core.
   - 6 threads with equal work measure 85.6 GFLOP/s = 6 × 14.3, so the slowest core sets the pace.
   - The machine's whole compute capacity is about 4 + 8 × 0.39 ≈ 7.1 "P-core equivalents", not 12.
   - `schedule(dynamic,1)` lets the fast cores take more tiles, which is why the covariance kernel
     keeps improving past 4 threads at all.
2. **End-to-end speedup is limited by Amdahl's law.** Eigen's eigensolver is serial and O(m³). Its
   share of the sequential time grows with m, from 5 % at m = 128 to 60 % at m = 2048, so the
   end-to-end speedup *falls* as m grows even though the covariance speedup rises. Growing t instead
   (more time bins, same m) shrinks the serial share and gives the best end-to-end result (3.23×).
3. **Static-scheduled kernels dip at 6–8 threads.** For score and means, the slowest E-core thread
   decides when the loop ends. The dip is visible in the scoring curve.
4. **The injection study scales best**: 7.23× at 16 threads. It has no serial fraction, each fit is
   independent, and `dynamic` scheduling balances P-cores against E-cores. Efficiency slightly
   above 100 % at 2–4 threads is within run-to-run noise, plus less frequency drop with fewer
   active cores.
5. **CPU utilisation** (CPU time / (wall × p), in `summary_pipeline.csv`) falls to 37 % at 16 threads
   for t = 32 768, m = 1024. Most of that is the serial eigensolver phase, when one thread works and
   the others are idle.

---

## 6. Roofline analysis

**Model:** attainable performance P = min(P_peak, B × AI), where AI = FLOPs / bytes.

**Ceilings, measured by `bench`** (`benchmarks/results/peak.csv`, `stream.csv`):

| | 1 thread | all threads (best) |
|---|---|---|
| FP64 peak, FMA microbenchmark | 36.4 GFLOP/s | 225.6 GFLOP/s (16 thr) |
| Memory bandwidth, STREAM triad (32 B/elem = 2 reads + 1 write + write-allocate) | 24.7 GB/s | 61.2 GB/s (16 thr) |
| Ridge point P_peak / B | 1.47 FLOP/B | 3.69 FLOP/B |

**Theoretical peak (an assumption, not measured):**
4 P-cores × 16 FLOP/cycle (2 × 256-bit FMA) × 2.6 GHz + 8 E-cores × 8 FLOP/cycle × 1.9 GHz
= **288 GFLOP/s**. The measured 225.6 GFLOP/s is 78 % of it. The theoretical DRAM bandwidth is not
stated, because the memory type cannot be read without root.

**Kernels at t = 32 768, m = 1024.** FLOPs and *compulsory* bytes (each matrix element moved once)
are counted analytically in `analyze_results.py`; times are measured. File:
`benchmarks/results/roofline_table.csv`, figure: `figures/roofline.png`.

| Kernel | FLOPs | Bytes | AI | Bound | Seq GFLOP/s (% of roof) | OMP 12 thr GFLOP/s (% of roof) |
|---|---|---|---|---|---|---|
| means | tm | 8tm | 0.125 | memory | 2.06 (67 %) | 6.37 (83 %) |
| center | tm | 16tm | 0.0625 | memory | 1.60 (103 %) | 3.33 (87 %) |
| covariance | t·m(m+1) | 8tm + 8m² | **124** | **compute** | 13.6 (37 %) | 63.0 (28 %) |
| score (fused, r = 6) | (4r+5)·tm | 8tm | 3.6 | memory (at the ridge) | 5.5 (15 %) | 29.7 (13 %) |

- **Means and centring** are memory-bound: about 1/8 and 1/16 FLOP per byte. They already reach
  67–103 % of the bandwidth roof, so more threads only help until the memory bus is saturated.
  Centring at 103 % of the single-thread roof is within the noise between a best-of-10
  microbenchmark and a median kernel time.
- **Covariance** has AI ≈ m/8 ≈ 124 FLOP/B, far right of the ridge, so it is **compute-bound** and
  cache blocking makes that AI achievable. It reaches 37 % of the single-core peak, and 28 % with
  12 threads, where the E-cores and the shared L2 pull the average down. The remaining gap is in
  the micro-kernel: a 1 × 4 register block needs 5 loads per 4 vector FMAs, so it is limited by load
  ports rather than FMA units. Tuned BLAS libraries use larger 2-D register blocks (e.g. 4 × 12) and
  packed panels to reach 80–90 %. That is the clear next optimisation; it is not done here.
- **Scoring** sits near the ridge (AI 3.6 with r = 6). It is well below its roof because its inner
  loops are short (r = 6 columns) and it makes two passes over each row block.
- **Theoretical vs measured.** The roof lines are measured. The 288 GFLOP/s line is a vendor-based
  estimate shown for reference only. All kernel points are measured.

---

## 7. Limitations
- **Detection only.** The paper's identification and quantification steps are not implemented.
- **Synthetic data only.** Real datasets (Abilene traffic matrices, CICIDS2017) were planned in
  `docs/PHASE1_DESIGN.md` but left out to meet the one-day deadline. Detection numbers describe the
  synthetic generator, not real networks.
- **Gaussian assumption.** The Q-statistic assumes Gaussian residuals. On the synthetic data the
  false-alarm rate on clean data was 2.7 % instead of 0.1 %, and h₀ < 0 forced the Box χ² fallback.
- **3σ rule.** It selected r = 1–2 here, not the paper's r = 4, and it is sensitive to anomalies in
  the training data.
- **Serial eigensolver.** Eigen's eigensolver is serial by design (the agreed scope). It caps
  end-to-end speedup (Amdahl), most severely at large m.
- **Benchmark matrices** are low-rank plus noise. They are workloads, not network models.
- **Hardware.** One laptop with firmware-capped clocks and a hybrid P/E-core CPU. Results will differ
  on homogeneous server CPUs. Windows builds are supported by the code but untested.
- **Covariance efficiency.** The hand-written kernel reaches only 28–37 % of peak; a tuned BLAS
  routine (`dsyrk`) would be faster. We used our own kernel deliberately, so the parallelism being
  measured is ours.

---

## 8. Viva preparation

**Q: What does the paper do, in one sentence?**
It uses PCA on all link time series together to split traffic into a low-dimensional "normal"
subspace and a residual subspace, and raises an alarm when the residual energy (SPE) of a time bin
exceeds a statistical threshold (the Q-statistic).

**Q: Why does an anomaly show up in the residual?**
Normal traffic is dominated by a few shared daily and weekly patterns, so it lies close to S. An
anomalous OD flow adds traffic along its own direction θ = Aᵢ/‖Aᵢ‖ in link space, which is generally
not inside S. That added traffic therefore appears in ỹ = (I − PPᵀ)y.

**Q: Why centre the columns, and why not standardise them?**
Centring makes PCA measure variance around the mean rather than the mean itself. The paper does not
standardise, because it wants the true byte variance: standardising would inflate quiet links.

**Q: What is the matrix orientation?**
Mathematically Y is t × m, rows are time and columns are links, as in the paper. In memory it is
column-major, so each link's time series is contiguous; that is the "link × time" layout. Σ = YᵀY/(t−1)
is m × m in either view.

**Q: How is the threshold computed? Is it tuned on labels?**
It is not tuned. δ²_α comes from the Q-statistic, using φ₁, φ₂, φ₃ (sums of powers of the residual
eigenvalues) and c_α (the normal quantile for a false-alarm rate α). We verified the empirical
false-alarm rate (0.97 % for α = 1 %) on Gaussian data.

**Q: What ambiguity did you find in the paper?**
The paper writes λ = ‖Yv‖². That is t times larger than a covariance eigenvalue, while SPE is a
single-row quantity, so used literally it would give a threshold about 1000× too high. We use
λ/(t−1), and a test demonstrates the difference.

**Q: Which operations did you parallelise, and how?**
- Means and centring: over links, `static`.
- Covariance: over 64×64 output tiles of the upper triangle, `dynamic,1`.
- Projection and SPE scoring: over blocks of time rows, `static`.
- 3σ statistics: `reduction(+)` and `reduction(max)`.
- Alarm counting: `reduction(+)`.
- Injection study: over experiments, `dynamic`.

**Q: Why are your OpenMP results bitwise identical to the sequential ones?**
Each output element is computed by one thread with exactly the same sequence of floating-point
operations as in the sequential code. Only the assignment of outputs to threads changes. A
reduction that splits a sum across threads would change the rounding order; we avoided that for the
numerical results.

**Q: Why not parallelise the reduction over t in the covariance?**
That would need one private m×m accumulator per thread (16 × 8 MB at m = 1024), a merge step, and it
would give results that depend on the thread count. Splitting by output tile needs none of this.

**Q: Why is the eigensolver not parallel?**
Householder tridiagonalisation and QR iterations are sequences of dependent steps. Only
fine-grained BLAS-2 parallelism exists inside each step, and it scales poorly. It is O(m³) against
the covariance's O(tm²), so it becomes the Amdahl bottleneck for large m: 60 % of runtime at
m = 2048, which limits end-to-end speedup to about 1.5×.

**Q: Explain your speedup curve.**
- Up to 4 threads (P-cores), covariance scales almost linearly (3.92×).
- From 5 to 12 threads the work moves onto E-cores, each about 2.5× slower (measured 14.3 against
  36.4 GFLOP/s), so the gains shrink.
- From 13 to 16 threads the P-cores' second hardware threads share execution units, so the gain is
  small.
- End-to-end speedup is further capped by the 22 % serial eigensolver: Amdahl bound 3.68 at 16
  threads, measured 2.42.

**Q: Is the covariance compute-bound or memory-bound?**
Compute-bound: AI ≈ m/8 = 124 FLOP/B, far above the ridge point of 3.7 FLOP/B. Means and centring are
memory-bound (AI 0.125 and 0.0625).

**Q: Why does OpenMP make the paper-sized problem slower?**
At 1008 × 60 the entire computation takes 0.76 ms. Starting threads and scheduling cost more than
the work, so OpenMP took 2.3 ms. Parallelism helps for large m and t, or for many independent fits
such as the injection study.

**Q: How did you avoid oversubscription with Eigen?**
`EIGEN_DONT_PARALLELIZE` is defined globally and no threaded BLAS is linked. In the injection study
only the outer loop is parallel and the inner kernels are sequential.

**Q: How did you validate correctness?**
- Hand-computed covariance on a 4×3 matrix.
- Comparison against Eigen's own matrix products.
- An independent NumPy SVD implementation.
- Perfectly correlated data (rank 1).
- An injected spike that must be detected.
- A 1e9 offset added to every value (numerical stability).
- Invalid input (NaN/Inf, ragged CSV, empty file).
- Bitwise equality between sequential and OpenMP at 1, 3, 8 and 16 threads.

**Q: What would you improve next?**
- A 2-D register-blocked covariance micro-kernel, or tuned BLAS `dsyrk` as a comparison.
- A parallel top-k eigensolver (subspace iteration), to remove the Amdahl bottleneck.
- Real data (Abilene traffic matrices).
- The identification step.
