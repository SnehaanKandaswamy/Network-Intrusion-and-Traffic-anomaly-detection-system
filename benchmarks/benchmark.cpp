// Benchmark harness.
//
// Modes:
//   bench pipeline --t 32768 --m 1024 --threads 1,2,4,8 --reps 5 --out file.csv
//       Generates a synthetic t x m traffic matrix (low-rank periodic structure
//       + noise + spikes, fixed seed), then times every pipeline stage for the
//       sequential implementation and for the OpenMP implementation at each
//       thread count. Also checks the OpenMP results against the sequential
//       ones (written to <out>.equivalence.csv).
//   bench stream --threads 1,2,4,8 --out file.csv
//       STREAM-style triad a = b + s*c to measure sustainable memory bandwidth.
//   bench peak --threads 1,2,4,8 --out file.csv
//       Register-resident FMA loop to measure attainable FP64 throughput.
//
// Every number written is measured on this machine at run time.

#include "kernels.hpp"
#include "pipeline.hpp"
#include "utils.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {

// Deterministic, platform-independent random numbers (SplitMix64 + Box-Muller),
// so the benchmark matrix is the same on every compiler / OS.
class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
    double normal() {
        const double u1 = uniform(), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

private:
    std::uint64_t state_;
};

// Benchmark workload: 10-minute bins, daily/half-daily/weekly patterns with a
// per-link mix, a per-link baseline, Gaussian noise and a few volume spikes.
pca::Matrix make_traffic(pca::Index t, pca::Index m, std::uint64_t seed) {
    Rng rng(seed);
    const double two_pi = 6.283185307179586;
    const double periods[3] = {144.0, 72.0, 1008.0};
    constexpr int kFactors = 6;
    pca::Matrix weights(kFactors, m);
    pca::Vector base(m), noise(m);
    for (pca::Index j = 0; j < m; ++j) {
        base(j) = 1e7 * (1.0 + 4.0 * rng.uniform());
        for (int f = 0; f < kFactors; ++f) weights(f, j) = base(j) * 0.25 * rng.normal() / (1 + f / 2);
        noise(j) = 0.02 * base(j);
    }
    pca::Matrix Y(t, m);
    for (pca::Index j = 0; j < m; ++j)
        for (pca::Index k = 0; k < t; ++k) {
            double v = base(j);
            for (int f = 0; f < kFactors; ++f) {
                const double phase = two_pi * static_cast<double>(k) / periods[f / 2];
                v += weights(f, j) * (f % 2 ? std::cos(phase) : std::sin(phase));
            }
            Y(k, j) = v + noise(j) * rng.normal();
        }
    const pca::Index n_spikes = std::max<pca::Index>(1, t / 2000);
    for (pca::Index s = 0; s < n_spikes; ++s) {
        const pca::Index k = static_cast<pca::Index>(rng.next() % static_cast<std::uint64_t>(t));
        for (int l = 0; l < 4; ++l) {
            const pca::Index j = static_cast<pca::Index>(rng.next() % static_cast<std::uint64_t>(m));
            Y(k, j) += 0.5 * base(j);
        }
    }
    return Y;
}

struct RunRecord {
    pca::StageTimes times;
    double wall = 0, cpu = 0;
    int r = 0;
    pca::Index alarms = 0;
    pca::Vector spe;
    pca::Vector eigenvalues;
    pca::Vector mean;
};

template <class K>
RunRecord run_pipeline(const pca::Matrix& original, pca::Matrix& work, const pca::PcaConfig& config) {
    work = original;  // untimed copy: every run starts from identical raw data
    RunRecord rec;
    const double cpu0 = pca::process_cpu_seconds();
    pca::Timer wall;
    const pca::PcaModel model = pca::fit<K>(work, config, rec.times);
    const pca::DetectionResult result = pca::score<K>(work, model, rec.times);
    rec.wall = wall.seconds();
    rec.cpu = pca::process_cpu_seconds() - cpu0;
    rec.r = model.r;
    rec.alarms = result.n_alarms;
    rec.spe = result.spe;
    rec.eigenvalues = model.eigenvalues;
    rec.mean = model.mean;
    return rec;
}

double max_rel_diff(const pca::Vector& a, const pca::Vector& b) {
    const double scale = std::max(a.cwiseAbs().maxCoeff(), 1e-300);
    return (a - b).cwiseAbs().maxCoeff() / scale;
}

int run_pipeline_mode(const pca::Args& args) {
    const pca::Index t = args.get_int("t", 32768);
    const pca::Index m = args.get_int("m", 512);
    const std::vector<int> threads = pca::parse_int_list(args.get("threads", "1,2,4,8"));
    const int reps = static_cast<int>(args.get_int("reps", 5));
    const std::string out = args.require("out");
    pca::PcaConfig config;
    config.method = pca::parse_rmethod(args.get("r-method", "sigma"));
    config.fixed_r = static_cast<int>(args.get_int("r", 6));

    std::cerr << "generating " << t << " x " << m << " matrix (" << std::fixed << std::setprecision(1)
              << 8.0 * double(t) * double(m) / 1048576.0 << " MiB)\n";
    const pca::Matrix original = make_traffic(t, m, 20040830ull + static_cast<std::uint64_t>(m));
    pca::Matrix work(t, m);

    std::ofstream csv(out);
    csv << "t,m,backend,threads,rep,r,alarms,means,center,covariance,eigen,select,score,fit_total,"
           "total,cpu_seconds,cpu_utilisation\n";
    std::ofstream eq(out + ".equivalence.csv");
    eq << "t,m,threads,max_rel_diff_mean,max_rel_diff_eigenvalues,max_rel_diff_spe,bitwise_identical_spe,"
          "same_r,same_alarms\n";
    csv << std::setprecision(9);
    eq << std::setprecision(6);

    auto write = [&](const char* backend, int p, int rep, const RunRecord& rec) {
        const auto& s = rec.times;
        csv << t << ',' << m << ',' << backend << ',' << p << ',' << rep << ',' << rec.r << ',' << rec.alarms << ','
            << s.means << ',' << s.center << ',' << s.covariance << ',' << s.eigen << ',' << s.select << ','
            << s.score << ',' << s.fit_total << ',' << rec.wall << ',' << rec.cpu << ','
            << rec.cpu / (rec.wall * p) << '\n';
    };

    // Sequential baseline: one warm-up run (not recorded) + reps runs.
    RunRecord reference = run_pipeline<pca::SeqKernels>(original, work, config);
    for (int rep = 0; rep < reps; ++rep) {
        const RunRecord rec = run_pipeline<pca::SeqKernels>(original, work, config);
        write("sequential", 1, rep, rec);
        std::cerr << "  seq       rep " << rep << ": " << std::setprecision(3) << rec.wall << " s\n";
    }

    for (const int p : threads) {
        omp_set_num_threads(p);
        const RunRecord first = run_pipeline<pca::OmpKernels>(original, work, config);  // warm-up
        eq << t << ',' << m << ',' << p << ',' << max_rel_diff(reference.mean, first.mean) << ','
           << max_rel_diff(reference.eigenvalues, first.eigenvalues) << ','
           << max_rel_diff(reference.spe, first.spe) << ',' << int(reference.spe == first.spe) << ','
           << int(reference.r == first.r) << ',' << int(reference.alarms == first.alarms) << '\n';
        for (int rep = 0; rep < reps; ++rep) {
            const RunRecord rec = run_pipeline<pca::OmpKernels>(original, work, config);
            write("openmp", p, rep, rec);
            std::cerr << "  omp p=" << std::setw(2) << p << " rep " << rep << ": " << rec.wall << " s\n";
        }
    }
    std::cerr << "peak RSS " << pca::peak_rss_mb() << " MiB\n";
    std::ofstream(out + ".meta.json") << "{\"t\": " << t << ", \"m\": " << m
                                      << ", \"peak_rss_mb\": " << pca::peak_rss_mb()
                                      << ", \"r\": " << reference.r << "}\n";
    return 0;
}

int run_stream_mode(const pca::Args& args) {
    const std::vector<int> threads = pca::parse_int_list(args.get("threads", "1,2,4,8"));
    const pca::Index n = args.get_int("n", 1 << 25);  // 3 arrays x 256 MiB >> 18 MiB L3
    const int reps = static_cast<int>(args.get_int("reps", 10));
    std::unique_ptr<double[]> a(new double[static_cast<std::size_t>(n)]);
    std::unique_ptr<double[]> b(new double[static_cast<std::size_t>(n)]);
    std::unique_ptr<double[]> c(new double[static_cast<std::size_t>(n)]);
    std::ofstream csv(args.require("out"));
    csv << "threads,best_seconds,bandwidth_gbs_24B,bandwidth_gbs_32B\n";
    for (const int p : threads) {
        omp_set_num_threads(p);
#pragma omp parallel for schedule(static)  // first touch by the same threads
        for (pca::Index i = 0; i < n; ++i) {
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 2.0;
        }
        double best = 1e300;
        for (int rep = 0; rep < reps; ++rep) {
            const double s = 0.5 + 1e-3 * rep;
            pca::Timer timer;
#pragma omp parallel for schedule(static)
            for (pca::Index i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
            best = std::min(best, timer.seconds());
        }
        double check = 0.0;
        for (pca::Index i = 0; i < n; i += 4096) check += a[i];
        // 24 B/element counts the bytes the kernel names (2 reads + 1 write);
        // 32 B/element adds the write-allocate read of `a` done by the cache.
        const double gb24 = 24.0 * double(n) / best / 1e9, gb32 = 32.0 * double(n) / best / 1e9;
        csv << p << ',' << best << ',' << gb24 << ',' << gb32 << '\n';
        std::cerr << "stream p=" << p << ": " << gb24 << " GB/s (24 B/elem), " << gb32
                  << " GB/s (32 B/elem)   [check " << check << "]\n";
    }
    return 0;
}

// 48 independent multiply-add chains (12 AVX2 registers) per thread: enough
// independent work to keep both FMA pipes busy despite 4-cycle FMA latency.
double fma_loop(long long iterations, double seed) {
    constexpr int kChains = 48;
    double x[kChains];
    for (int j = 0; j < kChains; ++j) x[j] = seed + 1e-3 * j;
    const double mul = 0.999999999, add = 1e-9;
    for (long long it = 0; it < iterations; ++it) {
#pragma omp simd
        for (int j = 0; j < kChains; ++j) x[j] = x[j] * mul + add;
    }
    double sum = 0.0;
    for (int j = 0; j < kChains; ++j) sum += x[j];
    return sum;
}

int run_peak_mode(const pca::Args& args) {
    const std::vector<int> threads = pca::parse_int_list(args.get("threads", "1,2,4,8"));
    const long long iterations = args.get_int("iterations", 200000000LL);
    std::ofstream csv(args.require("out"));
    csv << "threads,seconds,gflops\n";
    for (const int p : threads) {
        omp_set_num_threads(p);
        double best = 1e300, sink = 0.0;
        for (int rep = 0; rep < 3; ++rep) {
            pca::Timer timer;
#pragma omp parallel reduction(+ : sink)
            sink += fma_loop(iterations, 1.0 + omp_get_thread_num());
            best = std::min(best, timer.seconds());
        }
        const double gflops = 2.0 * 48.0 * double(iterations) * p / best / 1e9;
        csv << p << ',' << best << ',' << gflops << '\n';
        std::cerr << "peak p=" << p << ": " << gflops << " GFLOP/s   [sink " << sink << "]\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: bench pipeline|stream|peak [options] --out FILE (see benchmarks/benchmark.cpp)\n";
            return 2;
        }
        const std::string mode = argv[1];
        const pca::Args args(argc - 1, argv + 1,
                             {"t", "m", "threads", "reps", "out", "r-method", "r", "n", "iterations"}, {});
        if (mode == "pipeline") return run_pipeline_mode(args);
        if (mode == "stream") return run_stream_mode(args);
        if (mode == "peak") return run_peak_mode(args);
        std::cerr << "unknown mode: " << mode << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
