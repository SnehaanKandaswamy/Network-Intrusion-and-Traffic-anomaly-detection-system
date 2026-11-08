// Parallel anomaly-injection experiment (after Lakhina et al. section 6.3).
//
//   injection_study --data Y_clean.csv --routing A.csv --sizes 3e7,1e7
//                   [--time-start 288 --time-end 432 --time-step 4]
//                   [--threads 1,2,4,8] [--reps 3] [--out benchmarks/results]
//
// Runs the full experiment once with the sequential implementation and then
// with the OpenMP implementation for each thread count; checks that every
// outcome is identical; writes timings and per-experiment outcomes.

#include "injection.hpp"
#include "matrix.hpp"
#include "utils.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

namespace {

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const pca::Args args(argc, argv,
                             {"data", "routing", "sizes", "time-start", "time-end", "time-step", "threads",
                              "reps", "out", "r-method", "r", "sigma", "alpha"},
                             {"help"});
        if (args.has("help") || !args.has("data") || !args.has("routing")) {
            std::cout << "Usage: injection_study --data Y_clean.csv --routing A.csv --sizes S1,S2\n"
                         "         [--time-start 288] [--time-end 432] [--time-step 4]\n"
                         "         [--threads 1,2,4,8] [--reps 3] [--out DIR]\n"
                         "         [--r-method sigma|fixed] [--r 4] [--sigma 3] [--alpha 0.001]\n";
            return args.has("help") ? 0 : 2;
        }
        const pca::LabelledMatrix data = pca::load_matrix(args.require("data"));
        const pca::LabelledMatrix routing = pca::load_matrix(args.require("routing"));
        pca::validate_measurements(data.values, "clean data");
        if (routing.values.rows() != data.values.cols())
            throw std::invalid_argument("routing matrix must have one row per link (" +
                                        std::to_string(data.values.cols()) + ")");

        pca::PcaConfig config;
        config.method = pca::parse_rmethod(args.get("r-method", "sigma"));
        config.fixed_r = static_cast<int>(args.get_int("r", 4));
        config.sigma = args.get_double("sigma", 3.0);
        config.alpha = args.get_double("alpha", 0.001);

        const std::vector<double> sizes = pca::parse_double_list(args.require("sizes"));
        const pca::Index t = data.values.rows();
        const pca::Index t0 = args.get_int("time-start", 0);
        const pca::Index t1 = std::min<pca::Index>(args.get_int("time-end", std::min<pca::Index>(t, 144)), t);
        const pca::Index step = std::max<long long>(1, args.get_int("time-step", 1));
        const std::vector<int> thread_counts = pca::parse_int_list(args.get("threads", "1,2,4,8"));
        const int reps = static_cast<int>(std::max<long long>(1, args.get_int("reps", 3)));
        const std::string out_dir = args.get("out", "benchmarks/results");
        pca::ensure_directory(out_dir);

        std::vector<pca::InjectionExperiment> plan;
        for (const double size : sizes)
            for (pca::Index time = t0; time < t1; time += step)
                for (pca::Index flow = 0; flow < routing.values.cols(); ++flow) plan.push_back({flow, time, size});
        std::cout << "injection study: " << plan.size() << " experiments (" << routing.values.cols()
                  << " flows x " << (t1 - t0 + step - 1) / step << " time bins x " << sizes.size()
                  << " sizes), each a full PCA re-fit of " << t << " x " << data.values.cols() << "\n";

        std::ofstream timing(out_dir + "/injection_timing.csv");
        timing << "backend,threads,rep,seconds,experiments\n";

        // ---- sequential reference
        std::vector<pca::InjectionOutcome> reference;
        std::vector<double> seq_times;
        for (int rep = 0; rep < reps; ++rep) {
            pca::Timer timer;
            reference = pca::seq::run_injection_study(data.values, routing.values, plan, config);
            seq_times.push_back(timer.seconds());
            timing << "sequential,1," << rep << ',' << seq_times.back() << ',' << plan.size() << '\n';
        }
        const double t_seq = median(seq_times);
        std::cout << std::fixed << std::setprecision(3) << "  sequential          : " << t_seq << " s\n";

        // ---- OpenMP
        bool all_equal = true;
        for (const int p : thread_counts) {
            omp_set_num_threads(p);
            std::vector<double> times;
            std::vector<pca::InjectionOutcome> outcomes;
            for (int rep = 0; rep < reps; ++rep) {
                pca::Timer timer;
                outcomes = pca::omp::run_injection_study(data.values, routing.values, plan, config);
                times.push_back(timer.seconds());
                timing << "openmp," << p << ',' << rep << ',' << times.back() << ',' << plan.size() << '\n';
            }
            std::size_t mismatches = 0;
            for (std::size_t i = 0; i < plan.size(); ++i)
                if (outcomes[i].detected != reference[i].detected || outcomes[i].r != reference[i].r ||
                    outcomes[i].spe != reference[i].spe)
                    ++mismatches;
            all_equal = all_equal && mismatches == 0;
            const double tp = median(times);
            std::cout << "  openmp " << std::setw(2) << p << " threads   : " << tp << " s   speedup "
                      << std::setprecision(2) << t_seq / tp << "   efficiency " << std::setprecision(1)
                      << 100.0 * t_seq / tp / p << " %   mismatches vs sequential: " << mismatches << "\n"
                      << std::setprecision(3);
        }

        // ---- per-experiment outcomes (from the sequential run)
        std::ofstream res(out_dir + "/injection_results.csv");
        res << std::setprecision(10) << "flow,flow_name,time,size,r,spe,delta2,detected\n";
        std::map<double, std::pair<std::size_t, std::size_t>> per_size;  // size -> (detected, total)
        for (std::size_t i = 0; i < plan.size(); ++i) {
            const auto& e = plan[i];
            const auto& o = reference[i];
            res << e.flow << ',' << routing.col_labels[static_cast<std::size_t>(e.flow)] << ',' << e.time << ','
                << e.size << ',' << o.r << ',' << o.spe << ',' << o.delta2 << ',' << int(o.detected) << '\n';
            per_size[e.size].first += o.detected ? 1 : 0;
            per_size[e.size].second += 1;
        }
        for (const auto& [size, counts] : per_size)
            std::cout << "  size " << std::scientific << std::setprecision(2) << size << std::fixed
                      << ": detection rate " << std::setprecision(1)
                      << 100.0 * double(counts.first) / double(counts.second) << " % (" << counts.first << "/"
                      << counts.second << ")\n";
        std::cout << (all_equal ? "  all OpenMP outcomes identical to sequential\n"
                                : "  WARNING: OpenMP outcomes differ from sequential\n")
                  << "  wrote " << out_dir << "/injection_timing.csv and injection_results.csv\n";
        return all_equal ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
