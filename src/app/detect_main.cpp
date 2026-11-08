// Detection driver. Built twice:
//   pca_seq : PCA_USE_OPENMP=0, sequential kernels, no OpenMP runtime linked
//   pca_omp : PCA_USE_OPENMP=1, OpenMP kernels
//
// Usage:
//   pca_omp --train Y_train.csv [--test Y_test.csv] --out results/ [--threads 8]
//           [--r-method sigma|fixed|variance] [--sigma 3] [--r 4] [--variance 0.9]
//           [--alpha 0.001] [--top-links 3]
// Without --test the model is fitted and evaluated on the same data (the
// paper's offline setting). With --test, mean, subspace and threshold come from
// --train only and are applied to --test.

#include "anomaly_detection.hpp"
#include "matrix.hpp"
#include "pca.hpp"
#include "pipeline.hpp"
#include "utils.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#if PCA_USE_OPENMP
#include <omp.h>
using Backend = pca::OmpKernels;
static const char* const kBackendName = "openmp";
#else
using Backend = pca::SeqKernels;
static const char* const kBackendName = "sequential";
#endif

namespace {

void print_usage() {
    std::cout <<
        "Usage: pca_seq|pca_omp --train FILE [--test FILE] [--out DIR] [options]\n"
        "  --train FILE      training measurements (.csv time x link, or .bin)\n"
        "  --test FILE       data to score with the trained model (default: --train)\n"
        "  --out DIR         output directory (default: results)\n"
        "  --threads N       OpenMP threads (pca_omp only; default OMP_NUM_THREADS)\n"
        "  --r-method M      sigma (paper, default) | fixed | variance\n"
        "  --sigma S         deviation threshold for the sigma rule (default 3)\n"
        "  --r R             r for --r-method fixed (default 4)\n"
        "  --variance F      variance fraction for --r-method variance (default 0.9)\n"
        "  --alpha A         false-alarm level of the Q-statistic (default 0.001 = 99.9%)\n"
        "  --top-links K     residual links reported per alarm (default 3)\n";
}

void write_model_json(const std::string& path, const pca::PcaModel& model,
                      const std::vector<std::string>& links) {
    std::ofstream out(path);
    out << std::setprecision(17);
    const double total = model.eigenvalues.cwiseMax(0.0).sum();
    out << "{\n";
    out << "  \"t_train\": " << model.t_train << ",\n";
    out << "  \"m\": " << model.P.rows() << ",\n";
    out << "  \"r\": " << model.r << ",\n";
    out << "  \"r_method\": \"" << pca::to_string(model.config.method) << "\",\n";
    out << "  \"sigma\": " << model.config.sigma << ",\n";
    out << "  \"axes_tested\": " << model.axes_tested << ",\n";
    out << "  \"r_note\": \"" << pca::json_escape(model.r_note) << "\",\n";
    out << "  \"alpha\": " << model.config.alpha << ",\n";
    out << "  \"delta2\": " << model.q.delta2 << ",\n";
    out << "  \"threshold_method\": \"" << model.q.method << "\",\n";
    out << "  \"phi\": [" << model.q.phi1 << ", " << model.q.phi2 << ", " << model.q.phi3 << "],\n";
    out << "  \"h0\": " << model.q.h0 << ",\n";
    out << "  \"c_alpha\": " << model.q.c_alpha << ",\n";
    out << "  \"total_variance\": " << total << ",\n";
    out << "  \"eigenvalues\": [";
    for (pca::Index i = 0; i < model.eigenvalues.size(); ++i)
        out << (i ? ", " : "") << model.eigenvalues(i);
    out << "],\n  \"mean\": [";
    for (pca::Index i = 0; i < model.mean.size(); ++i) out << (i ? ", " : "") << model.mean(i);
    out << "],\n  \"links\": [";
    for (std::size_t i = 0; i < links.size(); ++i)
        out << (i ? ", " : "") << '"' << pca::json_escape(links[i]) << '"';
    out << "]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const pca::Args args(argc, argv,
                             {"train", "test", "out", "threads", "r-method", "sigma", "r", "variance",
                              "alpha", "top-links"},
                             {"help"});
        if (args.has("help") || !args.has("train")) {
            print_usage();
            return args.has("help") ? 0 : 2;
        }

        int threads = 1;
#if PCA_USE_OPENMP
        if (args.has("threads")) omp_set_num_threads(static_cast<int>(args.get_int("threads", 1)));
        threads = omp_get_max_threads();
#else
        if (args.has("threads") && args.get_int("threads", 1) != 1)
            std::cerr << "note: pca_seq is sequential; --threads is ignored\n";
#endif

        pca::PcaConfig config;
        config.method = pca::parse_rmethod(args.get("r-method", "sigma"));
        config.sigma = args.get_double("sigma", 3.0);
        config.fixed_r = static_cast<int>(args.get_int("r", 4));
        config.variance_fraction = args.get_double("variance", 0.90);
        config.alpha = args.get_double("alpha", 0.001);
        const int top_k = static_cast<int>(args.get_int("top-links", 3));
        const std::string out_dir = args.get("out", "results");
        pca::ensure_directory(out_dir);

        const double cpu0 = pca::process_cpu_seconds();
        pca::Timer wall;

        // ---- load
        pca::Timer timer;
        pca::LabelledMatrix train = pca::load_matrix(args.require("train"));
        const bool in_sample = !args.has("test");
        pca::LabelledMatrix test;
        if (!in_sample) {
            test = pca::load_matrix(args.require("test"));
            pca::validate_measurements(test.values, "test matrix");
            if (test.values.cols() != train.values.cols())
                throw std::invalid_argument("train and test have different numbers of links");
            if (test.col_labels != train.col_labels)
                std::cerr << "warning: train and test link names differ; assuming same order\n";
        }
        const double load_seconds = timer.seconds();

        // ---- fit (centres train.values in place)
        pca::StageTimes times;
        const pca::PcaModel model = pca::fit<Backend>(train.values, config, times);

        // ---- score
        double test_center_seconds = 0.0;
        const pca::LabelledMatrix& scored = in_sample ? train : test;
        if (!in_sample) {
            timer.reset();
            Backend::center_columns(test.values, model.mean);  // train mean, not test mean
            test_center_seconds = timer.seconds();
        }
        const pca::DetectionResult result = pca::score<Backend>(scored.values, model, times);
        const double compute_seconds = times.fit_total + test_center_seconds + times.score;

        // ---- outputs
        timer.reset();
        write_model_json(out_dir + "/model.json", model, train.col_labels);
        {
            std::ofstream spe_out(out_dir + "/spe.csv");
            spe_out << std::setprecision(12) << "time_index,time,spe,state_norm2,threshold,anomaly\n";
            for (pca::Index k = 0; k < result.spe.size(); ++k)
                spe_out << k << ',' << scored.row_labels[static_cast<std::size_t>(k)] << ','
                        << result.spe(k) << ',' << result.state_norm2(k) << ',' << model.q.delta2 << ','
                        << int(result.flags[static_cast<std::size_t>(k)]) << '\n';
        }
        {
            std::ofstream alarms(out_dir + "/alarms.csv");
            alarms << std::setprecision(12) << "time_index,time,spe,spe_over_threshold,top_links\n";
            for (pca::Index k = 0; k < result.spe.size(); ++k) {
                if (!result.flags[static_cast<std::size_t>(k)]) continue;
                alarms << k << ',' << scored.row_labels[static_cast<std::size_t>(k)] << ',' << result.spe(k)
                       << ',' << result.spe(k) / model.q.delta2 << ',';
                const auto links = pca::top_residual_links(scored.values, k, model.P, top_k);
                for (std::size_t i = 0; i < links.size(); ++i)
                    alarms << (i ? ";" : "") << train.col_labels[static_cast<std::size_t>(links[i].first)] << ':'
                           << std::setprecision(3) << links[i].second << std::setprecision(12);
                alarms << '\n';
            }
        }
        const double write_seconds = timer.seconds();
        const double total_seconds = wall.seconds();
        const double cpu_seconds = pca::process_cpu_seconds() - cpu0;
        {
            std::ofstream tj(out_dir + "/timings.json");
            tj << std::setprecision(9) << "{\n"
               << "  \"backend\": \"" << kBackendName << "\",\n"
               << "  \"threads\": " << threads << ",\n"
               << "  \"t_train\": " << model.t_train << ",\n"
               << "  \"t_scored\": " << scored.values.rows() << ",\n"
               << "  \"m\": " << scored.values.cols() << ",\n"
               << "  \"load\": " << load_seconds << ",\n"
               << "  \"means\": " << times.means << ",\n"
               << "  \"center\": " << times.center + test_center_seconds << ",\n"
               << "  \"covariance\": " << times.covariance << ",\n"
               << "  \"eigen\": " << times.eigen << ",\n"
               << "  \"select\": " << times.select << ",\n"
               << "  \"score\": " << times.score << ",\n"
               << "  \"compute_total\": " << compute_seconds << ",\n"
               << "  \"write\": " << write_seconds << ",\n"
               << "  \"end_to_end\": " << total_seconds << ",\n"
               << "  \"cpu_seconds\": " << cpu_seconds << ",\n"
               << "  \"peak_rss_mb\": " << pca::peak_rss_mb() << "\n}\n";
        }

        const double captured = model.eigenvalues.head(model.r).sum() / model.eigenvalues.cwiseMax(0.0).sum();
        std::cout << std::setprecision(4)
                  << "backend           : " << kBackendName << " (" << threads << " thread"
                  << (threads == 1 ? "" : "s") << ")\n"
                  << "training matrix   : " << model.t_train << " time bins x " << model.P.rows() << " links\n"
                  << "scored matrix     : " << scored.values.rows() << " time bins"
                  << (in_sample ? " (in-sample, as in the paper)" : " (held-out test data)") << "\n"
                  << "normal subspace   : r = " << model.r << " (" << pca::to_string(config.method)
                  << " rule), captures " << 100.0 * captured << " % of variance\n";
        if (!model.r_note.empty()) std::cout << "                    " << model.r_note << "\n";
        std::cout << "threshold delta^2 : " << model.q.delta2 << " (" << model.q.method << ", "
                  << 100.0 * (1.0 - config.alpha) << " % confidence)\n"
                  << "alarms            : " << result.n_alarms << " of " << result.spe.size() << " time bins\n"
                  << "compute time      : " << compute_seconds << " s  (cov " << times.covariance
                  << " s, eigen " << times.eigen << " s, score " << times.score << " s)\n"
                  << "outputs           : " << out_dir << "/{model.json,spe.csv,alarms.csv,timings.json}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
