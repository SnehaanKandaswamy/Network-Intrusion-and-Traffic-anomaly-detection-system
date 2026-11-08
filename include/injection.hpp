#pragma once
// Synthetic anomaly-injection study (after Lakhina et al. section 6.3).
//
// One experiment: take anomaly-free link traffic Y, add `size` bytes to OD flow
// `flow` at time bin `time` (y_time += size * A[:, flow], i.e. every link on the
// flow's path carries the extra bytes), re-fit the whole PCA model on the
// modified matrix, and test whether SPE(time) > delta^2.
//
// Experiments are independent, which makes this an embarrassingly parallel,
// throughput-bound workload. The OpenMP version parallelises the OUTER loop over
// experiments; each experiment runs the sequential kernels, so there is no
// nested parallelism.

#include "pca.hpp"

#include <vector>

namespace pca {

struct InjectionExperiment {
    Index flow = 0;
    Index time = 0;
    double size = 0.0;
};

struct InjectionOutcome {
    int r = 0;
    double spe = 0.0;
    double delta2 = 0.0;
    bool detected = false;
};

InjectionOutcome run_one_injection(const Matrix& Y_clean, const Matrix& routing,
                                   const InjectionExperiment& experiment,
                                   const PcaConfig& config);

namespace seq {
std::vector<InjectionOutcome> run_injection_study(const Matrix& Y_clean, const Matrix& routing,
                                                  const std::vector<InjectionExperiment>& plan,
                                                  const PcaConfig& config);
}
namespace omp {
std::vector<InjectionOutcome> run_injection_study(const Matrix& Y_clean, const Matrix& routing,
                                                  const std::vector<InjectionExperiment>& plan,
                                                  const PcaConfig& config);
}

}  // namespace pca
