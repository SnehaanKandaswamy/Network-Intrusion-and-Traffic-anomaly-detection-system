// Experiment-level (coarse-grained) parallelism for the injection study.
#include "injection.hpp"

#include <omp.h>

#include <exception>

namespace pca::omp {

std::vector<InjectionOutcome> run_injection_study(const Matrix& Y_clean, const Matrix& routing,
                                                  const std::vector<InjectionExperiment>& plan,
                                                  const PcaConfig& config) {
    std::vector<InjectionOutcome> outcomes(plan.size());
    const Index n = static_cast<Index>(plan.size());
    std::exception_ptr error = nullptr;

    // Each iteration is a complete PCA fit (~1 ms for 1008 x 60). Iterations
    // are independent and write only outcomes[i], so there are no races.
    // dynamic,1 balances P-cores against E-cores. The inner kernels are the
    // sequential ones, so there is exactly one level of parallelism.
#pragma omp parallel for schedule(dynamic, 1)
    for (Index i = 0; i < n; ++i) {
        try {
            outcomes[static_cast<std::size_t>(i)] =
                run_one_injection(Y_clean, routing, plan[static_cast<std::size_t>(i)], config);
        } catch (...) {
            // Exceptions must not cross the parallel-region boundary.
#pragma omp critical(injection_error)
            if (!error) error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);
    return outcomes;
}

}  // namespace pca::omp
