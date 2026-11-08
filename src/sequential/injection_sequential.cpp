#include "injection.hpp"
#include "pipeline.hpp"

#include <stdexcept>

namespace pca {

InjectionOutcome run_one_injection(const Matrix& Y_clean, const Matrix& routing,
                                   const InjectionExperiment& e, const PcaConfig& config) {
    if (e.time < 0 || e.time >= Y_clean.rows()) throw std::out_of_range("injection time out of range");
    if (e.flow < 0 || e.flow >= routing.cols()) throw std::out_of_range("injection flow out of range");

    Matrix Y = Y_clean;                                       // private copy
    Y.row(e.time) += e.size * routing.col(e.flow).transpose();  // y += b * A_i

    StageTimes times;
    const PcaModel model = fit<SeqKernels>(Y, config, times);  // Y is now centred

    const Vector y = Y.row(e.time).transpose();
    const Vector residual = y - model.P * (model.P.transpose() * y);
    InjectionOutcome out;
    out.r = model.r;
    out.spe = residual.squaredNorm();
    out.delta2 = model.q.delta2;
    out.detected = out.spe > out.delta2;
    return out;
}

namespace seq {

std::vector<InjectionOutcome> run_injection_study(const Matrix& Y_clean, const Matrix& routing,
                                                  const std::vector<InjectionExperiment>& plan,
                                                  const PcaConfig& config) {
    std::vector<InjectionOutcome> outcomes(plan.size());
    for (std::size_t i = 0; i < plan.size(); ++i)
        outcomes[i] = run_one_injection(Y_clean, routing, plan[i], config);
    return outcomes;
}

}  // namespace seq
}  // namespace pca
