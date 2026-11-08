#pragma once
// The detection pipeline, written once and instantiated for either backend:
//   fit<SeqKernels>(...)  or  fit<OmpKernels>(...)
//
//   1. mu = column means             (kernel)
//   2. Y <- Y - 1 mu^T               (kernel)
//   3. Sigma = Y^T Y / (t-1)         (kernel)
//   4. Sigma = V Lambda V^T          (Eigen, serial in both builds)
//   5. choose r, P = V[:, 0:r]       (sigma rule uses projection kernels)
//   6. delta^2 from the Q-statistic  (serial, O(m))
//   7. SPE and alarms                (kernels)

#include "anomaly_detection.hpp"
#include "kernels.hpp"
#include "pca.hpp"
#include "utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace pca {

// The paper's subspace-separation rule (section 4.3). Axes are tested in order,
// in batches of 8, so only as many projections are computed as needed.
template <class K>
int select_r_sigma(const Matrix& Yc, const Matrix& V, Index usable_axes, double sigma,
                   Index& axes_tested) {
    constexpr Index kBatch = 8;
    axes_tested = 0;
    for (Index i0 = 0; i0 < usable_axes; i0 += kBatch) {
        const Index nb = std::min(kBatch, usable_axes - i0);
        const Matrix U = K::project(Yc, V.middleCols(i0, nb));
        const Vector dev = K::max_sigma_deviation(U);
        for (Index j = 0; j < nb; ++j) {
            axes_tested = i0 + j + 1;
            if (dev(j) > sigma) return static_cast<int>(i0 + j);
        }
    }
    // No axis tripped the rule: every axis with non-zero variance is normal.
    return static_cast<int>(usable_axes);
}

// Fits the model on Y. Y is centred IN PLACE (saves a t x m copy); afterwards it
// holds the centred training data.
template <class K>
PcaModel fit(Matrix& Y, const PcaConfig& config, StageTimes& times) {
    validate_measurements(Y, "training matrix");
    PcaModel model;
    model.config = config;
    model.t_train = Y.rows();
    const Index m = Y.cols();

    Timer total;
    Timer timer;
    model.mean = K::column_means(Y);
    times.means = timer.seconds();

    timer.reset();
    K::center_columns(Y, model.mean);
    times.center = timer.seconds();

    timer.reset();
    const Matrix C = K::covariance(Y);
    times.covariance = timer.seconds();

    timer.reset();
    eigen_decompose(C, model.eigenvalues, model.eigenvectors);
    times.eigen = timer.seconds();

    timer.reset();
    int r = 0;
    switch (config.method) {
        case RMethod::Sigma:
            r = select_r_sigma<K>(Y, model.eigenvectors, numerical_rank(model.eigenvalues),
                                  config.sigma, model.axes_tested);
            break;
        case RMethod::Fixed:
            r = config.fixed_r;
            break;
        case RMethod::Variance:
            r = r_from_variance(model.eigenvalues, config.variance_fraction);
            break;
    }
    model.r = clamp_r(r, m, model.r_note);
    model.P = model.eigenvectors.leftCols(model.r);
    model.q = q_statistic(model.eigenvalues, model.r, config.alpha);
    times.select = timer.seconds();
    times.fit_total = total.seconds();
    return model;
}

// Scores already-centred data (training data after fit(), or test data after
// K::center_columns(Ytest, model.mean)).
template <class K>
DetectionResult score(const Matrix& Yc, const PcaModel& model, StageTimes& times) {
    if (Yc.cols() != model.P.rows())
        throw std::invalid_argument("score: matrix has a different number of links than the model");
    Timer timer;
    DetectionResult result;
    K::spe_scores(Yc, model.P, result.spe, result.state_norm2);
    result.n_alarms = K::apply_threshold(result.spe, model.q.delta2, result.flags);
    times.score = timer.seconds();
    return result;
}

}  // namespace pca
