#pragma once
// Detection statistics from Lakhina et al. (SIGCOMM 2004), section 5.1.
//
//   y_hat   = P P^T y               (modelled part, normal subspace S)
//   y_tilde = (I - P P^T) y         (residual part, anomalous subspace S~)
//   SPE     = ||y_tilde||^2         (squared prediction error = anomaly score)
//   alarm   if SPE > delta^2_alpha  (Q-statistic, Jackson & Mudholkar 1979)

#include "matrix.hpp"

#include <utility>
#include <vector>

namespace pca {

struct QStatistic {
    double delta2 = 0.0;  // threshold delta^2_alpha on the SPE
    double phi1 = 0.0, phi2 = 0.0, phi3 = 0.0;
    double h0 = 0.0;
    double c_alpha = 0.0;  // (1 - alpha) quantile of N(0,1)
    // "jackson-mudholkar" normally; "box-chi2" if h0 <= 0 makes the
    // Jackson-Mudholkar expression undefined; "degenerate-floor" if the
    // residual subspace carries (numerically) no variance at all.
    const char* method = "jackson-mudholkar";
};

// Inverse of the standard normal CDF (Acklam's approximation + one Halley step;
// absolute error < 1e-12 on (0,1)).
double inverse_normal_cdf(double p);

// eigenvalues: covariance eigenvalues sorted in decreasing order.
// r: dimension of the normal subspace; residual eigenvalues are r..m-1.
// lambda_j are covariance eigenvalues, i.e. ||Y v_j||^2 / (t-1). The paper
// writes ||Y v_j||^2; dividing by (t-1) puts lambda on the same per-sample
// scale as the SPE (ambiguity 1 in docs/PHASE1_DESIGN.md).
QStatistic q_statistic(const Vector& eigenvalues, int r, double alpha);

struct DetectionResult {
    Vector spe;                        // anomaly score per time bin
    Vector state_norm2;                // ||y||^2 per time bin (for plots)
    std::vector<unsigned char> flags;  // 1 if SPE > delta^2
    Index n_alarms = 0;
};

// For one (already centred) time bin: the k links with the largest squared
// residual, as (link index, share of the SPE). Explanatory output only; this is
// not the paper's OD-flow identification step (out of scope).
std::vector<std::pair<Index, double>> top_residual_links(const Matrix& Yc, Index row,
                                                        const Matrix& P, int k);

}  // namespace pca
