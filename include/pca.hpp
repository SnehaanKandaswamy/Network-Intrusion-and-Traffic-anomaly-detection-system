#pragma once
// PCA model, configuration, and the serial pieces shared by both builds.

#include "anomaly_detection.hpp"
#include "matrix.hpp"

#include <string>

namespace pca {

// How the normal-subspace dimension r is chosen.
//  Sigma    : the paper's rule (section 4.3) - walk the principal axes in order;
//             the first axis whose temporal projection u_i contains a deviation
//             larger than `sigma` standard deviations from its mean, and all
//             later axes, form the anomalous subspace.
//  Fixed    : r = fixed_r.
//  Variance : smallest r whose components capture `variance_fraction` of the
//             total variance.
enum class RMethod { Sigma, Fixed, Variance };

struct PcaConfig {
    RMethod method = RMethod::Sigma;
    double sigma = 3.0;
    int fixed_r = 4;
    double variance_fraction = 0.90;
    double alpha = 0.001;  // false-alarm level; 0.001 = 99.9 % confidence
};

struct PcaModel {
    Vector mean;          // column (link) means of the training data
    Vector eigenvalues;   // covariance eigenvalues, decreasing
    Matrix eigenvectors;  // column i = principal axis v_i
    int r = 0;            // normal-subspace dimension
    Matrix P;             // m x r basis of the normal subspace
    QStatistic q;
    PcaConfig config;
    Index t_train = 0;
    Index axes_tested = 0;  // principal axes examined by the sigma rule
    std::string r_note;     // explanation when r had to be clamped
};

// Wall-clock seconds per pipeline stage.
struct StageTimes {
    double means = 0, center = 0, covariance = 0, eigen = 0, select = 0;
    double fit_total = 0;
    double score = 0;
};

// Symmetric eigendecomposition with Eigen's SelfAdjointEigenSolver (serial in
// both builds). Returns eigenvalues in decreasing order; each eigenvector is
// sign-normalised so its largest-magnitude entry is positive.
void eigen_decompose(const Matrix& C, Vector& values, Matrix& vectors);

// Number of eigenvalues larger than 1e-12 * lambda_1 (numerical rank).
Index numerical_rank(const Vector& eigenvalues);

int r_from_variance(const Vector& eigenvalues, double fraction);

// Clamps r to [1, m-1] (there must be at least one normal and one residual
// axis) and records why in `note` if it changed.
int clamp_r(int r, Index m, std::string& note);

std::string to_string(RMethod method);
RMethod parse_rmethod(const std::string& text);

}  // namespace pca
