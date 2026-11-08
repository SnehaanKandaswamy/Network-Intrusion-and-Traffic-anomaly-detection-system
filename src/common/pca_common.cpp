// Serial parts of the PCA that are shared by both builds.
#include "pca.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pca {

void eigen_decompose(const Matrix& C, Vector& values, Matrix& vectors) {
    // Householder tridiagonalisation + implicit symmetric QR, O(m^3). Each step
    // depends on the previous one, so it is kept serial (Amdahl fraction).
    Eigen::SelfAdjointEigenSolver<Matrix> solver(C);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("eigendecomposition of the covariance matrix failed");
    const Index m = C.rows();
    values.resize(m);
    vectors.resize(m, m);
    for (Index i = 0; i < m; ++i) {  // Eigen returns ascending order
        values(i) = solver.eigenvalues()(m - 1 - i);
        vectors.col(i) = solver.eigenvectors().col(m - 1 - i);
        Index largest = 0;
        vectors.col(i).cwiseAbs().maxCoeff(&largest);
        if (vectors(largest, i) < 0) vectors.col(i) = -vectors.col(i);
    }
}

Index numerical_rank(const Vector& eigenvalues) {
    if (eigenvalues.size() == 0 || eigenvalues(0) <= 0.0) return 0;
    const double cutoff = 1e-12 * eigenvalues(0);
    Index rank = 0;
    while (rank < eigenvalues.size() && eigenvalues(rank) > cutoff) ++rank;
    return rank;
}

int r_from_variance(const Vector& eigenvalues, double fraction) {
    if (!(fraction > 0.0 && fraction <= 1.0))
        throw std::invalid_argument("variance fraction must be in (0, 1]");
    const double total = eigenvalues.cwiseMax(0.0).sum();
    double running = 0.0;
    for (Index i = 0; i < eigenvalues.size(); ++i) {
        running += std::max(eigenvalues(i), 0.0);
        if (running >= fraction * total) return static_cast<int>(i + 1);
    }
    return static_cast<int>(eigenvalues.size());
}

int clamp_r(int r, Index m, std::string& note) {
    const int upper = static_cast<int>(m - 1);
    if (r < 1) {
        note = "rule selected r=" + std::to_string(r) + "; clamped to 1 (need >= 1 normal axis)";
        return 1;
    }
    if (r > upper) {
        note = "rule selected r=" + std::to_string(r) + "; clamped to m-1=" + std::to_string(upper) +
               " (need >= 1 residual axis)";
        return upper;
    }
    return r;
}

std::string to_string(RMethod method) {
    switch (method) {
        case RMethod::Sigma: return "sigma";
        case RMethod::Fixed: return "fixed";
        case RMethod::Variance: return "variance";
    }
    return "unknown";
}

RMethod parse_rmethod(const std::string& text) {
    if (text == "sigma") return RMethod::Sigma;
    if (text == "fixed") return RMethod::Fixed;
    if (text == "variance") return RMethod::Variance;
    throw std::invalid_argument("--r-method must be sigma, fixed or variance");
}

}  // namespace pca
