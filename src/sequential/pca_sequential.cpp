// Sequential kernels. Compiled WITHOUT the OpenMP runtime (only -fopenmp-simd,
// which enables `#pragma omp simd` vectorisation hints and creates no threads).
#include "kernels.hpp"
#include "kernels_inl.hpp"

#include <algorithm>
#include <cmath>

namespace pca::seq {

Vector column_means(const Matrix& Y) {
    const Index t = Y.rows(), m = Y.cols();
    Vector mean(m);
    for (Index j = 0; j < m; ++j) mean(j) = detail::column_sum(Y.col(j).data(), t) / static_cast<double>(t);
    return mean;
}

void center_columns(Matrix& Y, const Vector& mean) {
    const Index t = Y.rows(), m = Y.cols();
    for (Index j = 0; j < m; ++j) detail::subtract_scalar(Y.col(j).data(), t, mean(j));
}

Matrix covariance(const Matrix& Yc) {
    const Index m = Yc.cols();
    Matrix C(m, m);
    for (const detail::Tile& tile : detail::upper_tiles(m)) detail::covariance_tile(Yc, C, tile);
    const double scale = 1.0 / static_cast<double>(Yc.rows() - 1);
    for (Index j = 0; j < m; ++j) detail::finish_covariance_column(C, j, scale);
    return C;
}

Matrix project(const Matrix& Yc, const Matrix& V) {
    const Index t = Yc.rows();
    Matrix U(t, V.cols());
    const Index block = detail::rows_per_block(Yc.cols());
    for (Index r0 = 0; r0 < t; r0 += block) detail::project_rows(Yc, V, U, r0, std::min(t, r0 + block));
    return U;
}

Vector max_sigma_deviation(const Matrix& U) {
    const Index t = U.rows();
    Vector out(U.cols());
    for (Index c = 0; c < U.cols(); ++c) {
        const double* u = U.col(c).data();
        double sum = 0.0;
        for (Index k = 0; k < t; ++k) sum += u[k];
        const double mean = sum / static_cast<double>(t);
        double ss = 0.0, max_dev = 0.0;
        for (Index k = 0; k < t; ++k) {
            const double d = u[k] - mean;
            ss += d * d;
            max_dev = std::max(max_dev, std::abs(d));
        }
        const double sd = std::sqrt(ss / static_cast<double>(t - 1));
        out(c) = sd > 0.0 ? max_dev / sd : 0.0;
    }
    return out;
}

void spe_scores(const Matrix& Yc, const Matrix& P, Vector& spe, Vector& state_norm2) {
    const Index t = Yc.rows();
    spe.resize(t);
    state_norm2.resize(t);
    const Index block = detail::rows_per_block(Yc.cols());
    std::vector<double> z(static_cast<std::size_t>(block * std::max<Index>(P.cols(), 1)));
    std::vector<double> resid(static_cast<std::size_t>(block));
    for (Index r0 = 0; r0 < t; r0 += block)
        detail::spe_rows(Yc, P, spe.data(), state_norm2.data(), r0, std::min(t, r0 + block), z.data(),
                         resid.data());
}

Index apply_threshold(const Vector& spe, double delta2, std::vector<unsigned char>& flags) {
    flags.assign(static_cast<std::size_t>(spe.size()), 0);
    Index count = 0;
    for (Index k = 0; k < spe.size(); ++k) {
        flags[static_cast<std::size_t>(k)] = spe(k) > delta2 ? 1 : 0;
        count += flags[static_cast<std::size_t>(k)];
    }
    return count;
}

}  // namespace pca::seq
