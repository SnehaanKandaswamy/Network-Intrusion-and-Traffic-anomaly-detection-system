// OpenMP-parallel kernels.
//
// Strategy: parallelise over independent OUTPUTS (columns, covariance tiles,
// blocks of time rows) so that every output element is computed by exactly one
// thread with the same arithmetic as the sequential code (kernels_inl.hpp).
// Consequences: no locks, no atomics, no false sharing on accumulators, and
// results bitwise-identical to the sequential build. Reductions are used where
// the result is a single scalar (alarm count, deviation statistics).
#include "kernels.hpp"
#include "kernels_inl.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>

namespace pca::omp {

// Uniform work per column -> static schedule (lowest overhead; each thread gets
// one contiguous range of columns, i.e. one contiguous region of memory).
Vector column_means(const Matrix& Y) {
    const Index t = Y.rows(), m = Y.cols();
    Vector mean(m);
#pragma omp parallel for schedule(static)
    for (Index j = 0; j < m; ++j) mean(j) = detail::column_sum(Y.col(j).data(), t) / static_cast<double>(t);
    return mean;
}

void center_columns(Matrix& Y, const Vector& mean) {
    const Index t = Y.rows(), m = Y.cols();
#pragma omp parallel for schedule(static)
    for (Index j = 0; j < m; ++j) detail::subtract_scalar(Y.col(j).data(), t, mean(j));
}

// Upper-triangular tiles have unequal cost (diagonal tiles do half the work)
// and this CPU mixes fast P-cores with slower E-cores, so tiles are handed out
// one at a time: schedule(dynamic, 1). There are (m/64)(m/64+1)/2 tiles; the
// scheduling overhead per tile is negligible next to 64*64*t FMAs.
Matrix covariance(const Matrix& Yc) {
    const Index m = Yc.cols();
    Matrix C(m, m);
    const std::vector<detail::Tile> tiles = detail::upper_tiles(m);
    const Index n_tiles = static_cast<Index>(tiles.size());
    const double scale = 1.0 / static_cast<double>(Yc.rows() - 1);
#pragma omp parallel
    {
#pragma omp for schedule(dynamic, 1)
        for (Index k = 0; k < n_tiles; ++k) detail::covariance_tile(Yc, C, tiles[static_cast<std::size_t>(k)]);
        // implicit barrier: every tile is finished before mirroring
#pragma omp for schedule(static)
        for (Index j = 0; j < m; ++j) detail::finish_covariance_column(C, j, scale);
    }
    return C;
}

Matrix project(const Matrix& Yc, const Matrix& V) {
    const Index t = Yc.rows();
    Matrix U(t, V.cols());
    const Index block = detail::rows_per_block(Yc.cols());
    const Index n_blocks = (t + block - 1) / block;
#pragma omp parallel for schedule(static)
    for (Index b = 0; b < n_blocks; ++b)
        detail::project_rows(Yc, V, U, b * block, std::min(t, (b + 1) * block));
    return U;
}

// Only a few columns (<= 8) but many rows: parallelise over rows with
// reductions. reduction(+) changes the summation order relative to the
// sequential loop, so mean/std can differ in the last bits; the result is only
// compared against `sigma`, and the tests check that the chosen r agrees.
Vector max_sigma_deviation(const Matrix& U) {
    const Index t = U.rows();
    Vector out(U.cols());
    for (Index c = 0; c < U.cols(); ++c) {
        const double* u = U.col(c).data();
        double sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : sum)
        for (Index k = 0; k < t; ++k) sum += u[k];
        const double mean = sum / static_cast<double>(t);
        double ss = 0.0, max_dev = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : ss) reduction(max : max_dev)
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

// Each block of time rows is independent. Workspace is allocated once per
// thread inside the parallel region (private, so no sharing).
void spe_scores(const Matrix& Yc, const Matrix& P, Vector& spe, Vector& state_norm2) {
    const Index t = Yc.rows();
    spe.resize(t);
    state_norm2.resize(t);
    const Index block = detail::rows_per_block(Yc.cols());
    const Index n_blocks = (t + block - 1) / block;
#pragma omp parallel
    {
        std::vector<double> z(static_cast<std::size_t>(block * std::max<Index>(P.cols(), 1)));
        std::vector<double> resid(static_cast<std::size_t>(block));
#pragma omp for schedule(static)
        for (Index b = 0; b < n_blocks; ++b)
            detail::spe_rows(Yc, P, spe.data(), state_norm2.data(), b * block,
                             std::min(t, (b + 1) * block), z.data(), resid.data());
    }
}

// Integer reduction: exact, so the count is identical to the sequential one.
Index apply_threshold(const Vector& spe, double delta2, std::vector<unsigned char>& flags) {
    const Index t = spe.size();
    flags.assign(static_cast<std::size_t>(t), 0);
    Index count = 0;
#pragma omp parallel for schedule(static) reduction(+ : count)
    for (Index k = 0; k < t; ++k) {
        const unsigned char f = spe(k) > delta2 ? 1 : 0;
        flags[static_cast<std::size_t>(k)] = f;
        count += f;
    }
    return count;
}

}  // namespace pca::omp
