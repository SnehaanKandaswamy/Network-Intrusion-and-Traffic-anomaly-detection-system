#pragma once
// Inner arithmetic shared by the sequential and OpenMP kernels.
//
// Every function here computes a disjoint piece of the output (one column, one
// covariance tile, one block of time rows) entirely by itself. The OpenMP
// version only decides WHICH thread runs which piece; the floating-point
// operations and their order are identical to the sequential version. That is
// why both builds produce bitwise-identical results for these kernels.
//
// `#pragma omp simd` asks the compiler to vectorise a loop (AVX2 here). It
// creates no threads; the sequential build enables it with -fopenmp-simd.

#include "matrix.hpp"

#include <algorithm>
#include <vector>

namespace pca::detail {

// Covariance tiling: a tile is a 64 x 64 block of output columns, and rows are
// streamed in chunks of 512 so that the column panels of one tile
// (2 x 64 x 512 x 8 B = 512 KiB) stay in the L2 cache while they are reused.
constexpr Index kColBlock = 64;
constexpr Index kRowChunk = 512;

struct Tile {
    Index i0, i1, j0, j1;  // output rows [i0,i1), columns [j0,j1), with j >= i used
};

inline std::vector<Tile> upper_tiles(Index m) {
    std::vector<Tile> tiles;
    for (Index i0 = 0; i0 < m; i0 += kColBlock)
        for (Index j0 = i0; j0 < m; j0 += kColBlock)
            tiles.push_back({i0, std::min(i0 + kColBlock, m), j0, std::min(j0 + kColBlock, m)});
    return tiles;
}

inline double column_sum(const double* x, Index n) {
    double s = 0.0;
#pragma omp simd reduction(+ : s)
    for (Index k = 0; k < n; ++k) s += x[k];
    return s;
}

inline void subtract_scalar(double* x, Index n, double value) {
#pragma omp simd
    for (Index k = 0; k < n; ++k) x[k] -= value;
}

// Accumulates the un-normalised products S(i,j) = sum_k Y(k,i) Y(k,j) for the
// upper-triangular part (j >= i) of one tile into C.
inline void covariance_tile(const Matrix& Y, Matrix& C, const Tile& tile) {
    const Index t = Y.rows();
    const double* base = Y.data();
    for (Index j = tile.j0; j < tile.j1; ++j)
        for (Index i = tile.i0; i < std::min(tile.i1, j + 1); ++i) C(i, j) = 0.0;

    for (Index k0 = 0; k0 < t; k0 += kRowChunk) {
        const Index len = std::min(kRowChunk, t - k0);
        for (Index i = tile.i0; i < tile.i1; ++i) {
            const double* a = base + i * t + k0;
            Index j = std::max(tile.j0, i);
            // Register blocking: one load of a[k] feeds four dot products.
            for (; j + 3 < tile.j1; j += 4) {
                const double* b0 = base + j * t + k0;
                const double* b1 = b0 + t;
                const double* b2 = b1 + t;
                const double* b3 = b2 + t;
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
#pragma omp simd reduction(+ : s0, s1, s2, s3)
                for (Index k = 0; k < len; ++k) {
                    const double av = a[k];
                    s0 += av * b0[k];
                    s1 += av * b1[k];
                    s2 += av * b2[k];
                    s3 += av * b3[k];
                }
                C(i, j) += s0;
                C(i, j + 1) += s1;
                C(i, j + 2) += s2;
                C(i, j + 3) += s3;
            }
            for (; j < tile.j1; ++j) {
                const double* b = base + j * t + k0;
                double s = 0.0;
#pragma omp simd reduction(+ : s)
                for (Index k = 0; k < len; ++k) s += a[k] * b[k];
                C(i, j) += s;
            }
        }
    }
}

// Scales column j of the upper triangle by 1/(t-1) and mirrors it into row j.
// Column j writes only C(0..j, j) and C(j, 0..j-1), so different j never touch
// the same element.
inline void finish_covariance_column(Matrix& C, Index j, double scale) {
    for (Index i = 0; i < j; ++i) {
        C(i, j) *= scale;
        C(j, i) = C(i, j);
    }
    C(j, j) *= scale;
}

// Number of time rows processed together by the per-row kernels: about 512 KiB
// of Y per block so the block is still in cache for the second pass.
inline Index rows_per_block(Index m) {
    Index rows = 65536 / std::max<Index>(m, 1);
    rows = std::max<Index>(16, std::min<Index>(1024, rows));
    return rows - rows % 8;
}

// U(r0:r1, :) = Y(r0:r1, :) V
inline void project_rows(const Matrix& Y, const Matrix& V, Matrix& U, Index r0, Index r1) {
    const Index m = Y.cols(), k = V.cols(), len = r1 - r0;
    for (Index c = 0; c < k; ++c) {
        double* u = &U(r0, c);
#pragma omp simd
        for (Index row = 0; row < len; ++row) u[row] = 0.0;
    }
    for (Index j = 0; j < m; ++j) {
        const double* y = &Y(r0, j);
        for (Index c = 0; c < k; ++c) {
            const double v = V(j, c);
            double* u = &U(r0, c);
#pragma omp simd
            for (Index row = 0; row < len; ++row) u[row] += v * y[row];
        }
    }
}

// Fused projection + reconstruction + residual + SPE for rows [r0, r1):
//   z      = P^T y            (scores, r values per row)
//   y_tilde = y - P z         (residual; never stored)
//   spe    = ||y_tilde||^2,  state = ||y||^2
// Workspace: z needs len*r doubles, resid needs len doubles.
inline void spe_rows(const Matrix& Y, const Matrix& P, double* spe, double* state,
                     Index r0, Index r1, double* z, double* resid) {
    const Index m = Y.cols(), r = P.cols(), len = r1 - r0;
    for (Index q = 0; q < len * r; ++q) z[q] = 0.0;
    for (Index row = 0; row < len; ++row) {
        spe[r0 + row] = 0.0;
        state[r0 + row] = 0.0;
    }
    // Pass 1: scores z = P^T y for every row of the block.
    for (Index j = 0; j < m; ++j) {
        const double* y = &Y(r0, j);
        for (Index c = 0; c < r; ++c) {
            const double p = P(j, c);
            double* zc = z + c * len;
#pragma omp simd
            for (Index row = 0; row < len; ++row) zc[row] += p * y[row];
        }
    }
    // Pass 2: residual of every link, accumulated into the SPE.
    double* s = spe + r0;
    double* n = state + r0;
    for (Index j = 0; j < m; ++j) {
        const double* y = &Y(r0, j);
#pragma omp simd
        for (Index row = 0; row < len; ++row) resid[row] = y[row];
        for (Index c = 0; c < r; ++c) {
            const double p = P(j, c);
            const double* zc = z + c * len;
#pragma omp simd
            for (Index row = 0; row < len; ++row) resid[row] -= p * zc[row];
        }
#pragma omp simd
        for (Index row = 0; row < len; ++row) {
            s[row] += resid[row] * resid[row];
            n[row] += y[row] * y[row];
        }
    }
}

}  // namespace pca::detail
