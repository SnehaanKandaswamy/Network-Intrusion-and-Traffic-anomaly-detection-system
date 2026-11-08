#pragma once
// The computationally intensive kernels. There are two independent
// implementations with identical signatures:
//   pca::seq  - src/sequential/pca_sequential.cpp (compiled without OpenMP)
//   pca::omp  - src/openmp/pca_openmp.cpp        (OpenMP-parallel)
// Both use the same inner (per-tile / per-block) arithmetic from
// kernels_inl.hpp, so they differ only in how work is distributed.

#include "matrix.hpp"

#include <vector>

namespace pca {

#define PCA_KERNEL_DECLARATIONS                                                          \
    /* mu_j = (1/t) sum_k Y_kj */                                                        \
    Vector column_means(const Matrix& Y);                                                \
    /* Y_kj <- Y_kj - mu_j (in place) */                                                 \
    void center_columns(Matrix& Y, const Vector& mean);                                  \
    /* Sigma = Yc^T Yc / (t-1), full symmetric m x m */                                  \
    Matrix covariance(const Matrix& Yc);                                                 \
    /* U = Yc V  (t x k), the temporal projections onto the axes in V */                 \
    Matrix project(const Matrix& Yc, const Matrix& V);                                   \
    /* for each column u of U: max_k |u_k - mean(u)| / std(u) */                         \
    Vector max_sigma_deviation(const Matrix& U);                                         \
    /* spe_k = ||(I - P P^T) y_k||^2 and state_k = ||y_k||^2, fused, one pass */         \
    void spe_scores(const Matrix& Yc, const Matrix& P, Vector& spe, Vector& state_norm2); \
    /* flags_k = spe_k > delta2; returns the number of alarms */                         \
    Index apply_threshold(const Vector& spe, double delta2, std::vector<unsigned char>& flags);

namespace seq {
PCA_KERNEL_DECLARATIONS
}
namespace omp {
PCA_KERNEL_DECLARATIONS
}

#undef PCA_KERNEL_DECLARATIONS

// Backend "tags" so the pipeline (pipeline.hpp) can be written once as a
// template and instantiated for either implementation.
#define PCA_BACKEND_STRUCT(NAME, NS)                                                       \
    struct NAME {                                                                          \
        static constexpr const char* name = #NS;                                           \
        static Vector column_means(const Matrix& Y) { return NS::column_means(Y); }        \
        static void center_columns(Matrix& Y, const Vector& mu) { NS::center_columns(Y, mu); } \
        static Matrix covariance(const Matrix& Yc) { return NS::covariance(Yc); }          \
        static Matrix project(const Matrix& Yc, const Matrix& V) { return NS::project(Yc, V); } \
        static Vector max_sigma_deviation(const Matrix& U) { return NS::max_sigma_deviation(U); } \
        static void spe_scores(const Matrix& Yc, const Matrix& P, Vector& s, Vector& n) {   \
            NS::spe_scores(Yc, P, s, n);                                                   \
        }                                                                                  \
        static Index apply_threshold(const Vector& s, double d, std::vector<unsigned char>& f) { \
            return NS::apply_threshold(s, d, f);                                           \
        }                                                                                  \
    };

PCA_BACKEND_STRUCT(SeqKernels, seq)
PCA_BACKEND_STRUCT(OmpKernels, omp)

#undef PCA_BACKEND_STRUCT

}  // namespace pca
