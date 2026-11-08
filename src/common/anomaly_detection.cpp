#include "anomaly_detection.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace pca {

double inverse_normal_cdf(double p) {
    if (!(p > 0.0 && p < 1.0)) throw std::domain_error("inverse_normal_cdf: p must be in (0,1)");
    // P. J. Acklam's rational approximation (relative error 1.15e-9) ...
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01,  -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    const double p_low = 0.02425;
    double x;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - p_low) {
        const double q = p - 0.5, r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    // ... refined by one Halley step on Phi(x) - p = 0 (full double precision).
    const double pi = 3.14159265358979323846;
    const double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
    const double u = e * std::sqrt(2.0 * pi) * std::exp(0.5 * x * x);
    return x - u / (1.0 + 0.5 * x * u);
}

QStatistic q_statistic(const Vector& eigenvalues, int r, double alpha) {
    if (!(alpha > 0.0 && alpha < 1.0)) throw std::invalid_argument("alpha must be in (0,1)");
    const Index m = eigenvalues.size();
    if (r < 1 || r >= m) throw std::invalid_argument("q_statistic: need 1 <= r < m");

    QStatistic q;
    q.c_alpha = inverse_normal_cdf(1.0 - alpha);
    double total = 0.0;
    for (Index j = 0; j < m; ++j) {
        const double lam = std::max(eigenvalues(j), 0.0);  // clamp round-off negatives
        total += lam;
        if (j >= r) {
            q.phi1 += lam;
            q.phi2 += lam * lam;
            q.phi3 += lam * lam * lam;
        }
    }
    // Residual subspace with (numerically) zero variance, e.g. perfectly
    // correlated links: use a tiny floor relative to the total variance so that
    // round-off in the SPE does not raise alarms.
    const double floor = 1e-12 * std::max(total, 1e-300);
    if (q.phi1 <= floor) {
        q.delta2 = floor;
        q.method = "degenerate-floor";
        return q;
    }

    q.h0 = 1.0 - 2.0 * q.phi1 * q.phi3 / (3.0 * q.phi2 * q.phi2);
    if (q.h0 > 1e-6) {
        const double term = q.c_alpha * std::sqrt(2.0 * q.phi2 * q.h0 * q.h0) / q.phi1 + 1.0 +
                            q.phi2 * q.h0 * (q.h0 - 1.0) / (q.phi1 * q.phi1);
        if (term > 0.0) {
            q.delta2 = q.phi1 * std::pow(term, 1.0 / q.h0);
            q.method = "jackson-mudholkar";
            return q;
        }
    }
    // Fallback (very skewed residual spectra make h0 <= 0): Box's approximation
    // SPE ~ g * chi2_h with g = phi2/phi1, h = phi1^2/phi2, using the
    // Wilson-Hilferty approximation for the chi-square quantile.
    const double g = q.phi2 / q.phi1;
    const double h = q.phi1 * q.phi1 / q.phi2;
    const double wh = 1.0 - 2.0 / (9.0 * h) + q.c_alpha * std::sqrt(2.0 / (9.0 * h));
    q.delta2 = g * h * wh * wh * wh;
    q.method = "box-chi2";
    return q;
}

std::vector<std::pair<Index, double>> top_residual_links(const Matrix& Yc, Index row,
                                                        const Matrix& P, int k) {
    const Vector y = Yc.row(row).transpose();
    const Vector residual = y - P * (P.transpose() * y);
    const Vector sq = residual.array().square();
    const double spe = sq.sum();
    std::vector<Index> order(static_cast<std::size_t>(sq.size()));
    std::iota(order.begin(), order.end(), Index{0});
    const auto count = static_cast<std::size_t>(std::min<Index>(k, sq.size()));
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count), order.end(),
                      [&](Index a, Index b) { return sq(a) > sq(b); });
    std::vector<std::pair<Index, double>> out;
    for (std::size_t i = 0; i < count; ++i)
        out.emplace_back(order[i], spe > 0.0 ? sq(order[i]) / spe : 0.0);
    return out;
}

}  // namespace pca
