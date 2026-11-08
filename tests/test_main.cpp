// Unit tests (run with `ctest` or ./pca_tests). No external test framework.

#include "injection.hpp"
#include "kernels.hpp"
#include "pipeline.hpp"

#include <omp.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        ++g_checks;                                                                 \
        if (!(cond)) {                                                              \
            ++g_failures;                                                           \
            std::cerr << "  FAILED " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        }                                                                           \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                  \
    do {                                                                                       \
        ++g_checks;                                                                            \
        const double va = (a), vb = (b);                                                       \
        if (!(std::abs(va - vb) <= (tol))) {                                                   \
            ++g_failures;                                                                      \
            std::cerr << "  FAILED " << __FILE__ << ":" << __LINE__ << ": " #a " = " << va      \
                      << ", " #b " = " << vb << ", tol " << (tol) << "\n";                     \
        }                                                                                      \
    } while (0)

template <class E>
bool throws(const std::function<void()>& f) {
    try {
        f();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

using pca::Index;
using pca::Matrix;
using pca::Vector;

// Periodic low-rank traffic + noise (like the paper's link data): rank-`rank`
// diurnal structure across m links.
Matrix periodic_traffic(Index t, Index m, int rank, double noise, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Matrix W(rank, m);
    for (Index i = 0; i < W.size(); ++i) W.data()[i] = normal(gen);
    Matrix S(t, rank);
    for (Index k = 0; k < t; ++k)
        for (int f = 0; f < rank; ++f) {
            const double period = 144.0 / (1 + f / 2);
            const double phase = 6.283185307179586 * double(k) / period;
            S(k, f) = (f % 2 ? std::cos(phase) : std::sin(phase)) * (10.0 / (1 + f));
        }
    Matrix Y = S * W;
    for (Index i = 0; i < Y.size(); ++i) Y.data()[i] += 100.0 + noise * normal(gen);
    return Y;
}

Matrix random_matrix(Index t, Index m, unsigned seed) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Matrix Y(t, m);
    for (Index i = 0; i < Y.size(); ++i) Y.data()[i] = normal(gen) * (1.0 + double(i % 7));
    return Y;
}

// ----------------------------------------------------------------- tests

void test_inverse_normal() {
    CHECK_NEAR(pca::inverse_normal_cdf(0.5), 0.0, 1e-12);
    CHECK_NEAR(pca::inverse_normal_cdf(0.975), 1.959963984540054, 1e-9);
    CHECK_NEAR(pca::inverse_normal_cdf(0.995), 2.575829303548901, 1e-9);
    CHECK_NEAR(pca::inverse_normal_cdf(0.999), 3.090232306167813, 1e-9);
    CHECK_NEAR(pca::inverse_normal_cdf(0.001), -3.090232306167813, 1e-9);
    CHECK(throws<std::domain_error>([] { pca::inverse_normal_cdf(1.0); }));
}

void test_small_known_matrix() {
    // 4 time bins x 3 links, values chosen so the answers are easy by hand.
    Matrix Y(4, 3);
    Y << 1, 2, 0,
         3, 6, 1,
         5, 10, 0,
         7, 14, 1;
    const Vector mean = pca::seq::column_means(Y);
    CHECK_NEAR(mean(0), 4.0, 1e-15);
    CHECK_NEAR(mean(1), 8.0, 1e-15);
    CHECK_NEAR(mean(2), 0.5, 1e-15);
    Matrix Yc = Y;
    pca::seq::center_columns(Yc, mean);
    const Matrix C = pca::seq::covariance(Yc);
    // link0 deviations -3,-1,1,3 -> var = 20/3; link1 = 2*link0 -> var 80/3, cov 40/3
    CHECK_NEAR(C(0, 0), 20.0 / 3.0, 1e-12);
    CHECK_NEAR(C(1, 1), 80.0 / 3.0, 1e-12);
    CHECK_NEAR(C(0, 1), 40.0 / 3.0, 1e-12);
    CHECK_NEAR(C(1, 0), 40.0 / 3.0, 1e-12);
    // link2 deviations -.5,.5,-.5,.5 -> var 1/3; cov with link0 = (1.5-.5-.5+1.5)/3 = 2/3
    CHECK_NEAR(C(2, 2), 1.0 / 3.0, 1e-12);
    CHECK_NEAR(C(0, 2), 2.0 / 3.0, 1e-12);
    CHECK_NEAR(C(2, 0), 2.0 / 3.0, 1e-12);
}

void test_covariance_against_eigen() {
    // Sizes that are not multiples of the 4-column register block or the
    // 64-column tile, and t not a multiple of the 512-row chunk.
    const Index sizes[][2] = {{37, 5}, {1000, 131}, {513, 64}, {2049, 67}, {10, 2}};
    for (const auto& s : sizes) {
        Matrix Y = random_matrix(s[0], s[1], 7);
        const Vector mu = pca::seq::column_means(Y);
        pca::seq::center_columns(Y, mu);
        const Matrix expected = (Y.transpose() * Y) / double(s[0] - 1);
        const Matrix Cs = pca::seq::covariance(Y);
        const Matrix Co = pca::omp::covariance(Y);
        CHECK((Cs - expected).cwiseAbs().maxCoeff() <= 1e-12 * expected.cwiseAbs().maxCoeff());
        CHECK((Co - expected).cwiseAbs().maxCoeff() <= 1e-12 * expected.cwiseAbs().maxCoeff());
        CHECK(Cs.isApprox(Cs.transpose(), 0.0));
    }
}

void test_seq_omp_equivalence() {
    const Index sizes[][2] = {{2000, 300}, {1008, 60}, {777, 129}};
    for (const auto& s : sizes) {
        const Matrix Y0 = periodic_traffic(s[0], s[1], 4, 1.0, 11);
        for (const int p : {1, 3, 8, 16}) {
            omp_set_num_threads(p);
            Matrix Ys = Y0, Yo = Y0;
            pca::StageTimes ts, to;
            const pca::PcaModel ms = pca::fit<pca::SeqKernels>(Ys, pca::PcaConfig{}, ts);
            const pca::PcaModel mo = pca::fit<pca::OmpKernels>(Yo, pca::PcaConfig{}, to);
            const pca::DetectionResult rs = pca::score<pca::SeqKernels>(Ys, ms, ts);
            const pca::DetectionResult ro = pca::score<pca::OmpKernels>(Yo, mo, to);
            // Output-partitioned kernels -> bitwise identical results.
            CHECK(ms.mean == mo.mean);
            CHECK(Ys == Yo);
            CHECK(ms.eigenvalues == mo.eigenvalues);
            CHECK(ms.r == mo.r);
            CHECK(ms.q.delta2 == mo.q.delta2);
            CHECK(rs.spe == ro.spe);
            CHECK(rs.flags == ro.flags);
            CHECK(rs.n_alarms == ro.n_alarms);
            // Projection and the reduction-based deviation statistic.
            const Matrix Us = pca::seq::project(Ys, ms.eigenvectors.leftCols(8));
            const Matrix Uo = pca::omp::project(Yo, mo.eigenvectors.leftCols(8));
            CHECK(Us == Uo);
            const Vector ds = pca::seq::max_sigma_deviation(Us);
            const Vector dd = pca::omp::max_sigma_deviation(Uo);
            CHECK((ds - dd).cwiseAbs().maxCoeff() <= 1e-10 * ds.cwiseAbs().maxCoeff());
        }
    }
    omp_set_num_threads(omp_get_num_procs());
}

void test_spe_against_reference() {
    Matrix Y = periodic_traffic(500, 40, 4, 0.5, 3);
    pca::StageTimes times;
    pca::PcaConfig cfg;
    cfg.method = pca::RMethod::Fixed;
    cfg.fixed_r = 4;
    const pca::PcaModel model = pca::fit<pca::SeqKernels>(Y, cfg, times);
    const pca::DetectionResult res = pca::score<pca::OmpKernels>(Y, model, times);
    const Matrix residual = Y - (Y * model.P) * model.P.transpose();
    for (Index k = 0; k < Y.rows(); ++k) {
        CHECK_NEAR(res.spe(k), residual.row(k).squaredNorm(), 1e-9 * (1.0 + res.spe(k)));
        CHECK_NEAR(res.state_norm2(k), Y.row(k).squaredNorm(), 1e-9 * (1.0 + res.state_norm2(k)));
    }
    // Eigen-pairs satisfy C v = lambda v and are ordered/sign-normalised.
    const Matrix C = Y.transpose() * Y / double(Y.rows() - 1);
    for (Index i = 0; i < 5; ++i) {
        const Vector v = model.eigenvectors.col(i);
        CHECK((C * v - model.eigenvalues(i) * v).norm() <= 1e-9 * model.eigenvalues(0));
        Index arg = 0;
        v.cwiseAbs().maxCoeff(&arg);
        CHECK(v(arg) > 0);
        if (i > 0) CHECK(model.eigenvalues(i) <= model.eigenvalues(i - 1));
    }
}

void test_perfectly_correlated() {
    // Every link carries a scaled copy of the same signal: rank 1.
    const Index t = 300, m = 10;
    Matrix Y(t, m);
    for (Index k = 0; k < t; ++k)
        for (Index j = 0; j < m; ++j) Y(k, j) = (1.0 + j) * (50.0 + 10.0 * std::sin(0.1 * double(k)));
    pca::StageTimes times;
    pca::PcaConfig cfg;
    cfg.method = pca::RMethod::Fixed;
    cfg.fixed_r = 1;
    const pca::PcaModel model = pca::fit<pca::OmpKernels>(Y, cfg, times);
    CHECK(pca::numerical_rank(model.eigenvalues) == 1);
    CHECK(model.eigenvalues(1) <= 1e-10 * model.eigenvalues(0));
    const pca::DetectionResult res = pca::score<pca::OmpKernels>(Y, model, times);
    CHECK(res.n_alarms == 0);
    CHECK(res.spe.maxCoeff() <= 1e-18 * res.state_norm2.maxCoeff() + 1e-20);
    CHECK(std::string(model.q.method) == "degenerate-floor");
    // The sigma rule stops at the numerical rank (no axis beyond it has variance).
    Matrix Y2(t, m);
    for (Index k = 0; k < t; ++k)
        for (Index j = 0; j < m; ++j) Y2(k, j) = (1.0 + j) * (50.0 + 10.0 * std::sin(0.1 * double(k)));
    const pca::PcaModel m2 = pca::fit<pca::SeqKernels>(Y2, pca::PcaConfig{}, times);
    CHECK(m2.r == 1);
}

void test_injected_anomaly_detected() {
    const Index t = 1008, m = 40;
    Matrix Y = periodic_traffic(t, m, 4, 0.5, 5);
    Matrix Yc = Y;
    // Anomaly: +12 units on 4 links at bin 600 (an OD flow crossing 4 links).
    const Index spike_time = 600;
    for (const Index j : {3, 9, 17, 30}) Y(spike_time, j) += 12.0;
    pca::StageTimes times;
    const pca::PcaModel model = pca::fit<pca::OmpKernels>(Y, pca::PcaConfig{}, times);
    const pca::DetectionResult res = pca::score<pca::OmpKernels>(Y, model, times);
    CHECK(model.r >= 1 && model.r <= 8);
    CHECK(res.flags[static_cast<std::size_t>(spike_time)] == 1);
    CHECK(res.n_alarms <= 1 + t / 100);  // near the nominal 0.1 % false-alarm rate
    // The residual on the anomalous links dominates.
    const auto top = pca::top_residual_links(Y, spike_time, model.P, 4);
    int hits = 0;
    for (const auto& [link, share] : top) hits += (link == 3 || link == 9 || link == 17 || link == 30);
    CHECK(hits >= 3);
    (void)Yc;
}

void test_q_statistic_false_alarm_rate() {
    // Gaussian data with a known spectrum: 3 dominant + 17 residual directions.
    // With alpha = 0.01, about 1 % of rows should exceed delta^2.
    const Index t = 200000, m = 20;
    std::mt19937_64 gen(42);
    std::normal_distribution<double> normal(0.0, 1.0);
    Matrix Y(t, m);
    for (Index j = 0; j < m; ++j) {
        const double sd = j < 3 ? 30.0 / (1 + j) : 1.0 + 0.1 * double(j);
        for (Index k = 0; k < t; ++k) Y(k, j) = sd * normal(gen);
    }
    for (const double alpha : {0.01, 0.001}) {
        Matrix Yw = Y;
        pca::PcaConfig cfg;
        cfg.method = pca::RMethod::Fixed;
        cfg.fixed_r = 3;
        cfg.alpha = alpha;
        pca::StageTimes times;
        const pca::PcaModel model = pca::fit<pca::OmpKernels>(Yw, cfg, times);
        const pca::DetectionResult res = pca::score<pca::OmpKernels>(Yw, model, times);
        const double rate = double(res.n_alarms) / double(t);
        std::cout << "    Q-statistic alpha=" << alpha << ": empirical false-alarm rate " << rate << "\n";
        CHECK(rate > 0.5 * alpha && rate < 1.5 * alpha);
        CHECK(std::string(model.q.method) == "jackson-mudholkar");
    }
    // Literal reading of the paper (lambda = ||Y v||^2, not divided by t-1)
    // gives a threshold ~t times larger and detects nothing: ambiguity 1.
    Matrix Yw = Y;
    pca::PcaConfig cfg;
    cfg.method = pca::RMethod::Fixed;
    cfg.fixed_r = 3;
    pca::StageTimes times;
    const pca::PcaModel model = pca::fit<pca::SeqKernels>(Yw, cfg, times);
    const pca::QStatistic literal = pca::q_statistic(model.eigenvalues * double(t - 1), 3, 0.001);
    const pca::DetectionResult res = pca::score<pca::SeqKernels>(Yw, model, times);
    CHECK(res.spe.maxCoeff() < literal.delta2);
}

void test_numerical_stability_offset() {
    // Adding a huge constant to every link must not change the result: the
    // mean is removed before the covariance is formed.
    const Matrix Y = periodic_traffic(1008, 30, 4, 0.5, 8);
    Matrix Y1 = Y;
    Matrix Y2 = Y.array() + 1e9;
    pca::StageTimes times;
    pca::PcaConfig cfg;
    cfg.method = pca::RMethod::Fixed;
    cfg.fixed_r = 4;
    const pca::PcaModel m1 = pca::fit<pca::OmpKernels>(Y1, cfg, times);
    const pca::PcaModel m2 = pca::fit<pca::OmpKernels>(Y2, cfg, times);
    const pca::DetectionResult r1 = pca::score<pca::OmpKernels>(Y1, m1, times);
    const pca::DetectionResult r2 = pca::score<pca::OmpKernels>(Y2, m2, times);
    CHECK((m1.eigenvalues.head(4) - m2.eigenvalues.head(4)).cwiseAbs().maxCoeff() <=
          1e-6 * m1.eigenvalues(0));
    CHECK((r1.spe - r2.spe).cwiseAbs().maxCoeff() <= 1e-4 * r1.spe.maxCoeff());
    CHECK(std::abs(m1.q.delta2 - m2.q.delta2) <= 1e-4 * m1.q.delta2);
}

void test_invalid_input() {
    Matrix Y = random_matrix(10, 4, 1);
    Y(3, 2) = std::nan("");
    CHECK(throws<std::invalid_argument>([&] { pca::validate_measurements(Y, "Y"); }));
    Y(3, 2) = INFINITY;
    CHECK(throws<std::invalid_argument>([&] { pca::validate_measurements(Y, "Y"); }));
    CHECK(throws<std::invalid_argument>([] { pca::validate_measurements(Matrix(1, 5), "Y"); }));
    CHECK(throws<std::invalid_argument>([] { pca::validate_measurements(Matrix::Ones(5, 1), "Y"); }));
    CHECK(throws<std::invalid_argument>([] { (void)pca::q_statistic(Vector::Ones(4), 4, 0.01); }));
    CHECK(throws<std::invalid_argument>([] { (void)pca::q_statistic(Vector::Ones(4), 2, 1.5); }));
    CHECK(throws<std::invalid_argument>([] { (void)pca::parse_rmethod("bogus"); }));

    const auto dir = std::filesystem::temp_directory_path() / "pca_tests";
    std::filesystem::create_directories(dir);
    const std::string ragged = (dir / "ragged.csv").string();
    std::ofstream(ragged) << "time,a,b\n0,1,2\n1,3\n";
    CHECK(throws<std::runtime_error>([&] { (void)pca::load_csv(ragged); }));
    const std::string text = (dir / "text.csv").string();
    std::ofstream(text) << "time,a,b\n0,1,x\n";
    CHECK(throws<std::runtime_error>([&] { (void)pca::load_csv(text); }));
    const std::string empty = (dir / "empty.csv").string();
    std::ofstream(empty).close();
    CHECK(throws<std::runtime_error>([&] { (void)pca::load_csv(empty); }));
    CHECK(throws<std::runtime_error>([&] { (void)pca::load_csv((dir / "missing.csv").string()); }));
    CHECK(throws<std::runtime_error>([&] { (void)pca::load_matrix((dir / "x.txt").string()); }));

    // A matrix with a NaN read from CSV is rejected by fit().
    const std::string nan_csv = (dir / "nan.csv").string();
    std::ofstream(nan_csv) << "time,a,b\n0,1,2\n1,nan,3\n2,4,5\n";
    pca::LabelledMatrix lm = pca::load_csv(nan_csv);
    pca::StageTimes times;
    CHECK(throws<std::invalid_argument>([&] { (void)pca::fit<pca::SeqKernels>(lm.values, pca::PcaConfig{}, times); }));
}

void test_io_roundtrip() {
    const auto dir = std::filesystem::temp_directory_path() / "pca_tests";
    std::filesystem::create_directories(dir);
    const Matrix Y = random_matrix(17, 5, 9);
    const std::string bin = (dir / "y.bin").string();
    pca::save_bin(bin, Y);
    CHECK(pca::load_bin(bin) == Y);
    const std::string csv = (dir / "y.csv").string();
    {
        std::ofstream out(csv);
        out.precision(17);
        out << "time,l0,l1,l2,l3,l4\r\n";
        for (Index k = 0; k < Y.rows(); ++k) {
            out << "t" << k;
            for (Index j = 0; j < Y.cols(); ++j) out << ',' << Y(k, j);
            out << "\r\n";  // Windows line endings are accepted
        }
    }
    const pca::LabelledMatrix lm = pca::load_csv(csv);
    CHECK(lm.values == Y);
    CHECK(lm.col_labels.size() == 5 && lm.col_labels[4] == "l4");
    CHECK(lm.row_labels.size() == 17 && lm.row_labels[16] == "t16");
}

void test_r_selection_rules() {
    Vector lam(5);
    lam << 50, 30, 10, 6, 4;  // total 100
    CHECK(pca::r_from_variance(lam, 0.80) == 2);
    CHECK(pca::r_from_variance(lam, 0.90) == 3);
    std::string note;
    CHECK(pca::clamp_r(0, 5, note) == 1 && !note.empty());
    note.clear();
    CHECK(pca::clamp_r(9, 5, note) == 4 && !note.empty());
    note.clear();
    CHECK(pca::clamp_r(2, 5, note) == 2 && note.empty());
}

void test_injection_study_equivalence() {
    const Index t = 300, m = 12;
    const Matrix Y = periodic_traffic(t, m, 3, 0.3, 21);
    Matrix A = Matrix::Zero(m, 6);  // 6 "flows", each crossing 3 links
    for (Index f = 0; f < 6; ++f)
        for (Index h = 0; h < 3; ++h) A((f * 2 + h) % m, f) = 1.0;
    std::vector<pca::InjectionExperiment> plan;
    for (Index f = 0; f < 6; ++f)
        for (Index k = 100; k < 110; ++k) plan.push_back({f, k, 15.0});
    pca::PcaConfig cfg;
    const auto s = pca::seq::run_injection_study(Y, A, plan, cfg);
    omp_set_num_threads(4);
    const auto o = pca::omp::run_injection_study(Y, A, plan, cfg);
    std::size_t detected = 0;
    bool equal = s.size() == o.size();
    for (std::size_t i = 0; equal && i < s.size(); ++i) {
        equal = s[i].detected == o[i].detected && s[i].spe == o[i].spe && s[i].r == o[i].r;
        detected += s[i].detected;
    }
    CHECK(equal);
    CHECK(detected >= plan.size() * 9 / 10);  // a large spike should almost always be found
    std::vector<pca::InjectionExperiment> bad = {{99, 0, 1.0}};
    CHECK(throws<std::out_of_range>([&] { (void)pca::omp::run_injection_study(Y, A, bad, cfg); }));
    omp_set_num_threads(omp_get_num_procs());
}

}  // namespace

int main() {
    const std::pair<const char*, void (*)()> tests[] = {
        {"inverse normal CDF", test_inverse_normal},
        {"small known matrix (means, covariance)", test_small_known_matrix},
        {"covariance vs Eigen reference, odd sizes", test_covariance_against_eigen},
        {"sequential vs OpenMP equivalence (1,3,8,16 threads)", test_seq_omp_equivalence},
        {"SPE and eigenpairs vs direct formulas", test_spe_against_reference},
        {"perfectly correlated links", test_perfectly_correlated},
        {"injected anomaly is detected", test_injected_anomaly_detected},
        {"Q-statistic false-alarm rate", test_q_statistic_false_alarm_rate},
        {"numerical stability (1e9 offset)", test_numerical_stability_offset},
        {"invalid input is rejected", test_invalid_input},
        {"CSV / binary I/O round trip", test_io_roundtrip},
        {"r selection rules", test_r_selection_rules},
        {"injection study seq vs OpenMP", test_injection_study_equivalence},
    };
    for (const auto& [name, fn] : tests) {
        const int before = g_failures;
        std::cout << "[ RUN  ] " << name << "\n";
        fn();
        std::cout << (g_failures == before ? "[  OK  ] " : "[ FAIL ] ") << name << "\n";
    }
    std::cout << "\n" << g_checks << " checks, " << g_failures << " failures\n";
    return g_failures == 0 ? 0 : 1;
}
