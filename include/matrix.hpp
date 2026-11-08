#pragma once
// Matrix type, file I/O and input validation.
//
// Orientation convention (see docs/PHASE1_DESIGN.md section 5):
//   Mathematics follows the paper: Y is t x m (rows = time bins, columns = links),
//   and y is one row (a snapshot of all links at one time).
//   Eigen stores matrices column-major, so in memory every link's time series is
//   contiguous; physically this is the "link x time" layout of the project
//   registration. All heavy kernels stream along these contiguous columns.

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace pca {

using Matrix = Eigen::MatrixXd;  // column-major
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;      // signed (std::ptrdiff_t), safe as an OpenMP loop index

struct LabelledMatrix {
    Matrix values;                        // rows x cols
    std::vector<std::string> row_labels;  // first CSV column (e.g. time stamps)
    std::vector<std::string> col_labels;  // CSV header without the first cell
};

// CSV layout: a header line "<label>,<col1>,...,<colN>" followed by one line per
// row "<row label>,<v1>,...,<vN>". Throws std::runtime_error on malformed input.
LabelledMatrix load_csv(const std::string& path);

// Binary layout: 8-byte magic "PCAMAT01", int64 rows, int64 cols, then
// rows*cols little-endian float64 values in column-major order.
Matrix load_bin(const std::string& path);
void save_bin(const std::string& path, const Matrix& values);

// Loads .csv or .bin by extension. For .bin files, labels are generated.
LabelledMatrix load_matrix(const std::string& path);

// Throws std::invalid_argument if the matrix cannot be used as a measurement
// matrix: fewer than 2 rows or 2 columns, or any NaN / infinite entry.
void validate_measurements(const Matrix& Y, const std::string& name);

}  // namespace pca
