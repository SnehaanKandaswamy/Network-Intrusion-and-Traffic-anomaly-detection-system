#include "matrix.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pca {
namespace {

constexpr char kMagic[8] = {'P', 'C', 'A', 'M', 'A', 'T', '0', '1'};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream stream(line);
    while (std::getline(stream, cell, ',')) cells.push_back(cell);
    if (!line.empty() && line.back() == ',') cells.emplace_back();
    return cells;
}

void strip_cr(std::string& line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
}

double parse_number(const std::string& cell, std::size_t line_no, const std::string& path) {
    if (cell.empty())
        throw std::runtime_error(path + ":" + std::to_string(line_no) + ": empty value");
    // strtod accepts "nan"/"inf"; validate_measurements() rejects them later
    // with a clearer message.
    char* end = nullptr;
    const double value = std::strtod(cell.c_str(), &end);
    while (end && (*end == ' ' || *end == '\t')) ++end;
    if (end == cell.c_str() || (end && *end != '\0'))
        throw std::runtime_error(path + ":" + std::to_string(line_no) + ": not a number: '" +
                                 cell + "'");
    return value;
}

bool has_suffix(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

LabelledMatrix load_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error(path + ": file is empty");
    strip_cr(line);
    const std::vector<std::string> header = split_csv_line(line);
    if (header.size() < 2)
        throw std::runtime_error(path + ": header needs a label column and at least one data column");

    LabelledMatrix out;
    out.col_labels.assign(header.begin() + 1, header.end());
    const std::size_t cols = out.col_labels.size();

    std::vector<double> values;  // row-major while reading
    std::size_t line_no = 1;
    while (std::getline(in, line)) {
        ++line_no;
        strip_cr(line);
        if (line.empty()) continue;
        const std::vector<std::string> cells = split_csv_line(line);
        if (cells.size() != cols + 1)
            throw std::runtime_error(path + ":" + std::to_string(line_no) + ": expected " +
                                     std::to_string(cols + 1) + " fields, found " +
                                     std::to_string(cells.size()));
        out.row_labels.push_back(cells[0]);
        for (std::size_t c = 1; c < cells.size(); ++c)
            values.push_back(parse_number(cells[c], line_no, path));
    }
    if (out.row_labels.empty()) throw std::runtime_error(path + ": no data rows");

    const Index rows = static_cast<Index>(out.row_labels.size());
    out.values.resize(rows, static_cast<Index>(cols));
    for (Index r = 0; r < rows; ++r)
        for (Index c = 0; c < static_cast<Index>(cols); ++c)
            out.values(r, c) = values[static_cast<std::size_t>(r) * cols + static_cast<std::size_t>(c)];
    return out;
}

Matrix load_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[8];
    std::int64_t rows = 0, cols = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&rows), sizeof rows);
    in.read(reinterpret_cast<char*>(&cols), sizeof cols);
    if (!in || std::memcmp(magic, kMagic, 8) != 0)
        throw std::runtime_error(path + ": not a PCAMAT01 binary matrix");
    if (rows <= 0 || cols <= 0) throw std::runtime_error(path + ": invalid dimensions");
    Matrix values(rows, cols);
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(sizeof(double) * static_cast<std::size_t>(rows * cols)));
    if (!in) throw std::runtime_error(path + ": truncated data");
    return values;
}

void save_bin(const std::string& path, const Matrix& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    const std::int64_t rows = values.rows(), cols = values.cols();
    out.write(kMagic, 8);
    out.write(reinterpret_cast<const char*>(&rows), sizeof rows);
    out.write(reinterpret_cast<const char*>(&cols), sizeof cols);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(sizeof(double) * static_cast<std::size_t>(rows * cols)));
    if (!out) throw std::runtime_error("write failed: " + path);
}

LabelledMatrix load_matrix(const std::string& path) {
    if (has_suffix(path, ".csv")) return load_csv(path);
    if (has_suffix(path, ".bin")) {
        LabelledMatrix out;
        out.values = load_bin(path);
        for (Index r = 0; r < out.values.rows(); ++r) out.row_labels.push_back(std::to_string(r));
        for (Index c = 0; c < out.values.cols(); ++c) out.col_labels.push_back("link" + std::to_string(c));
        return out;
    }
    throw std::runtime_error(path + ": unsupported extension (use .csv or .bin)");
}

void validate_measurements(const Matrix& Y, const std::string& name) {
    if (Y.rows() < 2)
        throw std::invalid_argument(name + ": need at least 2 time bins, got " + std::to_string(Y.rows()));
    if (Y.cols() < 2)
        throw std::invalid_argument(name + ": need at least 2 links, got " + std::to_string(Y.cols()));
    for (Index c = 0; c < Y.cols(); ++c)
        for (Index r = 0; r < Y.rows(); ++r)
            if (!std::isfinite(Y(r, c)))
                throw std::invalid_argument(name + ": non-finite value at time " + std::to_string(r) +
                                            ", link " + std::to_string(c));
}

}  // namespace pca
