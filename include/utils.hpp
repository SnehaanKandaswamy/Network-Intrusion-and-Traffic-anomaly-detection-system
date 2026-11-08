#pragma once
// Timing, resource measurement and a minimal command-line parser.

#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace pca {

class Timer {
public:
    Timer() : start_(Clock::now()) {}
    void reset() { start_ = Clock::now(); }
    double seconds() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

double peak_rss_mb();           // peak resident set size of this process (MiB)
double process_cpu_seconds();   // user + system CPU time of this process (s)

// Parses "--key value" pairs and "--flag" switches. Unknown keys are an error.
class Args {
public:
    Args(int argc, char** argv, const std::set<std::string>& value_keys,
         const std::set<std::string>& flag_keys);
    bool has(const std::string& key) const;
    std::string get(const std::string& key, const std::string& fallback) const;
    std::string require(const std::string& key) const;
    double get_double(const std::string& key, double fallback) const;
    long long get_int(const std::string& key, long long fallback) const;

private:
    std::map<std::string, std::string> values_;
};

std::vector<std::string> split(const std::string& text, char sep);
std::vector<int> parse_int_list(const std::string& text);
std::vector<double> parse_double_list(const std::string& text);

void ensure_directory(const std::string& path);
std::string json_escape(const std::string& text);

}  // namespace pca
