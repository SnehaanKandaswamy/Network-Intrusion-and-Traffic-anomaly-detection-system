#include "utils.hpp"

#include <filesystem>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace pca {

double peak_rss_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters))
        return static_cast<double>(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
    return 0.0;
#else
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);  // bytes
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;  // KiB
#endif
#endif
}

double process_cpu_seconds() {
#if defined(_WIN32)
    FILETIME creation, exit, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) return 0.0;
    auto to_seconds = [](const FILETIME& ft) {
        ULARGE_INTEGER v;
        v.LowPart = ft.dwLowDateTime;
        v.HighPart = ft.dwHighDateTime;
        return static_cast<double>(v.QuadPart) * 1e-7;  // 100 ns units
    };
    return to_seconds(kernel) + to_seconds(user);
#else
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
           1e-6 * static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
#endif
}

Args::Args(int argc, char** argv, const std::set<std::string>& value_keys,
           const std::set<std::string>& flag_keys) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument: " + arg);
        const std::string key = arg.substr(2);
        if (flag_keys.count(key)) {
            values_[key] = "1";
        } else if (value_keys.count(key)) {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            values_[key] = argv[++i];
        } else {
            throw std::invalid_argument("unknown option: " + arg + " (see --help)");
        }
    }
}

bool Args::has(const std::string& key) const { return values_.count(key) != 0; }

std::string Args::get(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
}

std::string Args::require(const std::string& key) const {
    const auto it = values_.find(key);
    if (it == values_.end()) throw std::invalid_argument("missing required option --" + key);
    return it->second;
}

double Args::get_double(const std::string& key, double fallback) const {
    if (!has(key)) return fallback;
    try {
        return std::stod(get(key, ""));
    } catch (const std::exception&) {
        throw std::invalid_argument("--" + key + " expects a number");
    }
}

long long Args::get_int(const std::string& key, long long fallback) const {
    if (!has(key)) return fallback;
    try {
        return std::stoll(get(key, ""));
    } catch (const std::exception&) {
        throw std::invalid_argument("--" + key + " expects an integer");
    }
}

std::vector<std::string> split(const std::string& text, char sep) {
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(text);
    while (std::getline(stream, part, sep))
        if (!part.empty()) parts.push_back(part);
    return parts;
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> out;
    for (const auto& p : split(text, ',')) out.push_back(std::stoi(p));
    if (out.empty()) throw std::invalid_argument("empty list: " + text);
    return out;
}

std::vector<double> parse_double_list(const std::string& text) {
    std::vector<double> out;
    for (const auto& p : split(text, ',')) out.push_back(std::stod(p));
    if (out.empty()) throw std::invalid_argument("empty list: " + text);
    return out;
}

void ensure_directory(const std::string& path) {
    if (!path.empty()) std::filesystem::create_directories(path);
}

std::string json_escape(const std::string& text) {
    std::string out;
    for (const char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

}  // namespace pca
