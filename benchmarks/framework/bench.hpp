#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ratio>
#include <string>
#include <vector>

namespace aifw::bench {

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double, std::milli>;

struct Result {
  std::string name;
  double mean_ms;
  double min_ms;
  double max_ms;
  double stddev_ms;
  size_t iterations;
};

struct BenchCase {
  std::string suite;
  std::string name;
  size_t iterations;
  std::function<void()> fn;
};

inline std::vector<BenchCase>& registry() {
  static std::vector<BenchCase> cases;
  return cases;
}

inline Result run_one(const BenchCase& bc) {
  std::vector<double> samples;
  samples.reserve(bc.iterations);

  // cache and branch predictor warmup
  for (size_t i = 0; i < std::min(bc.iterations / 10 + 1, size_t(5)); ++i)
    bc.fn();

  for (size_t i = 0; i < bc.iterations; ++i) {
    auto t0 = Clock::now();
    bc.fn();
    auto t1 = Clock::now();
    samples.push_back(Duration(t1 - t0).count());
  }

  double sum = 0;
  double mn = samples[0];
  double mx = samples[0];
  for (double s : samples) {
    sum += s;
    mn = std::min(mn, s);
    mx = std::max(mx, s);
  }
  double mean = sum / samples.size();

  double var = 0;
  for (double s : samples) var += (s - mean) * (s - mean);
  double stddev = std::sqrt(var / samples.size());

  return {bc.name, mean, mn, mx, stddev, bc.iterations};
}

inline int run_all() {
  constexpr int kName = 36;
  constexpr int kCol = 10;

  std::cout << "\n"
            << std::left << std::setw(kName) << "benchmark" << std::right
            << std::setw(kCol) << "mean ms" << std::setw(kCol) << "min ms"
            << std::setw(kCol) << "max ms" << std::setw(kCol) << "stddev"
            << std::setw(kCol) << "iters"
            << "\n"
            << std::string(kName + kCol * 5, '-') << "\n";

  std::string current_suite;

  for (auto& bc : registry()) {
    if (bc.suite != current_suite) {
      current_suite = bc.suite;
      std::cout << "\n[ " << bc.suite << " ]\n";
    }

    Result r = run_one(bc);

    std::cout << std::left << std::setw(kName) << (" " + r.name) << std::right
              << std::fixed << std::setprecision(4) << std::setw(kCol)
              << r.mean_ms << std::setw(kCol) << r.min_ms << std::setw(kCol)
              << r.max_ms << std::setw(kCol) << r.stddev_ms << std::setw(kCol)
              << r.iterations << "\n";
  }

  std::cout << "\n";
  return 0;
}

#define BENCH(suite, name, iterations)                         \
  static void bench_##suite##_##name();                        \
  static const bool _reg_##suite##_##name = []() {             \
    ::aifw::bench::registry().push_back(                       \
        {#suite, #name, iterations, &bench_##suite##_##name}); \
    return true;                                               \
  }();                                                         \
  static void bench_##suite##_##name()

#define BENCH_MAIN() \
  int main() { return ::aifw::bench::run_all(); }

}  // namespace aifw::bench
