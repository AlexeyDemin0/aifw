#pragma once

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aifw::test {

namespace detail {

constexpr const char* kGreen = "\033[32m";
constexpr const char* kRed = "\033[31m";
constexpr const char* kYellow = "\033[33m";
constexpr const char* kReset = "\033[0m";

}  // namespace detail

struct TestCase {
  std::string suite;
  std::string name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> cases;
  return cases;
}

inline int run_all() {
  int passed = 0;
  int failed = 0;
  std::string current_suite;

  for (auto& tc : registry()) {
    if (tc.suite != current_suite) {
      current_suite = tc.suite;
      std::cout << "\n[ " << tc.suite << " ]\n";
    }

    try {
      tc.fn();
      std::cout << detail::kGreen << " PASS" << detail::kReset << " " << tc.name
                << "\n";
      ++passed;
    } catch (const std::exception& e) {
      std::cout << detail::kRed << " FAIL" << detail::kReset << " " << tc.name
                << "\n"
                << "        " << detail::kYellow << e.what() << detail::kReset
                << "\n";
      ++failed;
    }
  }

  std::cout << "\n-------------------------------------\n"
            << detail::kGreen << " passed: " << passed << detail::kReset << "\n"
            << detail::kRed << " failed: " << failed << detail::kReset << "\n";
  return failed == 0 ? 0 : 1;
}

inline void expect_true(
    bool cond, const char* expr, const char* file, int line
) {
  if (!cond)
    throw std::runtime_error(
        std::string(expr) + " [" + file + ":" + std::to_string(line) + "]"
    );
}

template <typename T>
void expect_eq(const T& a, const T& b, const char* file, int line) {
  if (a != b)
    throw std::runtime_error(
        "expected equal [" + std::string(file) + ":" + std::to_string(line) +
        "]"
    );
}

template <typename T>
void expect_near(T a, T b, T eps, const char* file, int line) {
  if (std::abs(a - b) > eps)
    throw std::runtime_error(
        "expected near: |" + std::to_string(a) + " - " + std::to_string(b) +
        "| > " + std::to_string(eps) + " [" + std::string(file) + ":" +
        std::to_string(line) + "]"
    );
}

template <typename Ex, typename Fn>
void expect_throws(Fn&& fn, const char* file, int line) {
  try {
    fn();
    throw std::runtime_error(
        std::string("expected exception not thrown") + " [" + file + ":" +
        std::to_string(line) + "]"
    );
  } catch (const Ex&) {  // NOLINT
    // expected
  }
}

#define TEST(suite, name)                          \
  static void test_##suite##_##name();             \
  static const bool _reg_##suite##_##name = []() { \
    ::aifw::test::registry().push_back(            \
        {#suite, #name, &test_##suite##_##name}    \
    );                                             \
    return true;                                   \
  }();                                             \
  static void test_##suite##_##name()

#define EXPECT_TRUE(cond) \
  ::aifw::test::expect_true((cond), #cond, __FILE__, __LINE__)

#define EXPECT_EQ(a, b) ::aifw::test::expect_eq((a), (b), __FILE__, __LINE__)

#define EXPECT_NEAR(a, b, eps) \
  ::aifw::test::expect_near((a), (b), (eps), __FILE__, __LINE__)

#define EXPECT_THROWS(ex, fn) \
  ::aifw::test::expect_throws<ex>((fn), __FILE__, __LINE__)

#define AIFW_TEST_MAIN() \
  int main() { return ::aifw::test::run_all(); }

}  // namespace aifw::test
