#pragma once

#include <cassert>
#include <stdexcept>

namespace aifw::core {

struct SafePolicy {
  static void expect(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
  }
};

struct UnsafePolicy {
  static void expect(bool cond, const char* msg) {}
};

#define AIFW_EXPECT(cond, msg)                  \
  do {                                          \
    if (!(cond)) throw std::runtime_error(msg); \
  } while (0)

#define AIFW_ASSERT(cond) assert(cond)

}  // namespace aifw::core
