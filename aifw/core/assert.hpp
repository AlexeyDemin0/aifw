#pragma once

#include <cassert>
// IWYU pragma: begin_exports
#include <stdexcept>
// IWYU pragma: end_exports

namespace aifw::core {

#define AIFW_EXPECT(cond, msg)                  \
  do {                                          \
    if (!(cond)) throw std::runtime_error(msg); \
  } while (0)

#define AIFW_ASSERT(cond) assert(cond)

}  // namespace aifw::core
