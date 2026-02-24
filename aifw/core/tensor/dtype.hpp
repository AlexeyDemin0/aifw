#pragma once

#include <cstddef>

namespace aifw::core {

enum class DType { Float32, Float64, Int32, Int64 };

inline size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::Float32:
      return 4;
    case DType::Float64:
      return 8;
    case DType::Int32:
      return 4;
    case DType::Int64:
      return 8;
  }
  return 0;
}

}  // namespace aifw::core
