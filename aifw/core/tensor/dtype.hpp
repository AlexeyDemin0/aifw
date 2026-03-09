#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace aifw::core {

enum class DType { Float32, Float64, Int32, Int64, Bool };

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
    case DType::Bool:
      return 1;
  }
  return 0;
}

template <typename Fn>
auto dtype_dispatch(DType dt, Fn&& fn) {
  switch (dt) {
    case DType::Float32:
      return fn.template operator()<float>();
    case DType::Float64:
      return fn.template operator()<double>();
    case DType::Int32:
      return fn.template operator()<int32_t>();
    case DType::Int64:
      return fn.template operator()<int64_t>();
    case DType::Bool:
      return fn.template operator()<bool>();
  }
  throw std::runtime_error("dtype_dispatch: unknown dtype");
}

}  // namespace aifw::core
