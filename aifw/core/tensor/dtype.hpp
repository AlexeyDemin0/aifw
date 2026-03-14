#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace aifw::core {

enum class DType : uint8_t { Float32, Float64, Int32, Int64, Bool };

template <typename T>
struct dtype_of {
  static_assert(false, "dtype_of: unsupported dtype");
};

template <>
struct dtype_of<float> {
  static constexpr DType value = DType::Float32;
};

template <>
struct dtype_of<double> {
  static constexpr DType value = DType::Float64;
};

template <>
struct dtype_of<int32_t> {
  static constexpr DType value = DType::Int32;
};

template <>
struct dtype_of<int64_t> {
  static constexpr DType value = DType::Int64;
};

template <>
struct dtype_of<bool> {
  static constexpr DType value = DType::Bool;
};

template <typename T>
inline constexpr DType dtype_of_v = dtype_of<T>::value;

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
