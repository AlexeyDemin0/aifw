#pragma once

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

#include "shape.hpp"

namespace aifw::core {

class Stride {
 public:
  Stride() = default;

  explicit Stride(std::vector<size_t> strides);
  explicit Stride(std::initializer_list<size_t> strides);

  size_t operator[](size_t i) const;

  const std::vector<size_t>& values() const;

 private:
  std::vector<size_t> strides_;
};

inline Stride::Stride(std::vector<size_t> strides)
    : strides_(std::move(strides)) {}

inline Stride::Stride(std::initializer_list<size_t> strides)
    : Stride(std::vector<size_t>(strides)) {}

inline size_t Stride::operator[](size_t i) const { return strides_[i]; }

inline const std::vector<size_t>& Stride::values() const { return strides_; }

inline Stride make_contiguous_stride(const Shape& shape) {
  std::vector<size_t> s(shape.rank());
  size_t acc = 1;
  for (size_t i = shape.rank(); i-- > 0;) {
    s[i] = acc;
    acc *= shape[i];
  }
  return Stride(std::move(s));
}

}  // namespace aifw::core
