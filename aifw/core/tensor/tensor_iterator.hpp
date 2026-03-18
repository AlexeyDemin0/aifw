#pragma once

#include <cstddef>
#include <vector>

#include "aifw/core/tensor/shape.hpp"
#include "aifw/core/tensor/stride.hpp"
#include "aifw/core/tensor/tensor.hpp"

namespace aifw::core {

struct ComputeOffsetPolicy {
  struct State {};

  static State init(const Tensor& t) { return {}; }

  static size_t offset_at(
      size_t flat, const Shape& shape, const Stride& stride, State& state
  ) {
    size_t off = 0;
    for (size_t d = shape.rank(); d-- > 0;) {
      off += (flat % shape[d]) * stride[d];
      flat /= shape[d];
    }
    return off;
  }
};

struct CacheOffsetPolicy {
  struct State {
    std::vector<size_t> offsets;
  };

  static State init(const Tensor& t) {
    State s;
    s.offsets.resize(t.numel());
    for (size_t i = 0; i < t.numel(); ++i)
      s.offsets[i] = compute_offset(i, t.shape(), t.stride());
    return s;
  }

  static size_t offset_at(
      size_t flat, const Shape& shape, const Stride& stride, State& state
  ) {
    return state.offsets[flat];
  }

 private:
  static size_t compute_offset(
      size_t flat, const Shape& shape, const Stride& stride
  ) {
    size_t off = 0;
    for (size_t d = shape.rank(); d-- > 0;) {
      off += (flat % shape[d]) * stride[d];
      flat /= shape[d];
    }
    return off;
  }
};

template <typename Policy = ComputeOffsetPolicy>
class TensorIterator {
 public:
  explicit TensorIterator(const Tensor& t)
      : shape_(t.shape()),
        stride_(t.stride()),
        offset_(t.offset()),
        state_(Policy::init(t)) {}

  size_t operator[](size_t flat) {
    return offset_ + Policy::offset_at(flat, shape_, stride_, state_);
  }

 private:
  Shape shape_;
  Stride stride_;
  size_t offset_;
  typename Policy::State state_;
};

using ComputeTensorIterator = TensorIterator<ComputeOffsetPolicy>;
using CacheTensorIterator = TensorIterator<CacheOffsetPolicy>;

}  // namespace aifw::core
