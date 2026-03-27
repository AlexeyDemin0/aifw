#pragma once

#include <cstring>
#include <utility>

#include "dtype.hpp"
#include "tensor.hpp"

namespace aifw::core {

inline Tensor zeros(IDevice& device, Shape shape, DType dt) {
  Tensor t(device, std::move(shape), dt);

  std::memset(t.data(), 0, t.numel() * dtype_size(dt));
  return t;
}

}  // namespace aifw::core
