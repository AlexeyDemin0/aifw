#pragma once

#include <cstring>

#include "../backend/backend.hpp"
#include "dtype.hpp"
#include "tensor.hpp"

namespace aifw::core {

inline Tensor zeros(IBackend& backend, Shape shape, DType dt) {
  Tensor t(backend, shape, dt);

  std::memset(t.data(), 0, t.numel() * dtype_size(dt));
  return t;
}

}  // namespace aifw::core
