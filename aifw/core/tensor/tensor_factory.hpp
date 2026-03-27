#pragma once

#include <utility>

#include "dtype.hpp"
#include "tensor.hpp"

namespace aifw::core {

inline Tensor zeros(IDevice& device, Shape shape, DType dt) {
  Tensor t(device, std::move(shape), dt);
  device.kernels().fill(t, 0.0);
  return t;
}

inline Tensor ones(IDevice& device, Shape shape, DType dt) {
  Tensor t(device, std::move(shape), dt);
  device.kernels().fill(t, 1.0);
  return t;
}

inline Tensor full(IDevice& device, Shape shape, DType dt, double value) {
  Tensor t(device, std::move(shape), dt);
  device.kernels().fill(t, value);
  return t;
}

}  // namespace aifw::core
