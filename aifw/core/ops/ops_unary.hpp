#pragma once

#include "../tensor/tensor.hpp"
#include "kernel_registry.hpp"

namespace aifw::core::ops {

namespace nocheck {

inline void relu(const Tensor& a, Tensor& out) {
  a.backend().kernels().relu(a, out);
}

inline Tensor relu(const Tensor& a) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  nocheck::relu(a, out);
  return out;
}

}  // namespace nocheck

inline void relu(const Tensor& a, Tensor& out) {
  AIFW_EXPECT(a.dtype() == out.dtype(), "relu: dtype mismatch");
  AIFW_EXPECT(&a.backend() == &out.backend(), "relu: different backends");
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(), "relu: shape mismatch"
  );
  nocheck::relu(a, out);
}

inline Tensor relu(const Tensor& a) { return nocheck::relu(a); }

}  // namespace aifw::core::ops
