#pragma once

#include "../tensor/tensor.hpp"

namespace aifw::core::ops {

namespace nocheck {

inline void relu(const Tensor& a, Tensor& out) {
  a.device().kernels().relu(a, out);
}

inline Tensor relu(const Tensor& a) {
  Tensor out(a.device(), a.shape(), a.dtype());
  nocheck::relu(a, out);
  return out;
}

}  // namespace nocheck

inline void relu(const Tensor& a, Tensor& out) {
  AIFW_EXPECT(a.dtype() == out.dtype(), "relu: dtype mismatch");
  AIFW_EXPECT(a.device_id() == out.device_id(), "relu: different backends");
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(), "relu: shape mismatch"
  );
  nocheck::relu(a, out);
}

inline Tensor relu(const Tensor& a) { return nocheck::relu(a); }

}  // namespace aifw::core::ops
