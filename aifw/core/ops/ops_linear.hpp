#pragma once

#include "../tensor/tensor.hpp"
#include "kernel_registry.hpp"

namespace aifw::core::ops {

namespace nocheck {

inline void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  a.backend().kernels().matmul(a, b, out);
}

inline Tensor matmul(const Tensor& a, const Tensor& b) {
  Shape out_shape{a.shape()[0], b.shape()[1]};
  Tensor out(a.backend(), std::move(out_shape), a.dtype());
  a.backend().kernels().matmul(a, b, out);
  return out;
}

}  // namespace nocheck

inline void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(&a.backend() == &b.backend(), "matmul: different backends");
  AIFW_EXPECT(&a.backend() == &out.backend(), "matmul: out different backend");
  AIFW_EXPECT(a.dtype() == b.dtype(), "matmul: dtype mismatch");
  AIFW_EXPECT(a.dtype() == out.dtype(), "matmul: out dtype mismatch");

  AIFW_EXPECT(a.shape().rank() == 2, "matmul: a must be 2D");
  AIFW_EXPECT(b.shape().rank() == 2, "matmul: b must be 2D");
  AIFW_EXPECT(out.shape().rank() == 2, "matmul: out must be 2D");

  AIFW_EXPECT(
      a.shape()[1] == b.shape()[0],
      "matmul: inner dims mismatch (a cols != b rows)"
  );

  AIFW_EXPECT(
      out.shape()[0] == a.shape()[0], "matmul: out rows must match a rows"
  );
  AIFW_EXPECT(
      out.shape()[1] == b.shape()[1], "matmul: out cols must match b cols"
  );
  nocheck::matmul(a, b, out);
}

inline Tensor matmul(const Tensor& a, const Tensor& b) {
  AIFW_EXPECT(&a.backend() == &b.backend(), "matmul: different backends");
  AIFW_EXPECT(a.dtype() == b.dtype(), "matmul: dtype mismatch");

  AIFW_EXPECT(a.shape().rank() == 2, "matmul: a must be 2D");
  AIFW_EXPECT(b.shape().rank() == 2, "matmul: b must be 2D");

  AIFW_EXPECT(
      a.shape()[1] == b.shape()[0],
      "matmul: inner dims mismatch (a cols != b rows)"
  );
  return nocheck::matmul(a, b);
}

}  // namespace aifw::core::ops
