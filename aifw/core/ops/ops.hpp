#pragma once

#include "../assert.hpp"
#include "../backend/backend.hpp"
#include "../tensor/tensor.hpp"
#include "kernel_registry.hpp"

namespace aifw::core::ops {

void add(const Tensor& a, const Tensor& b, Tensor& out);
void sub(const Tensor& a, const Tensor& b, Tensor& out);
void mul(const Tensor& a, const Tensor& b, Tensor& out);
void div(const Tensor& a, const Tensor& b, Tensor& out);
void matmul(const Tensor& a, const Tensor& b, Tensor& out);
void relu(const Tensor& a, Tensor& out);

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& a);

//

inline void add(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(
      &a.backend() == &b.backend(), "add: a and b are on different backends"
  );
  AIFW_EXPECT(
      &a.backend() == &out.backend(),
      "add: out is on a different backend than inputs"
  );

  AIFW_EXPECT(a.dtype() == b.dtype(), "add: dtype mismatch between a and b");
  AIFW_EXPECT(
      a.dtype() == out.dtype(), "add: dtype mismatch between input and out"
  );

  AIFW_EXPECT(
      a.shape().values() == b.shape().values(),
      "add: shape mismatch between a and b"
  );
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(),
      "add: shape mismatch between input and out"
  );

  a.backend().kernels().add(a, b, out);
}

inline void sub(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(
      &a.backend() == &b.backend(), "sub: a and b are on different backends"
  );
  AIFW_EXPECT(
      &a.backend() == &out.backend(),
      "sub: out is on a different backend than inputs"
  );

  AIFW_EXPECT(a.dtype() == b.dtype(), "sub: dtype mismatch between a and b");
  AIFW_EXPECT(
      a.dtype() == out.dtype(), "sub: dtype mismatch between input and out"
  );

  AIFW_EXPECT(
      a.shape().values() == b.shape().values(),
      "sub: shape mismatch between a and b"
  );
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(),
      "sub: shape mismatch between input and out"
  );

  a.backend().kernels().sub(a, b, out);
}

inline void mul(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(
      &a.backend() == &b.backend(), "mul: a and b are on different backends"
  );
  AIFW_EXPECT(
      &a.backend() == &out.backend(),
      "mul: out is on a different backend than inputs"
  );

  AIFW_EXPECT(a.dtype() == b.dtype(), "mul: dtype mismatch between a and b");
  AIFW_EXPECT(
      a.dtype() == out.dtype(), "mul: dtype mismatch between input and out"
  );

  AIFW_EXPECT(
      a.shape().values() == b.shape().values(),
      "mul: shape mismatch between a and b"
  );
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(),
      "mul: shape mismatch between input and out"
  );

  a.backend().kernels().mul(a, b, out);
}

inline void div(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(
      &a.backend() == &b.backend(), "div: a and b are on different backends"
  );
  AIFW_EXPECT(
      &a.backend() == &out.backend(),
      "div: out is on a different backend than inputs"
  );

  AIFW_EXPECT(a.dtype() == b.dtype(), "div: dtype mismatch between a and b");
  AIFW_EXPECT(
      a.dtype() == out.dtype(), "div: dtype mismatch between input and out"
  );

  AIFW_EXPECT(
      a.shape().values() == b.shape().values(),
      "div: shape mismatch between a and b"
  );
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(),
      "div: shape mismatch between input and out"
  );

  a.backend().kernels().div(a, b, out);
}

inline void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(
      &a.backend() == &b.backend(), "matmul: a and b are on different backends"
  );
  AIFW_EXPECT(
      &a.backend() == &out.backend(),
      "matmul: out is on a different backend than inputs"
  );

  AIFW_EXPECT(a.dtype() == b.dtype(), "matmul: dtype mismatch between a and b");
  AIFW_EXPECT(
      a.dtype() == out.dtype(), "matmul: dtype mismatch between input and out"
  );

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

  a.backend().kernels().matmul(a, b, out);
}

inline void relu(const Tensor& a, Tensor& out) {
  AIFW_EXPECT(
      &a.backend() == &out.backend(),
      "relu: out is on a different backend than input"
  );

  AIFW_EXPECT(
      a.dtype() == out.dtype(), "relu: dtype mismatch between input and out"
  );

  AIFW_EXPECT(
      a.shape().values() == out.shape().values(),
      "relu: shape mismatch between input and out"
  );

  a.backend().kernels().relu(a, out);
}

inline Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::add(a, b, out);
  return out;
}

inline Tensor sub(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::sub(a, b, out);
  return out;
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::mul(a, b, out);
  return out;
}

inline Tensor div(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::div(a, b, out);
  return out;
}

inline Tensor matmul(const Tensor& a, const Tensor& b) {
  Shape out_shape{a.shape()[0], b.shape()[1]};
  Tensor out(a.backend(), out_shape, a.dtype());
  ops::matmul(a, b, out);
  return out;
}

inline Tensor relu(const Tensor& a) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::relu(a, out);
  return out;
}

}  // namespace aifw::core::ops
