#pragma once

#include "../backend/backend.hpp"
#include "../contracts.hpp"
#include "../tensor/tensor.hpp"
#include "kernel_registry.hpp"

namespace aifw::core::ops {

template <typename Policy = SafePolicy>
void add(const Tensor& a, const Tensor& b, Tensor& out);
template <typename Policy = SafePolicy>
void sub(const Tensor& a, const Tensor& b, Tensor& out);
template <typename Policy = SafePolicy>
void mul(const Tensor& a, const Tensor& b, Tensor& out);
template <typename Policy = SafePolicy>
void div(const Tensor& a, const Tensor& b, Tensor& out);
template <typename Policy = SafePolicy>
void matmul(const Tensor& a, const Tensor& b, Tensor& out);
template <typename Policy = SafePolicy>
void relu(const Tensor& a, Tensor& out);

template <typename Policy = SafePolicy>
Tensor add(const Tensor& a, const Tensor& b);
template <typename Policy = SafePolicy>
Tensor sub(const Tensor& a, const Tensor& b);
template <typename Policy = SafePolicy>
Tensor mul(const Tensor& a, const Tensor& b);
template <typename Policy = SafePolicy>
Tensor div(const Tensor& a, const Tensor& b);
template <typename Policy = SafePolicy>
Tensor matmul(const Tensor& a, const Tensor& b);
template <typename Policy = SafePolicy>
Tensor relu(const Tensor& a);

//

template <typename Policy>
inline void add(const Tensor& a, const Tensor& b, Tensor& out) {
  Policy::expect(
      &a.backend() == &b.backend(), "add: a and b are on different backends"
  );
  Policy::expect(
      &a.backend() == &out.backend(),
      "add: out is on a different backend than inputs"
  );

  Policy::expect(a.dtype() == b.dtype(), "add: dtype mismatch between a and b");
  Policy::expect(
      a.dtype() == out.dtype(), "add: dtype mismatch between input and out"
  );

  Policy::expect(
      a.shape().values() == b.shape().values(),
      "add: shape mismatch between a and b"
  );
  Policy::expect(
      a.shape().values() == out.shape().values(),
      "add: shape mismatch between input and out"
  );

  a.backend().kernels().add(a, b, out);
}

template <typename Policy>
inline void sub(const Tensor& a, const Tensor& b, Tensor& out) {
  Policy::expect(
      &a.backend() == &b.backend(), "sub: a and b are on different backends"
  );
  Policy::expect(
      &a.backend() == &out.backend(),
      "sub: out is on a different backend than inputs"
  );

  Policy::expect(a.dtype() == b.dtype(), "sub: dtype mismatch between a and b");
  Policy::expect(
      a.dtype() == out.dtype(), "sub: dtype mismatch between input and out"
  );

  Policy::expect(
      a.shape().values() == b.shape().values(),
      "sub: shape mismatch between a and b"
  );
  Policy::expect(
      a.shape().values() == out.shape().values(),
      "sub: shape mismatch between input and out"
  );

  a.backend().kernels().sub(a, b, out);
}

template <typename Policy>
inline void mul(const Tensor& a, const Tensor& b, Tensor& out) {
  Policy::expect(
      &a.backend() == &b.backend(), "mul: a and b are on different backends"
  );
  Policy::expect(
      &a.backend() == &out.backend(),
      "mul: out is on a different backend than inputs"
  );

  Policy::expect(a.dtype() == b.dtype(), "mul: dtype mismatch between a and b");
  Policy::expect(
      a.dtype() == out.dtype(), "mul: dtype mismatch between input and out"
  );

  Policy::expect(
      a.shape().values() == b.shape().values(),
      "mul: shape mismatch between a and b"
  );
  Policy::expect(
      a.shape().values() == out.shape().values(),
      "mul: shape mismatch between input and out"
  );

  a.backend().kernels().mul(a, b, out);
}

template <typename Policy>
inline void div(const Tensor& a, const Tensor& b, Tensor& out) {
  Policy::expect(
      &a.backend() == &b.backend(), "div: a and b are on different backends"
  );
  Policy::expect(
      &a.backend() == &out.backend(),
      "div: out is on a different backend than inputs"
  );

  Policy::expect(a.dtype() == b.dtype(), "div: dtype mismatch between a and b");
  Policy::expect(
      a.dtype() == out.dtype(), "div: dtype mismatch between input and out"
  );

  Policy::expect(
      a.shape().values() == b.shape().values(),
      "div: shape mismatch between a and b"
  );
  Policy::expect(
      a.shape().values() == out.shape().values(),
      "div: shape mismatch between input and out"
  );

  a.backend().kernels().div(a, b, out);
}

template <typename Policy>
inline void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  Policy::expect(
      &a.backend() == &b.backend(), "matmul: a and b are on different backends"
  );
  Policy::expect(
      &a.backend() == &out.backend(),
      "matmul: out is on a different backend than inputs"
  );

  Policy::expect(
      a.dtype() == b.dtype(), "matmul: dtype mismatch between a and b"
  );
  Policy::expect(
      a.dtype() == out.dtype(), "matmul: dtype mismatch between input and out"
  );

  Policy::expect(a.shape().rank() == 2, "matmul: a must be 2D");
  Policy::expect(b.shape().rank() == 2, "matmul: b must be 2D");
  Policy::expect(out.shape().rank() == 2, "matmul: out must be 2D");

  Policy::expect(
      a.shape()[1] == b.shape()[0],
      "matmul: inner dims mismatch (a cols != b rows)"
  );

  Policy::expect(
      out.shape()[0] == a.shape()[0], "matmul: out rows must match a rows"
  );
  Policy::expect(
      out.shape()[1] == b.shape()[1], "matmul: out cols must match b cols"
  );

  a.backend().kernels().matmul(a, b, out);
}

template <typename Policy>
inline void relu(const Tensor& a, Tensor& out) {
  Policy::expect(
      &a.backend() == &out.backend(),
      "relu: out is on a different backend than input"
  );

  Policy::expect(
      a.dtype() == out.dtype(), "relu: dtype mismatch between input and out"
  );

  Policy::expect(
      a.shape().values() == out.shape().values(),
      "relu: shape mismatch between input and out"
  );

  a.backend().kernels().relu(a, out);
}

template <typename Policy>
inline Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::add<Policy>(a, b, out);
  return out;
}

template <typename Policy>
inline Tensor sub(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::sub<Policy>(a, b, out);
  return out;
}

template <typename Policy>
inline Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::mul<Policy>(a, b, out);
  return out;
}

template <typename Policy>
inline Tensor div(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::div<Policy>(a, b, out);
  return out;
}

template <typename Policy>
inline Tensor matmul(const Tensor& a, const Tensor& b) {
  Shape out_shape{a.shape()[0], b.shape()[1]};
  Tensor out(a.backend(), out_shape, a.dtype());
  ops::matmul<Policy>(a, b, out);
  return out;
}

template <typename Policy>
inline Tensor relu(const Tensor& a) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  ops::relu<Policy>(a, out);
  return out;
}

}  // namespace aifw::core::ops
