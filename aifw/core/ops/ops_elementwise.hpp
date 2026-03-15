#pragma once

#include "../tensor/tensor.hpp"
#include "kernel_registry.hpp"

namespace aifw::core::ops {

namespace nocheck {

inline void add(const Tensor& a, const Tensor& b, Tensor& out) {
  a.backend().kernels().add(a, b, out);
}

inline Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  nocheck::add(a, b, out);
  return out;
}

inline void sub(const Tensor& a, const Tensor& b, Tensor& out) {
  a.backend().kernels().sub(a, b, out);
}

inline Tensor sub(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  nocheck::sub(a, b, out);
  return out;
}

inline void mul(const Tensor& a, const Tensor& b, Tensor& out) {
  a.backend().kernels().mul(a, b, out);
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  nocheck::mul(a, b, out);
  return out;
}

inline void div(const Tensor& a, const Tensor& b, Tensor& out) {
  a.backend().kernels().div(a, b, out);
}

inline Tensor div(const Tensor& a, const Tensor& b) {
  Tensor out(a.backend(), a.shape(), a.dtype());
  nocheck::div(a, b, out);
  return out;
}

}  // namespace nocheck

inline void add(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(a.dtype() == b.dtype(), "add: dtype mismatch");
  AIFW_EXPECT(&a.backend() == &b.backend(), "add: different backends");
  AIFW_EXPECT(a.shape().values() == b.shape().values(), "add: shape mismatch");
  AIFW_EXPECT(a.dtype() == out.dtype(), "add: out dtype mismatch");
  AIFW_EXPECT(&a.backend() == &out.backend(), "add: out different backend");
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(), "add: out shape mismatch"
  );
  nocheck::add(a, b, out);
}

inline Tensor add(const Tensor& a, const Tensor& b) {
  AIFW_EXPECT(a.dtype() == b.dtype(), "add: dtype mismatch");
  AIFW_EXPECT(&a.backend() == &b.backend(), "add: different backends");
  AIFW_EXPECT(a.shape().values() == b.shape().values(), "add: shape mismatch");
  return nocheck::add(a, b);
}

inline void sub(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(a.dtype() == b.dtype(), "sub: dtype mismatch");
  AIFW_EXPECT(&a.backend() == &b.backend(), "sub: different backends");
  AIFW_EXPECT(a.shape().values() == b.shape().values(), "sub: shape mismatch");
  AIFW_EXPECT(a.dtype() == out.dtype(), "sub: out dtype mismatch");
  AIFW_EXPECT(&a.backend() == &out.backend(), "sub: out different backend");
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(), "sub: out shape mismatch"
  );
  nocheck::sub(a, b, out);
}

inline Tensor sub(const Tensor& a, const Tensor& b) {
  AIFW_EXPECT(a.dtype() == b.dtype(), "sub: dtype mismatch");
  AIFW_EXPECT(&a.backend() == &b.backend(), "sub: different backends");
  AIFW_EXPECT(a.shape().values() == b.shape().values(), "sub: shape mismatch");
  return nocheck::sub(a, b);
}

inline void mul(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(a.dtype() == b.dtype(), "mul: dtype mismatch");
  AIFW_EXPECT(&a.backend() == &b.backend(), "mul: different backends");
  AIFW_EXPECT(a.shape().values() == b.shape().values(), "mul: shape mismatch");
  AIFW_EXPECT(a.dtype() == out.dtype(), "mul: out dtype mismatch");
  AIFW_EXPECT(&a.backend() == &out.backend(), "mul: out different backend");
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(), "mul: out shape mismatch"
  );
  nocheck::mul(a, b, out);
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
  AIFW_EXPECT(a.dtype() == b.dtype(), "mul: dtype mismatch");
  AIFW_EXPECT(&a.backend() == &b.backend(), "mul: different backends");
  AIFW_EXPECT(a.shape().values() == b.shape().values(), "mul: shape mismatch");
  return nocheck::mul(a, b);
}

inline void div(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_EXPECT(a.dtype() == b.dtype(), "div: dtype mismatch");
  AIFW_EXPECT(&a.backend() == &b.backend(), "div: different backends");
  AIFW_EXPECT(a.shape().values() == b.shape().values(), "div: shape mismatch");
  AIFW_EXPECT(a.dtype() == out.dtype(), "div: out dtype mismatch");
  AIFW_EXPECT(&a.backend() == &out.backend(), "div: out different backend");
  AIFW_EXPECT(
      a.shape().values() == out.shape().values(), "div: out shape mismatch"
  );
  nocheck::div(a, b, out);
}

inline Tensor div(const Tensor& a, const Tensor& b) {
  AIFW_EXPECT(a.dtype() == b.dtype(), "div: dtype mismatch");
  AIFW_EXPECT(&a.backend() == &b.backend(), "div: different backends");
  AIFW_EXPECT(a.shape().values() == b.shape().values(), "div: shape mismatch");
  return nocheck::div(a, b);
}

}  // namespace aifw::core::ops
