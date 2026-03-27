#pragma once

#include "../tensor/tensor.hpp"

namespace aifw::core::ops {

namespace detail {

inline void validate_binary_op(const Tensor& a, const Tensor& b) {
  AIFW_EXPECT(a.dtype() == b.dtype(), "validate_binary_op: dtype mismatch");
  AIFW_EXPECT(
      a.device_id() == b.device_id(), "validate_binary_op: different backends"
  );
  AIFW_EXPECT(
      a.shape().values() == b.shape().values(),
      "validate_binary_op: shape mismatch"
  );
}

}  // namespace detail

namespace nocheck {

inline void add(const Tensor& a, const Tensor& b, Tensor& out) {
  a.device().kernels().add(a, b, out);
}

inline Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out(a.device(), a.shape(), a.dtype());
  nocheck::add(a, b, out);
  return out;
}

inline void sub(const Tensor& a, const Tensor& b, Tensor& out) {
  a.device().kernels().sub(a, b, out);
}

inline Tensor sub(const Tensor& a, const Tensor& b) {
  Tensor out(a.device(), a.shape(), a.dtype());
  nocheck::sub(a, b, out);
  return out;
}

inline void mul(const Tensor& a, const Tensor& b, Tensor& out) {
  a.device().kernels().mul(a, b, out);
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
  Tensor out(a.device(), a.shape(), a.dtype());
  nocheck::mul(a, b, out);
  return out;
}

inline void div(const Tensor& a, const Tensor& b, Tensor& out) {
  a.device().kernels().div(a, b, out);
}

inline Tensor div(const Tensor& a, const Tensor& b) {
  Tensor out(a.device(), a.shape(), a.dtype());
  nocheck::div(a, b, out);
  return out;
}

}  // namespace nocheck

inline void add(const Tensor& a, const Tensor& b, Tensor& out) {
  detail::validate_binary_op(a, b);
  detail::validate_binary_op(a, out);
  nocheck::add(a, b, out);
}

inline Tensor add(const Tensor& a, const Tensor& b) {
  detail::validate_binary_op(a, b);
  return nocheck::add(a, b);
}

inline void sub(const Tensor& a, const Tensor& b, Tensor& out) {
  detail::validate_binary_op(a, b);
  detail::validate_binary_op(a, out);
  nocheck::sub(a, b, out);
}

inline Tensor sub(const Tensor& a, const Tensor& b) {
  detail::validate_binary_op(a, b);
  return nocheck::sub(a, b);
}

inline void mul(const Tensor& a, const Tensor& b, Tensor& out) {
  detail::validate_binary_op(a, b);
  detail::validate_binary_op(a, out);
  nocheck::mul(a, b, out);
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
  detail::validate_binary_op(a, b);
  return nocheck::mul(a, b);
}

inline void div(const Tensor& a, const Tensor& b, Tensor& out) {
  detail::validate_binary_op(a, b);
  detail::validate_binary_op(a, out);
  nocheck::div(a, b, out);
}

inline Tensor div(const Tensor& a, const Tensor& b) {
  detail::validate_binary_op(a, b);
  return nocheck::div(a, b);
}

}  // namespace aifw::core::ops
