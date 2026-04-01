#pragma once

#include <string>

#include "../tensor/tensor.hpp"

namespace aifw::core::ops {

namespace detail {

inline void validate_binary_op(
    const Tensor& a, const Tensor& b, const char* op
) {
  AIFW_EXPECT(a.dtype() == b.dtype(), std::string(op) + ": dtype mismatch");
  AIFW_EXPECT(
      a.device_id() == b.device_id(), std::string(op) + ": device mismatch"
  );
  AIFW_EXPECT(
      a.shape().values() == b.shape().values(),
      std::string(op) + ": shape mismatch"
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

#define AIFW_ELEMENTWISE_OP(name)                                   \
  inline void name(const Tensor& a, const Tensor& b, Tensor& out) { \
    detail::validate_binary_op(a, b, #name);                        \
    detail::validate_binary_op(a, out, #name);                      \
    nocheck::name(a, b, out);                                       \
  }                                                                 \
                                                                    \
  inline Tensor name(const Tensor& a, const Tensor& b) {            \
    detail::validate_binary_op(a, b, #name);                        \
    return nocheck::name(a, b);                                     \
  }

AIFW_ELEMENTWISE_OP(add);
AIFW_ELEMENTWISE_OP(sub);
AIFW_ELEMENTWISE_OP(mul);
AIFW_ELEMENTWISE_OP(div);

#undef AIFW_ELEMENTWISE_OP

}  // namespace aifw::core::ops
