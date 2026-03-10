#pragma once

#include <cstddef>

#include "../contracts.hpp"
#include "../tensor/tensor.hpp"
#include "kernel_registry.hpp"

namespace aifw::core {

class CpuKernelRegistry final : public IKernelRegistry {
 public:
  void add(const Tensor& a, const Tensor& b, Tensor& out) override;
  void sub(const Tensor& a, const Tensor& b, Tensor& out) override;
  void mul(const Tensor& a, const Tensor& b, Tensor& out) override;
  void div(const Tensor& a, const Tensor& b, Tensor& out) override;
  void matmul(const Tensor& a, const Tensor& b, Tensor& out) override;
  void relu(const Tensor& a, Tensor& out) override;

 private:
  template <typename Op>
  static void elementwise(const Tensor& a, const Tensor& b, Tensor& out, Op op);
};

inline void CpuKernelRegistry::add(
    const Tensor& a, const Tensor& b, Tensor& out
) {
  elementwise(a, b, out, [](auto x, auto y) { return x + y; });
}

inline void CpuKernelRegistry::sub(
    const Tensor& a, const Tensor& b, Tensor& out
) {
  elementwise(a, b, out, [](auto x, auto y) { return x - y; });
}

inline void CpuKernelRegistry::mul(
    const Tensor& a, const Tensor& b, Tensor& out
) {
  elementwise(a, b, out, [](auto x, auto y) { return x * y; });
}

inline void CpuKernelRegistry::div(
    const Tensor& a, const Tensor& b, Tensor& out
) {
  elementwise(a, b, out, [](auto x, auto y) { return x / y; });
}

inline void CpuKernelRegistry::relu(const Tensor& a, Tensor& out) {
  dtype_dispatch(a.dtype(), [&]<typename T>() {
    const T* pa = a.data_as<T>();
    T* po = out.data_as<T>();
    for (size_t i = 0; i < a.numel(); ++i) po[i] = pa[i] > T{0} ? pa[i] : T{0};
  });
}

template <typename Op>
void CpuKernelRegistry::elementwise(
    const Tensor& a, const Tensor& b, Tensor& out, Op op
) {
  AIFW_ASSERT(a.numel() == b.numel());

  dtype_dispatch(a.dtype(), [&]<typename T>() {
    const T* pa = a.data_as<T>();
    const T* pb = b.data_as<T>();
    T* po = out.data_as<T>();
    for (size_t i = 0; i < a.numel(); ++i) po[i] = op(pa[i], pb[i]);
  });
}

}  // namespace aifw::core
