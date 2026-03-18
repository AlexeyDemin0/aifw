#pragma once

#include <cstddef>

#include "../assert.hpp"
#include "../tensor/tensor.hpp"
#include "../tensor/tensor_iterator.hpp"
#include "dispatch.hpp"
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

  template <typename Op>
  static void elementwise_contiguous(
      const Tensor& a, const Tensor& b, Tensor& out, Op op
  );

  template <typename Op>
  static void elementwise_strided(
      const Tensor& a, const Tensor& b, Tensor& out, Op op
  );
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
  if (detail::resolve_path(a, out) == detail::ExecPath::Contiguous) {
    dtype_dispatch(a.dtype(), [&]<typename T>() {
      const T* pa = a.data_as<T>();
      T* po = out.data_as<T>();
      for (size_t i = 0; i < a.numel(); ++i)
        po[i] = pa[i] > T{0} ? pa[i] : T{0};
    });
  } else {
    dtype_dispatch(a.dtype(), [&]<typename T>() {
      TensorIterator ia(a);
      TensorIterator io(out);
      const T* base_a = static_cast<const T*>(a.data());
      T* base_o = static_cast<T*>(out.data());
      for (size_t i = 0; i < a.numel(); ++i)
        base_o[io[i]] = base_a[ia[i]] > T{0} ? base_a[ia[i]] : T{0};
    });
  }
}

template <typename Op>
void CpuKernelRegistry::elementwise(
    const Tensor& a, const Tensor& b, Tensor& out, Op op
) {
  if (detail::resolve_path(a, b, out) == detail::ExecPath::Contiguous)
    elementwise_contiguous(a, b, out, op);
  else
    elementwise_strided(a, b, out, op);
}

template <typename Op>
void CpuKernelRegistry::elementwise_contiguous(
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

template <typename Op>
void CpuKernelRegistry::elementwise_strided(
    const Tensor& a, const Tensor& b, Tensor& out, Op op
) {
  AIFW_ASSERT(a.numel() == b.numel());

  dtype_dispatch(a.dtype(), [&]<typename T>() {
    TensorIterator ia(a);
    TensorIterator ib(b);
    TensorIterator io(out);
    const T* base_a = static_cast<const T*>(a.data());
    const T* base_b = static_cast<const T*>(b.data());
    T* base_o = static_cast<T*>(out.data());
    for (size_t i = 0; i < a.numel(); ++i)
      base_o[io[i]] = op(base_a[ia[i]], base_b[ib[i]]);
  });
}

}  // namespace aifw::core
