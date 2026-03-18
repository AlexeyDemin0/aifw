#include "cpu_kernel_registry.hpp"

#include <cstddef>
#include <cstring>

#include "../assert.hpp"
#include "../tensor/dtype.hpp"

namespace aifw::core {

void CpuKernelRegistry::matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_ASSERT(a.shape().rank() == 2);
  AIFW_ASSERT(b.shape().rank() == 2);
  AIFW_ASSERT(out.shape().rank() == 2);
  AIFW_ASSERT(a.shape()[1] == b.shape()[0]);

  std::memset(out.data(), 0, out.numel() * dtype_size(out.dtype()));

  dtype_dispatch(a.dtype(), [&]<typename T>() {
    const size_t M = a.shape()[0];
    const size_t K = a.shape()[1];
    const size_t N = b.shape()[1];

    const size_t sa0 = a.stride()[0];
    const size_t sa1 = a.stride()[1];

    const size_t sb0 = b.stride()[0];
    const size_t sb1 = b.stride()[1];

    const size_t so0 = out.stride()[0];
    const size_t so1 = out.stride()[1];

    const auto* pa = static_cast<const T*>(a.data()) + a.offset();
    const auto* pb = static_cast<const T*>(b.data()) + b.offset();
    T* po = static_cast<T*>(out.data()) + out.offset();

    for (size_t m = 0; m < M; ++m) {
      for (size_t k = 0; k < K; ++k) {
        const T a_mk = pa[m * sa0 + k * sa1];
        for (size_t n = 0; n < N; ++n)
          po[m * so0 + n * so1] += a_mk * pb[k * sb0 + n * sb1];
      }
    }
  });
}

}  // namespace aifw::core
