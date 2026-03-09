#include "cpu_kernel_registry.hpp"

#include <cstddef>
#include <cstring>

#include "../contracts.hpp"
#include "../tensor/dtype.hpp"

namespace aifw::core {

void CpuKernelRegistry::matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_ASSERT(a.shape().rank() == 2);
  AIFW_ASSERT(b.shape().rank() == 2);
  AIFW_ASSERT(a.shape()[1] == b.shape()[0]);

  std::memset(out.data(), 0, out.numel() * dtype_size(out.dtype()));

  dtype_dispatch(a.dtype(), [&]<typename T>() {
    const size_t M = a.shape()[0];
    const size_t K = a.shape()[1];
    const size_t N = b.shape()[1];

    const auto* pa = a.data_as<T>();
    const auto* pb = b.data_as<T>();
    T* po = out.data_as<T>();

    for (size_t m = 0; m < M; ++m) {
      for (size_t k = 0; k < K; ++k) {
        const T a_mk = pa[m * K + k];
        for (size_t n = 0; n < N; ++n) po[m * N + n] += a_mk * pb[k * N + n];
      }
    }
  });
}

}  // namespace aifw::core
