#include "cpu_kernel_registry.hpp"

#include <cstddef>

#include "../contracts.hpp"

namespace aifw::core {

void CpuKernelRegistry::matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_ASSERT(a.shape().rank() == 2);
  AIFW_ASSERT(b.shape().rank() == 2);
  AIFW_ASSERT(a.shape()[1] == b.shape()[0]);

  dtype_dispatch(a.dtype(), [&]<typename T>() {
    const size_t M = a.shape()[0];
    const size_t K = a.shape()[1];
    const size_t N = b.shape()[1];

    const auto* pa = a.data_as<T>();
    const auto* pb = b.data_as<T>();
    T* po = out.data_as<T>();

    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        T acc{0};
        for (size_t k = 0; k < K; ++k) acc += pa[m * K + k] * pb[k * N + n];
        po[m * N + n] = acc;
      }
    }
  });
}

}  // namespace aifw::core
