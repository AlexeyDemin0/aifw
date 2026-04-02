#include "cpu_kernel_registry.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "aifw/core/assert.hpp"
#include "aifw/core/ops/dispatch.hpp"
#include "aifw/core/tensor/dtype.hpp"
#include "aifw/core/tensor/tensor_iterator.hpp"

namespace aifw::core {

void CpuKernelRegistry::matmul(const Tensor& a, const Tensor& b, Tensor& out) {
  AIFW_ASSERT(a.shape().rank() == 2);
  AIFW_ASSERT(b.shape().rank() == 2);
  AIFW_ASSERT(out.shape().rank() == 2);
  AIFW_ASSERT(a.shape()[1] == b.shape()[0]);

  fill(out, 0.0f);

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

void CpuKernelRegistry::fill_diagonal(Tensor& t, double value) {
  AIFW_ASSERT(t.shape().rank() == 2);

  const size_t n = std::min(t.shape()[0], t.shape()[1]);

  dtype_dispatch(t.dtype(), [&]<typename T>() {
    const T val = static_cast<T>(value);
    if (detail::resolve_path(t) == detail::ExecPath::Contiguous) {
      T* ptr = t.data_as<T>();
      const size_t cols = t.shape()[1];
      for (size_t i = 0; i < n; ++i) ptr[i * cols + i] = val;
    } else {
      auto* base = static_cast<T*>(t.data());
      for (size_t i = 0; i < n; ++i)
        base[t.offset() + i * t.stride()[0] + i * t.stride()[1]] = val;
    }
  });
}

void CpuKernelRegistry::arange(Tensor& t, double start, double step) {
  AIFW_ASSERT(t.shape().rank() == 1);
  dtype_dispatch(t.dtype(), [&]<typename T>() {
    if (detail::resolve_path(t) == detail::ExecPath::Contiguous) {
      T* ptr = t.data_as<T>();
      for (size_t i = 0; i < t.numel(); ++i)
        ptr[i] = static_cast<T>(start + static_cast<double>(i) * step);
    } else {
      TensorIterator it(t);
      auto* base = static_cast<T*>(t.data());
      for (size_t i = 0; i < t.numel(); ++i)
        base[it[i]] = static_cast<T>(start + static_cast<double>(i) * step);
    }
  });
}

void CpuKernelRegistry::sum(
    const Tensor& input,
    Tensor& out,
    const std::vector<size_t>& axes,
    bool keepdims
) {
  fill(out, 0.0);

  const size_t rank = input.shape().rank();
  const size_t out_rank = out.shape().rank();

  dtype_dispatch(input.dtype(), [&]<typename T>() {
    TensorIterator in_it(input);
    const auto* in_base = static_cast<const T*>(input.data());
    auto* out_ptr = static_cast<T*>(out.data());

    const size_t n = input.numel();

    for (size_t flat = 0; flat < n; ++flat) {
      std::vector<size_t> coords(rank);
      size_t tmp = flat;
      for (size_t d = rank; d-- > 0;) {
        coords[d] = tmp % input.shape()[d];
        tmp /= input.shape()[d];
      }

      std::vector<size_t> out_coords;
      out_coords.reserve(out_rank);

      for (size_t d = 0; d < rank; ++d) {
        const bool is_reduced = std::ranges::binary_search(axes, d);
        if (is_reduced) {
          if (keepdims) out_coords.push_back(0);
        } else
          out_coords.push_back(coords[d]);
      }

      if (out_coords.empty()) out_coords.push_back(0);

      size_t out_off = out.offset();
      for (size_t d = 0; d < out_coords.size(); ++d)
        out_off += out_coords[d] * out.stride()[d];

      out_ptr[out_off] += in_base[in_it[flat]];
    }
  });
}

}  // namespace aifw::core
