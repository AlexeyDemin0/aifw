#pragma once

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "aifw/core/assert.hpp"
#include "aifw/core/device/idevice.hpp"
#include "aifw/core/ops/host_accessor.hpp"
#include "dtype.hpp"
#include "shape.hpp"
#include "tensor.hpp"

namespace aifw::core {

inline Tensor zeros(IDevice& device, Shape shape, DType dt) {
  Tensor t(device, std::move(shape), dt);
  device.kernels().fill(t, 0.0);
  return t;
}

inline Tensor ones(IDevice& device, Shape shape, DType dt) {
  Tensor t(device, std::move(shape), dt);
  device.kernels().fill(t, 1.0);
  return t;
}

inline Tensor full(IDevice& device, Shape shape, DType dt, double value) {
  Tensor t(device, std::move(shape), dt);
  device.kernels().fill(t, value);
  return t;
}

inline Tensor eye(IDevice& device, size_t n, DType dt) {
  Tensor t = zeros(device, Shape{n, n}, dt);
  device.kernels().fill_diagonal(t, 1.0);
  return t;
}

inline Tensor diag(const Tensor& vec) {
  AIFW_EXPECT(vec.shape().rank() == 1, "diag: input must be 1D");

  const size_t n = vec.shape()[0];
  Tensor t = zeros(vec.device(), Shape{n, n}, vec.dtype());

  dtype_dispatch(vec.dtype(), [&]<typename T>() {
    for (size_t i = 0; i < n; ++i) {
      const T val = ops::host_get<T>(vec, i);
      ops::host_set(t, val, i, i);
    }
  });

  return t;
}

inline Tensor arange(
    IDevice& device, double start, double stop, double step, DType dt
) {
  AIFW_EXPECT(step != 0.0, "arange: step cannot be zero");
  AIFW_EXPECT((stop - start) / step > 0.0, "arange: empty range");

  const auto n = static_cast<size_t>(std::ceil((stop - start) / step));

  Tensor t(device, Shape{n}, dt);
  device.kernels().arange(t, start, step);
  return t;
}

inline Tensor arange(IDevice& device, size_t n, DType dt) {
  return arange(device, 0.0, static_cast<double>(n), 1.0, dt);
}

inline Tensor linspace(
    IDevice& device, double start, double stop, size_t n, DType dt
) {
  AIFW_EXPECT(n >= 2, "linspace: n must be >= 2");

  const double step = (stop - start) / static_cast<double>(n - 1);
  Tensor t(device, Shape{n}, dt);
  device.kernels().arange(t, start, step);

  dtype_dispatch(dt, [&]<typename T>() {
    ops::host_set(t, static_cast<T>(stop), n - 1);
  });
  return t;
}

template <typename T>
Tensor from_data(IDevice& device, std::vector<T> data, Shape shape) {
  AIFW_EXPECT(
      data.size() == shape.numel(),
      "from_data: data size does not match shape numel"
  );

  Tensor t(device, std::move(shape), dtype_of_v<T>);
  device.write_bytes(
      static_cast<void*>(t.data_as<T>()), data.data(), data.size() * sizeof(T)
  );
  return t;
}

template <typename T>
Tensor from_data(IDevice& device, std::vector<T> data) {
  const size_t n = data.size();
  return from_data(device, std::move(data), Shape{n});
}

}  // namespace aifw::core
