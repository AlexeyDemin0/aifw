
#include <cstddef>
#include <stdexcept>

#include "../framework/test.hpp"
#include "aifw/core/ops/host_accessor.hpp"
#include "aifw/core/ops/ops_linear.hpp"
#include "aifw/core/runtime/device_registry.hpp"
#include "aifw/core/tensor/dtype.hpp"
#include "aifw/core/tensor/shape.hpp"
#include "aifw/core/tensor/tensor_factory.hpp"

using namespace aifw::core;

TEST(TensorFactory, zeros_are_zero) {
  auto t = zeros(cpu(), Shape{3, 3}, DType::Float32);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      EXPECT_NEAR(ops::host_get<float>(t, i, j), 0.0f, 1e-6f);
}

TEST(TensorFactory, ones_are_ones) {
  auto t = ones(cpu(), Shape{4}, DType::Float32);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(ops::host_get<float>(t, i), 1.0f, 1e-6f);
}

TEST(TensorFactory, full_fills_value) {
  auto t = full(cpu(), Shape{2, 3}, DType::Float64, 3.14);
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      EXPECT_NEAR(ops::host_get<double>(t, i, j), 3.14, 1e-10);
}

TEST(TensorFactory, eye_shape) {
  auto t = eye(cpu(), 4, DType::Float32);
  EXPECT_EQ(t.shape()[0], size_t(4));
  EXPECT_EQ(t.shape()[1], size_t(4));
}

TEST(TensorFactory, eye_diagonal_ones) {
  auto t = eye(cpu(), 3, DType::Float32);
  for (size_t i = 0; i < 3; ++i)
    EXPECT_NEAR(ops::host_get<float>(t, i, i), 1.0f, 1e-6f);
}

TEST(TensorFactory, eye_off_diagonal_zeros) {
  auto t = eye(cpu(), 3, DType::Float32);
  EXPECT_NEAR(ops::host_get<float>(t, 0, 1), 0.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(t, 1, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(t, 0, 2), 0.0f, 1e-6f);
}

TEST(TensorFactory, eye_matmul_identity) {
  auto a = from_data<float>(cpu(), {1, 2, 3, 4}, Shape{2, 2});
  auto i = eye(cpu(), 2, DType::Float32);
  auto out = ops::matmul(a, i);

  EXPECT_NEAR(ops::host_get<float>(out, 0, 0), 1.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(out, 0, 1), 2.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(out, 1, 0), 3.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(out, 1, 1), 4.0f, 1e-6f);
}

TEST(TensorFactory, diag_shape) {
  auto v = from_data<float>(cpu(), {1, 2, 3});
  auto d = diag(v);
  EXPECT_EQ(d.shape()[0], size_t(3));
  EXPECT_EQ(d.shape()[1], size_t(3));
}

TEST(TensorFactory, diag_values) {
  auto v = from_data<float>(cpu(), {1, 2, 3});
  auto d = diag(v);
  EXPECT_NEAR(ops::host_get<float>(d, 0, 0), 1.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(d, 1, 1), 2.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(d, 2, 2), 3.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(d, 0, 1), 0.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(d, 1, 0), 0.0f, 1e-6f);
}

TEST(TensorFactory, diag_non_1d_throws) {
  auto t = zeros(cpu(), Shape{2, 2}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { diag(t); });
}

TEST(TensorFactory, arange_n) {
  auto t = arange(cpu(), size_t(5), DType::Float32);
  EXPECT_EQ(t.shape()[0], size_t(5));
  EXPECT_NEAR(ops::host_get<float>(t, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(t, 4), 4.0f, 1e-6f);
}

TEST(TensorFactory, arange_start_stop_step) {
  auto t = arange(cpu(), 1.0, 3.0, 0.5, DType::Float64);
  EXPECT_EQ(t.shape()[0], size_t(4));
  EXPECT_NEAR(ops::host_get<double>(t, 0), 1.0, 1e-10);
  EXPECT_NEAR(ops::host_get<double>(t, 1), 1.5, 1e-10);
  EXPECT_NEAR(ops::host_get<double>(t, 2), 2.0, 1e-10);
  EXPECT_NEAR(ops::host_get<double>(t, 3), 2.5, 1e-10);
}

TEST(TensorFactory, arange_zero_step_throws) {
  EXPECT_THROWS(std::runtime_error, [&]() {
    arange(cpu(), 0.0, 5.0, 0.0, DType::Float32);
  });
}

TEST(TensorFactory, arange_empty_range_throws) {
  EXPECT_THROWS(std::runtime_error, [&]() {
    arange(cpu(), 5.0, 0.0, 1.0, DType::Float32);
  });
}

TEST(TensorFactory, linspace_endpoints) {
  auto t = linspace(cpu(), 0.0, 1.0, 5, DType::Float64);
  EXPECT_EQ(t.shape()[0], size_t(5));
  EXPECT_NEAR(ops::host_get<double>(t, 0), 0.0, 1e-10);
  EXPECT_NEAR(ops::host_get<double>(t, 4), 1.0, 1e-10);
}

TEST(TensorFactory, linspace_midpoints) {
  auto t = linspace(cpu(), 0.0, 1.0, 3, DType::Float64);
  EXPECT_NEAR(ops::host_get<double>(t, 1), 0.5, 1e-10);
}

TEST(TensorFactory, linspace_too_few_throws) {
  EXPECT_THROWS(std::runtime_error, [&]() {
    linspace(cpu(), 0.0, 1.0, 1, DType::Float32);
  });
}

TEST(TensorFactory, from_data_1d) {
  auto t = from_data<float>(cpu(), {1, 2, 3});
  EXPECT_EQ(t.shape()[0], size_t(3));
  EXPECT_NEAR(ops::host_get<float>(t, 0), 1.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(t, 2), 3.0f, 1e-6f);
}

TEST(TensorFactory, from_data_2d) {
  auto t = from_data<float>(cpu(), {1, 2, 3, 4, 5, 6}, Shape{2, 3});
  EXPECT_NEAR(ops::host_get<float>(t, 0, 0), 1.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(t, 1, 2), 6.0f, 1e-6f);
}

TEST(TensorFactory, from_data_size_mismatch_throws) {
  EXPECT_THROWS(std::runtime_error, [&]() {
    from_data<float>(cpu(), {1.0f, 2.0f, 3.0f}, Shape{2, 3});
  });
}

TEST(TensorFactory, from_data_int32) {
  auto t = from_data<int32_t>(cpu(), {10, 20, 30});
  EXPECT_EQ(ops::host_get<int32_t>(t, 1), int32_t(20));
}
