
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "../framework/test.hpp"
#include "aifw/core/ops/host_accessor.hpp"
#include "aifw/core/runtime/device_registry.hpp"
#include "aifw/core/tensor/dtype.hpp"
#include "aifw/core/tensor/shape.hpp"
#include "aifw/core/tensor/stride.hpp"
#include "aifw/core/tensor/tensor.hpp"
#include "aifw/core/tensor/tensor_factory.hpp"

using namespace aifw::core;

TEST(HostAccessor, get_1d) {
  auto t = zeros(cpu(), Shape{4}, DType::Float32);
  t.at<float>(2) = 7.0f;

  EXPECT_NEAR(ops::host_get<float>(t, 2), 7.0f, 1e-6f);
}

TEST(HostAccessor, get_2d) {
  auto t = zeros(cpu(), Shape{3, 4}, DType::Float32);
  t.at<float>(1, 2) = 5.0f;

  EXPECT_NEAR(ops::host_get<float>(t, 1, 2), 5.0f, 1e-6f);
}

TEST(HostAccessor, get_dtype_mismatch_throws) {
  auto t = zeros(cpu(), Shape{4}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { ops::host_get<double>(t, 0); });
}

TEST(HostAccessor, set_1d) {
  auto t = zeros(cpu(), Shape{4}, DType::Float32);
  ops::host_set<float>(t, 9.0f, 3);

  EXPECT_NEAR(ops::host_get<float>(t, 3), 9.0f, 1e-6f);
}

TEST(HostAccessor, set_2d) {
  auto t = zeros(cpu(), Shape{2, 3}, DType::Float32);
  ops::host_set<float>(t, 3.0f, 1, 2);

  EXPECT_NEAR(ops::host_get<float>(t, 1, 2), 3.0f, 1e-6f);
}

TEST(HostAccessor, set_dtype_mismatch_throws) {
  auto t = zeros(cpu(), Shape{4}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { ops::host_set<int32_t>(t, 1, 0); });
}

TEST(HostAccessor, get_on_strided_view) {
  auto base = zeros(cpu(), Shape{6}, DType::Float32);
  for (size_t i = 0; i < 6; ++i) ops::host_set(base, static_cast<float>(i), i);

  Tensor v = base.view(Shape{3}, Stride{2}, 0);

  EXPECT_NEAR(ops::host_get<float>(v, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(v, 1), 2.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(v, 2), 4.0f, 1e-6f);
}

TEST(HostAccessor, get_on_strided_view_writes_through) {
  auto base = zeros(cpu(), Shape{6}, DType::Float32);
  Tensor v = base.view(Shape{3}, Stride{2}, 0);

  ops::host_set(v, 99.0f, 1);

  EXPECT_NEAR(ops::host_get<float>(base, 1), 0.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(base, 2), 99.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(base, 3), 0.0f, 1e-6f);
}

TEST(HostAccessor, get_on_view_with_offset) {
  auto base = zeros(cpu(), Shape{8}, DType::Float32);
  for (size_t i = 0; i < 8; ++i) ops::host_set(base, static_cast<float>(i), i);

  Tensor v = base.view(Shape{3}, make_contiguous_stride(Shape{3}), 3);

  EXPECT_NEAR(ops::host_get<float>(v, 0), 3.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(v, 2), 5.0f, 1e-6f);
}

TEST(HostAccessor, int32_roundtrip) {
  auto t = zeros(cpu(), Shape{4}, DType::Int32);
  ops::host_set(t, 42, 2);
  EXPECT_EQ(ops::host_get<int32_t>(t, 2), 42);
}

TEST(HostAccessor, float64_roundtrip) {
  auto t = zeros(cpu(), Shape{4}, DType::Float64);
  ops::host_set(t, 3.141592653589793, 0);  // NOLINT
  EXPECT_NEAR(ops::host_get<double>(t, 0), 3.141592653589793, 1e-15);
}
