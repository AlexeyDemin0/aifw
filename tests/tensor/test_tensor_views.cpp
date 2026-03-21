
#include <stdexcept>

#include "../framework/test.hpp"
#include "aifw/core/backend/cpu_backend.hpp"
#include "aifw/core/ops/ops_elementwise.hpp"
#include "aifw/core/ops/ops_linear.hpp"
#include "aifw/core/tensor/tensor.hpp"
#include "aifw/core/tensor/tensor_factory.hpp"
#include "aifw/core/tensor/tensor_view.hpp"

using namespace aifw::core;

TEST(TensorViews, reshape_preserves_data) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{2, 3}, DType::Float32);
  t.at<float>(0, 0) = 1.0f;
  t.at<float>(1, 2) = 5.0f;

  auto flat = t.reshape(Shape{6});
  EXPECT_NEAR(flat.at<float>(0), 1.0f, 1e-6f);
  EXPECT_NEAR(flat.at<float>(5), 5.0f, 1e-6f);
}

TEST(TensorViews, reshape_numel_mismatch_throws) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{2, 3}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { t.reshape(Shape{5}); });
}

TEST(TensorViews, reshape_shares_storage) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{4}, DType::Float32);
  auto r = t.reshape(Shape{2, 2});

  r.at<float>(0, 1) = 99.0f;
  EXPECT_NEAR(t.at<float>(1), 99.0f, 1e-6f);
}

TEST(TensorViews, transpose_2d_shape) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{3, 4}, DType::Float32);
  auto tr = transpose(t, 0, 1);

  EXPECT_EQ(tr.shape()[0], size_t(4));
  EXPECT_EQ(tr.shape()[1], size_t(3));
}

TEST(TensorViews, transpose_2d_values) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{2, 3}, DType::Float32);
  t.at<float>(0, 0) = 1.0f;
  t.at<float>(0, 1) = 2.0f;
  t.at<float>(0, 2) = 3.0f;
  t.at<float>(1, 0) = 4.0f;
  t.at<float>(1, 1) = 5.0f;
  t.at<float>(1, 2) = 6.0f;

  auto tr = transpose(t, 0, 1);

  EXPECT_NEAR(tr.at<float>(0, 1), 4.0f, 1e-6f);
  EXPECT_NEAR(tr.at<float>(2, 0), 3.0f, 1e-6f);
}

TEST(TensorViews, transpose_shares_storage) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{2, 2}, DType::Float32);
  auto tr = transpose(t, 0, 1);

  tr.at<float>(0, 1) = 42.0f;
  EXPECT_NEAR(t.at<float>(1, 0), 42.0f, 1e-6f);
}

TEST(TensorViews, transpose_dim_out_of_range_throws) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{2, 3}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { transpose(t, 0, 5); });
}

TEST(TensorViews, permute_3d_shape) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{2, 3, 4}, DType::Float32);
  auto p = permute(t, {2, 0, 1});

  EXPECT_EQ(p.shape()[0], size_t(4));
  EXPECT_EQ(p.shape()[1], size_t(2));
  EXPECT_EQ(p.shape()[2], size_t(3));
}

TEST(TensorViews, permute_duplicate_dim_throws) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{2, 3, 4}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { permute(t, {0, 0, 1}); });
}

TEST(TensorViews, permute_2d) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{2, 3}, DType::Float32);
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      t.at<float>(i, j) = static_cast<float>(i * 3 + j);

  auto p = permute(t, {1, 0});
  EXPECT_NEAR(p.at<float>(1, 0), t.at<float>(0, 1), 1e-6f);
  EXPECT_NEAR(p.at<float>(2, 1), t.at<float>(1, 2), 1e-6f);
}

TEST(TensorViews, slice_rows) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{4, 3}, DType::Float32);
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 3; ++j)
      t.at<float>(i, j) = static_cast<float>(i * 3 + j);

  auto s = slice(t, 0, 1, 3);
  EXPECT_EQ(s.shape()[0], size_t(2));
  EXPECT_EQ(s.shape()[1], size_t(3));
  EXPECT_NEAR(s.at<float>(0, 0), 3.0f, 1e-6f);
  EXPECT_NEAR(s.at<float>(1, 2), 8.0f, 1e-6f);
}

TEST(TensorViews, slice_shares_storage) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{4}, DType::Float32);
  auto s = slice(t, 0, 1, 3);

  s.at<float>(0) = 5.0f;
  s.at<float>(1) = 6.0f;
  EXPECT_NEAR(t.at<float>(1), 5.0f, 1e-6f);
  EXPECT_NEAR(t.at<float>(2), 6.0f, 1e-6f);
}

TEST(TensorViews, slice_invalid_range_throws) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{4}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { slice(t, 0, 2, 2); });
  EXPECT_THROWS(std::runtime_error, [&]() { slice(t, 0, 3, 5); });
}

TEST(TensorViews, squeeze_remove_ones) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{1, 3, 1, 4}, DType::Float32);
  auto s = squeeze(t);

  EXPECT_EQ(s.shape().rank(), size_t(2));
  EXPECT_EQ(s.shape()[0], size_t(3));
  EXPECT_EQ(s.shape()[1], size_t(4));
}

TEST(TensorViews, squeeze_specific_dim) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{1, 3, 4}, DType::Float32);
  auto s = squeeze(t, 0);

  EXPECT_EQ(s.shape().rank(), size_t(2));
  EXPECT_EQ(s.shape()[0], size_t(3));
}

TEST(TensorViews, squeeze_non_one_dim_throws) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{2, 3}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { squeeze(t, 0); });
}

TEST(TensorViews, unsqueeze_insert_dim) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{3, 4}, DType::Float32);
  auto u = unsqueeze(t, 0);

  EXPECT_EQ(u.shape().rank(), size_t(3));
  EXPECT_EQ(u.shape()[0], size_t(1));
  EXPECT_EQ(u.shape()[1], size_t(3));
  EXPECT_EQ(u.shape()[2], size_t(4));
}

TEST(TensorViews, unsqueeze_squeeze_roundtrip) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{3, 4}, DType::Float32);
  auto roundtrip = squeeze(unsqueeze(t, 1), 1);

  EXPECT_EQ(roundtrip.shape().rank(), size_t(2));
  EXPECT_EQ(roundtrip.shape()[0], size_t(3));
  EXPECT_EQ(roundtrip.shape()[1], size_t(4));
}

TEST(TensorViews, flatten_all) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{2, 3, 4}, DType::Float32);
  auto f = flatten(t);

  EXPECT_EQ(f.shape().rank(), size_t(1));
  EXPECT_EQ(f.shape()[0], size_t(24));
}

TEST(TensorViews, flatten_partial) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{2, 3, 4}, DType::Float32);
  auto f = flatten(t, 1, 2);

  EXPECT_EQ(f.shape().rank(), size_t(2));
  EXPECT_EQ(f.shape()[0], size_t(2));
  EXPECT_EQ(f.shape()[1], size_t(12));
}

TEST(TensorViews, flatten_preserves_data) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{2, 3}, DType::Float32);
  t.at<float>(1, 2) = 9.0f;

  auto f = flatten(t);
  EXPECT_NEAR(f.at<float>(5), 9.0f, 1e-6f);
}

TEST(TensorViews, expand_1d_to_2d_via_unsqueeze) {
  CpuBackend cpu;
  auto bias = zeros(cpu, Shape{4}, DType::Float32);
  bias.at<float>(2) = 7.0f;

  auto row = unsqueeze(bias, 0);
  auto expanded = expand(row, Shape{3, 4});

  EXPECT_NEAR(expanded.at<float>(0, 2), 7.0f, 1e-6f);
  EXPECT_NEAR(expanded.at<float>(1, 2), 7.0f, 1e-6f);
  EXPECT_NEAR(expanded.at<float>(2, 2), 7.0f, 1e-6f);
}

TEST(TensorViews, expand_write_through) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{1, 3}, DType::Float32);
  auto e = expand(t, Shape{4, 3});

  e.at<float>(2, 1) = 5.0f;
  EXPECT_NEAR(t.at<float>(0, 1), 5.0f, 1e-6f);
}

TEST(TensorViews, ops_on_transposed) {
  CpuBackend cpu;
  Tensor a(cpu, Shape{2, 3}, DType::Float32);
  Tensor b(cpu, Shape{2, 3}, DType::Float32);

  a.at<float>(0, 0) = 1.0f;
  a.at<float>(0, 1) = 2.0f;
  a.at<float>(0, 2) = 3.0f;
  a.at<float>(1, 0) = 4.0f;
  a.at<float>(1, 1) = 5.0f;
  a.at<float>(1, 2) = 6.0f;

  b.at<float>(0, 0) = 1.0f;
  b.at<float>(0, 1) = 0.0f;
  b.at<float>(0, 2) = 0.0f;
  b.at<float>(1, 0) = 0.0f;
  b.at<float>(1, 1) = 1.0f;
  b.at<float>(1, 2) = 0.0f;

  auto bt = transpose(b, 0, 1);
  auto out = ops::matmul(a, bt);

  EXPECT_NEAR(out.at<float>(0, 0), 1.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(1, 1), 5.0f, 1e-6f);
}

TEST(TensorViews, slice_then_add) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{6}, DType::Float32);
  for (size_t i = 0; i < 6; ++i) t.at<float>(i) = static_cast<float>(i);

  auto first = slice(t, 0, 0, 3);
  auto second = slice(t, 0, 3, 6);
  auto out = ops::add(first, second);

  EXPECT_NEAR(out.at<float>(0), 3.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(1), 5.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(2), 7.0f, 1e-6f);
}
