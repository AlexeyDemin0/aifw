
#include "../framework/test.hpp"
#include "aifw/core/backend/cpu_backend.hpp"
#include "aifw/core/ops/ops_elementwise.hpp"
#include "aifw/core/ops/ops_linear.hpp"
#include "aifw/core/ops/ops_unary.hpp"
#include "aifw/core/tensor/stride.hpp"
#include "aifw/core/tensor/tensor.hpp"
#include "aifw/core/tensor/tensor_factory.hpp"
#include "aifw/core/tensor/tensor_iterator.hpp"

using namespace aifw::core;

TEST(TensorIterator, compute_contiguous_1d) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{5}, DType::Float32);
  ComputeTensorIterator it(t);
  for (size_t i = 0; i < 5; ++i) EXPECT_EQ(it[i], i);
}

TEST(TensorIterator, compute_contiguous_2d) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{3, 4}, DType::Float32);
  ComputeTensorIterator it(t);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 4; ++j) EXPECT_EQ(it[i * 4 + j], i * 4 + j);
}

TEST(TensorIterator, compute_view_with_offset) {
  CpuBackend cpu;
  Tensor base(cpu, Shape{8}, DType::Float32);
  Tensor v = base.view(Shape{4}, make_contiguous_stride(Shape{4}), 2);
  ComputeTensorIterator it(v);
  EXPECT_EQ(it[0], size_t(2));
  EXPECT_EQ(it[1], size_t(3));
  EXPECT_EQ(it[2], size_t(4));
  EXPECT_EQ(it[3], size_t(5));
}

TEST(TensorIterator, compute_transposed_2d) {
  CpuBackend cpu;
  Tensor base(cpu, Shape{3, 2}, DType::Float32);
  Tensor t = base.view(Shape{2, 3}, Stride{1, 2}, 0);
  ComputeTensorIterator it(t);
  EXPECT_EQ(it[0], size_t(0));
  EXPECT_EQ(it[1], size_t(2));
  EXPECT_EQ(it[2], size_t(4));
  EXPECT_EQ(it[3], size_t(1));
  EXPECT_EQ(it[4], size_t(3));
  EXPECT_EQ(it[5], size_t(5));
}

TEST(TensorIterator, cache_matches_compute) {
  CpuBackend cpu;
  Tensor base(cpu, Shape{3, 2}, DType::Float32);
  Tensor v = base.view(Shape{2, 3}, Stride{1, 2}, 0);

  CacheTensorIterator cache(v);
  ComputeTensorIterator compute(v);

  for (size_t i = 0; i < v.numel(); ++i) EXPECT_EQ(cache[i], compute[i]);
}

TEST(TensorIterator, cache_matches_compute_with_offset) {
  CpuBackend cpu;
  Tensor base(cpu, Shape{10}, DType::Float32);
  Tensor v = base.view(Shape{5}, make_contiguous_stride(Shape{5}), 3);

  CacheTensorIterator cache(v);
  ComputeTensorIterator compute(v);

  for (size_t i = 0; i < v.numel(); ++i) EXPECT_EQ(cache[i], compute[i]);
}

TEST(StridedOps, add_on_view_with_offset) {
  CpuBackend cpu;
  auto base = zeros(cpu, Shape{8}, DType::Float32);

  for (size_t i = 0; i < 8; ++i) base.at<float>(i) = static_cast<float>(i);

  Tensor a = base.view(Shape{4}, make_contiguous_stride(Shape{4}), 2);
  Tensor b = base.view(Shape{4}, make_contiguous_stride(Shape{4}), 0);

  auto out = ops::add(a, b);

  EXPECT_NEAR(out.at<float>(0), 2.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(1), 4.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(2), 6.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(3), 8.0f, 1e-6f);
}

TEST(StridedOps, add_non_contiguous_stride) {
  CpuBackend cpu;
  auto base = zeros(cpu, Shape{8}, DType::Float32);
  for (size_t i = 0; i < 8; ++i) base.at<float>(i) = static_cast<float>(i);

  Tensor a = base.view(Shape{4}, Stride{2}, 0);
  Tensor b = base.view(Shape{4}, Stride{2}, 1);

  auto out = ops::add(a, b);

  EXPECT_NEAR(out.at<float>(0), 1.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(1), 5.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(2), 9.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(3), 13.0f, 1e-6f);
}

TEST(StridedOps, relu_on_strided_view) {
  CpuBackend cpu;
  auto base = zeros(cpu, Shape{6}, DType::Float32);
  base.at<float>(0) = -1.0f;
  base.at<float>(2) = -2.0f;
  base.at<float>(4) = 3.0f;

  Tensor v = base.view(Shape{3}, Stride{2}, 0);
  auto out = ops::relu(v);

  EXPECT_NEAR(out.at<float>(0), 0.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(1), 0.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(2), 3.0f, 1e-6f);
}

TEST(StridedOps, matmul_transposed_b) {
  CpuBackend cpu;

  Tensor a(cpu, Shape{2, 2}, DType::Float32);
  a.at<float>(0, 0) = 1.0f;
  a.at<float>(0, 1) = 2.0f;
  a.at<float>(1, 0) = 3.0f;
  a.at<float>(1, 1) = 4.0f;

  Tensor b_storage(cpu, Shape{2, 2}, DType::Float32);
  b_storage.at<float>(0, 0) = 5.0f;
  b_storage.at<float>(0, 1) = 6.0f;
  b_storage.at<float>(1, 0) = 7.0f;
  b_storage.at<float>(1, 1) = 8.0f;

  Tensor b = b_storage.view(Shape{2, 2}, Stride{1, 2}, 0);

  auto out = ops::matmul(a, b);

  EXPECT_NEAR(out.at<float>(0, 0), 17.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(0, 1), 23.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(1, 0), 39.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(1, 1), 53.0f, 1e-6f);
}
