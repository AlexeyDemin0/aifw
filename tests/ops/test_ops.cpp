#include <stdexcept>

#include "../../aifw/core/backend/cpu_backend.hpp"
#include "../../aifw/core/ops/ops.hpp"
#include "../../aifw/core/tensor/tensor_factory.hpp"
#include "../framework/test.hpp"

using namespace aifw::core;

TEST(Ops, add_float32) {
  CpuBackend cpu;
  auto a = zeros(cpu, Shape{4}, DType::Float32);
  auto b = zeros(cpu, Shape{4}, DType::Float32);

  a.at<float>(0) = 1.0f;
  b.at<float>(0) = 2.0f;
  a.at<float>(1) = 3.0f;
  b.at<float>(1) = 4.0f;

  auto out = ops::add(a, b);

  EXPECT_NEAR(out.at<float>(0), 3.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(1), 7.0f, 1e-6f);
}

TEST(Ops, add_shape_mismatch_throws) {
  CpuBackend cpu;
  auto a = zeros(cpu, Shape{4}, DType::Float32);
  auto b = zeros(cpu, Shape{2}, DType::Float32);
  Tensor out(cpu, Shape{4}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { ops::add(a, b, out); });
}

TEST(Ops, matmul_2x2) {
  CpuBackend cpu;
  Tensor a(cpu, Shape{2, 2}, DType::Float32);
  Tensor b(cpu, Shape{2, 2}, DType::Float32);

  a.at<float>(0, 0) = 1.0f;
  a.at<float>(0, 1) = 2.0f;
  a.at<float>(1, 0) = 3.0f;
  a.at<float>(1, 1) = 4.0f;

  b.at<float>(0, 0) = 5.0f;
  b.at<float>(0, 1) = 6.0f;
  b.at<float>(1, 0) = 7.0f;
  b.at<float>(1, 1) = 8.0f;

  auto out = ops::matmul(a, b);

  EXPECT_NEAR(out.at<float>(0, 0), 19.0f, 1e-5f);
  EXPECT_NEAR(out.at<float>(0, 1), 22.0f, 1e-5f);
  EXPECT_NEAR(out.at<float>(1, 0), 43.0f, 1e-5f);
  EXPECT_NEAR(out.at<float>(1, 1), 50.0f, 1e-5f);
}

TEST(Ops, relu_zeroes_negatives) {
  CpuBackend cpu;
  Tensor a(cpu, Shape{4}, DType::Float32);
  a.at<float>(0) = -2.0f;
  a.at<float>(1) = 0.0f;
  a.at<float>(2) = 3.0f;
  a.at<float>(3) = -1.0f;

  auto out = ops::relu(a);
  EXPECT_NEAR(out.at<float>(0), 0.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(1), 0.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(2), 3.0f, 1e-6f);
  EXPECT_NEAR(out.at<float>(3), 0.0f, 1e-6f);
}
