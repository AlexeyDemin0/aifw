#include <cstddef>
#include <stdexcept>

#include "../../aifw/core/backend/cpu_backend.hpp"
#include "../../aifw/core/tensor/tensor.hpp"
#include "../../aifw/core/tensor/tensor_factory.hpp"
#include "../framework/test.hpp"

using namespace aifw::core;

TEST(Tensor, zeros_are_zero) {
  CpuBackend cpu;
  auto t = zeros(cpu, Shape{2, 3}, DType::Float32);
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j) EXPECT_NEAR(t.at<float>(i, j), 0.0f, 1e-6f);
}

TEST(Tensor, at_read_write) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{2, 3}, DType::Float32);
  t.at<float>(0, 0) = 1.0f;
  t.at<float>(1, 1) = 2.0f;
  EXPECT_NEAR(t.at<float>(0, 0), 1.0f, 1e-6f);
  EXPECT_NEAR(t.at<float>(1, 1), 2.0f, 1e-6f);
}

TEST(Tensor, dtype_mismatch_throws) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{2, 2}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { t.at<double>(0, 0); });
}

TEST(Tensor, is_contiguous) {
  CpuBackend cpu;
  Tensor t(cpu, Shape{3, 3}, DType::Float32);
  EXPECT_TRUE(t.is_contiguous());
}
