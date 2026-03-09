#include <cstdint>

#include "../../aifw/core/tensor/dtype.hpp"
#include "../framework/test.hpp"

using namespace aifw::core;

TEST(DType, sizes) {
  EXPECT_EQ(dtype_size(DType::Float32), size_t(4));
  EXPECT_EQ(dtype_size(DType::Float64), size_t(8));
  EXPECT_EQ(dtype_size(DType::Int32), size_t(4));
  EXPECT_EQ(dtype_size(DType::Int64), size_t(8));
  EXPECT_EQ(dtype_size(DType::Bool), size_t(1));
}

TEST(DType, dtype_of) {
  EXPECT_EQ(dtype_of_v<float>, DType::Float32);
  EXPECT_EQ(dtype_of_v<double>, DType::Float64);
  EXPECT_EQ(dtype_of_v<int32_t>, DType::Int32);
  EXPECT_EQ(dtype_of_v<int64_t>, DType::Int64);
  EXPECT_EQ(dtype_of_v<bool>, DType::Bool);
}
