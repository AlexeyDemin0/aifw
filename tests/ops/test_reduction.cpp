#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../framework/test.hpp"
#include "aifw/core/ops/host_accessor.hpp"
#include "aifw/core/ops/ops_reduction.hpp"
#include "aifw/core/runtime/device_registry.hpp"
#include "aifw/core/tensor/tensor_factory.hpp"

using namespace aifw::core;
using namespace aifw::core::ops;

TEST(Sum, all_elements) {
  // {2,3}: sum of all = 0+1+2+3+4+5 = 15
  auto t = from_data<float>(cpu(), {0, 1, 2, 3, 4, 5}, Shape{2, 3});
  auto s = sum(t);

  EXPECT_EQ(s.shape()[0], size_t(1));
  EXPECT_NEAR(host_get<float>(s, 0), 15.0f, 1e-5f);
}

TEST(Sum, axis0_no_keepdims) {
  // {2,3}, axis=0 -> {3}
  // col sums: [0+3, 1+4, 2+5] = [3, 5, 7]
  auto t = from_data<float>(cpu(), {0, 1, 2, 3, 4, 5}, Shape{2, 3});
  auto s = sum(t, 0);

  EXPECT_EQ(s.shape().rank(), size_t(1));
  EXPECT_EQ(s.shape()[0], size_t(3));
  EXPECT_NEAR(host_get<float>(s, 0), 3.0f, 1e-5f);
  EXPECT_NEAR(host_get<float>(s, 1), 5.0f, 1e-5f);
  EXPECT_NEAR(host_get<float>(s, 2), 7.0f, 1e-5f);
}

TEST(Sum, axis1_no_keepdims) {
  // {2,3}, axis=1 -> {2}
  // row sums: [0+1+2, 3+4+5] = [3, 12]
  auto t = from_data<float>(cpu(), {0, 1, 2, 3, 4, 5}, Shape{2, 3});
  auto s = sum(t, 1);

  EXPECT_EQ(s.shape().rank(), size_t(1));
  EXPECT_EQ(s.shape()[0], size_t(2));
  EXPECT_NEAR(host_get<float>(s, 0), 3.0f, 1e-5f);
  EXPECT_NEAR(host_get<float>(s, 1), 12.0f, 1e-5f);
}

TEST(Sum, axis0_keepdims) {
  // {2,3}, axis=0, keepdims -> {1,3}
  auto t = from_data<float>(cpu(), {0, 1, 2, 3, 4, 5}, Shape{2, 3});
  auto s = sum(t, std::vector<size_t>{0}, true);

  EXPECT_EQ(s.shape().rank(), size_t(2));
  EXPECT_EQ(s.shape()[0], size_t(1));
  EXPECT_EQ(s.shape()[1], size_t(3));
  EXPECT_NEAR(host_get<float>(s, 0, 0), 3.0f, 1e-5f);
  EXPECT_NEAR(host_get<float>(s, 0, 2), 7.0f, 1e-5f);
}

TEST(Sum, axis1_keepdims) {
  // {2,3}, axis=1, keepdims -> {2,1}
  auto t = from_data<float>(cpu(), {0, 1, 2, 3, 4, 5}, Shape{2, 3});
  auto s = sum(t, std::vector<size_t>{1}, true);

  EXPECT_EQ(s.shape().rank(), size_t(2));
  EXPECT_EQ(s.shape()[0], size_t(2));
  EXPECT_EQ(s.shape()[1], size_t(1));
  EXPECT_NEAR(host_get<float>(s, 0, 0), 3.0f, 1e-5f);
  EXPECT_NEAR(host_get<float>(s, 1, 0), 12.0f, 1e-5f);
}

TEST(Sum, multiple_axes) {
  // {2,3,4}, axes={0,2} -> {3}
  auto t = ones(cpu(), Shape{2, 3, 4}, DType::Float32);
  auto s = sum(t, std::vector<size_t>{0, 2});

  EXPECT_EQ(s.shape().rank(), size_t(1));
  EXPECT_EQ(s.shape()[0], size_t(3));
  for (size_t i = 0; i < 3; ++i)
    EXPECT_NEAR(host_get<float>(s, i), 8.0f, 1e-5f);
}

TEST(Sum, multiple_axes_keepdims) {
  // {2,3,4}, axes={0,2}, keepdims -> {1,3,1}
  auto t = ones(cpu(), Shape{2, 3, 4}, DType::Float32);
  auto s = sum(t, std::vector<size_t>{0, 2}, true);

  EXPECT_EQ(s.shape().rank(), size_t(3));
  EXPECT_EQ(s.shape()[0], size_t(1));
  EXPECT_EQ(s.shape()[1], size_t(3));
  EXPECT_EQ(s.shape()[2], size_t(1));
}

TEST(Sum, empty_axes_noop) {
  auto t = from_data<float>(cpu(), {1, 2, 3, 4}, Shape{2, 2});
  auto s = sum(t, std::vector<size_t>{});

  EXPECT_EQ(s.shape().values(), t.shape().values());
  EXPECT_NEAR(host_get<float>(s, 0, 0), 1.0f, 1e-6f);
  EXPECT_NEAR(host_get<float>(s, 1, 1), 4.0f, 1e-6f);
}

TEST(Sum, axis_out_of_range_throws) {
  auto t = zeros(cpu(), Shape{3, 4}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { sum(t, std::vector<size_t>{5}); });
}

TEST(Sum, strided_tensor) {
  auto base = from_data<float>(cpu(), {0, 1, 2, 3, 4, 5, 6, 7}, Shape{8});
  // stride=2 -> [0,2,4,6], shape {4}
  auto view = base.view(Shape{4}, Stride{2}, 0);
  auto s = sum(view);
  EXPECT_NEAR(host_get<float>(s, 0), 12.0f, 1e-5f);  // 0+2+4+6
}

TEST(SumTo, noop_same_shape) {
  auto t = from_data<float>(cpu(), {1, 2, 3, 4}, Shape{2, 2});
  auto s = sum_to(t, Shape{2, 2});

  EXPECT_EQ(s.shape().values(), t.shape().values());
  EXPECT_NEAR(host_get<float>(s, 0, 0), 1.0f, 1e-6f);
}

TEST(SumTo, reduce_axis0) {
  // {3,4} -> {4}:
  auto t = from_data<float>(
      cpu(), {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, Shape{3, 4}
  );
  auto s = sum_to(t, Shape{4});

  EXPECT_EQ(s.shape().rank(), size_t(1));
  EXPECT_EQ(s.shape()[0], size_t(4));
  EXPECT_NEAR(host_get<float>(s, 0), 3.0f, 1e-5f);
  EXPECT_NEAR(host_get<float>(s, 3), 12.0f, 1e-5f);
}

TEST(SumTo, reduce_axis1_keepdims) {
  // {3,4} -> {3,1}
  auto t = from_data<float>(
      cpu(), {1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1}, Shape{3, 4}
  );
  auto s = sum_to(t, Shape{3, 1});

  EXPECT_EQ(s.shape()[0], size_t(3));
  EXPECT_EQ(s.shape()[1], size_t(1));
  EXPECT_NEAR(host_get<float>(s, 0, 0), 10.0f, 1e-5f);  // 1+2+3+4
  EXPECT_NEAR(host_get<float>(s, 1, 0), 26.0f, 1e-5f);  // 5+6+7+8
  EXPECT_NEAR(host_get<float>(s, 2, 0), 4.0f, 1e-5f);   // 1+1+1+1
}

TEST(SumTo, reduce_multiple_axes) {
  // {2,3,4} -> {4}
  auto t = ones(cpu(), Shape{2, 3, 4}, DType::Float32);
  auto s = sum_to(t, Shape{4});

  EXPECT_EQ(s.shape().rank(), size_t(1));
  EXPECT_EQ(s.shape()[0], size_t(4));
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(host_get<float>(s, i), 6.0f, 1e-5f);
}

TEST(SumTo, inverse_of_broadcast) {
  auto t = from_data<float>(cpu(), {1.0f, 2.0f, 3.0f, 4.0f});
  auto bcast = broadcast_to(t, Shape{3, 4});
  auto back = sum_to(bcast, Shape{4});

  EXPECT_NEAR(host_get<float>(back, 0), 3.0f, 1e-5f);
  EXPECT_NEAR(host_get<float>(back, 3), 12.0f, 1e-5f);
}

TEST(Mean, all_elements) {
  auto t = from_data<float>(cpu(), {0, 1, 2, 3, 4, 5}, Shape{2, 3});
  auto m = mean(t);
  EXPECT_NEAR(host_get<float>(m, 0), 2.5f, 1e-5f);  // (0+1+2+3+4+5)/6
}

TEST(Mean, axis0) {
  // {2,3}, axis=0 -> {3}
  auto t = from_data<float>(cpu(), {0, 2, 4, 6, 8, 10}, Shape{2, 3});
  auto m = mean(t, 0);

  EXPECT_NEAR(host_get<float>(m, 0), 3.0f, 1e-5f);  // (0+6)/2
  EXPECT_NEAR(host_get<float>(m, 1), 5.0f, 1e-5f);  // (2+8)/2
  EXPECT_NEAR(host_get<float>(m, 2), 7.0f, 1e-5f);  // (4+10)/2
}

TEST(Mean, keepdims) {
  auto t = from_data<float>(cpu(), {1, 2, 3, 4}, Shape{2, 2});
  auto m = mean(t, std::vector<size_t>{1}, true);

  EXPECT_EQ(m.shape()[0], size_t(2));
  EXPECT_EQ(m.shape()[1], size_t(1));
  EXPECT_NEAR(host_get<float>(m, 0, 0), 1.5f, 1e-5f);  // (1+2)/2
  EXPECT_NEAR(host_get<float>(m, 1, 0), 3.5f, 1e-5f);  // (3+4)/2
}
