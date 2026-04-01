
#include <cstddef>

#include "../framework/test.hpp"
#include "aifw/core/ops/broadcast.hpp"
#include "aifw/core/ops/host_accessor.hpp"
#include "aifw/core/ops/ops_elementwise.hpp"
#include "aifw/core/ops/ops_linear.hpp"
#include "aifw/core/runtime/device_registry.hpp"
#include "aifw/core/tensor/shape.hpp"
#include "aifw/core/tensor/tensor.hpp"
#include "aifw/core/tensor/tensor_factory.hpp"

using namespace aifw::core;

TEST(BroadcastShape, same_shape) {
  Shape r = broadcast_shape(Shape{3, 4}, Shape{3, 4});
  EXPECT_EQ(r[0], size_t(3));
  EXPECT_EQ(r[1], size_t(4));
}

TEST(BroadcastShape, scalar_broadcast) {
  Shape r = broadcast_shape(Shape{1}, Shape{3, 4});
  EXPECT_EQ(r.rank(), size_t(2));
  EXPECT_EQ(r[0], size_t(3));
  EXPECT_EQ(r[1], size_t(4));
}

TEST(BroadcastShape, rank_extension) {
  Shape r = broadcast_shape(Shape{4}, Shape{3, 4});
  EXPECT_EQ(r.rank(), size_t(2));
  EXPECT_EQ(r[0], size_t(3));
  EXPECT_EQ(r[1], size_t(4));
}

TEST(BroadcastShape, both_sides) {
  Shape r = broadcast_shape(Shape{3, 1}, Shape{1, 4});
  EXPECT_EQ(r[0], size_t(3));
  EXPECT_EQ(r[1], size_t(4));
}

TEST(BroadcastShape, 3d) {
  Shape r = broadcast_shape(Shape{2, 1, 4}, Shape{3, 4});
  EXPECT_EQ(r.rank(), size_t(3));
  EXPECT_EQ(r[0], size_t(2));
  EXPECT_EQ(r[1], size_t(3));
  EXPECT_EQ(r[2], size_t(4));
}

TEST(BroadcastShape, incompatible_throws) {
  EXPECT_THROWS(std::runtime_error, [&]() {
    broadcast_shape(Shape{3, 4}, Shape{2, 4});
  });
}

TEST(BroadcastShape, incompatible_non_one_throws) {
  EXPECT_THROWS(std::runtime_error, [&]() {
    broadcast_shape(Shape{3}, Shape{4});
  });
}

TEST(BroadcastTo, 1d_to_2d) {
  auto t = from_data<float>(cpu(), {1, 2, 3, 4});
  auto e = broadcast_to(t, Shape{3, 4});

  EXPECT_EQ(e.shape()[0], size_t(3));
  EXPECT_EQ(e.shape()[1], size_t(4));

  for (size_t row = 0; row < 3; ++row)
    for (size_t col = 0; col < 4; ++col)
      EXPECT_NEAR(
          ops::host_get<float>(e, row, col), static_cast<float>(col + 1), 1e-6f
      );
}

TEST(BroadcastTo, column_to_matrix) {
  auto t = from_data<float>(cpu(), {1, 2, 3}, Shape{3, 1});
  auto e = broadcast_to(t, Shape{3, 4});

  for (size_t row = 0; row < 3; ++row)
    for (size_t col = 0; col < 4; ++col)
      EXPECT_NEAR(
          ops::host_get<float>(e, row, col), static_cast<float>(row + 1), 1e-6f
      );
}

TEST(BroadcastTo, no_copy) {
  auto t = zeros(cpu(), Shape{1, 4}, DType::Float32);
  auto e = broadcast_to(t, Shape{3, 4});

  ops::host_set(t, 99.0f, 0, 2);
  EXPECT_NEAR(ops::host_get<float>(e, 0, 2), 99.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(e, 1, 2), 99.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(e, 2, 2), 99.0f, 1e-6f);
}

TEST(BroadcastTo, rank_less_than_tensor_throws) {
  auto t = zeros(cpu(), Shape{3, 4}, DType::Float32);
  EXPECT_THROWS(std::runtime_error, [&]() { broadcast_to(t, Shape{4}); });
}

TEST(BroadcastOps, add_bias_to_batch) {
  auto batch = from_data<float>(
      cpu(), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, Shape{3, 4}
  );
  auto bias = from_data<float>(cpu(), {1, 2, 3, 4});

  auto [bias_e, batch_e] = broadcast(bias, batch);
  auto out = ops::add(bias_e, batch_e);

  EXPECT_EQ(out.shape()[0], size_t(3));
  EXPECT_EQ(out.shape()[1], size_t(4));

  // row 0: [1+1, 2+2, 3+3, 4+4]
  EXPECT_NEAR(ops::host_get<float>(out, 0, 0), 2.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(out, 0, 3), 8.0f, 1e-6f);

  // row 1: [5+1, 6+2, 7+3, 8+4]
  EXPECT_NEAR(ops::host_get<float>(out, 1, 0), 6.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(out, 1, 3), 12.0f, 1e-6f);

  // row 2: [9+1, 10+2, 11+3, 12+4]
  EXPECT_NEAR(ops::host_get<float>(out, 2, 0), 10.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(out, 2, 3), 16.0f, 1e-6f);
}

TEST(BroadcastOps, add_column_vector) {
  auto col = from_data<float>(cpu(), {1, 2, 3}, Shape{3, 1});
  auto mat = from_data<float>(
      cpu(), {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, Shape{3, 4}
  );

  auto [col_e, mat_e] = broadcast(col, mat);
  auto out = ops::add(col_e, mat_e);

  EXPECT_NEAR(ops::host_get<float>(out, 0, 0), 2.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(out, 1, 0), 3.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(out, 2, 0), 4.0f, 1e-6f);
  EXPECT_NEAR(ops::host_get<float>(out, 2, 3), 4.0f, 1e-6f);
}

TEST(BroadcastOps, mul_scale_rows) {
  auto scale = from_data<float>(cpu(), {2, 3, 4}, Shape{3, 1});
  auto mat = ones(cpu(), Shape{3, 4}, DType::Float32);

  auto [scale_e, mat_e] = broadcast(scale, mat);
  auto out = ops::mul(scale_e, mat_e);

  for (size_t col = 0; col < 4; ++col) {
    EXPECT_NEAR(ops::host_get<float>(out, 0, col), 2.0f, 1e-6f);
    EXPECT_NEAR(ops::host_get<float>(out, 1, col), 3.0f, 1e-6f);
    EXPECT_NEAR(ops::host_get<float>(out, 2, col), 4.0f, 1e-6f);
  }
}

TEST(BroadcastOps, add_3d) {
  auto a = ones(cpu(), Shape{2, 1, 4}, DType::Float32);
  auto b = full(cpu(), Shape{2, 3, 4}, DType::Float32, 2.0);

  auto [a_e, b_e] = broadcast(a, b);
  auto out = ops::add(a_e, b_e);

  EXPECT_EQ(out.shape()[0], size_t(2));
  EXPECT_EQ(out.shape()[1], size_t(3));
  EXPECT_EQ(out.shape()[2], size_t(4));

  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 3; ++j)
      for (size_t k = 0; k < 4; ++k)
        EXPECT_NEAR(ops::host_get<float>(out, i, j, k), 3.0f, 1e-6f);
}

TEST(BroadcastOps, linear_layer_forward) {
  // out = matmul(x, W) + b
  // x: {batch=2, in=3}, W: {in=3, out=4}, b: {out=4}
  auto x = from_data<float>(cpu(), {1, 0, 0, 0, 1, 0}, Shape{2, 3});

  auto W = from_data<float>(
      cpu(), {1, 2, 3, 4, 5, 6, 7, 8, 9}, Shape{3, 3}
  );                                            // {3,3}
  auto b = from_data<float>(cpu(), {1, 2, 3});  // bias {3}

  auto xW = ops::matmul(x, W);  // {2,3}

  auto [xW_e, b_e] = broadcast(xW, b);
  auto out = ops::add(xW_e, b_e);  // {2,3} + {3} → {2,3}

  // row 0: x[0]=[1,0,0], xW[0]=[1,0,0], +b=[2,2,3]
  EXPECT_NEAR(ops::host_get<float>(out, 0, 0), 2.0f, 1e-5f);
  EXPECT_NEAR(ops::host_get<float>(out, 0, 1), 4.0f, 1e-5f);
  EXPECT_NEAR(ops::host_get<float>(out, 0, 2), 6.0f, 1e-5f);
  // row 1: x[1]=[0,1,0], xW[1]=[0,1,0], +b=[1,3,3]
  EXPECT_NEAR(ops::host_get<float>(out, 1, 0), 5.0f, 1e-5f);
  EXPECT_NEAR(ops::host_get<float>(out, 1, 1), 7.0f, 1e-5f);
  EXPECT_NEAR(ops::host_get<float>(out, 1, 2), 9.0f, 1e-5f);
}
