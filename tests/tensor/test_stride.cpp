#include <cstddef>

#include "../../aifw/core/tensor/stride.hpp"
#include "../framework/test.hpp"

using namespace aifw::core;

TEST(Stride, contiguous_1d) {
  Shape s{8};
  Stride st = make_contiguous_stride(s);
  EXPECT_EQ(st[0], size_t(1));
}

TEST(Stride, contiguous_2d) {
  Shape s{3, 4};
  Stride st = make_contiguous_stride(s);
  EXPECT_EQ(st[0], size_t(4));
  EXPECT_EQ(st[1], size_t(1));
}

TEST(Stride, contiguous_3d) {
  Shape s{2, 3, 4};
  Stride st = make_contiguous_stride(s);
  EXPECT_EQ(st[0], size_t(12));
  EXPECT_EQ(st[1], size_t(4));
  EXPECT_EQ(st[2], size_t(1));
}
