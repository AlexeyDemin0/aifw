#include <cstddef>

#include "../../aifw/core/tensor/shape.hpp"
#include "../framework/test.hpp"

using namespace aifw::core;

TEST(Shape, numel_basic) {
  Shape s{2, 3, 4};
  EXPECT_EQ(s.numel(), size_t(24));
}

TEST(Shape, numel_empty) {
  Shape s{};
  EXPECT_EQ(s.numel(), size_t(0));
}

TEST(Shape, rank) {
  Shape s{2, 3};
  EXPECT_EQ(s.rank(), size_t(2));
}

TEST(Shape, index_access) {
  Shape s{5, 7};
  EXPECT_EQ(s[0], size_t(5));
  EXPECT_EQ(s[1], size_t(7));
}
