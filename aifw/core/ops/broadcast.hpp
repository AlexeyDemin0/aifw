#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "aifw/core/assert.hpp"
#include "aifw/core/tensor/shape.hpp"
#include "aifw/core/tensor/tensor.hpp"
#include "aifw/core/tensor/tensor_view.hpp"

namespace aifw::core {

inline Shape broadcast_shape(const Shape& a, const Shape& b) {
  const size_t rank_a = a.rank();
  const size_t rank_b = b.rank();
  const size_t rank = std::max(rank_a, rank_b);

  std::vector<size_t> result(rank);

  for (size_t i = 0; i < rank; ++i) {
    const size_t dim_a = (i < rank - rank_a) ? 1 : a[i - (rank - rank_a)];
    const size_t dim_b = (i < rank - rank_b) ? 1 : b[i - (rank - rank_b)];

    if (dim_a == dim_b)
      result[i] = dim_a;
    else if (dim_a == 1)
      result[i] = dim_b;
    else if (dim_b == 1)
      result[i] = dim_a;
    else
      throw std::runtime_error(
          "broadcast_shape: incompatible dims at axis " + std::to_string(i) +
          ": " + std::to_string(dim_a) + " vs " + std::to_string(dim_b)
      );
  }

  return Shape(std::move(result));
}

inline Tensor broadcast_to(const Tensor& t, const Shape& target) {
  AIFW_EXPECT(
      target.rank() >= t.shape().rank(),
      "broadcast_to: target rank is less than tensor rank"
  );

  Tensor expanded = t;
  while (expanded.shape().rank() < target.rank())
    expanded = unsqueeze(expanded, 0);

  return expand(expanded, target);
}

struct BroadcastPair {
  Tensor a;
  Tensor b;
};

inline BroadcastPair broadcast(const Tensor& a, const Tensor& b) {
  const Shape out_shape = broadcast_shape(a.shape(), b.shape());
  return {broadcast_to(a, out_shape), broadcast_to(b, out_shape)};
}

}  // namespace aifw::core
