#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "aifw/core/assert.hpp"
#include "aifw/core/ops/ops_elementwise.hpp"
#include "aifw/core/tensor/shape.hpp"
#include "aifw/core/tensor/tensor.hpp"
#include "aifw/core/tensor/tensor_factory.hpp"
#include "aifw/core/tensor/tensor_view.hpp"

namespace aifw::core::ops {

namespace detail {

inline std::vector<size_t> normalize_axes(
    const std::vector<size_t>& axes, size_t rank
) {
  std::vector<size_t> result = axes;
  std::ranges::sort(result);

  auto [first, last] = std::ranges::unique(result);
  result.erase(first, last);

  for (size_t ax : result) AIFW_EXPECT(ax < rank, "sum: axis out of range");

  return result;
}

inline Shape output_shape(
    const Shape& input, const std::vector<size_t>& sorted_axes, bool keepdims
) {
  std::vector<size_t> dims;
  dims.reserve(input.rank());

  for (size_t d = 0; d < input.rank(); ++d) {
    const bool is_reduced = std::ranges::binary_search(sorted_axes, d);

    if (is_reduced) {
      if (keepdims) dims.push_back(1);
    } else
      dims.push_back(input[d]);
  }

  if (dims.empty()) dims.push_back(1);

  return Shape(std::move(dims));
}

inline std::vector<size_t> all_axes(size_t rank) {
  std::vector<size_t> axes(rank);
  std::iota(axes.begin(), axes.end(), size_t(0));
  return axes;
}

inline std::vector<size_t> axes_for_sum_to(
    const Shape& grad_shape, const Shape& target_shape
) {
  AIFW_EXPECT(
      grad_shape.rank() >= target_shape.rank(),
      "sum_to: grad rank must be >= target rank"
  );

  const size_t rank = grad_shape.rank();
  const size_t rank_diff = rank - target_shape.rank();

  std::vector<size_t> axes;

  for (size_t d = 0; d < rank; ++d) {
    if (d < rank_diff) {
      axes.push_back(d);
    } else {
      const size_t target_d = d - rank_diff;
      if (target_shape[target_d] == 1 && grad_shape[d] != 1) {
        axes.push_back(d);
      }
    }
  }
  return axes;
}

}  // namespace detail

inline Tensor sum(
    const Tensor& t, const std::vector<size_t>& axes, bool keepdims = false
) {
  const auto sorted_axes = detail::normalize_axes(axes, t.shape().rank());

  if (sorted_axes.empty()) return t;

  const Shape out_shape =
      detail::output_shape(t.shape(), sorted_axes, keepdims);
  Tensor out(t.device(), out_shape, t.dtype());

  t.device().kernels().sum(t, out, sorted_axes, keepdims);
  return out;
}

inline Tensor sum(const Tensor& t) {
  return sum(t, detail::all_axes(t.shape().rank()), false);
}

inline Tensor sum(const Tensor& t, size_t axis, bool keepdims = false) {
  return sum(t, std::vector<size_t>{axis}, keepdims);
}

inline Tensor sum_to(const Tensor& t, const Shape& target) {
  if (t.shape().values() == target.values()) return t;

  AIFW_EXPECT(
      t.shape().rank() >= target.rank(),
      "sum_to: tensor rank must be >= target rank"
  );

  const auto axes = detail::axes_for_sum_to(t.shape(), target);

  Tensor result = sum(t, axes, true);

  if (result.shape().rank() > target.rank()) {
    const size_t rank_diff = result.shape().rank() - target.rank();
    for (size_t i = 0; i < rank_diff; ++i) result = squeeze(result, 0);
  }

  AIFW_EXPECT(
      result.shape().values() == target.values(),
      "sum_to: result shape does not match target"
  );

  return result;
}

inline Tensor mean(
    const Tensor& t, const std::vector<size_t>& axes, bool keepdims = false
) {
  const auto sorted_axes = detail::normalize_axes(axes, t.shape().rank());

  size_t count = 1;
  for (size_t ax : sorted_axes) count *= t.shape()[ax];

  AIFW_EXPECT(count > 0, "mean: cannot reduce over empty axes");

  Tensor s = sum(t, axes, keepdims);

  Tensor divisor =
      full(t.device(), Shape{1}, t.dtype(), static_cast<double>(count));
  return broadcast_div(s, divisor);
}

inline Tensor mean(const Tensor& t) {
  return mean(t, detail::all_axes(t.shape().rank()), false);
}

inline Tensor mean(const Tensor& t, size_t axis, bool keepdims = false) {
  return mean(t, std::vector<size_t>{axis}, keepdims);
}

}  // namespace aifw::core::ops
