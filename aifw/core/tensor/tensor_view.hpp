#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "aifw/core/assert.hpp"
#include "aifw/core/tensor/tensor.hpp"

namespace aifw::core {

inline Tensor transpose(const Tensor& t, size_t dim0, size_t dim1) {
  const size_t rank = t.shape().rank();
  AIFW_EXPECT(dim0 < rank, "transpose: dim0 out of range");
  AIFW_EXPECT(dim1 < rank, "transpose: dim1 out of range");

  auto dims = t.shape().values();
  auto strides = t.stride().values();
  std::swap(dims[dim0], dims[dim1]);
  std::swap(strides[dim0], strides[dim1]);

  return t.view(Shape(std::move(dims)), Stride(std::move(strides)), t.offset());
}

inline Tensor permute(const Tensor& t, std::vector<size_t> order) {
  const size_t rank = t.shape().rank();
  AIFW_EXPECT(order.size() == rank, "permute: order size != rank");

  std::vector<bool> seen(rank, false);
  for (size_t d : order) {
    AIFW_EXPECT(d < rank, "permute: order contains out-of-range dim");
    AIFW_EXPECT(!seen[d], "permute: duplicate dim in order");
    seen[d] = true;
  }

  std::vector<size_t> new_dims(rank);
  std::vector<size_t> new_strides(rank);
  for (size_t i = 0; i < rank; ++i) {
    new_dims[i] = t.shape()[order[i]];
    new_strides[i] = t.stride()[order[i]];
  }

  return t.view(
      Shape(std::move(new_dims)), Stride(std::move(new_strides)), t.offset()
  );
}

inline Tensor slice(const Tensor& t, size_t dim, size_t start, size_t stop) {
  const size_t rank = t.shape().rank();
  AIFW_EXPECT(dim < rank, "slice: dim out of range");
  AIFW_EXPECT(start < stop, "slice: start must be < stop");
  AIFW_EXPECT(stop <= t.shape()[dim], "slice: stop out of range");

  auto dims = t.shape().values();
  dims[dim] = stop - start;

  const size_t new_offset = t.offset() + start * t.stride()[dim];

  return t.view(Shape(std::move(dims)), t.stride(), new_offset);
}

inline Tensor squeeze(const Tensor& t, size_t dim) {
  const size_t rank = t.shape().rank();
  AIFW_EXPECT(dim < rank, "squeeze: dim out of range");
  AIFW_EXPECT(t.shape()[dim] == 1, "squeeze: dim is not size 1");

  auto dims = t.shape().values();
  auto strides = t.stride().values();
  dims.erase(dims.begin() + static_cast<ptrdiff_t>(dim));
  strides.erase(strides.begin() + static_cast<ptrdiff_t>(dim));

  return t.view(Shape(std::move(dims)), Stride(std::move(strides)), t.offset());
}

inline Tensor squeeze(const Tensor& t) {
  auto dims = t.shape().values();
  auto strides = t.stride().values();

  std::vector<size_t> new_dims;
  std::vector<size_t> new_strides;
  for (size_t d = 0; d < dims.size(); ++d) {
    if (dims[d] != 1) {
      new_dims.push_back(dims[d]);
      new_strides.push_back(strides[d]);
    }
  }

  if (new_dims.empty()) {
    new_dims.push_back(1);
    new_strides.push_back(1);
  }

  return t.view(
      Shape(std::move(new_dims)), Stride(std::move(new_strides)), t.offset()
  );
}

inline Tensor unsqueeze(const Tensor& t, size_t dim) {
  const size_t rank = t.shape().rank();
  AIFW_EXPECT(dim <= rank, "unsqueeze: dim out of range");

  auto dims = t.shape().values();
  auto strides = t.stride().values();

  const auto pos = static_cast<ptrdiff_t>(dim);
  dims.insert(dims.begin() + pos, 1);
  strides.insert(strides.begin() + pos, 0);

  return t.view(Shape(std::move(dims)), Stride(std::move(strides)), t.offset());
}

inline Tensor flatten(const Tensor& t, size_t start_dim, size_t end_dim) {
  const size_t rank = t.shape().rank();
  AIFW_EXPECT(start_dim <= end_dim, "flatten: start_dim > end_dim");
  AIFW_EXPECT(end_dim < rank, "flatten: end_dim out of range");

  for (size_t d = start_dim; d < end_dim; ++d) {
    AIFW_EXPECT(
        t.stride()[d] == t.shape()[d + 1] * t.stride()[d + 1],
        "flatten: tensor is not contiguous along flatten dims"
    );
  }

  size_t merged = 1;
  for (size_t d = start_dim; d <= end_dim; ++d) merged *= t.shape()[d];

  auto dims = t.shape().values();
  auto strides = t.stride().values();

  dims.erase(
      dims.begin() + static_cast<ptrdiff_t>(start_dim + 1),
      dims.begin() + static_cast<ptrdiff_t>(end_dim + 1)
  );
  strides.erase(
      strides.begin() + static_cast<ptrdiff_t>(start_dim + 1),
      strides.begin() + static_cast<ptrdiff_t>(end_dim + 1)
  );
  dims[start_dim] = merged;

  return t.view(Shape(std::move(dims)), Stride(std::move(strides)), t.offset());
}

inline Tensor flatten(const Tensor& t) {
  AIFW_EXPECT(t.is_contiguous(), "flatten: tensor must be contiguous");
  return t.reshape(Shape{t.numel()});
}

inline Tensor expand(const Tensor& t, Shape new_shape) {
  const size_t rank = t.shape().rank();
  AIFW_EXPECT(new_shape.rank() == rank, "expand: rank mismatch");

  auto strides = t.stride().values();
  for (size_t d = 0; d < rank; ++d) {
    if (t.shape()[d] == new_shape[d]) {
      //
    } else if (t.shape()[d] == 1) {
      strides[d] = 0;
    } else {
      throw std::runtime_error(
          "expand: dim " + std::to_string(d) +
          " cannot be expanded (source size != 1 and != target size)"
      );
    }
  }

  return t.view(std::move(new_shape), Stride(std::move(strides)), t.offset());
}

}  // namespace aifw::core
