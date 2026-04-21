#pragma once

#include <cstddef>
#include <vector>

#include "aifw/autograd/edge.hpp"
#include "aifw/core/tensor/tensor.hpp"

namespace aifw::autograd {

class Node {
 public:
  virtual ~Node() = default;

  virtual std::vector<core::Tensor> apply(
      const std::vector<core::Tensor>& grad_outputs
  ) = 0;

  virtual void release_resources() {}

  const std::vector<Edge>& next_edges() const { return next_edges_; }
  size_t num_outputs() const { return num_outputs_; }

 protected:
  std::vector<Edge> next_edges_;
  size_t num_outputs_ = 1;
};

}  // namespace aifw::autograd
