#pragma once

#include <cstddef>
#include <memory>

#include "aifw/core/tensor/tensor.hpp"

namespace aifw::autograd {

class Node;

struct AutogradMeta {
  bool requires_grad = false;
  core::Tensor grad;
  std::shared_ptr<Node> grad_fn;
  size_t output_nr = 0;
};

inline bool requires_grad(const core::Tensor& t) {
  const auto& m = t.autograd_meta();
  return m && m->requires_grad;
}

inline bool is_leaf(const core::Tensor& t) {
  const auto& m = t.autograd_meta();
  return !m || m->grad_fn == nullptr;
}

}  // namespace aifw::autograd
