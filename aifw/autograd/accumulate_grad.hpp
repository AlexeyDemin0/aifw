#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "aifw/autograd/autograd_meta.hpp"
#include "aifw/autograd/node.hpp"
#include "aifw/core/assert.hpp"
#include "aifw/core/ops/ops_elementwise.hpp"
#include "aifw/core/tensor/tensor.hpp"

namespace aifw::autograd {

class AccumulateGrad final : public Node {
 public:
  explicit AccumulateGrad(std::shared_ptr<AutogradMeta> meta)
      : meta_(std::move(meta)) {
    num_outputs_ = 1;
  }

  std::vector<core::Tensor> apply(
      const std::vector<core::Tensor>& grad_outputs
  ) override {
    AIFW_EXPECT(grad_outputs.size() == 1, "AccumulateGrad: expected 1 grad");
    const core::Tensor& g = grad_outputs[0];

    if (!meta_->grad.defined()) {
      meta_->grad = g;
    } else {
      meta_->grad = core::ops::add(meta_->grad, g);
    }
    return {};
  }

 private:
  std::shared_ptr<AutogradMeta> meta_;
};

}  // namespace aifw::autograd
