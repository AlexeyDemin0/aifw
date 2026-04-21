#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "aifw/autograd/autograd_meta.hpp"
#include "aifw/autograd/node.hpp"
#include "aifw/core/assert.hpp"
#include "aifw/core/ops/ops_elementwise.hpp"
#include "aifw/core/tensor/tensor.hpp"
#include "aifw/core/tensor/tensor_factory.hpp"

namespace aifw::autograd {

class Engine {
 public:
  void backward(
      const core::Tensor& root,
      std::optional<core::Tensor> grad_output = std::nullopt,
      bool retain_graph = false
  );

 private:
  void accumulate(Node* node, size_t output_nr, const core::Tensor& g);

  static void toposort(
      Node* root, std::unordered_set<Node*>& visited, std::vector<Node*>& order
  );

  std::unordered_map<Node*, std::vector<core::Tensor>> buffers_;
};

inline void Engine::backward(
    const core::Tensor& root,
    std::optional<core::Tensor> grad_output,
    bool retain_graph
) {
  const auto& meta = root.autograd_meta();
  AIFW_EXPECT(
      meta && meta->grad_fn,
      "backward: root has no grad_fn - nothing to differentiate"
  );

  core::Tensor g = grad_output
                       ? std::move(*grad_output)
                       : core::ones(root.device(), root.shape(), root.dtype());

  Node* root_node = meta->grad_fn.get();
  auto& root_buf = buffers_[root_node];
  root_buf.resize(root_node->num_outputs());
  root_buf[meta->output_nr] = g;

  std::unordered_set<Node*> visited;
  std::vector<Node*> order;
  toposort(root_node, visited, order);

  for (Node* node : order) {
    auto& grads = buffers_[node];
    auto grad_inputs = node->apply(grads);

    const auto& edges = node->next_edges();
    AIFW_EXPECT(
        grad_inputs.size() == edges.size(),
        "Engine: backward returned wrong number of grads"
    );
    for (size_t i = 0; i < edges.size(); ++i) {
      if (edges[i].is_valid()) {
        accumulate(edges[i].function.get(), edges[i].output_nr, grad_inputs[i]);
      }
    }

    if (!retain_graph) node->release_resources();
  }

  buffers_.clear();
}

inline void Engine::accumulate(
    Node* node, size_t output_nr, const core::Tensor& g
) {
  auto& buf = buffers_[node];
  if (buf.size() <= output_nr) buf.resize(node->num_outputs());
  if (buf[output_nr].defined()) {
    buf[output_nr] = g;
  } else {
    buf[output_nr] = core::ops::add(buf[output_nr], g);
  }
}

inline void Engine::toposort(
    Node* root, std::unordered_set<Node*>& visited, std::vector<Node*>& order
) {
  struct Frame {
    Node* node;
    size_t next_child_idx;
  };

  std::vector<Frame> stack;
  stack.push_back({root, 0});
  visited.insert(root);

  while (!stack.empty()) {
    auto& [node, idx] = stack.back();
    const auto& edges = node->next_edges();
    if (idx == edges.size()) {
      order.push_back(node);
      stack.pop_back();
      continue;
    }
    Node* child = edges[idx].function.get();
    ++idx;
    if (child && visited.insert(child).second) {
      stack.push_back({child, 0});
    }
  }

  std::ranges::reverse(order);
}

inline void backward(
    const core::Tensor& root,
    std::optional<core::Tensor> grad_output = std::nullopt,
    bool retain_graph = false
) {
  Engine e;
  e.backward(root, std::move(grad_output), retain_graph);
}

}  // namespace aifw::autograd
