#pragma once

#include <cstddef>
#include <memory>

namespace aifw::autograd {

class Node;

struct Edge {
  std::shared_ptr<Node> function;
  size_t output_nr = 0;

  bool is_valid() const { return function != nullptr; }
};

}  // namespace aifw::autograd
