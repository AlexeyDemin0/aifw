#pragma once

#include <cstdint>

#include "aifw/core/tensor/tensor.hpp"

namespace aifw::core::detail {

enum class ExecPath : uint8_t {
  Contiguous,
  Strided,
};

inline bool all_contiguous(const Tensor& t) { return t.is_contiguous(); }

template <typename... Rest>
inline bool all_contiguous(const Tensor& first, const Rest&... rest) {
  return first.is_contiguous() && all_contiguous(rest...);
}

template <typename... Tensors>
inline ExecPath resolve_path(const Tensors&... tensors) {
  return all_contiguous(tensors...) ? ExecPath::Contiguous : ExecPath::Strided;
}

}  // namespace aifw::core::detail
