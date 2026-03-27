#pragma once

#include <cstddef>
#include <initializer_list>
#include <type_traits>

#include "aifw/core/assert.hpp"
#include "aifw/core/tensor/dtype.hpp"
#include "aifw/core/tensor/tensor.hpp"

namespace aifw::core::ops {

namespace detail {

template <typename T, typename... Ix>
const void* device_element_ptr(const Tensor& t, Ix... ix) {
  static_assert(
      (std::is_convertible_v<Ix, size_t> && ...),
      "host_accessor: indices must be convertible to size_t"
  );
  std::initializer_list<size_t> indices{static_cast<size_t>(ix)...};
  const size_t off = t.compute_offset(indices);
  return static_cast<const T*>(t.data()) + off - t.offset();
}

template <typename T, typename... Ix>
void* device_element_ptr_mut(Tensor& t, Ix... ix) {
  std::initializer_list<size_t> indices{static_cast<size_t>(ix)...};
  const size_t off = t.compute_offset(indices);
  return static_cast<T*>(t.data()) + off - t.offset();
}

}  // namespace detail

template <typename T, typename... Ix>
T host_get(const Tensor& t, Ix... ix) {
  AIFW_EXPECT(t.dtype() == dtype_of_v<T>, "host_get: dtype mismatch");

  const void* src = detail::device_element_ptr<T>(t, ix...);
  T result;
  t.device().read_bytes(&result, src, sizeof(T));
  return result;
}

template <typename T, typename... Ix>
T host_set(const Tensor& t, T value, Ix... ix) {
  AIFW_EXPECT(t.dtype() == dtype_of_v<T>, "host_set: dtype mismatch");
  void* dst = detail::device_element_ptr_mut<T>(t, ix...);
  t.device().write_bytes(dst, &value, sizeof(T));
}

}  // namespace aifw::core::ops
