#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "../backend/backend.hpp"
#include "dtype.hpp"
#include "shape.hpp"
#include "storage.hpp"
#include "stride.hpp"

namespace aifw::core {

class Tensor {
 public:
  Tensor(IBackend& backend, Shape shape, DType dtype);

  void* data();
  const void* data() const;

  template <typename T>
  T* data_as();
  template <typename T>
  const T* data_as() const;

  const Shape& shape() const;
  const Stride& stride() const;
  DType dtype() const;
  size_t numel() const;

  size_t compute_offset(std::initializer_list<size_t> indices) const;

 private:
  template <typename T>
  void validate_type() const;

  IBackend* backend_;
  Shape shape_;
  Stride stride_;
  DType dtype_;
  std::shared_ptr<Storage> storage_;
  size_t offset_ = 0;
};

inline Tensor::Tensor(IBackend& backend, Shape shape, DType dtype)
    : backend_(&backend),
      shape_(shape),
      stride_(make_contiguous_stride(shape_)),
      dtype_(dtype),
      storage_(std::make_shared<Storage>(backend, shape_.numel())) {}


inline void* Tensor::data() { return storage_->data(); }

inline const void* Tensor::data() const { return storage_->data(); }

template <typename T>
inline T* Tensor::data_as() {
  validate_type<T>();
  return static_cast<T*>(storage_->data()) + offset_;
}

template <typename T>
inline const T* Tensor::data_as() const {
  validate_type<T>();
  return static_cast<const T*>(storage_->data()) + offset_;
}

inline const Shape& Tensor::shape() const { return shape_; }

inline const Stride& Tensor::stride() const { return stride_; }

inline DType Tensor::dtype() const { return dtype_; }

inline size_t Tensor::numel() const { return shape_.numel(); }

inline size_t Tensor::compute_offset(
    std::initializer_list<size_t> indices) const {
  if (indices.size() != shape_.rank())
    throw std::runtime_error("rank mismatch");

  size_t off = offset_;
  size_t dim = 0;

  for (auto idx : indices) off += idx * stride_[dim++];
  return off;
}

template <typename T>
inline void Tensor::validate_type() const {
  if constexpr (std::is_same_v<T, float>) {
    if (dtype_ != DType::Float32) throw std::runtime_error("dtype mismatch");
  } else if constexpr (std::is_same_v<T, double>) {
    if (dtype_ != DType::Float64) throw std::runtime_error("dtype mismatch");
  } else if constexpr (std::is_same_v<T, int32_t>) {
    if (dtype_ != DType::Int32) throw std::runtime_error("dtype mismatch");
  } else if constexpr (std::is_same_v<T, int64_t>) {
    if (dtype_ != DType::Int64) throw std::runtime_error("dtype mismatch");
  } else {
    static_assert(sizeof(T) == 0, "Unsupported dtype");
  }
}

}  // namespace aifw::core
