#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../assert.hpp"
#include "../device/idevice.hpp"
#include "dtype.hpp"
#include "shape.hpp"
#include "storage.hpp"
#include "stride.hpp"

namespace aifw::autograd {

struct AutogradMeta;

}  // namespace aifw::autograd

namespace aifw::core {

class Tensor {
 public:
  Tensor() = default;

  Tensor(IDevice& device, Shape shape, DType dtype);
  Tensor(
      IDevice& device,
      Shape shape,
      Stride stride,
      DType dtype,
      std::shared_ptr<Storage> storage,
      size_t offset
  );

  template <typename T, typename... Ix>
  T& at(Ix... ix);
  template <typename T, typename... Ix>
  const T& at(Ix... ix) const;

  bool defined() const noexcept { return storage_ != nullptr; }
  explicit operator bool() const noexcept { return defined(); }

  void* data();
  const void* data() const;

  template <typename T>
  T* data_as();
  template <typename T>
  const T* data_as() const;

  IDevice& device() const;
  Device device_id() const;
  const Shape& shape() const;
  const Stride& stride() const;
  size_t offset() const;
  DType dtype() const;
  size_t numel() const;

  size_t compute_offset(std::initializer_list<size_t> indices) const;

  Tensor view(Shape new_shape, Stride new_stride, size_t new_offset) const;
  Tensor reshape(Shape new_shape) const;

  bool is_contiguous() const;

  const std::shared_ptr<autograd::AutogradMeta>& autograd_meta() const {
    return autograd_meta_;
  }
  void set_autograd_meta(std::shared_ptr<autograd::AutogradMeta> m) {
    autograd_meta_ = std::move(m);
  }

 private:
  template <typename T>
  void validate_type() const;

  IDevice* device_ = nullptr;
  Shape shape_;
  Stride stride_;
  DType dtype_;
  std::shared_ptr<Storage> storage_;
  size_t offset_ = 0;

  std::shared_ptr<autograd::AutogradMeta> autograd_meta_;
};

inline Tensor::Tensor(IDevice& device, Shape shape, DType dtype)
    : device_(&device),
      shape_(std::move(shape)),
      stride_(make_contiguous_stride(shape_)),
      dtype_(dtype),
      storage_(
          std::make_shared<Storage>(
              device.allocator(), shape_.numel() * dtype_size(dtype_)
          )
      ) {}

inline Tensor::Tensor(
    IDevice& device,
    Shape shape,
    Stride stride,
    DType dtype,
    std::shared_ptr<Storage> storage,
    size_t offset
)
    : device_(&device),
      shape_(std::move(shape)),
      stride_(std::move(stride)),
      dtype_(dtype),
      storage_(std::move(storage)),
      offset_(offset) {}

template <typename T, typename... Ix>
inline T& Tensor::at(Ix... ix) {
  AIFW_ASSERT(defined());
  AIFW_ASSERT(device_id().type == DeviceType::Cpu);
  validate_type<T>();
  std::initializer_list<size_t> list{static_cast<size_t>(ix)...};
  auto* ptr = static_cast<T*>(data());
  return ptr[compute_offset(list)];
}

template <typename T, typename... Ix>
inline const T& Tensor::at(Ix... ix) const {
  AIFW_ASSERT(defined());
  AIFW_ASSERT(device_id().type == DeviceType::Cpu);
  validate_type<T>();
  std::initializer_list<size_t> list{static_cast<size_t>(ix)...};
  const auto* ptr = static_cast<const T*>(data());
  return ptr[compute_offset(list)];
}

inline void* Tensor::data() {
  AIFW_ASSERT(defined());
  return storage_->data();
}

inline const void* Tensor::data() const {
  AIFW_ASSERT(defined());
  return storage_->data();
}

template <typename T>
inline T* Tensor::data_as() {
  AIFW_ASSERT(defined());
  validate_type<T>();
  return static_cast<T*>(storage_->data()) + offset_;
}

template <typename T>
inline const T* Tensor::data_as() const {
  AIFW_ASSERT(defined());
  validate_type<T>();
  return static_cast<const T*>(storage_->data()) + offset_;
}

inline IDevice& Tensor::device() const {
  AIFW_ASSERT(defined());
  return *device_;
}

inline Device Tensor::device_id() const {
  AIFW_ASSERT(defined());
  return device_->device();
}

inline const Shape& Tensor::shape() const {
  AIFW_ASSERT(defined());
  return shape_;
}

inline const Stride& Tensor::stride() const {
  AIFW_ASSERT(defined());
  return stride_;
}

inline size_t Tensor::offset() const {
  AIFW_ASSERT(defined());
  return offset_;
}

inline DType Tensor::dtype() const {
  AIFW_ASSERT(defined());
  return dtype_;
}

inline size_t Tensor::numel() const {
  AIFW_ASSERT(defined());
  return shape_.numel();
}

inline size_t Tensor::compute_offset(
    std::initializer_list<size_t> indices
) const {
  if (indices.size() != shape_.rank())
    throw std::runtime_error("rank mismatch");

  size_t off = offset_;
  size_t dim = 0;

  for (auto idx : indices) off += idx * stride_[dim++];
  return off;
}

inline Tensor Tensor::view(
    Shape new_shape, Stride new_stride, size_t new_offset
) const {
  return Tensor(
      *device_,
      std::move(new_shape),
      std::move(new_stride),
      dtype_,
      storage_,
      new_offset
  );
}

inline Tensor Tensor::reshape(Shape new_shape) const {
  AIFW_EXPECT(new_shape.numel() == shape_.numel(), "reshape: numel mismatch");
  AIFW_EXPECT(is_contiguous(), "reshape: tensor must be contiguous");

  Stride new_stride = make_contiguous_stride(new_shape);
  return Tensor(
      *device_,
      std::move(new_shape),
      std::move(new_stride),
      dtype_,
      storage_,
      offset_
  );
}

inline bool Tensor::is_contiguous() const {
  return stride_.values() == make_contiguous_stride(shape_).values();
}

template <typename T>
inline void Tensor::validate_type() const {
  AIFW_EXPECT(dtype_ == dtype_of_v<T>, "validate_type: dtype mismatch");
}

}  // namespace aifw::core
