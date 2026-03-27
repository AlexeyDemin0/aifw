#pragma once

#include <cstddef>

#include "../device/iallocator.hpp"

namespace aifw::core {

class Storage {
 public:
  Storage() = default;
  Storage(IAllocator& allocator, size_t bytes);

  Storage(const Storage& other) = delete;
  Storage& operator=(const Storage& other) = delete;

  Storage(Storage&& other) noexcept;
  Storage& operator=(Storage&& other) noexcept;

  ~Storage();

  void* data();
  const void* data() const;

  size_t size() const;

 private:
  void realise();
  void move_from(Storage& other);

  IAllocator* allocator_ = nullptr;
  size_t size_ = 0;
  void* data_ = nullptr;
};

inline Storage::Storage(IAllocator& allocator, size_t bytes)
    : allocator_(&allocator), size_(bytes), data_(allocator.allocate(bytes)) {}

inline Storage::Storage(Storage&& other) noexcept { move_from(other); }

inline Storage& Storage::operator=(Storage&& other) noexcept {
  if (this != &other) {
    realise();
    move_from(other);
  }
  return *this;
}

inline Storage::~Storage() { realise(); }

inline void* Storage::data() { return data_; }

inline const void* Storage::data() const { return data_; }

inline size_t Storage::size() const { return size_; }

inline void Storage::realise() {
  if (data_ && allocator_) allocator_->deallocate(data_);
  data_ = nullptr;
}

inline void Storage::move_from(Storage& other) {
  allocator_ = other.allocator_;
  data_ = other.data_;
  size_ = other.size_;

  other.allocator_ = nullptr;
  other.data_ = nullptr;
  other.size_ = 0;
}

}  // namespace aifw::core
