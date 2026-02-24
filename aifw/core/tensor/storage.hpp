#pragma once

#include <cstddef>

#include "../backend/backend.hpp"

namespace aifw::core {

class Storage {
 public:
  Storage() = default;
  Storage(IBackend& backend, size_t bytes);

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

  IBackend* backend_ = nullptr;
  size_t size_ = 0;
  void* data_ = nullptr;
};

inline Storage::Storage(IBackend& backend, size_t bytes)
    : backend_(&backend), size_(bytes), data_(backend.allocate(bytes)) {}

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
  if (data_ && backend_) backend_->deallocate(data_);
  data_ = nullptr;
}

inline void Storage::move_from(Storage& other) {
  backend_ = other.backend_;
  data_ = other.data_;
  size_ = other.size_;

  other.data_ = nullptr;
  other.size_ = 0;
}

}  // namespace aifw::core
