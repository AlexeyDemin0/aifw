#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <utility>
#include <vector>

namespace aifw::core {

class Shape {
 public:
  Shape() = default;
  explicit Shape(std::vector<size_t> dims);
  explicit Shape(std::initializer_list<size_t> dims);

  size_t operator[](size_t i) const;

  size_t rank() const;
  const std::vector<size_t>& values();
  size_t numel() const;

 private:
  std::vector<size_t> dims_;
};

inline Shape::Shape(std::vector<size_t> dims) : dims_(std::move(dims)) {}

inline Shape::Shape(std::initializer_list<size_t> dims)
    : Shape(std::vector<size_t>(dims)) {}

inline size_t Shape::operator[](size_t i) const { return dims_[i]; }

inline size_t Shape::rank() const { return dims_.size(); }

inline const std::vector<size_t>& Shape::values() { return dims_; }

inline size_t Shape::numel() const {
  if (dims_.empty()) return 0;
  return std::accumulate(dims_.begin(), dims_.end(), size_t(1),
                         std::multiplies<>());
}

}  // namespace aifw::core
