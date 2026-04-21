#pragma once

namespace aifw::autograd {

class GradMode {
 public:
  static bool enabled() { return flag_; }
  static void set(bool v) { flag_ = v; }

 private:
  thread_local static inline bool flag_ = true;
};

class NoGradGuard {
 public:
  NoGradGuard() : prev_(GradMode::enabled()) { GradMode::set(false); }
  ~NoGradGuard() { GradMode::set(prev_); }

 private:
  bool prev_;
};

}  // namespace aifw::autograd
