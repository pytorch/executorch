#pragma once

#include <executorch/backends/cuda/runtime/c10/util/Exception.h>

#include <cstdint>
#include <ostream>

namespace executorch::backends::cuda::c10 {
enum class Layout : int8_t {
  Strided,
  Sparse,
  SparseCsr,
  Mkldnn,
  SparseCsc,
  SparseBsr,
  SparseBsc,
  Jagged,
  NumOptions
};

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kSparseCsr = Layout::SparseCsr;
constexpr auto kMkldnn = Layout::Mkldnn;
constexpr auto kSparseCsc = Layout::SparseCsc;
constexpr auto kSparseBsr = Layout::SparseBsr;
constexpr auto kSparseBsc = Layout::SparseBsc;
constexpr auto kJagged = Layout::Jagged;

inline std::ostream &operator<<(std::ostream &stream, Layout layout) {
  switch (layout) {
  case kStrided:
    return stream << "Strided";
  case kSparse:
    return stream << "Sparse";
  case kSparseCsr:
    return stream << "SparseCsr";
  case kSparseCsc:
    return stream << "SparseCsc";
  case kSparseBsr:
    return stream << "SparseBsr";
  case kSparseBsc:
    return stream << "SparseBsc";
  case kMkldnn:
    return stream << "Mkldnn";
  case kJagged:
    return stream << "Jagged";
  default:
    STANDALONE_CHECK(false, "Unknown layout");
  }
}

} // namespace executorch::backends::cuda::c10
