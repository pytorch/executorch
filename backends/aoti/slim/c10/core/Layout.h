#pragma once

#include <executorch/backends/aoti/slim/c10/util/Exception.h>

#include <cstdint>
#include <ostream>

namespace executorch::backends::aoti::slim::c10 {
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

inline std::ostream& operator<<(std::ostream& stream, c10::Layout layout) {
  switch (layout) {
    case c10::kStrided:
      return stream << "Strided";
    case c10::kSparse:
      return stream << "Sparse";
    case c10::kSparseCsr:
      return stream << "SparseCsr";
    case c10::kSparseCsc:
      return stream << "SparseCsc";
    case c10::kSparseBsr:
      return stream << "SparseBsr";
    case c10::kSparseBsc:
      return stream << "SparseBsc";
    case c10::kMkldnn:
      return stream << "Mkldnn";
    case c10::kJagged:
      return stream << "Jagged";
    default:
      STANDALONE_CHECK(false, "Unknown layout");
  }
}

} // namespace executorch::backends::aoti::slim::c10
