#pragma once

#include <cstdint>

namespace torch {
namespace executor {
#define EXECUTORCH_FORALL_TAGS(_) \
  _(None)                         \
  _(Tensor)                       \
  _(String)                       \
  _(Double)                       \
  _(Int)                          \
  _(Bool)                         \
  _(ListBool)                     \
  _(ListDouble)                   \
  _(ListInt)                      \
  _(ListTensor)                   \
  _(ListScalar)                   \
  _(ListOptionalTensor)

enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
  EXECUTORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};

} // namespace executor
} // namespace torch
