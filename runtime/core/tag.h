/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/defines.h>
#include <executorch/runtime/platform/compiler.h>
#include <cstdint>

// X11 headers via volk define None, so we need to undef it
#if defined(__linux__)
#undef None
#endif

namespace executorch {
namespace runtime {

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

/**
 * The dynamic type of an EValue.
 */
enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
  EXECUTORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};

#if ET_ENABLE_ENUM_STRINGS
inline const char* tag_to_string(Tag tag) {
  switch (tag) {
#define CASE_TAG(x) \
  case Tag::x:      \
    return #x;
    EXECUTORCH_FORALL_TAGS(CASE_TAG)
#undef CASE_TAG
    default:
      return "Unknown";
  }
}
#endif // ET_ENABLE_ENUM_STRINGS

/**
 * Convert a tag value to a string representation. If ET_ENABLE_ENUM_STRINGS is
 * set (it is on by default), this will return a string name (for example,
 * "Tensor"). Otherwise, it will return a string representation of the index
 * value ("1").
 *
 * If the user buffer is not large enough to hold the string representation, the
 * string will be truncated.
 *
 * The return value is the number of characters written, or in the case of
 * truncation, the number of characters that would be written if the buffer was
 * large enough.
 */
size_t tag_to_string(Tag tag, char* buffer, size_t buffer_size);

/* The size of the buffer needed to hold the longest tag string, including the
 * null terminator. This value is expected to be updated manually, but it
 * checked in test_tag.cpp.
 */
constexpr size_t kTagNameBufferSize = 19;

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::Tag;
} // namespace executor
} // namespace torch
