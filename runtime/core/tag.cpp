/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/tag.h>

#include <cstdio>

namespace executorch {
namespace runtime {

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
size_t tag_to_string(Tag tag, char* buffer, size_t buffer_size) {
#if ET_ENABLE_ENUM_STRINGS
  const char* name_str;
#define DEFINE_CASE(name) \
  case Tag::name:         \
    name_str = #name;     \
    break;

  switch (tag) {
    EXECUTORCH_FORALL_TAGS(DEFINE_CASE)
    default:
      name_str = "Unknown";
      break;
  }

  return snprintf(buffer, buffer_size, "%s", name_str);
#undef DEFINE_CASE
#else
  return snprintf(buffer, buffer_size, "%d", static_cast<int>(tag));
#endif // ET_ENABLE_ENUM_TO_STRING
}

} // namespace runtime
} // namespace executorch
