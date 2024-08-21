/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

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

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::Tag;
} // namespace executor
} // namespace torch
