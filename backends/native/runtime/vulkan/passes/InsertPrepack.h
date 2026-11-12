// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

#include <executorch/backends/native/runtime/Method.h>

namespace ptn::vulkan {

size_t insert_prepack_nodes(Method& method);

} // namespace ptn::vulkan
