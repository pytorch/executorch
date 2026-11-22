// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn::vulkan {

// ET-VK's linear multiplies over one batch dim, -3, and ignores any before it,
// so a linear of a rank-4 input runs on the input reshaped to rank 3 and has
// its output reshaped back. Throws for a dynamically shaped rank-4 input and
// for a rank-4 input whose output is not rank 4. Returns the number of linears
// rewritten.
size_t flatten_linear_batches(Graph& graph);

} // namespace ptn::vulkan
