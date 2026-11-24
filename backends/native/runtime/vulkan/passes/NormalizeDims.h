// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn::vulkan {

// Rewrites every dimension argument of an aten op (dim, dims, dim0, dim1,
// dim2, start_dim, end_dim) as dim + rank when negative, so nothing after it
// depends on whether an ET-VK kernel normalizes dims: some map dim to WHCN as
// rank - 1 - dim. The rank is the first tensor input's, plus one for unsqueeze,
// whose dim indexes its output. Throws for a symbolic dim, a dim out of range,
// or an op with no tensor input to take the rank from. Only scheduled nodes are
// visited. Returns the number of dims rewritten.
size_t make_dims_non_negative(Graph& graph);

} // namespace ptn::vulkan
