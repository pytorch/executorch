// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn::vulkan {

size_t lower_rms_norms(Graph& graph);

} // namespace ptn::vulkan
