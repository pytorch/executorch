/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/xnnpack/runtime/XNNExecutor.h>
#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/runtime/platform/compiler.h>
#include <xnnpack.h>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

class XNNCompiler {
 public:
  // Takes Flatbuffer Serialized XNNPACK Model and rebuilds the xnn-subgraph
  // returns an executor object that holds the xnn runtime object which we
  // can then use to set inputs and run inference using the xnn graph.
  ET_NODISCARD static executorch::runtime::Error compileModel(
      const void* buffer_pointer,
      size_t num_bytes,
      XNNExecutor* executor,
      XNNWeightsCache* weights_cache,
      xnn_workspace_t workspace,
      const NamedDataMap* named_data_map);
};

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
