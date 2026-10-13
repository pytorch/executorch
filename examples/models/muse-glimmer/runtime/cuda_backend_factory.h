/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/examples/models/muse-glimmer/runtime/backend_factory.h>

namespace executorch::extension::llm {

// The CUDA batching backend: a CudaExecutor over the cell-layout artifact
// export_solo_batching.py writes. Text only; the artifact has no vision path.
// A zero max_session_tokens shares the cell pool evenly between max_sessions,
// capped by the exported context.
runtime::Result<MuseGlimmerBackend> create_muse_glimmer_cuda_backend(
    const MuseGlimmerBackendConfig& config);

} // namespace executorch::extension::llm
