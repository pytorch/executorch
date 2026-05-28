/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// EXPERIMENTAL. The c-shim AOTI dispatches to for
// ``executorch_weight_offload::probe(w, probe_id)``. Body lives in
// probe_op.cpp; routes through ProbeRegistry to Session::serve when
// offload is opted in, identity-passthrough otherwise.

#include <cstdint>

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

#ifdef __cplusplus
extern "C" {
#endif

// AOTI c-shim for ``executorch_weight_offload::probe(w, probe_id)``.
//
// AOTI's name mangling convention is ``aoti_torch_<device>_<opname>``
// (see PyTorch's ``cpp_wrapper_cpu.py``). For
// ``executorch_weight_offload::probe(Tensor w, int probe_id) -> Tensor``
// on CUDA, the generated ``wrapper.cpp`` emits a direct call to
// ``aoti_torch_cuda_probe(input_handle, probe_id, &output_handle)``.
// The C signature of this symbol is the string the CUDA backend hands
// AOTI via the ``aot_inductor.custom_ops_to_c_shims`` config (see
// ``backends/cuda/cuda_backend.py``). ``Tensor*`` here is
// ``SlimTensor*``; AOTI's ``AtenTensorHandle`` is pointer-compatible.
//
// At runtime, the symbol must be globally visible from the host process
// so the dynamically-loaded AOTI ``.so`` can resolve it. The CUDA
// backend's ``platform.cpp`` dlopen handshake promotes
// ``libaoti_cuda_shims.so``'s symbols to global on first
// ``load_library`` call.
//
// ``probe_id`` self-identifies each ``(consumer, weight)`` call site so
// the runtime needs no schedule cursor — it indexes the payload's
// ``schedule[probe_id]`` directly. The pass assigns contiguous
// ``probe_id`` values 0..N-1 in graph order.
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_cuda_probe(Tensor* input, int64_t probe_id, Tensor** output);

// Test-only: read and reset the process-global probe call counter the
// c-shim increments on every invocation. Exposed as a separate symbol
// so a C++ unit test that links ``aoti_cuda_shims`` directly can verify
// the dispatch wiring without running the full executor. The Python
// e2e test counts ``[ET_WEIGHT_OFFLOAD_PROBE]`` log lines on stderr
// instead (gated on the ``EXECUTORCH_WEIGHT_OFFLOAD_PROBE_TRACE`` env
// var).
AOTI_SHIM_EXPORT int64_t weight_offload_probe_count_and_reset();

#ifdef __cplusplus
} // extern "C"
#endif

} // namespace executorch::backends::cuda
