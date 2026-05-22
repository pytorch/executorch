/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// EXPERIMENTAL -- identity passthrough. See ``probe_op.h`` for the
// staging banner and the planned Session-managed lookup that replaces
// this body. The current implementation validates that AOTI emits one
// call per FX-graph probe node — distinct ``probe_id`` constants make
// otherwise-identical probe calls syntactically distinct from
// inductor's POV, so the multi-consumer case in
// ``test_weight_offload_probe_dispatch.py`` survives CSE.

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/weight_offload/probe_op.h>
#include <executorch/backends/cuda/runtime/weight_offload/probe_registry.h>

namespace executorch::backends::cuda {

namespace {

// Process-global call counter. Lives at namespace scope inside the
// shared library so a test linking ``aoti_cuda_shims`` directly can
// observe it through ``weight_offload_probe_count_and_reset``.
std::atomic<int64_t> g_probe_call_count{0};

// Cached env-var read. ``std::getenv`` is racy but the value here is
// only read on first call (function-local static); the trace channel
// is an opt-in test affordance, not a production-tunable.
bool probe_trace_enabled() {
  static const bool enabled =
      std::getenv("EXECUTORCH_WEIGHT_OFFLOAD_PROBE_TRACE") != nullptr;
  return enabled;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError
aoti_torch_cuda_probe(Tensor* input, int64_t probe_id, Tensor** output) {
  ET_CHECK_OR_RETURN_ERROR(
      input != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_probe: input is null");
  ET_CHECK_OR_RETURN_ERROR(
      output != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_probe: output is null");

  g_probe_call_count.fetch_add(1, std::memory_order_relaxed);

  if (probe_trace_enabled()) {
    // Single-line, easy to grep from a subprocess's captured stderr.
    std::fprintf(
        stderr, "[ET_WEIGHT_OFFLOAD_PROBE] probe_id=%" PRId64 "\n", probe_id);
  }

  // Dispatch through the ProbeRegistry. The semantics are:
  //   * Lookup hit  -> a Session owns this dummy pointer; forward
  //                    the probe call to its serve callback.
  //   * Lookup miss, registry empty -> no offload is active; fall
  //                    back to the identity passthrough (preserves
  //                    the manual-probe dispatch tests from commit
  //                    1, where the c-shim is exercised without
  //                    any Session ever being created).
  //   * Lookup miss, registry NOT empty -> offload IS active but
  //                    this pointer is unbound. Hard-fail loudly;
  //                    otherwise the identity passthrough would
  //                    silently read the eager AOTI constant and
  //                    mask the broken binding.
  void* input_data = nullptr;
  auto err =
      ::executorch::backends::aoti::aoti_torch_get_data_ptr(input, &input_data);
  if (err != ::executorch::runtime::Error::Ok) {
    return err;
  }

  auto& registry = weight_offload::ProbeRegistry::instance();
  auto lookup = registry.lookup(input_data);
  if (lookup.found) {
    return lookup.callback(lookup.context, input, probe_id, output);
  }
  if (registry.has_any_context()) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] probe input data_ptr=%p probe_id=%" PRId64
        " has no Session binding but the registry has at least one active "
        "context; refusing to silently identity-passthrough\n",
        input_data,
        probe_id);
    return ::executorch::runtime::Error::Internal;
  }

  return aoti_torch_new_tensor_handle(input, output);
}

int64_t weight_offload_probe_count_and_reset() {
  return g_probe_call_count.exchange(0, std::memory_order_relaxed);
}

#ifdef __cplusplus
} // extern "C"
#endif

} // namespace executorch::backends::cuda
