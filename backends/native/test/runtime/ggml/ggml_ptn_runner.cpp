// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// The ggml engine host for PTN test drivers.

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngineFlags.h>
#include <executorch/backends/native/test/runtime/PtnRunner.h>
#include <gflags/gflags.h>

DEFINE_bool(
    f32_gpu_kv_cache_for_testing,
    false,
    "Testing only. Default: on CUDA and HIP, an in-graph F32 KV cache that "
    "only attention reads is stored in F16, as in llama.cpp. Set: it stays "
    "F32. An off-graph cache takes its storage from the cache config instead. "
    "Ops: KV-cache buffers read by scaled_dot_product_attention and the "
    "custom SDPA; CUDA and HIP only.");

namespace ptn::testing {

// cppcheck-suppress unusedFunction
std::unique_ptr<EngineHost> make_engine_host(const HostOptions& host_options) {
  GgmlEngineOptions options = ggml_engine_options_from_flags();
  options.f32_gpu_kv_cache_for_testing = FLAGS_f32_gpu_kv_cache_for_testing;
  options.allow_activation_quant_for_weight_only =
      host_options.allow_activation_quant_for_weight_only;
  return GgmlEngineHost::create(options);
}

} // namespace ptn::testing
