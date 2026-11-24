// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/examples/hf_llm/runtime/LlmRunner.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngineFlags.h>
#include <gflags/gflags.h>

DECLARE_string(kv_dtype);

DEFINE_bool(
    allow_activation_quant_for_weight_only,
    true,
    "Run matmuls of weight-only quantized weights, such as GGUF k-quants, "
    "with ggml's quantized activations, as llama.cpp does; false dequantizes "
    "those weights to F32 and runs them without, for debugging.");

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Generate text from an LLM .ptn with the native ggml engine.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ptn::register_ggml_cache_builders();
  ptn::GgmlEngineOptions options = ptn::ggml_engine_options_from_flags();
  options.f32_gpu_kv_cache_for_testing = FLAGS_kv_dtype == "fp32";
  options.allow_activation_quant_for_weight_only =
      FLAGS_allow_activation_quant_for_weight_only;
  return ptn::run_llm(
      [options] { return ptn::GgmlEngineHost::create(options); },
      ptn::kGgmlCacheBackendId,
      /*half_kv_cache=*/options.device == ptn::GgmlDevice::Gpu);
}
