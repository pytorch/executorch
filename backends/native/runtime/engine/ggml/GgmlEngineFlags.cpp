// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngineFlags.h>

#include <stdexcept>

#include <gflags/gflags.h>

DEFINE_string(
    device,
    "cpu",
    "ggml device: cpu, or gpu for the GPU backend this binary links.");
DEFINE_int32(device_index, 0, "GPU device index.");
DEFINE_int32(cpu_threads, 0, "ggml CPU threads; 0 keeps ggml's default of 4.");
DEFINE_int32(
    max_cached_graphs,
    1,
    "Graphs kept per executable by input shapes; 0 rebuilds every call.");
DEFINE_bool(
    print_graph_ops,
    false,
    "Print each built graph's node count per ggml op to stderr.");
DEFINE_bool(
    exact_topk_for_testing,
    false,
    "Testing only. Default: topk is one argsort, as in llama.cpp, which leaves "
    "a row holding NaN unordered. Set: NaN ranks above every value as eager "
    "does, at a NaN key map and a second argsort. Ops: topk.");
DEFINE_bool(
    exact_gelu_for_testing,
    false,
    "Testing only. Default: tanh GELU is one ggml_gelu, as in llama.cpp, which "
    "ggml-cpu reads from an F16 table. Set: the CPU builds it from exact ops "
    "as eager does, at five more kernels. Ops: gelu with approximate=tanh; "
    "CPU only.");
DEFINE_bool(
    cuda_f32_accumulation_for_testing,
    false,
    "Testing only. Default: matrix multiplies with an F16 operand accumulate "
    "in F16 on CUDA, as in llama.cpp. Set: they accumulate in F32, at 2-3x "
    "prefill and 4x decode GEMM time. Ops: F16 linear, matmul, mm, bmm and "
    "convolution; CUDA only.");

namespace ptn {

// cppcheck-suppress unusedFunction ; called by the ggml runners
// (run_hf_llm_ggml, ggml_ptn_runner)
GgmlEngineOptions ggml_engine_options_from_flags() {
  GgmlEngineOptions options;
  if (FLAGS_device == "gpu") {
    options.device = GgmlDevice::Gpu;
  } else if (FLAGS_device != "cpu") {
    throw std::invalid_argument("--device must be cpu or gpu");
  }
  options.device_index = FLAGS_device_index;
  options.cpu_threads = FLAGS_cpu_threads;
  if (FLAGS_max_cached_graphs < 0) {
    throw std::invalid_argument("--max_cached_graphs must be non-negative");
  }
  options.max_cached_graphs = static_cast<size_t>(FLAGS_max_cached_graphs);
  options.print_graph_ops = FLAGS_print_graph_ops;
  options.exact_topk_for_testing = FLAGS_exact_topk_for_testing;
  options.exact_gelu_for_testing = FLAGS_exact_gelu_for_testing;
  options.cuda_f32_accumulation_for_testing =
      FLAGS_cuda_f32_accumulation_for_testing;
  return options;
}

} // namespace ptn
