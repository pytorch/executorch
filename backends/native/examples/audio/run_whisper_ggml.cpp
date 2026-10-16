// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdexcept>

#include <executorch/backends/native/examples/audio/WhisperRunner.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <gflags/gflags.h>

DEFINE_string(
    device,
    "cpu",
    "ggml device: cpu, or gpu for the GPU backend this binary links.");
DEFINE_int32(device_index, 0, "GPU device index.");
DEFINE_int32(
    cpu_threads,
    0,
    "ggml CPU threads; zero keeps the backend default.");
DEFINE_int32(
    max_cached_graphs,
    1,
    "Graphs kept per executable by input shapes; 0 rebuilds every call.");

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Transcribe a Whisper .ptn with the native ggml engine.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return ptn::run_whisper([] {
    ptn::GgmlEngineOptions options;
    if (FLAGS_device == "gpu") {
      options.device = ptn::GgmlDevice::Gpu;
    } else if (FLAGS_device != "cpu") {
      throw std::invalid_argument("--device must be cpu or gpu");
    }
    options.device_index = FLAGS_device_index;
    options.cpu_threads = FLAGS_cpu_threads;
    if (FLAGS_max_cached_graphs < 0) {
      throw std::invalid_argument("--max_cached_graphs must be non-negative");
    }
    options.max_cached_graphs = static_cast<size_t>(FLAGS_max_cached_graphs);
    return ptn::GgmlEngineHost::create(options);
  });
}
