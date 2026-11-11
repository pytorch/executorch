// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdexcept>

#include <executorch/backends/native/examples/audio/WhisperRunner.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <gflags/gflags.h>

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
      "Transcribe a Whisper .ptn with the native ggml CPU engine.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return ptn::run_whisper([] {
    ptn::GgmlEngineOptions options;
    options.cpu_threads = FLAGS_cpu_threads;
    if (FLAGS_max_cached_graphs < 0) {
      throw std::invalid_argument("--max_cached_graphs must be non-negative");
    }
    options.max_cached_graphs = static_cast<size_t>(FLAGS_max_cached_graphs);
    return ptn::GgmlEngineHost::create(options);
  });
}
