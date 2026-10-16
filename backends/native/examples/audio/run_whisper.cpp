// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/examples/audio/WhisperRunner.h>
#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>
#include <gflags/gflags.h>

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Transcribe a Whisper .ptn with the native Vulkan engine.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return ptn::run_whisper([] { return ptn::VulkanEngineHost::create(); });
}
