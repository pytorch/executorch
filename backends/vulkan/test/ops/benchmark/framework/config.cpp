// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "config.h"

namespace executorch {
namespace vulkan {
namespace prototyping {

// Output and latency printing utilities
namespace {
static int print_output_enabled = 0;
static int print_latencies_enabled = 0;
static int gpu_timestamps_enabled = 0;
static int debugging_enabled = 0;
} // namespace

bool print_output() {
  return print_output_enabled > 0;
}

void set_print_output(bool print_output) {
  print_output_enabled = print_output ? 1 : 0;
}

bool print_latencies() {
  return print_latencies_enabled > 0;
}

void set_print_latencies(bool print_latencies) {
  print_latencies_enabled = print_latencies ? 1 : 0;
}

bool use_gpu_timestamps() {
  return gpu_timestamps_enabled > 0;
}

void set_use_gpu_timestamps(bool use_timestamps) {
  gpu_timestamps_enabled = use_timestamps ? 1 : 0;
}

bool debugging() {
  return debugging_enabled > 0;
}

void set_debugging(bool enable_debugging) {
  debugging_enabled = enable_debugging ? 1 : 0;
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
