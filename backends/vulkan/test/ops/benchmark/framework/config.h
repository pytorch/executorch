// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace executorch {
namespace vulkan {
namespace prototyping {

//
// Global configuration options
//

bool print_output();
void set_print_output(bool print_output);

bool print_latencies();
void set_print_latencies(bool print_latencies);

bool use_gpu_timestamps();
void set_use_gpu_timestamps(bool use_timestamps);

bool debugging();
void set_debugging(bool enable_debugging);

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
