// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/cpu/runtime/KernelProvider.h>

namespace executorch::backends::cpu {
std::unique_ptr<KernelProvider> create_et_provider();
} // namespace executorch::backends::cpu
