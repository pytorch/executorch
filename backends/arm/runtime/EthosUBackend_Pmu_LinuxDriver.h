/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/arm/runtime/EthosUBackend_Pmu.h>
#include <executorch/runtime/core/error.h>

namespace EthosU {
class Inference;
class Device;
} // namespace EthosU

namespace executorch::backends::arm {

executorch::runtime::Error ethosu_linux_inference_begin(
    EthosUInvocationContext& invocation);
void ethosu_linux_inference_end(
    const EthosU::Inference* inference,
    const EthosUInvocationContext& invocation,
    bool success) noexcept;

} // namespace executorch::backends::arm
