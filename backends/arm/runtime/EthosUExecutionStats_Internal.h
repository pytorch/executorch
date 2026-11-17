/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/arm/runtime/EthosUBackend_Pmu.h>
#include <executorch/backends/arm/runtime/EthosUExecutionStats.h>

namespace executorch::backends::arm {

executorch::runtime::ArrayRef<ethosu_pmu_event_type> ethosu_prepare_invocation(
    EthosUInvocationContext& invocation,
    uint32_t product);
void ethosu_execution_begin(EthosUInvocationContext& invocation);
void ethosu_execution_end(
    const EthosUInvocationContext& invocation,
    const EthosUExecutionRecord* record);

#if defined(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX)
bool ethosu_linux_pmu_event_supported(
    uint32_t product,
    ethosu_pmu_event_type event);
#endif

} // namespace executorch::backends::arm
