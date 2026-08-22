/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstddef>
#include <cstring>

// Weak default for arm_ethos_io_memcpy. Firmware targets can provide a
// strong-symbol override (e.g. routing through DMA on Cortex-M55) without
// touching the upstream EthosUBackend code. Lives in its own translation
// unit so the compiler in the call-site TUs cannot inline this body and
// bypass the link-time override (same trick as bolt_arm_memcpy_external).
extern "C" __attribute__((weak)) void
arm_ethos_io_memcpy(void* dst, const void* src, size_t size) {
  std::memcpy(dst, src, size);
}
