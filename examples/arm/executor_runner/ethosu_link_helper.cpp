/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Helper to force-link the Ethos-U backend when building the portable runner.

#if defined(EXECUTORCH_BUILD_ARM_ETHOSU_LINUX)
#include <executorch/runtime/core/error.h>

extern "C" ::executorch::runtime::Error
executorch_delegate_EthosUBackend_registered();

namespace {
struct EthosULinkHook {
  EthosULinkHook() {
    // Force linker to keep the Ethos-U backend object file.
    (void)executorch_delegate_EthosUBackend_registered();
  }
};

static EthosULinkHook g_link_hook;
} // namespace
#endif // EXECUTORCH_BUILD_ARM_ETHOSU_LINUX
