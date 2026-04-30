/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/RuntimeContext.h>

namespace executorch {
namespace backends {
namespace native {

/**
 * RuntimeContext for the HostPool — the canonical host-buffer pool that
 * owns the host-resident home for every boundary value (graph IO and
 * cross-runtime intermediates). Carries no compute state; HostPool does
 * not run kernels.
 */
class HostPoolRuntimeContext final : public RuntimeContext {};

}  // namespace native
}  // namespace backends
}  // namespace executorch
