/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace executorch {
namespace backends {
namespace espressif {

/// Backend ID that `EspNNBackend.preprocess()` writes into lowered programs.
constexpr const char* kEspNNBackendId = "EspNNBackend";

/**
 * Counters accumulated by every EspNNBackend delegate since startup or the
 * last reset_delegate_stats() call.
 */
struct DelegateStats {
  /// Number of delegate instances initialized.
  uint32_t delegates;
  /// Number of delegate execute() calls.
  uint32_t calls;
  /// Total time spent in execute(), including input and output conversion.
  int64_t total_us;
  /// Sum of the activation arenas held by the initialized delegates.
  size_t arena_bytes;
};

/**
 * Registers EspNNBackend with the ExecuTorch runtime.
 *
 * Registration also happens from a static initializer when the component is
 * linked whole-archive; calling this is only needed when the linker could
 * drop that initializer. Safe to call more than once.
 *
 * @retval true The backend is registered.
 */
bool register_espnn_backend();

/// Returns a snapshot of the delegate counters.
DelegateStats delegate_stats();

/// Zeroes all delegate counters.
void reset_delegate_stats();

} // namespace espressif
} // namespace backends
} // namespace executorch
