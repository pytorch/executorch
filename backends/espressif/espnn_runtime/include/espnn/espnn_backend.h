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

/// Backend option (bool) that calls reset_delegate_states() when true:
/// `executorch::runtime::set_option(kEspNNBackendId, options)`.
constexpr char kResetStatesOption[] = "reset_states";

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
  /// Sums of the separate ESP-NN scratch buffers, the requantization tables
  /// and the per-operation and per-tensor tables of the initialized delegates.
  size_t scratch_bytes;
  size_t parameter_bytes;
  size_t table_bytes;
  /// Sum of the buffers that delegates allocated on their first execute()
  /// for inputs and outputs they cannot use in place.
  size_t staging_bytes;
  /// Delegate-owned state tensors (mutable buffers), and how many of them
  /// are updated in place rather than copied after the last operator.
  uint32_t states;
  uint32_t states_updated_in_place;
};

/**
 * Registers EspNNBackend with the ExecuTorch runtime.
 *
 * Registration also happens from a static initializer once the delegate is
 * linked: when the espnn_runtime component is linked whole-archive or, in the
 * generated ExecuTorch ESP-IDF component, when the kernel bundle records
 * EspNNBackend or the project calls executorch_add_backend(EspNNBackend).
 * Without either, that component does not link the delegate, and calls to
 * these functions fail to link. Calling this is only needed when the linker
 * could drop the initializer. Safe to call more than once.
 *
 * @retval true The backend is registered.
 */
bool register_espnn_backend();

/// Returns a snapshot of the delegate counters.
DelegateStats delegate_stats();

/// Zeroes all delegate counters.
void reset_delegate_stats();

/**
 * Restores the state that delegates own (the mutable buffers lowered with
 * `delegate_mutable_buffers=True`) to its initial value, in every loaded
 * EspNNBackend delegate, before that delegate's next execute().
 *
 * Each loaded method instance has its own state, kept between executions
 * until this is called or the method is destroyed; loading the method again
 * also starts from the initial state. Safe to call from any task, including
 * while a method executes on another one: that execution finishes with the
 * old state and the next one starts from the initial state.
 */
void reset_delegate_states();

} // namespace espressif
} // namespace backends
} // namespace executorch
