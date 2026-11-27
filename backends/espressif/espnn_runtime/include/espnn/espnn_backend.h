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

/// Backend option that resets delegate-owned state, through
/// `executorch::runtime::set_option(kEspNNBackendId, options)`: true calls
/// reset_delegate_states(), and a reset group (int) calls
/// reset_delegate_states(group).
constexpr char kResetStatesOption[] = "reset_states";

/// Load-time backend option (int, 0 by default) that puts the delegates of a
/// method in a reset group from 0 to kResetGroups - 1. Pass it to
/// `Program::load_method()` in a LoadBackendOptionsMap.
constexpr char kResetGroupOption[] = "reset_group";
constexpr int kResetGroups = 8;

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
  /// Sum of the activation arenas allocated by the delegates initialized in
  /// that time. Destroying a delegate does not subtract its memory.
  size_t arena_bytes;
  /// Sums of the separate ESP-NN scratch buffers, the requantization tables
  /// and the per-operation and per-tensor tables they allocated.
  size_t scratch_bytes;
  size_t parameter_bytes;
  size_t table_bytes;
  /// Sum of the buffers that delegates allocated at init() for outputs
  /// whose memory turns out not to be 16-byte aligned, and the number of
  /// outputs that execute() wrote through them.
  size_t staging_bytes;
  uint32_t staged_outputs;
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
 * also starts from the initial state. Buffers that stay in the program are
 * not reset. Safe to call from any task.
 */
void reset_delegate_states();

/**
 * Same, for the delegates loaded with kResetGroupOption set to @p group.
 *
 * @retval false @p group is not from 0 to kResetGroups - 1.
 */
bool reset_delegate_states(int group);

} // namespace espressif
} // namespace backends
} // namespace executorch
