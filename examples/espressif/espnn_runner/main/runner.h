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
#include <cstring>

namespace executorch {
namespace examples {
namespace espressif {

/// Whether @p data is an XTB1 bundle of XTS3 int8 graphs.
inline bool is_int8_bundle(const uint8_t* data, size_t size) {
  return data != nullptr && size >= 12 && std::memcmp(data, "XTB1", 4) == 0;
}

/// Whether @p data is an XTP1 bundle of ExecuTorch programs.
inline bool is_pte_bundle(const uint8_t* data, size_t size) {
  return data != nullptr && size >= 12 && std::memcmp(data, "XTP1", 4) == 0;
}

/**
 * Runs every graph in an XTB1 bundle, checks it against the bundled vectors
 * and prints one SUITE CASE line per graph.
 *
 * @retval true Every case passed.
 */
bool run_int8_suite(const uint8_t* data, size_t size);
/**
 * Loads every program in an XTP1 bundle with the ExecuTorch runtime, checks
 * it against the bundled vectors and prints one ET CASE line per program.
 *
 * @retval true Every case passed.
 */
bool run_pte_suite(const uint8_t* data, size_t size);

} // namespace espressif
} // namespace examples
} // namespace executorch
