/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

namespace executorch {
namespace runtime {
namespace etensor {

// NOLINTNEXTLINE(misc-unused-using-decls)
using std::nullopt;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::nullopt_t;
// NOLINTNEXTLINE(misc-unused-using-decls)
using std::optional;

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::nullopt;
using ::executorch::runtime::etensor::nullopt_t;
using ::executorch::runtime::etensor::optional;
} // namespace executor
} // namespace torch
