/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/portable_type/half.h>

namespace executorch {
namespace runtime {
namespace etensor {

/**
 * An implementation of complex numbers, compatible with c10/util/complex.h from
 * pytorch core.
 */
template <typename T>
struct alignas(sizeof(T) * 2) complex {
  T real_ = T(0);
  T imag_ = T(0);
};

/**
 * Specialization for Half, which is not a primitive C numeric type.
 */
template <>
struct alignas(4) complex<Half> {
  Half real_;
  Half imag_;
};

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::complex;
} // namespace executor
} // namespace torch
