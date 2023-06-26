// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <executorch/core/kernel_types/lean/half.h>

namespace torch {
namespace executor {

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

} // namespace executor
} // namespace torch
