/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Forked from
 * https://github.com/pytorch/pytorch/blob/master/c10/core/ScalarType.h
 *
 * Everything but the ScalarType definition is in util/ScalarTypeUtil.h
 *
 * Note that these files do not need to be strictly identical to the pytorch
 * core file, as far as names go. The only critical piece is that the types and
 * indices of the main ScalarType enum line up, so that serialization is
 * compatible between the two.
 *
 * Modifications for ExecuTorch:
 * - Namespace torch::executor instead of c10
 * - Macro prefix ET_ instead of AT_
 * - Use ET_CHECK_MSG() instead of TORCH_CHECK()
 * - Don't define standalone constants like `kByte`, `kInt` to keep the
 *   namespace clean
 * - Remove operator<< to avoid a dependency on ostream and stdlib
 * - Make `static inline` functions `inline` to avoid creating multiple
 *   copies of them. See
 *   https://gist.github.com/htfy96/50308afc11678d2e3766a36aa60d5f75#conclusion.
 * - Remove deprecated definitions
 * - Minor cleanup for internal consistency
 */

#pragma once

#include <cstdint>

#include <executorch/runtime/core/portable_type/bfloat16.h>
#include <executorch/runtime/core/portable_type/bits_types.h>
#include <executorch/runtime/core/portable_type/complex.h>
#include <executorch/runtime/core/portable_type/half.h>
#include <executorch/runtime/core/portable_type/qint_types.h>

namespace executorch {
namespace runtime {
namespace etensor {

/**
 * Calls the provided macro on every ScalarType, providing the C type and the
 * ScalarType name to each call.
 *
 * The indices and C types must be consistent with
 * AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS in the core pytorch file
 * c10/core/ScalarType.h. This ensures that ExecuTorch serialization is
 * compatible with ATen serialization.
 *
 * @param _ A macro that takes two parameters: the name of a C type, and the
 *     name of the corresponding ScalarType enumerator.
 */
#define ET_FORALL_SCALAR_TYPES(_)                                             \
  _(uint8_t, Byte) /* 0 */                                                    \
  _(int8_t, Char) /* 1 */                                                     \
  _(int16_t, Short) /* 2 */                                                   \
  _(int32_t, Int) /* 3 */                                                     \
  _(int64_t, Long) /* 4 */                                                    \
  _(::torch::executor::Half, Half) /* 5 */                                    \
  _(float, Float) /* 6 */                                                     \
  _(double, Double) /* 7 */                                                   \
  _(::torch::executor::complex<::torch::executor::Half>, ComplexHalf) /* 8 */ \
  _(::torch::executor::complex<float>, ComplexFloat) /* 9 */                  \
  _(::torch::executor::complex<double>, ComplexDouble) /* 10 */               \
  _(bool, Bool) /* 11 */                                                      \
  _(::torch::executor::qint8, QInt8) /* 12 */                                 \
  _(::torch::executor::quint8, QUInt8) /* 13 */                               \
  _(::torch::executor::qint32, QInt32) /* 14 */                               \
  _(::torch::executor::BFloat16, BFloat16) /* 15 */                           \
  _(::torch::executor::quint4x2, QUInt4x2) /* 16 */                           \
  _(::torch::executor::quint2x4, QUInt2x4) /* 17 */                           \
  _(::torch::executor::bits1x8, Bits1x8) /* 18 */                             \
  _(::torch::executor::bits2x4, Bits2x4) /* 19 */                             \
  _(::torch::executor::bits4x2, Bits4x2) /* 20 */                             \
  _(::torch::executor::bits8, Bits8) /* 21 */                                 \
  _(::torch::executor::bits16, Bits16) /* 22 */

/**
 * Data types (dtypes) that can be used as element types in ETensors.
 */
enum class ScalarType : int8_t {
/// Define an enumerator for each ScalarType.
#define DEFINE_ENUM(unused, name) name,
  ET_FORALL_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM

  /// An explicitly undefined ScalarType. Does not map to any C type.
  Undefined,
  /// The number of ScalarType enumerators.
  NumOptions,
};

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::ScalarType;
} // namespace executor
} // namespace torch
