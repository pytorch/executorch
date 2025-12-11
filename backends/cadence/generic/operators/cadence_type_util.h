/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file cadence_type_util.h
 * @brief Common type macros for Cadence quantized operators
 *
 * This header provides utility macros for iterating over supported quantized
 * data types in Cadence operators. These macros are used with switch statements
 * to dispatch to type-specific implementations.
 */

/**
 * Macro to iterate over standard Cadence quantized types (uint8_t, int8_t)
 *
 * Usage:
 *   ET_FORALL_CADENCE_QUANTIZED_TYPES(MACRO)
 *
 * Where MACRO is defined as: #define MACRO(ctype, name) ...
 * - ctype: C++ type (uint8_t or int8_t)
 * - name: ExecutorTorch ScalarType name suffix (Byte or Char)
 *
 * Example:
 *   #define HANDLE_TYPE(ctype, name) \
 *     case ScalarType::name: \
 *       return process<ctype>(tensor); \
 *       break;
 *
 *   ScalarType dtype = tensor.scalar_type();
 *   switch (dtype) {
 *     ET_FORALL_CADENCE_QUANTIZED_TYPES(HANDLE_TYPE)
 *     default:
 *       ET_CHECK_MSG(false, "Unsupported dtype");
 *   }
 */
#define ET_FORALL_CADENCE_QUANTIZED_TYPES(_) \
  _(uint8_t, Byte)                           \
  _(int8_t, Char)

/**
 * Macro to iterate over extended Cadence quantized types including int16_t
 *
 * Usage:
 *   ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(MACRO)
 *
 * Where MACRO is defined as: #define MACRO(ctype, name) ...
 * - ctype: C++ type (uint8_t, int8_t, or int16_t)
 * - name: ExecutorTorch ScalarType name suffix (Byte, Char, or Short)
 *
 * This macro includes int16_t support for operators that can handle 16-bit
 * quantized values (e.g., quantized_linear, quantized_fully_connected).
 */
#define ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(_) \
  _(uint8_t, Byte)                                      \
  _(int8_t, Char)                                       \
  _(int16_t, Short)
