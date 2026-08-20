/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Dummy source file for non-Xtensa builds
 * This file is used when building the vision-nnlib library on platforms
 * other than Xtensa, providing empty stubs for compatibility.
 * The actual function implementations are provided as stubs via DISCARD_FUN
 * in headers when COMPILER_XTENSA is not defined.
 */

// This file intentionally contains no function definitions and no includes.
// When COMPILER_XTENSA is not defined, all functions are stubbed out
// using the DISCARD_FUN macro in the header files.
