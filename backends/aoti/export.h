/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Define export macro for Windows DLL
// When building the aoti_cuda_backend library, EXPORT_AOTI_FUNCTIONS is defined
// by CMake, which causes this macro to export symbols using
// __declspec(dllexport). When consuming the library, the macro imports symbols
// using
// __declspec(dllimport). On non-Windows platforms, the macro is empty and has
// no effect.
#ifdef _WIN32
#ifdef EXPORT_AOTI_FUNCTIONS
#define AOTI_SHIM_EXPORT __declspec(dllexport)
#else
#define AOTI_SHIM_EXPORT __declspec(dllimport)
#endif
#else
#define AOTI_SHIM_EXPORT
#endif
