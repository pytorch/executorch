/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The shared GPU registry must have one definition across all delegate shared
// objects in a process. Export its public API from
// executorch_gpu_shared_runtime.
#if defined(_WIN32)
#if defined(EXECUTORCH_GPU_SHARED_BUILDING)
#define EXECUTORCH_GPU_SHARED_API __declspec(dllexport)
#else
#define EXECUTORCH_GPU_SHARED_API __declspec(dllimport)
#endif
#else
#define EXECUTORCH_GPU_SHARED_API __attribute__((visibility("default")))
#endif
