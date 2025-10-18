// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define CUDA_BACKEND_INIT_EXPORT __declspec(dllexport)
#define CUDA_BACKEND_INIT_IMPORT __declspec(dllimport)
#else
#define CUDA_BACKEND_INIT_EXPORT __attribute__((visibility("default")))
#define CUDA_BACKEND_INIT_IMPORT
#endif

// When building the DLL, define BUILDING_CUDA_BACKEND
// When using the DLL, this will import the function
#ifdef BUILDING_CUDA_BACKEND
#define CUDA_BACKEND_INIT_API CUDA_BACKEND_INIT_EXPORT
#else
#define CUDA_BACKEND_INIT_API CUDA_BACKEND_INIT_IMPORT
#endif

/**
 * Initialize the CUDA backend and register it with the ExecutorTorch runtime.
 * On Windows, this must be called explicitly before loading models that use
 * the CUDA backend. On other platforms, the backend is registered automatically
 * via static initialization.
 */
CUDA_BACKEND_INIT_API void InitCudaBackend();

#ifdef __cplusplus
}
#endif
