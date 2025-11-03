#pragma once

#include <executorch/backends/cuda/runtime/c10/util/Exception.h>

using AOTITorchError = int32_t;
#define AOTI_TORCH_SUCCESS 0
#define AOTI_TORCH_FAILURE 1

#define AOTI_TORCH_CHECK(...) STANDALONE_CHECK(__VA_ARGS__)
#define AOTI_TORCH_WARN(...) STANDALONE_WARN(__VA_ARGS__)

#ifdef _WIN32
#ifdef INLINE_SHIM

// The proper way to do this is to separate the impl. to .cpp files, but I just
// used this hack for the demo
#define AOTI_TORCH_EXPORT inline
#else
#ifdef EXPORT_AOTI_FUNCTIONS
#define AOTI_TORCH_EXPORT                                                      \
  __declspec(dllexport) // used to produce executorch.lib
#else
#define AOTI_TORCH_EXPORT __declspec(dllimport) // not really used for the demo
#endif
#endif
#else
// Linux/Unix
#ifdef EXPORT_AOTI_FUNCTIONS
#define AOTI_TORCH_EXPORT __attribute__((visibility("default")))
#else
#define AOTI_TORCH_EXPORT inline
#endif
#endif // _WIN32

// using namespace executorch::backends::cuda will make sure AOTI-generated code continue to
// work without any change, e.g. c10::DeviceType::CUDA will actually refer to
// executorch::backends::cuda::c10::DeviceType::CUDA
using namespace executorch::backends::cuda;
using namespace executorch::backends::cuda::c10;
