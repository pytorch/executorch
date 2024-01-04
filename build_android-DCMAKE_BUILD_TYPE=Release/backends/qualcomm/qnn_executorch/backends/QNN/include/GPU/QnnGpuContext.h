//==============================================================================
//
// Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  A header which defines the QNN GPU specialization of the QnnContext.h interface.
 */

#ifndef QNN_GPU_CONTEXT_H
#define QNN_GPU_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief This enum defines QNN GPU custom context config options.
 */
typedef enum {
  /// Sets performance hint options via QnnGpuContext_PerfHint_t
  QNN_GPU_CONTEXT_CONFIG_OPTION_PERF_HINT = 0,
  /// If non-zero, OpenGL buffers will be used
  QNN_GPU_CONTEXT_CONFIG_OPTION_USE_GL_BUFFERS = 1,
  /// The kernel disk cache directory. Must be non-null
  QNN_GPU_CONTEXT_CONFIG_OPTION_KERNEL_REPO_DIR = 2,
  /// If non-zero, the kernel disk cache will be ignored when initializing
  QNN_GPU_CONTEXT_CONFIG_OPTION_INVALIDATE_KERNEL_REPO = 3,
  /// Unused, present to ensure 32 bits.
  QNN_GPU_CONTEXT_CONFIG_OPTION_UNDEFINED = 0x7FFFFFFF
} QnnGpuContext_ConfigOption_t;

/**
 * @brief An enum which defines the different GPU performance hint options.
 */
typedef enum {
  /// Sets the GPU performance hint to high performance, this is the default
  QNN_GPU_CONTEXT_PERF_HINT_HIGH = 0,
  /// Sets the GPU performance hint to normal performance
  QNN_GPU_CONTEXT_PERF_HINT_NORMAL = 1,
  /// Sets the GPU performance hint to low performance
  QNN_GPU_CONTEXT_PERF_HINT_LOW = 2
} QnnGpuContext_PerfHint_t;

/**
 * @brief A struct which defines the QNN GPU context custom configuration options.
 *        Objects of this type are to be referenced through QnnContext_CustomConfig_t.
 */
typedef struct {
  QnnGpuContext_ConfigOption_t option;
  union UNNAMED {
    QnnGpuContext_PerfHint_t perfHint;
    uint8_t useGLBuffers;
    const char* kernelRepoDir;
    uint8_t invalidateKernelRepo;
  };
} QnnGpuContext_CustomConfig_t;

// clang-format off
/// QnnGpuContext_CustomConfig_t initializer macro
#define QNN_GPU_CONTEXT_CUSTOM_CONFIG_INIT                        \
  {                                                               \
    QNN_GPU_CONTEXT_CONFIG_OPTION_UNDEFINED, /*option*/           \
    {                                                             \
    QNN_GPU_CONTEXT_PERF_HINT_HIGH           /*perfHint*/         \
    }                                                             \
  }
// clang-format on

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
