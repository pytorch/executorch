//==============================================================================
//
// Copyright (c) 2020-2021 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  A header which defines the QNN GPU specialization of the QnnGraph.h interface.
 */

#ifndef QNN_GPU_GRAPH_H
#define QNN_GPU_GRAPH_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#include "QnnGraph.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief An enum which defines the different tensor optimization options. A
 *        tensor may be optimized to the specified QnnGpu_Precision_t when it
 *        is a graph tensor that is not a graph input or a graph output and
 *        does not connect two operations from different op packages.
 */
typedef enum {
  /// Sets the precision mode to floating point 32-bit (FP32)
  QNN_GPU_PRECISION_FP32 = 0,
  /// Sets the precision mode to floating point 16-bit (FP16)
  QNN_GPU_PRECISION_FP16 = 1,
  /// Sets the precision mode to FP16 for storage and FP32 for calculations
  QNN_GPU_PRECISION_HYBRID = 2,
  /// Uses the tensor data type provided by the user (default)
  QNN_GPU_PRECISION_USER_PROVIDED = 3,
} QnnGpu_Precision_t;

/**
 * @brief A struct which defines the QNN GPU graph custom configuration options.
 *        Objects of this type are to be referenced through QnnGraph_CustomConfig_t.
 */
typedef struct {
  QnnGpu_Precision_t precision;
  uint8_t disableMemoryOptimizations;
  uint8_t disableNodeOptimizations;
  uint8_t disableQueueRecording;
} QnnGpuGraph_CustomConfig_t;

// clang-format off
/// QnnGpuGraph_CustomConfig_t initializer macro
#define QNN_GPU_GRAPH_CUSTOM_CONFIG_INIT                              \
  {                                                                   \
    QNN_GPU_PRECISION_USER_PROVIDED,   /*precision*/                  \
    0u,                                /*disableMemoryOptimizations*/ \
    0u,                                /*disableNodeOptimizations*/   \
    0u                                 /*disableQueueRecording*/      \
  }
// clang-format on

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
