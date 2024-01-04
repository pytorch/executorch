//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief QNN HTP component Graph API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnGraph.h for HTP backend
 */

#ifndef QNN_HTP_GRAPH_H
#define QNN_HTP_GRAPH_H

#include "QnnGraph.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief This enum provides different HTP graph optimization
 *        options that can be used to finalize the graph
 *        for optimum performance.
 */
typedef enum {
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_SCHEDULE_THRESHOLD         = 1,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_RETRIES           = 2,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG = 3,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC                = 4,
  
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_UNKNOWN                    = 0x7fffffff
} QnnHtpGraph_OptimizationType_t;

// clang-format off

/**
 * @brief Struct describing the set of optimization types
 *        and the values associated with each optimization type.
 *
 *        Below is the Map between QnnHtpGraph_OptimizationType_t and allowable values:
 *
 *        \verbatim embed:rst:leading-asterisk
 *        +----+------------------------------------------------------------+---------------------------------------------------------------------+
 *        | #  | OptimizationType option                                    | Allowable values                                                    |
 *        +====+============================================================+=====================================================================+
 *        | 1  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_SCHEDULE_THRESHOLD         | Reserved                                                            |
 *        +----+------------------------------------------------------------+---------------------------------------------------------------------+
 *        | 2  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_RETRIES           | Reserved                                                            |
 *        +----+------------------------------------------------------------+---------------------------------------------------------------------+
 *        | 3  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG | Defines the optimization strategy used by the HTP backend           |
 *        |    |                                                            |                                                                     |
 *        |    |                                                            |   1 = Faster preparation time, less optimal graph                   |
 *        |    |                                                            |                                                                     |
 *        |    |                                                            |   2 = Longer preparation time, more optimal graph                   |
 *        |    |                                                            |                                                                     |
 *        |    |                                                            |   3 = Longest preparation time, most likely even more optimal graph:|
 *        |    |                                                            |       QNN_HTP_DEVICE_CONFIG_OPTION_SOC configuration will be taken  |
 *        |    |                                                            |       into account when possible, details see HTP Backend Specific  |
 *        |    |                                                            |       Page                                                          |
 *        +----+------------------------------------------------------------+---------------------------------------------------------------------+
 *        | 4  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC                | Reserved                                                            |
 *        +----+------------------------------------------------------------+---------------------------------------------------------------------+
 *        \endverbatim
 */
typedef struct {
  QnnHtpGraph_OptimizationType_t type;
  float floatValue;
} QnnHtpGraph_OptimizationOption_t;

/// QnnHtpGraph_OptimizationOption_t initializer macro
#define QNN_HTP_GRAPH_OPTIMIZATION_OPTION_INIT              \
  {                                                         \
    QNN_HTP_GRAPH_OPTIMIZATION_TYPE_UNKNOWN, /*type*/       \
    0.0f                                     /*floatValue*/ \
  }
// clang-format on

/**
 * @brief This enum provides different HTP graph configuration
 *        options associated with QnnGraph
 */
typedef enum {
  QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION                       = 1,
  QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION                          = 2,
  QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE                          = 3,
  QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF = 4,
  QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF        = 5,
  QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS                    = 6,
  
  QNN_HTP_GRAPH_CONFIG_OPTION_UNKNOWN                            = 0x7fffffff
} QnnHtpGraph_ConfigOption_t;

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

// clang-format off

/**
 * @brief        Structure describing the set of configurations supported by graph.
 *               Objects of this type are to be referenced through QnnGraph_CustomConfig_t.
 *
 *               The struct has two fields - option and a union of corresponding config values
 *               Based on the option corresponding item in the union can be used to specify
 *               config.
 *
 *               Below is the Map between QnnHtpGraph_ConfigOption_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+-------------------------------------------------------------------+------------------------------------+
 *               | #  | Config Option                                                     | Configuration Struct/value         |
 *               +====+===================================================================+====================================+
 *               | 1  | QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION                          | QnnHtpGraph_OptimizationOption_t   |
 *               +----+-------------------------------------------------------------------+------------------------------------+
 *               | 2  | QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION                             | Qnn_Precision_t                    |
 *               +----+-------------------------------------------------------------------+------------------------------------+
 *               | 3  | QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE                             | uint32_t                           |
 *               +----+-------------------------------------------------------------------+------------------------------------+
 *               | 4  | QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF    | bool                               |
 *               +----+-------------------------------------------------------------------+------------------------------------+
 *               | 5  | QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF           | bool                               |
 *               +----+-------------------------------------------------------------------+------------------------------------+
 *               | 6  | QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS                       | uint32_t                           |
 *               +----+-------------------------------------------------------------------+------------------------------------+
 *               \endverbatim
 *
 *               NOTE: Option #6 (i.e. QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS), can only be
 *               set prior to the first execution of the graph. Proceeding executions will not use
 *               the updated value if user does change it after the first execution.
 */
typedef struct {
  QnnHtpGraph_ConfigOption_t option;
  union {
    QnnHtpGraph_OptimizationOption_t optimizationOption;
    Qnn_Precision_t precision;
    uint32_t vtcmSizeInMB;
    bool foldReluActivationIntoConvOff;
    bool shortDepthConvOnHmxOff;
    uint64_t numHvxThreads;
    
  };
} QnnHtpGraph_CustomConfig_t;

// clang-format on
/// QnnHtpGraph_CustomConfig_t initializer macro
#define QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT                            \
  {                                                                 \
    QNN_HTP_GRAPH_CONFIG_OPTION_UNKNOWN, /*option*/                 \
    {                                                               \
      QNN_HTP_GRAPH_OPTIMIZATION_OPTION_INIT /*optimizationOption*/ \
    }                                                               \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
