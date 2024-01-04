//=============================================================================
//
//  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief  Property component API.
 *
 *          Provides means for client to discover capabilities of a backend.
 */

#ifndef QNN_PROPERTY_H
#define QNN_PROPERTY_H

#include "QnnCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================
///
/// Definition of QNN_PROPERTY_GROUP_CORE property group.
///

/**
 * @brief Property group for the QNN core property group.
 */
#define QNN_PROPERTY_GROUP_CORE 0x00000001

///
/// Definition of QNN_PROPERTY_GROUP_BACKEND property group. This group is Core (non-optional) API.
///

/**
 * @brief Property group for the QNN Backend API property group. This is a non-optional API
 *        component and cannot be used as a property key.
 */
#define QNN_PROPERTY_GROUP_BACKEND (QNN_PROPERTY_GROUP_CORE + 100)

/**
 * @brief Property key for determining if a backend supports op package registration.
 *        This is a capability.
 */
#define QNN_PROPERTY_BACKEND_SUPPORT_OP_PACKAGE (QNN_PROPERTY_GROUP_BACKEND + 4)

/**
 * @brief Property key for determining whether or not a backend supports the
 *        QNN_BACKEND_CONFIG_OPTION_PLATFORM configuration.
 */
#define QNN_PROPERTY_BACKEND_SUPPORT_PLATFORM_OPTIONS (QNN_PROPERTY_GROUP_BACKEND + 5)

/**
 * @brief Property key for determining whether a backend supports graph composition.
 *        The following are considered graph composition APIs:
 *        - QnnContext_create
 *        - QnnGraph_create
 *        - QnnGraph_addNode
 *        - QnnGraph_finalize
 *        - QnnTensor_createContextTensor
 *        - QnnTensor_createGraphTensor
 *        - QnnBackend_validateOpConfig
 */
#define QNN_PROPERTY_BACKEND_SUPPORT_COMPOSITION (QNN_PROPERTY_GROUP_BACKEND + 6)

///
/// Definition of QNN_PROPERTY_GROUP_CONTEXT property group. This group is Core (non-optional) API.
///

/**
 * @brief Property group for the QNN Context API property group. This is a non-optional API
 *        component and cannot be used as a property key.
 */
#define QNN_PROPERTY_GROUP_CONTEXT (QNN_PROPERTY_GROUP_CORE + 200)

/**
 * @brief Property key for determining whether or not a backend supports
 *        get binary context. This is a capability.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CACHING (QNN_PROPERTY_GROUP_CONTEXT + 1)

/**
 * @brief Property key for determining whether or not a backend supports
 *        context configurations. This is a capability.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CONFIGURATION (QNN_PROPERTY_GROUP_CONTEXT + 4)

/**
 * @brief Property key for determining whether or not a backend supports graph enablement in a
 *        context. See QNN_CONTEXT_CONFIG_ENABLE_GRAPHS. This is a capability.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_ENABLE_GRAPHS (QNN_PROPERTY_GROUP_CONTEXT + 5)

/**
 * @brief Property key for determining whether or not a backend supports memory limits in a
 *        context. See QNN_CONTEXT_CONFIG_MEMORY_LIMIT. This is a capability.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_MEMORY_LIMIT_HINT (QNN_PROPERTY_GROUP_CONTEXT + 6)

/**
 * @brief Property key for determining whether or not a backend supports context binaries that are
 *        readable throughout the lifetime of the context. See
 *        QNN_CONTEXT_CONFIG_PERSISTENT_BINARY. This is a capability.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_PERSISTENT_BINARY (QNN_PROPERTY_GROUP_CONTEXT + 7)

///
/// Definition of QNN_PROPERTY_GROUP_GRAPH property group. This group is Core (non-optional) API.
///

/**
 * @brief Property group for the QNN Graph API property group. This is a non-optional API
 *        component and cannot be used as a property key.
 */
#define QNN_PROPERTY_GROUP_GRAPH (QNN_PROPERTY_GROUP_CORE + 300)

/**
 * @brief Property key for determining whether or not a backend supports
 *        graph configuration. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_CONFIG (QNN_PROPERTY_GROUP_GRAPH + 1)

/**
 * @brief Property key for determining whether or not a backend supports
 *        signals. This is a capability.
 * @note This capability is equivalent to all of QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_SIGNAL,
 *       QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_SIGNAL, and
 *       QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_ASYNC_SIGNAL having support.
 * @note DEPRECATED: Use QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_SIGNAL,
 *       QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_SIGNAL, or
 *       QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_ASYNC_SIGNAL for QnnGraph API support for QnnSignal.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_SIGNALS (QNN_PROPERTY_GROUP_GRAPH + 2)

/**
 * @brief Property key for determining whether or not a backend supports
 *        asynchronous graph execution. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION (QNN_PROPERTY_GROUP_GRAPH + 3)

/**
 * @brief Property key for determining whether or not a backend supports
 *        execution of graphs will null inputs. This implies that the graph
 *        will contain no APP_WRITE tensors.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_NULL_INPUTS (QNN_PROPERTY_GROUP_GRAPH + 4)

/**
 * @brief Property key for determining whether or not a backend supports
 *        graph level priority control. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_PRIORITY_CONTROL (QNN_PROPERTY_GROUP_GRAPH + 5)

/**
 * @brief Property key for determining whether or not a backend supports QnnSignal for
 *        QnnGraph_finalize. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_SIGNAL (QNN_PROPERTY_GROUP_GRAPH + 6)

/**
 * @brief Property key for determining whether or not a backend supports QnnSignal for
 *        QnnGraph_execute. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_SIGNAL (QNN_PROPERTY_GROUP_GRAPH + 7)

/**
 * @brief Property key for determining whether or not a backend supports QnnSignal for
 *        QnnGraph_executeAsync. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_ASYNC_SIGNAL (QNN_PROPERTY_GROUP_GRAPH + 8)

/**
 * @brief Property key for determining whether a backend supports graph-level
 *        continuous profiling. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_CONTINUOUS_PROFILING (QNN_PROPERTY_GROUP_GRAPH + 9)

/**
 * @brief Property key for determining whether or not a backend supports
 *        graph execution. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE (QNN_PROPERTY_GROUP_GRAPH + 10)

/**
 * @brief Property key for determining whether a backend supports batch multiplier.
 *        This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_BATCH_MULTIPLE (QNN_PROPERTY_GROUP_GRAPH + 11)

/**
 * @brief Property key for determining whether a backend supports per-API profiling data
 *        for graph execute. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_PER_API_PROFILING (QNN_PROPERTY_GROUP_GRAPH + 12)

/**
 * @brief Property key for determining whether or not a backend supports
 *        subgraphs. This is a capability.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_SUBGRAPH (QNN_PROPERTY_GROUP_GRAPH + 13)

///
/// Definition of QNN_PROPERTY_GROUP_OP_PACKAGE property group. This group is Optional portion of
/// API.
///

/**
 * @brief Property group for the QNN Op Package API property group.This can be used as a key to
 *        check if Op Package API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_OP_PACKAGE (QNN_PROPERTY_GROUP_CORE + 400)

/**
 * @brief Property key for determining whether or not an op package supports validation.
 *        This is a capability.
 */
#define QNN_PROPERTY_OP_PACKAGE_SUPPORTS_VALIDATION (QNN_PROPERTY_GROUP_OP_PACKAGE + 1)

/**
 * @brief Property key for determining whether or not an op package supports op implementation
 *         creation and freeing. This is a capability.
 */
#define QNN_PROPERTY_OP_PACKAGE_SUPPORTS_OP_IMPLS (QNN_PROPERTY_GROUP_OP_PACKAGE + 2)

/**
 * @brief Property key for determining whether or not an op package supports duplication of
 *        operation names, such that there are duplicated op_package_name::op_name combinations.
 *        This is a capability.
 */
#define QNN_PROPERTY_OP_PACKAGE_SUPPORTS_DUPLICATE_NAMES (QNN_PROPERTY_GROUP_OP_PACKAGE + 3)

///
/// Definition of QNN_PROPERTY_GROUP_TENSOR property group. This group is Core (non-optional) API.
///

/**
 * @brief Property group for the QNN Tensor API property group. This is a non-optional API
 *        component and cannot be used as a property key.
 */
#define QNN_PROPERTY_GROUP_TENSOR (QNN_PROPERTY_GROUP_CORE + 500)

/**
 * @brief Property key to determine whether or not a backend supports Qnn_MemHandle_t type tensors.
 *        This is a capability.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_MEMHANDLE_TYPE (QNN_PROPERTY_GROUP_TENSOR + 1)

/**
 * @brief Property key to determine whether or not a backend supports creating context tensors.
 *        This is a capability.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_CONTEXT_TENSORS (QNN_PROPERTY_GROUP_TENSOR + 2)

///
/// Definition of QNN_PROPERTY_GROUP_ERROR property group. This group is Optional portion of API.
///

/**
 * @brief Property key for the QNN Error API property group. This can be used as a key to
 *        check if Error API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_ERROR (QNN_PROPERTY_GROUP_CORE + 1000)

///
/// Definition of QNN_PROPERTY_GROUP_MEMORY property group. This group is an optional API.
///

/**
 * @brief Property group for the QNN Memory API property group. This can be used as a key to
 *        check if Memory API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_MEMORY (QNN_PROPERTY_GROUP_CORE + 1100)

///
/// Definition of QNN_PROPERTY_GROUP_SIGNAL property group. This group is an optional API.
///

/**
 * @brief Property group for signal support. This can be used as a key to
 *        check if Signal API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_SIGNAL (QNN_PROPERTY_GROUP_CORE + 1200)

/**
 * @brief Property key to determine whether or not a backend supports abort signals.
 *        This is a capability.
 */
#define QNN_PROPERTY_SIGNAL_SUPPORT_ABORT QNN_PROPERTY_GROUP_SIGNAL + 1

/**
 * @brief Property key to determine whether or not a backend supports timeout signals.
 *        This is a capability.
 */
#define QNN_PROPERTY_SIGNAL_SUPPORT_TIMEOUT QNN_PROPERTY_GROUP_SIGNAL + 2

///
/// Definition of QNN_PROPERTY_GROUP_LOG property group. This group is an optional API.
///

/**
 * @brief Property group for log support. This can be used as a key to
 *        check if Log API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_LOG (QNN_PROPERTY_GROUP_CORE + 1300)

/**
 * @brief Property key for determining whether a backend supports logging with the
 *        system's default stream (callback=NULL). This is a capability.
 */
#define QNN_PROPERTY_LOG_SUPPORTS_DEFAULT_STREAM (QNN_PROPERTY_GROUP_LOG + 1)

///
/// Definition of QNN_PROPERTY_GROUP_PROFILE property group. This group is an optional API.
///

/**
 * @brief Property group for profile support. This can be used as a key to
 *        check if Profile API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_PROFILE (QNN_PROPERTY_GROUP_CORE + 1400)

/**
 * @brief Property key for determining whether a backend supports the custom
 *        profile configuration. This is a capability.
 */
#define QNN_PROPERTY_PROFILE_SUPPORT_CUSTOM_CONFIG (QNN_PROPERTY_GROUP_PROFILE + 1)

/**
 * @brief Property key for determining whether a backend supports the maximum
 *        events profile configuration. This is a capability.
 */
#define QNN_PROPERTY_PROFILE_SUPPORT_MAX_EVENTS_CONFIG (QNN_PROPERTY_GROUP_PROFILE + 2)

/**
 * @brief Property key for determining whether a backend supports the extended
 *        event data. This is a capability.
 */
#define QNN_PROPERTY_PROFILE_SUPPORTS_EXTENDED_EVENT (QNN_PROPERTY_GROUP_PROFILE + 3)

/**
 * @brief Property key for determining whether a backend supports optrace
 *        event data. This is a capability.
 */
#define QNN_PROPERTY_PROFILE_SUPPORT_OPTRACE_CONFIG (QNN_PROPERTY_GROUP_PROFILE + 4)

/**
 * @brief Property group for device support. This can be used as a key to
 *        check if Device API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_DEVICE (QNN_PROPERTY_GROUP_CORE + 1500)

/**
 * @brief Property key for determining if a backend supports device infrastructure.
 *        This is a capability.
 */
#define QNN_PROPERTY_DEVICE_SUPPORT_INFRASTRUCTURE (QNN_PROPERTY_GROUP_DEVICE + 1)

///
/// Definition of QNN_PROPERTY_GROUP_CUSTOM property group. This group represents backend defined
/// properties.
///

/**
 * @brief Property group for custom backend properties.
 */
#define QNN_PROPERTY_GROUP_CUSTOM (QNN_PROPERTY_GROUP_CORE + 2000)

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief Type used for unique property identifiers.
 */
typedef uint32_t QnnProperty_Key_t;

/**
 * @brief QNN Property API result / error codes.
 */
typedef enum {
  QNN_PROPERTY_MIN_ERROR = QNN_MIN_ERROR_PROPERTY,
  //////////////////////////////////////////////

  QNN_PROPERTY_NO_ERROR = QNN_SUCCESS,
  /// Property in question is supported
  QNN_PROPERTY_SUPPORTED = QNN_SUCCESS,
  /// Property in question not supported.
  QNN_PROPERTY_NOT_SUPPORTED = QNN_COMMON_ERROR_NOT_SUPPORTED,

  // Remaining values signal errors.

  /// The the property key was not known to a backend.
  QNN_PROPERTY_ERROR_UNKNOWN_KEY = QNN_MIN_ERROR_PROPERTY + 0,

  //////////////////////////////////////////////
  QNN_PROPERTY_MAX_ERROR = QNN_MAX_ERROR_PROPERTY,
  // Unused, present to ensure 32 bits.
  QNN_PROPERTY_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnProperty_Error_t;

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Queries a capability of the backend.
 *
 * @note Safe to call any time, backend does not have to be created.
 *
 * @param[in] key Key which identifies the capability within group.
 *
 * @return Error code:
 *         - QNN_PROPERTY_SUPPORTED: if the backend supports capability.
 *         - QNN_PROPERTY_ERROR_UNKNOWN_KEY: The provided key is not valid.
 *         - QNN_PROPERTY_NOT_SUPPORTED: if the backend does not support capability.
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnProperty_hasCapability(QnnProperty_Key_t key);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
