//==============================================================================
//
// Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  Context component API.
 *
 *          Requires Backend to be initialized.
 *          Graphs and Tensors are created within Context.
 *          Context content once created can be cached into a binary form.
 */

#ifndef QNN_CONTEXT_H
#define QNN_CONTEXT_H

#include "QnnCommon.h"
#include "QnnTypes.h"

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
 * @brief QNN Context API result / error codes.
 */
typedef enum {
  QNN_CONTEXT_MIN_ERROR = QNN_MIN_ERROR_CONTEXT,
  ////////////////////////////////////////////

  /// Qnn context success
  QNN_CONTEXT_NO_ERROR = QNN_SUCCESS,
  /// There is optional API component that is not supported yet. See QnnProperty.
  QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// Context-specific memory allocation/deallocation failure
  QNN_CONTEXT_ERROR_MEM_ALLOC = QNN_COMMON_ERROR_MEM_ALLOC,
  /// An argument to QNN context API is deemed invalid by a backend
  QNN_CONTEXT_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_CONTEXT + 0,
  /// A QNN context has not yet been created in the backend
  QNN_CONTEXT_ERROR_CTX_DOES_NOT_EXIST = QNN_MIN_ERROR_CONTEXT + 1,
  /// Invalid/NULL QNN context handle
  QNN_CONTEXT_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_CONTEXT + 2,
  /// Attempting an operation when graphs in a context haven't been finalized
  QNN_CONTEXT_ERROR_NOT_FINALIZED = QNN_MIN_ERROR_CONTEXT + 3,
  /// Attempt to access context binary with an incompatible version
  QNN_CONTEXT_ERROR_BINARY_VERSION = QNN_MIN_ERROR_CONTEXT + 4,
  /// Failure to create context from binary
  QNN_CONTEXT_ERROR_CREATE_FROM_BINARY = QNN_MIN_ERROR_CONTEXT + 5,
  /// Failure to get size of a QNN serialized context
  QNN_CONTEXT_ERROR_GET_BINARY_SIZE_FAILED = QNN_MIN_ERROR_CONTEXT + 6,
  /// Failure to generate a QNN serialized context
  QNN_CONTEXT_ERROR_GET_BINARY_FAILED = QNN_MIN_ERROR_CONTEXT + 7,
  /// Invalid context binary configuration
  QNN_CONTEXT_ERROR_BINARY_CONFIGURATION = QNN_MIN_ERROR_CONTEXT + 8,
  /// Failure to set profile
  QNN_CONTEXT_ERROR_SET_PROFILE = QNN_MIN_ERROR_CONTEXT + 9,
  /// Invalid config
  QNN_CONTEXT_ERROR_INVALID_CONFIG = QNN_MIN_ERROR_CONTEXT + 10,
  ////////////////////////////////////////////
  QNN_CONTEXT_MAX_ERROR = QNN_MAX_ERROR_CONTEXT,
  // Unused, present to ensure 32 bits.
  QNN_CONTEXT_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnContext_Error_t;

/**
 * @brief Context specific object for custom configuration
 *
 * Please refer to documentation provided by the backend for usage information
 */
typedef void* QnnContext_CustomConfig_t;

/**
 * @brief This enum defines context config options.
 */
typedef enum {
  /// Sets context custom options via QnnContext_CustomConfig_t
  QNN_CONTEXT_CONFIG_OPTION_CUSTOM = 0,
  /// Sets the default priority for graphs in this context. QNN_GRAPH_CONFIG_OPTION_PRIORITY can be
  /// used to override this default.
  QNN_CONTEXT_CONFIG_OPTION_PRIORITY = 1,
  /// Sets the error reporting level.
  QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING = 2,
  /// Sets the string used for custom oem functionality. This config option is DEPRECATED.
  QNN_CONTEXT_CONFIG_OPTION_OEM_STRING = 3,
  /// Sets async execution queue depth for all graphs in this context. This option represents the
  /// number of executions that can be in the queue at a given time before QnnGraph_executeAsync()
  /// will start blocking until a new spot is available. Queue depth is subject to a maximum limit
  /// determined by the backend and available system resources. The default depth is
  /// backend-specific, refer to SDK documentation.
  QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH = 4,
  /// Null terminated array of null terminated strings listing the names of the graphs to
  /// deserialize from a context binary. All graphs are enabled by default. An error is generated if
  /// an invalid graph name is provided.
  QNN_CONTEXT_CONFIG_ENABLE_GRAPHS = 5,
  /// Sets the peak memory limit hint of a deserialized context in megabytes
  QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT = 6,
  /// Indicates that the context binary pointer is available during QnnContext_createFromBinary and
  /// until QnnContext_free is called.
  QNN_CONTEXT_CONFIG_PERSISTENT_BINARY = 7,
  // Unused, present to ensure 32 bits.
  QNN_CONTEXT_CONFIG_UNDEFINED = 0x7FFFFFFF
} QnnContext_ConfigOption_t;

typedef enum {
  /// Sets a numeric value for the maximum queue depth
  QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_NUMERIC = 0,

  // Unused, present to ensure 32 bits
  QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_UNDEFINED = 0x7FFFFFF
} QnnContext_AsyncExecutionQueueDepthType_t;

/**
 * @brief This struct provides async execution queue depth.
 */
typedef struct {
  QnnContext_AsyncExecutionQueueDepthType_t type;
  union UNNAMED {
    uint32_t depth;
  };
} QnnContext_AsyncExecutionQueueDepth_t;

/// QnnContext_AsyncExecutionQueueDepth_t initializer macro
#define QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_INIT                 \
  {                                                                  \
    QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_UNDEFINED, /*type*/ \
    {                                                                \
      0 /*depth*/                                                    \
    }                                                                \
  }

/**
 * @brief This struct provides context configuration.
 */
typedef struct {
  QnnContext_ConfigOption_t option;
  union UNNAMED {
    /// Used with QNN_CONTEXT_CONFIG_OPTION_CUSTOM.
    QnnContext_CustomConfig_t customConfig;
    /// Used with QNN_CONTEXT_CONFIG_OPTION_PRIORITY.
    Qnn_Priority_t priority;
    /// Used with QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING.
    Qnn_ErrorReportingConfig_t errorConfig;
    /// DEPRECATED. Used with QNN_CONTEXT_CONFIG_OPTION_OEM_STRING
    const char* oemString;
    /// Used with QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH
    QnnContext_AsyncExecutionQueueDepth_t asyncExeQueueDepth;
    /// Used with QNN_CONTEXT_CONFIG_ENABLE_GRAPHS
    const char* const* enableGraphs;
    /// Used with QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT
    uint64_t memoryLimitHint;
    /// Used with QNN_CONTEXT_CONFIG_PERSISTENT_BINARY
    uint8_t isPersistentBinary;
  };
} QnnContext_Config_t;

/// QnnContext_Config_t initializer macro
#define QNN_CONTEXT_CONFIG_INIT              \
  {                                          \
    QNN_CONTEXT_CONFIG_UNDEFINED, /*option*/ \
    {                                        \
      NULL /*customConfig*/                  \
    }                                        \
  }

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief A function to create a context.
 *        Context holds graphs, operations and tensors
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] device A device handle to set hardware affinity for the created context. NULL value
 *                   can be supplied for device handle and it is equivalent to calling
 *                   QnnDevice_create() with NULL config.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. All config options have default
 *                   value, in case not provided. If same config option type is provided multiple
 *                   times, the last option value will be used.
 *
 * @param[out] context A handle to the created context.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: at least one argument is invalid
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: failure in allocating memory when creating context
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _backend_ or _device_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: an optional feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_CONFIG: one or more config values is invalid
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_create(Qnn_BackendHandle_t backend,
                                    Qnn_DeviceHandle_t device,
                                    const QnnContext_Config_t** config,
                                    Qnn_ContextHandle_t* context);

/**
 * @brief A function to set/modify configuration options on an already generated context.
 *        Backends are not required to support this API.
 *
 * @param[in] context A context handle.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. All config options have default
 *                   value, in case not provided. If same config option type is provided multiple
 *                   times, the last option value will be used. If a backend cannot support all
 *                   provided configs it will fail.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: at least one config option is invalid
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: an optional feature is not supported
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_setConfig(Qnn_ContextHandle_t context,
                                       const QnnContext_Config_t** config);

/**
 * @brief A function to get the size of memory to be allocated to hold
 *        the context content in binary (serialized) form.
 *        This function must be called after all entities in the context have been finalized.
 *
 * @param[in] context A context handle.
 *
 * @param[out] binaryBufferSize The amount of memory in bytes a client will need to allocate
 *                              to hold context content in binary form.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _binaryBufferSize_ is NULL
 *         - QNN_CONTEXT_ERROR_NOT_FINALIZED: if there were any non-finalized entities in the
 *           context
 *         - QNN_CONTEXT_ERROR_GET_BINARY_SIZE_FAILED: Operation failure due to other factors
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: Attempting to get binary size for a
 *           context re-created from a cached binary.
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: Not enough memory is available to retrieve the context
 *           content.
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_getBinarySize(Qnn_ContextHandle_t context,
                                           Qnn_ContextBinarySize_t* binaryBufferSize);

/**
 * @brief A function to get the context content in binary (serialized) form.
 *        The binary can be used to re-create context by using QnnContext_createFromBinary(). This
 *        function must be called after all entities in the context have been finalized. Unconsumed
 *        tensors are not included in the binary. Client is responsible for allocating sufficient
 *        and valid memory to hold serialized context content produced by this method. It is
 *        recommended the user calls QnnContext_getBinarySize() to allocate a buffer of sufficient
 *        space to hold the binary.
 *
 * @param[in] context A context handle.
 *
 * @param[in] binaryBuffer Pointer to the user-allocated context binary memory.
 *
 * @param[in] binaryBufferSize Size of _binaryBuffer_ to populate context binary with, in bytes.
 *
 * @param[out] writtenBufferSize Amount of memory actually written into _binaryBuffer_, in bytes.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: one of the arguments to the API is invalid/NULL
 *         - QNN_CONTEXT_ERROR_NOT_FINALIZED: if there were any non-finalized entities in the
 *           context
 *         - QNN_CONTEXT_ERROR_GET_BINARY_FAILED: Operation failure due to other factors
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: Attempting to get binary for a
 *           context re-created from a cached binary.
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: Not enough memory is available to retrieve the context
 *           content.
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_getBinary(Qnn_ContextHandle_t context,
                                       void* binaryBuffer,
                                       Qnn_ContextBinarySize_t binaryBufferSize,
                                       Qnn_ContextBinarySize_t* writtenBufferSize);

/**
 * @brief A function to create a context from a stored binary.
 *        The binary was previously obtained via QnnContext_getBinary() and stored by a client. The
 *        content of a context created in this way cannot be further altered, meaning *no* new
 *        nodes or tensors can be added to the context. Creating context by deserializing provided
 *        binary is meant for fast content creation, ready to execute on.
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] device A device handle to set hardware affinity for the created context. NULL value
 *                   can be supplied for device handle and it is equivalent to calling
 *                   QnnDevice_create() with NULL config.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. In case they are not provided,
 *                   all config options have a default value in accordance with the serialized
 *                   context. If the same config option type is provided multiple times, the last
 *                   option value will be used.
 *
 * @param[in] binaryBuffer A pointer to the context binary.
 *
 * @param[in] binaryBufferSize Holds the size of the context binary.
 *
 * @param[out] context A handle to the created context.
 *
 * @param[in] profile The profile handle on which metrics are populated and can be queried. Use
 *            NULL handle to disable profile collection. A handle being re-used would reset and is
 *            populated with values from the current call.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _binaryBuffer_ or _context_ is NULL
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: memory allocation error while creating context
 *         - QNN_CONTEXT_ERROR_CREATE_FROM_BINARY: failed to deserialize binary and
 *           create context from it
 *         - QNN_CONTEXT_ERROR_BINARY_VERSION: incompatible version of the binary
 *         - QNN_CONTEXT_ERROR_BINARY_CONFIGURATION: binary is not configured for this device
 *         - QNN_CONTEXT_ERROR_SET_PROFILE: failed to set profiling info
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _backend_, __profile_, or _device_ is not a
 *           valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_CONFIG: one or more config values is invalid
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_createFromBinary(Qnn_BackendHandle_t backend,
                                              Qnn_DeviceHandle_t device,
                                              const QnnContext_Config_t** config,
                                              const void* binaryBuffer,
                                              Qnn_ContextBinarySize_t binaryBufferSize,
                                              Qnn_ContextHandle_t* context,
                                              Qnn_ProfileHandle_t profile);

/**
 * @brief A function to free the context and all associated graphs, operations & tensors
 *
 * @param[in] context A context handle.
 *
 * @param[in] profile The profile handle on which metrics are populated and can be queried. Use
 *                    NULL handle to disable profile collection. A handle being re-used would reset
 *                    and is populated with values from the current call.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _profile_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: an error is encountered with de-allocation of associated
 *           memory
 *         - QNN_CONTEXT_ERROR_SET_PROFILE: failed to set profiling info
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_free(Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_CONTEXT_H
