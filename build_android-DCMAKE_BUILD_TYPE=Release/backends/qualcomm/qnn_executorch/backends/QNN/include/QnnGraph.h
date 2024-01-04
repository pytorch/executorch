//==============================================================================
//
// Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  Graph component API
 *
 *          Requires Backend to be initialized.
 *          Provides composable graph API. Graph is created inside Context.
 *          Nodes are added to the graph. Nodes are connected with Tensors.
 *          Once finalized graph can be executed.
 */

#ifndef QNN_GRAPH_H
#define QNN_GRAPH_H

#include "QnnCommon.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN Graph API result / error codes.
 */
typedef enum {
  QNN_GRAPH_MIN_ERROR = QNN_MIN_ERROR_GRAPH,
  ////////////////////////////////////////

  /// Qnn Graph success
  QNN_GRAPH_NO_ERROR = QNN_SUCCESS,
  /// There is optional API component that is not supported yet. See QnnProperty.
  QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// General error relating to memory allocation in processing graph API
  QNN_GRAPH_ERROR_MEM_ALLOC = QNN_COMMON_ERROR_MEM_ALLOC,
  /// General type of graph error, which has not been identified as any
  /// other error type. Any Graph API can return this error code.
  QNN_GRAPH_ERROR_GENERAL = QNN_COMMON_ERROR_GENERAL,
  /// An argument to QNN API is deemed invalid by a backend
  QNN_GRAPH_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_GRAPH + 0,
  /// Invalid graph handle
  QNN_GRAPH_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_GRAPH + 1,
  /// No graph with specified info is registered in the backend
  QNN_GRAPH_ERROR_GRAPH_DOES_NOT_EXIST = QNN_MIN_ERROR_GRAPH + 2,
  /// Invalid or duplicate graph name
  QNN_GRAPH_ERROR_INVALID_NAME = QNN_MIN_ERROR_GRAPH + 3,
  /// Invalid or NULL QNN tensor
  QNN_GRAPH_ERROR_INVALID_TENSOR = QNN_MIN_ERROR_GRAPH + 4,
  /// Some elements in the op config data are invalid
  QNN_GRAPH_ERROR_INVALID_OP_CONFIG = QNN_MIN_ERROR_GRAPH + 5,
  /// Failure to set profile
  QNN_GRAPH_ERROR_SET_PROFILE = QNN_MIN_ERROR_GRAPH + 6,
  /// Node added before its dependent node(s)
  QNN_GRAPH_ERROR_UNCONNECTED_NODE = QNN_MIN_ERROR_GRAPH + 7,

  /// Failure in creating graph with specified configuration
  QNN_GRAPH_ERROR_CREATE_FAILED = QNN_MIN_ERROR_GRAPH + 20,
  /// Graph couldn't be optimized with specified list of ops or config
  QNN_GRAPH_ERROR_OPTIMIZATION_FAILED = QNN_MIN_ERROR_GRAPH + 21,
  /// Graph finalize failed
  QNN_GRAPH_ERROR_FINALIZE_FAILED = QNN_MIN_ERROR_GRAPH + 22,
  /// Graph attempted to be executed before being finalized
  QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED = QNN_MIN_ERROR_GRAPH + 23,
  /// Graph attempted to be modified after being finalized
  QNN_GRAPH_ERROR_GRAPH_FINALIZED = QNN_MIN_ERROR_GRAPH + 24,
  /// FIFO queue cannot register any more async execution requests
  QNN_GRAPH_ERROR_EXECUTION_ASYNC_FIFO_FULL = QNN_MIN_ERROR_GRAPH + 25,

  /// A control signal object was provided to a call, but that signal object
  /// is already in-use by another call.
  QNN_GRAPH_ERROR_SIGNAL_IN_USE = QNN_MIN_ERROR_GRAPH + 30,
  /// Return when a call is aborted early due to a QnnSignal_trigger call issued
  /// to the observed signal object.
  QNN_GRAPH_ERROR_ABORTED = QNN_MIN_ERROR_GRAPH + 31,
  /// A profile handle was bound to a graph, but that profile handle is
  /// already in-use by another graph.
  QNN_GRAPH_ERROR_PROFILE_IN_USE = QNN_MIN_ERROR_GRAPH + 32,
  /// Return when a call is aborted early due to a QnnSignal timeout
  QNN_GRAPH_ERROR_TIMED_OUT = QNN_MIN_ERROR_GRAPH + 33,

  /// Operation not permitted on a subgraph
  QNN_GRAPH_ERROR_SUBGRAPH = QNN_MIN_ERROR_GRAPH + 34,

  /// Graph is not enabled
  QNN_GRAPH_ERROR_DISABLED = QNN_MIN_ERROR_GRAPH + 35,

  ////////////////////////////////////////
  QNN_GRAPH_MAX_ERROR = QNN_MAX_ERROR_GRAPH,
  // Unused, present to ensure 32 bits.
  QNN_GRAPH_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnGraph_Error_t;

/**
 * @brief This enum defines graph config options.
 */
typedef enum {
  /// Sets backend custom configs, see backend specific documentation.
  QNN_GRAPH_CONFIG_OPTION_CUSTOM = 0,
  /// Sets priority of a graph within the context. This config overrides
  /// QNN_CONTEXT_CONFIG_OPTION_PRIORITY which provides the default graph priority.
  QNN_GRAPH_CONFIG_OPTION_PRIORITY = 3,
  /// Enables continuous profiling of a graph. This can include finalize and execute data. The
  /// profile handle will be bound to the graph until a new handle is bound or the graph has been
  /// freed. This feature is mutually exclusive with the per-API profile handles. A
  /// Qnn_ProfileHandle_t bound to a graph can be concurrently used with QnnProfile_get* APIs. A
  /// Qnn_ProfileHandle_t may only be bound to one graph at a time. A different Qnn_ProfileHandle_t
  /// may be bound to the graph via QnnGraph_setConfig.
  QNN_GRAPH_CONFIG_OPTION_PROFILE_HANDLE = 4,
  // Unused, present to ensure 32 bits.
  QNN_GRAPH_CONFIG_OPTION_UNDEFINED = 0x7FFFFFFF
} QnnGraph_ConfigOption_t;

/**
 * @brief Graph specific object for custom configuration
 *
 * Please refer to documentation provided by the backend for usage information
 */
typedef void* QnnGraph_CustomConfig_t;

/**
 * @brief This struct provides graph configuration.
 */
typedef struct {
  QnnGraph_ConfigOption_t option;
  union UNNAMED {
    QnnGraph_CustomConfig_t customConfig;
    Qnn_Priority_t priority;
    Qnn_ProfileHandle_t profileHandle;
  };
} QnnGraph_Config_t;

/// QnnGraph_Config_t initializer macro
#define QNN_GRAPH_CONFIG_INIT                     \
  {                                               \
    QNN_GRAPH_CONFIG_OPTION_UNDEFINED, /*option*/ \
    {                                             \
      NULL /*customConfig*/                       \
    }                                             \
  }

/**
 * @brief This struct provides status associated with Qnn_NotifyFn_t() function.
 */
typedef struct {
  Qnn_ErrorHandle_t error;
} Qnn_NotifyStatus_t;

/// Qnn_NotifyStatus_t initializer macro
#define QNN_NOTIFY_STATUS_INIT \
  { 0u /*error*/ }

/**
 * @brief A client-defined callback function. It is not guaranteed that a spot in the execution
 *        queue is free once this callback is called. i.e. it cannot be inferred that once a
 *        callback is received, the next call to QnnGraph_executeAsync() will not block due to the
 *        queue being full.
 *
 * @param[in] notifyParam Client supplied data object which may be used to identify
 *                        which function this callback applies to.
 *
 * @param[in] notifyStatus Execution status associate with callback.
 *
 * @return None
 *
 */
typedef void (*Qnn_NotifyFn_t)(void* notifyParam, Qnn_NotifyStatus_t notifyStatus);

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief A function to create an empty graph.
 *        The function returns an opaque object to be used on all graph APIs
 *        (addNode, finalize, execute, ...)
 *
 * @param[in] contextHandle A handle to the context in which the graph would be created.
 *
 * @param[in] graphName A string which identifies the graph. Graph name allows retrieval of the
 *                      graph after creating the context from cached binary.  _graphName_ must be
 *                      unique within the _context_.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. All config options have default
 *                   value, in case not provided. If same config option type is provided multiple
 *                   times, the last option value will be used.
 *
 * @param[out] graphHandle The created graph handle.
 *
 * @return Error code:
 *         - QNN_SUCCESS: the graph was successfully created
 *         - QNN_GRAPH_ERROR_INVALID_ARGUMENT: _graph_ is NULL or at least one config option was
 *           invalid
 *         - QNN_GRAPH_ERROR_INVALID_NAME: _graphName_ is NULL or not unique within the
 *           _context_
 *         - QNN_GRAPH_ERROR_INVALID_HANDLE: _context_ is not a valid handle
 *         - QNN_GRAPH_ERROR_MEM_ALLOC: create failed due to memory/resource allocation
 *         - QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE: some API feature is not supported yet, e.g.
 *           config option
 *         - QNN_GRAPH_ERROR_CREATE_FAILED: create failed due to some other reason
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: create failed when context is
 *           re-created from binary using QnnContext_createFromBinary().
 *         - QNN_GRAPH_ERROR_PROFILE_IN_USE: when a profile handle is passed as graph config, that
 *           profile handle can only be bound to one graph at a time
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnGraph_create(Qnn_ContextHandle_t contextHandle,
                                  const char* graphName,
                                  const QnnGraph_Config_t** config,
                                  Qnn_GraphHandle_t* graphHandle);

/**
 * @brief A function to create an empty graph which will be a subgraph of another graph.
 *        The function returns an opaque object to be used to add nodes to the subgraph.
 *        A subgraph can not be explicitly finalized or executed. Only a graph with no
 *        parent graphs can be finalized and executed.
 *
 * @param[in] graphHandle Handle to the graph in which the subgraph is created.
 *
 * @param[in] graphName A string which identifies the graph. Graph name allows retrieval of the
 *                      graph after creating the context from cached binary. _graphName_ must be
 *                      unique within the _context_.
 *
 * @param[out] subgraphHandle The created subgraph handle.
 *
 * @note A subgraph can have another subgraph as a parent.
 *
 * @note Nodes and tensors can be added to a subgraph before and/or after the subgraph handle has
 *       been included as part of an op config added as a node.
 *
 * @return Error code:
 *         - QNN_SUCCESS: the graph was successfully created
 *         - QNN_GRAPH_ERROR_INVALID_ARGUMENT: _subgraphHandle_ is NULL
 *         - QNN_GRAPH_ERROR_INVALID_NAME: _graphName_ is NULL or not unique within the
 *           _context_
 *         - QNN_GRAPH_ERROR_INVALID_HANDLE: _graphHandle_ is not a valid handle
 *         - QNN_GRAPH_ERROR_MEM_ALLOC: create failed due to memory/resource allocation
 *         - QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE: This API is not yet supported
 *         - QNN_GRAPH_ERROR_CREATE_FAILED: create failed due to some other reason
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: create failed when context is
 *           re-created from binary using QnnContext_createFromBinary().
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnGraph_createSubgraph(Qnn_GraphHandle_t graphHandle,
                                          const char* graphName,
                                          Qnn_GraphHandle_t* subgraphHandle);

/**
 * @brief A function to set/modify configuration options on an already created graph.
 *        Backends are not required to support this API.
 *
 * @param[in] graphHandle A graph handle.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. All config options have default
 *                   value, in case not provided. If same config option type is provided multiple
 *                   times, the last option value will be used. If a backend cannot support all
 *                   provided configs it will fail.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_GRAPH_ERROR_INVALID_HANDLE: _graphHandle_ is not a valid handle
 *         - QNN_GRAPH_ERROR_INVALID_ARGUMENT: at least one config option is invalid
 *         - QNN_GRAPH_ERROR_GRAPH_FINALIZED: at least one valid config option is not valid
 *           on a finalized graph
 *         - QNN_GRAPH_ERROR_SUBGRAPH: operation not permitted on a subgraph
 *         - QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE: at least one valid config option is not supported
 *         - QNN_GRAPH_ERROR_PROFILE_IN_USE: when a profile handle is passed as graph config, that
 *           profile handle can only be bound to one graph at a time
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnGraph_setConfig(Qnn_GraphHandle_t graphHandle,
                                     const QnnGraph_Config_t** config);

/**
 * @brief A function to add a node to the graph
 *
 * @param[in] graphHandle The graph or sub-graph handle to add the node to.
 *
 * @note The following conditions should be honored by tensors specified as
 *       part of opConfig:
 *       1. No tensor in the list opConfig.outputTensors can be of type
 *          QNN_TENSOR_TYPE_APP_WRITE or QNN_TENSOR_TYPE_STATIC.
 *       2. All parameters in the opConfig that happen to be tensors must be
 *          of the type QNN_TENSOR_TYPE_STATIC.
 *       3. Tensors express connectivity between nodes. However, it is permissible
 *          for tensors to remain 'unconsumed' in a graph, i.e.,
 *          not act as inputs to any other node in the graph.
 *
 * @note QnnGraph does not validate opConfig used in creating node beyond checks for basic sanity.
 *       A thorough validation of opConfig for this node defined in a certain op package
 *       has to be done via QnnBackend_validateOpConfig().
 *
 * @note Nodes must be added in dependency order. i.e. all QNN_TENSOR_TYPE_NATIVE inputs to the node
 *       must be outputs of a previously added node.
 *
 * @param[in] opConfig A struct containing the configuration of the operation which should be
 *                     added as a node in the graph. The tensor objects in this structure for
 *                     inputs and outputs to the node must be created with APIs in QnnTensor.h
 *                     which register them with a backend. Unrecognized tensors in the opConfig
 *                     result in failure. Since the tensor ID is provided by the backend and is
 *                     unique, it is sufficient to only specify a valid tensor ID in the
 *                     Qnn_Tensor_t structures associated with the opConfig. All other fields
 *                     including any static data are ignored by the backend when parsing these
 *                     tensors.
 *
 * @return Error code
 *         - QNN_SUCCESS: the node is successfully added to the graph
 *         - QNN_GRAPH_ERROR_INVALID_OP_CONFIG: misconfigured operation - invalid op config
 *           Thrown when a BE cannot match package name and/or op name with any
 *           registered op packages, or when
 *           tensor metadata for tensors in opConfig differs from that used in
 *           registering them with a graph using QnnTensor_createGraphTensor().
 *         - QNN_GRAPH_ERROR_INVALID_TENSOR: when tensor objects within opConfig are invalid
 *         - QNN_GRAPH_ERROR_INVALID_HANDLE: _graph_ is not a valid handle
 *         - QNN_GRAPH_ERROR_GRAPH_FINALIZED: add nodes on a finalized graph
 *         - QNN_GRAPH_ERROR_UNCONNECTED_NODE: node added before its dependent node(s)
 *         - QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE: some API feature is not supported yet
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnGraph_addNode(Qnn_GraphHandle_t graphHandle, Qnn_OpConfig_t opConfig);

/**
 * @brief A function to finalize the graph. The runtime will process the
 *        graph, validate that all operations are created successfully and
 *        that connectivity is correct.
 *
 * @param[in] graphHandle Handle to the graph to be finalized.
 *
 * @param[in] profileHandle The profile handle on which metrics is populated and can be queried.
 *                          Use NULL handle to disable profile collection. A handle being re-used
 *                          would reset and is populated with values from the current call. This
 *                          handle must be NULL when a continuous profile handle has been configured
 *                          via the QNN_GRAPH_CONFIG_OPTION_PROFILE_HANDLE option
 *
 * @param[in] signalHandle Signal object to control the execution of the finalize process. NULL may
 *                         be passed to indicate that no execution control is requested, and the
 *                         finalize operation should continue to completion uninterrupted.
 *                         The signal object, if not NULL, is considered to be in-use for
 *                         the duration of the call.
 *
 * @note Graphs that contain zero nodes will fail to finalize.
 *
 * @return Error code:
 *         - QNN_SUCCESS: the graph is finalized successfully
 *         - QNN_GRAPH_ERROR_INVALID_HANDLE: _graph_ is not a valid handle
 *         - QNN_GRAPH_ERROR_INVALID_ARGUMENT:
 *            - invalid param passed in OR
 *            - continuous graph profiling is enabled and the per-API handle is not NULL.
 *         - QNN_GRAPH_ERROR_CREATE_FAILED: op/kernel creation failed
 *         - QNN_GRAPH_ERROR_OPTIMIZATION_FAILED: graph optimization failed
 *         - QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE: some API feature is not supported yet,
 *           e.g. signal or profile
 *         - QNN_GRAPH_ERROR_SET_PROFILE: set profile failed
 *         - QNN_GRAPH_ERROR_SIGNAL_IN_USE: the supplied control signal is
 *           already in-use by another call.
 *         - QNN_GRAPH_ERROR_ABORTED: the call is aborted before completion due to user cancellation
 *         - QNN_GRAPH_ERROR_TIMED_OUT: the call is aborted before completion due to a timeout
 *         - QNN_GRAPH_ERROR_FINALIZE_FAILED: finalize failed for some other reason
 *         - QNN_GRAPH_ERROR_SUBGRAPH: operation not permitted on a subgraph
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnGraph_finalize(Qnn_GraphHandle_t graphHandle,
                                    Qnn_ProfileHandle_t profileHandle,
                                    Qnn_SignalHandle_t signalHandle);

/**
 * @brief A function to retrieve a graph based on name.
 *        This function is typically used when a context was created from cached binary. The
 *        re-created context has graph(s) which are also re-created. The function returns the graph
 *        handle to be used for all graph APIs (addNode, finalize, execute, ...).
 *
 * @param[in] contextHandle An opaque ID to the context.
 *
 * @param[in] graphName A string which identifies the graph.
 *
 * @param[out] graphHandle A pointer to the graph handle that is being retrieved.
 *
 * @return Error code:
 *         - QNN_SUCCESS: the graph was successfully retrieved
 *         - QNN_GRAPH_ERROR_INVALID_NAME: _graphName_ or _graph_ is NULL
 *         - QNN_GRAPH_ERROR_INVALID_HANDLE: _context_ is not a valid handle
 *         - QNN_GRAPH_ERROR_GRAPH_DOES_NOT_EXIST: graph not found/created
 *         - QNN_GRAPH_ERROR_SUBGRAPH: operation not permitted on a subgraph
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnGraph_retrieve(Qnn_ContextHandle_t contextHandle,
                                    const char* graphName,
                                    Qnn_GraphHandle_t* graphHandle);

/**
 * @brief Synchronously execute a finalized graph.
 *
 * @param[in] graphHandle Handle of finalized graph to execute.
 *
 * @param[in] inputs Array of tensors with which to populate graph inputs.
 *
 * @param[in] numInputs Number of input tensors.
 *
 * @param[out] outputs Array of output tensors which the graph will populate with output values.
 *
 * @param[in] numOutputs Number of output tensors.
 *
 * @param[in] profileHandle The profile handle on which metrics is populated and can be queried.
 *                          Use NULL handle to disable profile collection. A handle being reused
 *                          would reset and is populated with values from the current call. This
 *                          handle must be NULL when a continuous profile handle has been configured
 *                          via the QNN_GRAPH_CONFIG_OPTION_PROFILE_HANDLE option
 *
 * @param[in] signalHandle Signal object which may be used to control the execution of this call.
 *                         NULL indicates execution should proceed as normal.
 *                         The signal object, if not NULL, is considered to be in-use
 *                         for the duration of the call.
 *
 * @note Tensors in _inputs_ and _outputs_ must carry the same ID that was assigned when they were
 *       created. Values for all other attributes in Qnn_Tensor_t are assumed from the point at
 *       which they were registered with a backend at the time of tensor creation, with the
 *       following exceptions:
 *       - Tensor data provided by client in structs such as _clientBuf_ can be changed between
 *         invocations to execute().
 *       - An _inputs_ or _outputs_ tensor Qnn_TensorV1_t _dimensions_ field, if non-null, should
 *         match the values provided at tensor creation, with the following exception. The batch
 *         dimension, as determined by the op definition, can be an integer multiple of the
 *         respective dimension provided at tensor creation. All _inputs_ and _outputs_ tensors
 *         must have the same batch multiple.
 *       - Additionally, an _outputs_ tensor Qnn_TensorV1_t _dimensions_ field, if non-null, can
 *         vary after graph execution. As determined by the op definition, non-batch dimensions may
 *         be less than the respective dimension at tensor creation.
 *       - Other fields like _dataType_ can also be permitted to change between invocations to
 *         execute() for certain ops that perform data type conversions.
 *       - Some backends may be able to execute a graph with no _inputs_ provided the graph has no
 *         application-writable tensors.
 *       - Graph I/O Tensors marked optional (i.e. omitted or marked as type=QNN_TENSOR_TYPE_NULL
 *         during QnnGraph_addNode()) cannot be supplied to QnnGraph_execute(). Clients mark
 *         tensors to be of type QNN_TENSOR_TYPE_NULL to indicate that they must be ignored when
 *         constructing a node that lists them as optional.
 *
 * @note If there are simultaneous calls to QnnGraph_execute() and QnnGraph_executeAsync(), the
 *       priority for enqueuing or executing is equal. Both functions operate on the same queue,
 *       the only difference in behavior is whether the function returns when the execution is
 *       enqueued, or when the execution finishes. If there are executions already enqueued, the
 *       execution will be added to the end of the queue, and QnnGraph_execute() will block while
 *       waiting in the queue.
 *
 * @return Error code:
 *         - QNN_SUCCESS: the graph was successfully executed
 *         - QNN_GRAPH_ERROR_INVALID_HANDLE: _graph_ is not a valid handle
 *         - QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED: graph was not finalized
 *         - QNN_GRAPH_ERROR_SUBGRAPH: cannot execute a subgraph
 *         - QNN_GRAPH_ERROR_INVALID_ARGUMENT:
 *            - _inputs_ or _outputs_ is NULL or ill-formed OR
 *            - _inputs_ is NOT NULL and _numInputs_ is 0 OR
 *            - _outputs_ is NOT NULL and _numOutputs_ is 0 OR
 *            - _profile_ handle is invalid OR
 *            - continuous graph profiling is enabled and the per-API handle is not NULL.
 *         - QNN_GRAPH_ERROR_INVALID_TENSOR: one or more tensors in _inputs_ or _outputs_
 *           is invalid or not recognized by graph
 *         - QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE: graph execution is not supported on this
 *           backend or some API feature is not supported yet, e.g. signal, profile, or batch
 *           multiplier
 *         - QNN_GRAPH_ERROR_SET_PROFILE: set profile failed
 *         - QNN_GRAPH_ERROR_SIGNAL_IN_USE: the supplied control signal is already in-use by
 *           another call.
 *         - QNN_GRAPH_ERROR_ABORTED: the call is aborted before completion due to user cancellation
 *         - QNN_GRAPH_ERROR_TIMED_OUT: the call is aborted before completion due to a timeout
 *         - QNN_GRAPH_ERROR_DISABLED: the graph was not enabled when the context was deserialized
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnGraph_execute(Qnn_GraphHandle_t graphHandle,
                                   const Qnn_Tensor_t* inputs,
                                   uint32_t numInputs,
                                   Qnn_Tensor_t* outputs,
                                   uint32_t numOutputs,
                                   Qnn_ProfileHandle_t profileHandle,
                                   Qnn_SignalHandle_t signalHandle);

/**
 * @brief Asynchronously execute a finalized graph. Graphs will be enqueued for execution in FIFO
 * order. There is no guarantee that graphs will finish execution in the same order they were
 * enqueued. If the the execution queue is full, this function will block until space is available.
 *
 * @param[in] graphHandle Handle of finalized graph to execute.
 *
 * @param[in] inputs Array of input tensors with which to populate graph inputs.
 *
 * @param[in] numInputs Number of input tensors.
 *
 * @param[out] outputs Array of tensors which the graph will populate with output values.
 *
 * @param[in] numOutputs Number of output tensors.
 *
 * @param[in] profileHandle The profile handle on which metrics is populated and can be queried.
 *                          Use NULL handle to disable profile collection. A handle being reused
 *                          would reset and is populated with values from the enqueued execute
 *                          call. Profile handle management/reuse across asynchronous calls is
 *                          client's responsibility. Behavior is undefined if same profile handle
 *                          is used by two enqueued execute instances at the same time. This
 *                          handle must be NULL when a continuous profile handle has been
 *                          configured via the QNN_GRAPH_CONFIG_OPTION_PROFILE_HANDLE option
 *
 * @param[in] signalHandle Signal object which may be used to control the execution of this call.
 *                         NULL indicates execution should proceed as normal. All pending
 *                         executions in the queue are affected by Signal control. Instance
 *                         executing when Signal control is issued may not be affected.
 *                         The signal object, if not NULL, is considered to be in-use
 *                         for the duration of the call. For timeout signals, the timeout
 *                         duration applies from the QnnGraph_executeAsync call until the
 *                         callback is called. The same Qnn_GraphHandle_t can be used
 *                         for multiple calls to QnnGraph_executeAsync, however, different
 *                         Qnn_SignalHandle_t must be supplied.
 *
 * @param[in] notifyFn Pointer to notification function, called when execution is finished. NULL
 *                     indicates no notification is requested. _notifyFn_ will be called in
 *                     context of backend owned thread, with priority equal or lower than client's
 *                     calling thread. Please note that a failed call to QnnGraph_executeAsync
 *                     does not call the notification function.
 *
 * @param[in] notifyParam Client-supplied data object which will be passed back via _notifyFn_ and
 *                        can be used to identify asynchronous execution instance. Can be NULL.
 *
 * @note Tensors in _inputs_ and _outputs_ must carry the same ID that was assigned when they were
 *       created. Values for all other attributes in Qnn_Tensor_t are assumed from the point at
 *       which they were registered with a backend at the time of tensor creation, with the
 *       following exceptions:
 *       - Tensor data provided by client in structs such as _clientBuf_ can be changed between
 *         invocations to execute().
 *       - An _inputs_ or _outputs_ tensor Qnn_TensorV1_t _dimensions_ field, if non-null, should
 *         match the values provided at tensor creation, with the following exception. The batch
 *         dimension, as determined by the op definition, can be an integer multiple of the
 *         respective dimension provided at tensor creation. All _inputs_ and _outputs_ tensors
 *         must have the same batch multiple.
 *       - Additionally, an _outputs_ tensor Qnn_TensorV1_t _dimensions_ field, if non-null, can
 *         vary after graph execution. As determined by the op definition, non-batch dimensions may
 *         be less than the respective dimension at tensor creation.
 *       - Some backends may be able to execute a graph with no _inputs_ provided the graph has no
 *         application-writable tensors.
 *       - Graph I/O Tensors marked optional (type=QNN_TENSOR_TYPE_NULL) cannot be supplied to
 *         QnnGraph_executeAsync(). Clients mark tensors to be of type QNN_TENSOR_TYPE_NULL to
 *         indicate that they must be ignored when constructing a node that lists them as optional.
 *
 * @note If there are simultaneous calls to QnnGraph_execute() and QnnGraph_executeAsync(), the
 *       priority for enqueuing or executing is equal. Both functions will add to the same queue,
 *       the only difference in behavior is whether the function returns when the execution is
 *       enqueued, or when the execution finishes.
 *
 * @return Error code:
 *         - QNN_SUCCESS: the graph was successfully executed
 *         - QNN_GRAPH_ERROR_INVALID_HANDLE: _graph_ is not a valid handle
 *         - QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED: graph was not finalized
 *         - QNN_GRAPH_ERROR_SUBGRAPH: cannot execute a subgraph
 *         - QNN_GRAPH_ERROR_INVALID_ARGUMENT:
 *            - _inputs_ or _outputs_ is NULL or ill-formed OR
 *            - _inputs_ is NOT NULL and _numInputs_ is 0 OR
 *            - _outputs_ is NOT NULL and _numOutputs_ is 0 OR
 *            - _profile_ handle is invalid OR
 *            - continuous graph profiling is enabled and the per-API handle is not NULL.
 *         - QNN_GRAPH_ERROR_INVALID_TENSOR: one or more tensors in _inputs_ or _outputs_
 *           is invalid or not recognized by graph
 *         - QNN_GRAPH_ERROR_UNSUPPORTED_FEATURE: asynchronous graph execution is not supported on
 *           this backend or some API feature is not supported yet, e.g. signal, profile, or batch
 *           multiplier
 *         - QNN_GRAPH_ERROR_SIGNAL_IN_USE: the supplied control signal is already in-use by
 *           another call.
 *         - QNN_GRAPH_ERROR_ABORTED: the call is aborted before completion due to user cancellation
 *         - QNN_GRAPH_ERROR_TIMED_OUT: the call is aborted before completion due to a timeout
 *         - QNN_GRAPH_ERROR_DISABLED: the graph was not enabled when the context was deserialized
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnGraph_executeAsync(Qnn_GraphHandle_t graphHandle,
                                        const Qnn_Tensor_t* inputs,
                                        uint32_t numInputs,
                                        Qnn_Tensor_t* outputs,
                                        uint32_t numOutputs,
                                        Qnn_ProfileHandle_t profileHandle,
                                        Qnn_SignalHandle_t signalHandle,
                                        Qnn_NotifyFn_t notifyFn,
                                        void* notifyParam);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_GRAPH_H
