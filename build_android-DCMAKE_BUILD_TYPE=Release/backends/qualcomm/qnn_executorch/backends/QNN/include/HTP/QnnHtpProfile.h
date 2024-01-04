//==============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief QNN HTP Profile component API.
 *
 *          Requires HTP backend to be initialized.
 *          Should be used with the QnnProfile API but has HTP backend
 *          specific definition for different QnnProfile data structures
 *
 */

#ifndef QNN_HTP_PROFILE_H
#define QNN_HTP_PROFILE_H

#include "QnnProfile.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================
/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the ARM processor
 *        when client invokes QnnContext_createFromBinary. The value
 *        returned is time in microseconds.
 *
 * @note context load binary host rpc time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_CONTEXT_LOAD_BIN_HOST_RPC_TIME_MICROSEC 1002

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the HTP processor
 *        when client invokes QnnContext_createFromBinary. The value
 *        returned is time in microseconds.
 *
 * @note context load binary htp rpc time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_CONTEXT_LOAD_BIN_HTP_RPC_TIME_MICROSEC 1003

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the time taken to create the context on the
 *        accelerator when client invokes QnnContext_createFromBinary.
 *        The value returned is time in microseconds.
 *
 * @note context load binary accelerator time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_CONTEXT_LOAD_BIN_ACCEL_TIME_MICROSEC 1004

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the ARM processor
 *        when client invokes QnnGraph_finalize.
 *        The value returned is time in microseconds.
 *
 * @note graph finalize host rpc time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_FINALIZE_HOST_RPC_TIME_MICROSEC 2001

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the HTP processor
 *        when client invokes QnnGraph_finalize.
 *        The value returned is time in microseconds.
 *
 * @note graph finalize htp rpc time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_FINALIZE_HTP_RPC_TIME_MICROSEC 2002

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to finalize the graph on the accelerator
 *        when client invokes QnnGraph_finalize.
 *        The value returned is time in microseconds.
 *
 * @note graph finalize accelerator time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_FINALIZE_ACCEL_TIME_MICROSEC 2003



/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the ARM processor
 *        when client invokes QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is time in microseconds.
 *
 * @note graph execute host rpc time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HOST_RPC_TIME_MICROSEC 3001

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the HTP processor
 *        when client invokes QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is time in microseconds.
 *
 * @note graph execute htp rpc time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HTP_RPC_TIME_MICROSEC 3002

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to execute the graph on the accelerator
 *        when client invokes QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is number of processor cycles taken.
 *
 * @note graph execute accelerator time maybe available only on
 *       QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note When QNN_PROFILE_LEVEL_DETAILED is used, this event can have
 *       multiple sub-events of type QNN_PROFILE_EVENTTYPE_NODE.
 *       There will be a sub-event for each node that was added to the graph
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE 3003

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to execute the graph on the accelerator
 *        when client invokes QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is time taken in microseconds
 *
 * @note graph execute accelerator time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note When QNN_PROFILE_LEVEL_DETAILED is used, this event can have
 *       multiple sub-events of type QNN_PROFILE_EVENTTYPE_NODE / QNN_PROFILE_EVENTUNIT_MICROSEC
 *       There will be a sub-event for each node that was added to the graph
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC 3004

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to time taken for miscellaneous work i.e. time
 *        that cannot be attributed to a node but are still needed to
 *        execute the graph on the accelerator. This occurs when client invokes
 *        QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is time taken in microseconds
 *
 * @note graph execute misc accelerator time is available only on
 *       QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_MISC_ACCEL_TIME_MICROSEC 3005

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to time taken for a graph yield instance to
 *        release all its resources to the other graph.
 *        The value returned is time taken in microseconds.
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_YIELD_INSTANCE_RELEASE_TIME 3006

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to time a graph spends waiting for a higher
 *        priority graph to finish execution.
 *        The value returned is time taken in microseconds
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_YIELD_INSTANCE_WAIT_TIME 3007

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to time a graph spends re-acquiring resources
 *        and restoring vtcm.
 *        The value returned is time taken in microseconds
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_YIELD_INSTANCE_RESTORE_TIME 3008

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the number of times that a yield occured
 *        during execution
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_YIELD_COUNT 3009

/**
 * @brief QnnProfile_EventType_t definition for time a graph waits to get
 *        VTCM. This should be constant UNLESS we need another graph to yield.
 *        The value returned is time taken in microseconds.
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_VTCM_ACQUIRE_TIME 3010

/**
 * @brief QnnProfile_EventType_t definition for time a graph waits to get
 *        HMX + HVX, and turn them all on.
 *        The value returned is time taken in microseconds.
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_RESOURCE_POWER_UP_TIME 3011

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the ARM processor
 *        when client invokes QnnContext_free which in consequence deinit graph.
 *        The value returned is time in microseconds.
 *
 * @note graph deinit host rpc time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_DEINIT_HOST_RPC_TIME_MICROSEC 4001

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the remote procedure call on the HTP processor
 *        when client invokes QnnContext_free which in consequence deinit graph.
 *        The value returned is time in microseconds.
 *
 * @note graph deinit htp rpc time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_DEINIT_HTP_RPC_TIME_MICROSEC 4002

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to the time taken to deinit graph on the
 *        accelerator when client invokes QnnContext_free which in consequence
 *        deinit graph. The value returned is time in microseconds.
 *
 * @note graph deinit accelerator time maybe available on both
 *       QNN_PROFILE_LEVEL_BASIC and QNN_PROFILE_LEVEL_DETAILED levels
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_DEINIT_ACCEL_TIME_MICROSEC 4003

/**
 * @brief QnnProfile_EventType_t definition to get data related to execution of
 *        an operation. This value represents the amount of time an op spends
 *        waiting for execution on the main thread since the last op on the main
 *        thread due to scheduling and can be interpreted appropriately in
 *        conjunction with the unit.
 *
 * @note node wait information is available on QNN_HTP_PROFILE_LEVEL_LINTING level
 */
#define QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT 5001

/**
 * @brief QnnProfile_EventType_t definition to get data related to execution of
 *        an operation. This value represents the amount of time at least one
 *        background op is running during the execution of an op on the main thread
 *        and can be interpreted appropriately in conjunction with the unit.
 *
 * @note node overlap information is available on QNN_HTP_PROFILE_LEVEL_LINTING level
 */
#define QNN_HTP_PROFILE_EVENTTYPE_NODE_OVERLAP 5002

/**
 * @brief QnnProfile_EventType_t definition to get data related to execution of
 *        an operation. This value represents the amount of time at least one
 *        background op that is not being waited upon to finish is running during
 *        the wait period of an op on the main thread and can be interpreted
 *        appropriately in conjunction with the unit.
 *
 * @note node wait overlap information is available on QNN_HTP_PROFILE_LEVEL_LINTING
 *       level
 */
#define QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT_OVERLAP 5003

/**
 * @brief QnnProfile_EventType_t definition to get data related to execution of
 *        an operation. This value represents a bitmask denoting the resources
 *        an op uses.
 *
 * @note node specific information is available on QNN_HTP_PROFILE_LEVEL_LINTING level
 */
#define QNN_HTP_PROFILE_EVENTTYPE_NODE_RESOURCEMASK 5004

/**
 * @brief QnnProfile_EventType_t definition to get data related to execution of
 *        an operation. This value represents the ID of an op running in parallel to
 *        an op running on the main thread or on HMX.
 *
 * @note node specific information is available on QNN_HTP_PROFILE_LEVEL_LINTING level
 */
#define QNN_HTP_PROFILE_EVENTTYPE_NODE_CRITICAL_BG_OP_ID 5005

/**
 * @brief QnnProfile_EventType_t definition to get data related to execution of
 *        an operation. This value represents the ID of an op running on threads other
 *        than the main or the HMX thread when the main and the HMX threads are not
 *        executing any op.
 *
 * @note node specific information is available on QNN_HTP_PROFILE_LEVEL_LINTING level
 */
#define QNN_HTP_PROFILE_EVENTTYPE_NODE_WAIT_BG_OP_ID 5006

/**
 * @brief QnnProfile_EventType_t definition to get profile information
 *        that corresponds to execute the graph's critical path on the accelerator
 *        when client invokes QnnGraph_execute or QnnGraph_executeAsync.
 *        The value returned is number of processor cycles taken.
 *
 * @note graph execute accelerator time maybe available only on
 *       QNN_HTP_PROFILE_LEVEL_LINTING levels
 *
 * @note When QNN_HTP_PROFILE_LEVEL_LINTING is used, this event can have
 *       multiple sub-events of type QNN_PROFILE_EVENTTYPE_NODE.
 *       There will be a sub-event for each node that was added to the graph
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_CRITICAL_ACCEL_TIME_CYCLE 6001

/**
 * @brief Linting QnnProfile_Level_t definition that allows collecting in-depth
 *        performance metrics for each op in the graph including main thread
 *        execution time and time spent on parallel background ops.
 */
#define QNN_HTP_PROFILE_LEVEL_LINTING 7001

/**
 * @brief QnnProfile_EventType_t definition to get number of HVX threads
 *        configured by a graph. Different graphs can have a different
 *        value.
 */
#define QNN_HTP_PROFILE_EVENTTYPE_GRAPH_NUMBER_OF_HVX_THREADS 8001

#ifdef __cplusplus
}
#endif

#endif  // QNN_HTP_PROFILE_H
