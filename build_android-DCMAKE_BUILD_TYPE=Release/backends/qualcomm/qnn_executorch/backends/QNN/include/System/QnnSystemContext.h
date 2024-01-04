//==============================================================================
//
// Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  QNN System Context API.
 *
 *          This is a system API header dedicated to extensions to QnnContext
 *          that provide backend-agnostic services to users.
 */

#ifndef QNN_SYSTEM_CONTEXT_H
#define QNN_SYSTEM_CONTEXT_H

#include "QnnDevice.h"
#include "QnnTypes.h"
#include "System/QnnSystemCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Error Codes
//=============================================================================

/**
 * @brief QNN System Context API result / error codes.
 */
typedef enum {
  QNN_SYSTEM_CONTEXT_MINERROR = QNN_MIN_ERROR_SYSTEM,
  //////////////////////////////////////////

  /// Qnn System Context success
  QNN_SYSTEM_CONTEXT_NO_ERROR = QNN_SUCCESS,
  /// There is optional API component that is not supported yet.
  QNN_SYSTEM_CONTEXT_ERROR_UNSUPPORTED_FEATURE = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// QNN System Context invalid handle
  QNN_SYSTEM_CONTEXT_ERROR_INVALID_HANDLE = QNN_SYSTEM_CONTEXT_MINERROR + 0,
  /// One or more arguments to a System Context API is/are NULL/invalid.
  QNN_SYSTEM_CONTEXT_ERROR_INVALID_ARGUMENT = QNN_SYSTEM_CONTEXT_MINERROR + 1,
  /// Generic Failure in achieving the objective of a System Context API
  QNN_SYSTEM_CONTEXT_ERROR_OPERATION_FAILED = QNN_SYSTEM_CONTEXT_MINERROR + 2,

  // Errors related to context caching
  /// Malformed context binary
  QNN_SYSTEM_CONTEXT_ERROR_MALFORMED_BINARY = QNN_SYSTEM_CONTEXT_MINERROR + 10,
  //////////////////////////////////////////
  QNN_SYSTEM_CONTEXT_MAXERROR = QNN_MAX_ERROR_SYSTEM
} QnnSystemContext_Error_t;

/*****************************************************************************/
/* Enums and data structures corresponding to QnnSystemContext               */
/*****************************************************************************/

typedef enum {
  QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1 = 0x01,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_CONTEXT_GRAPH_INFO_UNDEFINED = 0x7FFFFFFF
} QnnSystemContext_GraphInfoVersion_t;

typedef enum {
  QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1 = 0x01,
  QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2 = 0x02,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_CONTEXT_BINARY_INFO_UNDEFINED = 0x7FFFFFFF
} QnnSystemContext_BinaryInfoVersion_t;

//=============================================================================
// Data structures representing context binary metadata contents
//=============================================================================

/**
 * @brief Struct that provides information about graphs registered with a context.
 *        This is version V1 of the structure.
 */
typedef struct {
  /// Name of graph
  const char* graphName;
  /// Number of input tensors to graph
  uint32_t numGraphInputs;
  /// List of input tensors to graph
  Qnn_Tensor_t* graphInputs;
  /// Number of output tensors from graph
  uint32_t numGraphOutputs;
  /// List of output tensors from graph
  Qnn_Tensor_t* graphOutputs;
} QnnSystemContext_GraphInfoV1_t;

// clang-format off
/// QnnSystemContext_GraphInfoV1_t initializer macro
#define QNN_SYSTEM_CONTEXT_GRAPH_INFO_V1_INIT  \
  {                                            \
     NULL,    /* graphName */                  \
     0,       /* numGraphInputs */             \
     NULL,    /* graphInputs */                \
     0,       /* numGraphOutputs */            \
     NULL,    /* graphOutputs */               \
  }
// clang-format on

typedef struct {
  QnnSystemContext_GraphInfoVersion_t version;
  union UNNAMED {
    QnnSystemContext_GraphInfoV1_t graphInfoV1;
  };
} QnnSystemContext_GraphInfo_t;

// clang-format off
/// QnnSystemContext_GraphInfo_t initializer macro
#define QNN_SYSTEM_CONTEXT_GRAPH_INFO_INIT                      \
  {                                                             \
    QNN_SYSTEM_CONTEXT_GRAPH_INFO_UNDEFINED,  /* version */     \
    {                                                           \
      QNN_SYSTEM_CONTEXT_GRAPH_INFO_V1_INIT  /* graphInfoV1 */  \
    }                                                           \
  }
// clang-format on

/**
 * @brief Struct that provides information about contents of a context binary.
 *        This is version V1 of the structure.
 */
typedef struct {
  /// Backend that this context binary is associated with
  uint32_t backendId;
  /// Build ID of QNN SDK used to create context binary
  const char* buildId;
  /// QNN core API version
  Qnn_Version_t coreApiVersion;
  /// Version of backend-specific API for the backend producing context binary
  Qnn_Version_t backendApiVersion;
  /// Version of the SOC for which context binary was generated
  const char* socVersion;
  /// Version of hardware info blob stored in the context binary
  Qnn_Version_t hwInfoBlobVersion;
  /// Version of the opaque context blob generated by backend that is packed into the context binary
  /// Note that the context blob is not part of metadata. It is described by the metadata
  Qnn_Version_t contextBlobVersion;
  /// Size of hardware info blob stored in the context binary, in bytes
  uint32_t hwInfoBlobSize;
  /// Hardware Info blob. Needs to be interpreted based on backend-specific instructions
  void* hwInfoBlob;
  /// Size of opaque backend-specific context blob, in bytes
  uint64_t contextBlobSize;

  // details about graphs stored in context
  /// Number of context tensors
  uint32_t numContextTensors;
  /// List of tensors registered to this context
  Qnn_Tensor_t* contextTensors;
  /// Number of graphs registered with this context
  uint32_t numGraphs;
  /// List of graphs registered to this context
  QnnSystemContext_GraphInfo_t* graphs;
} QnnSystemContext_BinaryInfoV1_t;

// clang-format off
/// QnnSystemContext_BinaryInfoV1_t initializer macro
#define QNN_SYSTEM_CONTEXT_BINARY_INFO_V1_INIT                \
  {                                                           \
    0,                               /* backendId */          \
    NULL,                            /* buildId */            \
    QNN_VERSION_INIT,                /* coreApiVersion */     \
    QNN_VERSION_INIT,                /* backendApiVersion */  \
    NULL,                            /* socVersion */         \
    QNN_VERSION_INIT,                /* hwInfoBlobVersion */  \
    QNN_VERSION_INIT,                /* contextBlobVersion */ \
    0,                               /* hwInfoBlobSize */     \
    NULL,                            /* hwInfoBlob */         \
    0,                               /* contextBlobSize */    \
    0,                               /* numContextTensors */  \
    NULL,                            /* contextTensors */     \
    0,                               /* numGraphs */          \
    NULL,                           /* graphs */              \
  }
// clang-format on

/**
 * @brief Struct that provides information about contents of a context binary.
 *        This is version V2 of the structure.
 */
typedef struct {
  /// Backend that this context binary is associated with
  uint32_t backendId;
  /// Build ID of QNN SDK used to create context binary
  const char* buildId;
  /// QNN core API version
  Qnn_Version_t coreApiVersion;
  /// Version of backend-specific API for the backend producing context binary
  Qnn_Version_t backendApiVersion;
  /// Version of the SOC for which context binary was generated
  const char* socVersion;
  /// Version of hardware info blob stored in the context binary
  Qnn_Version_t hwInfoBlobVersion;
  /// Version of the opaque context blob generated by backend that is packed into the context binary
  /// Note that the context blob is not part of metadata. It is described by the metadata
  Qnn_Version_t contextBlobVersion;
  /// Size of hardware info blob stored in the context binary, in bytes
  uint32_t hwInfoBlobSize;
  /// Hardware Info blob. Needs to be interpreted based on backend-specific instructions
  void* hwInfoBlob;
  /// Size of opaque backend-specific context blob, in bytes
  uint64_t contextBlobSize;

  // details about graphs stored in context
  /// Number of context tensors
  uint32_t numContextTensors;
  /// List of tensors registered to this context
  Qnn_Tensor_t* contextTensors;
  /// Number of graphs registered with this context
  uint32_t numGraphs;
  /// List of graphs registered to this context
  QnnSystemContext_GraphInfo_t* graphs;
  /// Device information associated with the context
  QnnDevice_PlatformInfo_t* platformInfo;
} QnnSystemContext_BinaryInfoV2_t;

// clang-format off
/// QnnSystemContext_BinaryInfoV2_t initializer macro
#define QNN_SYSTEM_CONTEXT_BINARY_INFO_V2_INIT                \
  {                                                           \
    0,                               /* backendId */          \
    NULL,                            /* buildId */            \
    QNN_VERSION_INIT,                /* coreApiVersion */     \
    QNN_VERSION_INIT,                /* backendApiVersion */  \
    NULL,                            /* socVersion */         \
    QNN_VERSION_INIT,                /* hwInfoBlobVersion */  \
    QNN_VERSION_INIT,                /* contextBlobVersion */ \
    0,                               /* hwInfoBlobSize */     \
    NULL,                            /* hwInfoBlob */         \
    0,                               /* contextBlobSize */    \
    0,                               /* numContextTensors */  \
    NULL,                            /* contextTensors */     \
    0,                               /* numGraphs */          \
    NULL,                            /* graphs */             \
    NULL                             /* platformInfo */       \
  }
// clang-format on

typedef struct {
  QnnSystemContext_BinaryInfoVersion_t version;
  union UNNAMED {
    QnnSystemContext_BinaryInfoV1_t contextBinaryInfoV1;
    QnnSystemContext_BinaryInfoV2_t contextBinaryInfoV2;
  };
} QnnSystemContext_BinaryInfo_t;

// clang-format off
/// QnnSystemContext_BinaryInfo_t initializer macro
#define QNN_SYSTEM_CONTEXT_BINARYINFO_INIT                             \
  {                                                                    \
    QNN_SYSTEM_CONTEXT_BINARY_INFO_UNDEFINED, /* version */            \
    {                                                                  \
      QNN_SYSTEM_CONTEXT_BINARY_INFO_V1_INIT /* contextBinaryInfoV1 */ \
    }                                                                  \
  }
// clang-format on

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief A typedef to indicate a QNN System context handle
 */
typedef void* QnnSystemContext_Handle_t;

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief A function to create an instance of the QNN system context
 *
 * @param[out] sysCtxHandle A handle to the created instance of a systemContext entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemContext entity
 *         - QNN_SYSTEM_CONTEXT_ERROR_INVALID_ARGUMENT: sysCtxHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemContext instance
 *         - QNN_SYSTEM_CONTEXT_ERROR_UNSUPPORTED_FEATURE: system context features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemContext_create(QnnSystemContext_Handle_t* sysCtxHandle);

/**
 * @brief A function to get context info from the serialized binary buffer.
 *
 * @deprecated Use QnnSystemContext_getMetadata instead
 *
 * @param[in]  sysCtxHandle     Handle to the systemContext object
 *
 * @param[in]  binaryBuffer     Serialized buffer representing a context binary.
 *
 * @param[in]  binaryBufferSize Size of context binary in bytes
 *
 * @param[out] binaryInfo       Pointer to memory that will be populated with
 *                              user-visible information about the context binary.
 *                              Memory for this information is internally allocated
 *                              and managed by QNN, and is associated with the
 *                              handle _sysCtxHandle_ created with QnnSystemContext_create().
 *                              This memory has to be released by calling
 *                              QnnSystemContext_free() when it is no longer needed.
 *
 * @param[out] binaryInfoSize   Size of metadata describing the contents
 *                              of the context binary, in bytes.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully returned context binary info to caller
 *         - QNN_SYSTEM_CONTEXT_ERROR_INVALID_HANDLE: Invalid System Context handle
 *         - QNN_SYSTEM_CONTEXT_ERROR_INVALID_ARGUMENT: One or more arguments to the API
 *           is/are NULL/invalid.
 *         - QNN_SYSTEM_CONTEXT_ERROR_OPERATION_FAILED: Failed to obtain context binary info
 *         - QNN_SYSTEM_CONTEXT_ERROR_MALFORMED_BINARY: The binary is either malformed or
 *           cannot be parsed successfully.
 *         - QNN_SYSTEM_CONTEXT_ERROR_UNSUPPORTED_FEATURE: not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemContext_getBinaryInfo(QnnSystemContext_Handle_t sysCtxHandle,
                                                 void* binaryBuffer,
                                                 uint64_t binaryBufferSize,
                                                 const QnnSystemContext_BinaryInfo_t** binaryInfo,
                                                 Qnn_ContextBinarySize_t* binaryInfoSize);

/**
 * @brief A function to get meta data from the serialized binary buffer.
 *
 * @param[in]  sysCtxHandle     Handle to the systemContext object
 *
 * @param[in]  binaryBuffer     Serialized buffer representing a const context binary.
 *
 * @param[in]  binaryBufferSize Size of context binary in bytes
 *
 * @param[out] binaryInfo       Pointer to memory that will be populated with
 *                              user-visible information about the context binary.
 *                              Memory for this information is internally allocated
 *                              and managed by QNN, and is associated with the
 *                              handle _sysCtxHandle_ created with QnnSystemContext_create().
 *                              This memory has to be released by calling
 *                              QnnSystemContext_free() when it is no longer needed.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully returned context binary info to caller
 *         - QNN_SYSTEM_CONTEXT_ERROR_INVALID_HANDLE: Invalid System Context handle
 *         - QNN_SYSTEM_CONTEXT_ERROR_INVALID_ARGUMENT: One or more arguments to the API
 *           is/are NULL/invalid.
 *         - QNN_SYSTEM_CONTEXT_ERROR_OPERATION_FAILED: Failed to obtain context binary info
 *         - QNN_SYSTEM_CONTEXT_ERROR_MALFORMED_BINARY: The binary is either malformed or
 *           cannot be parsed successfully.
 *         - QNN_SYSTEM_CONTEXT_ERROR_UNSUPPORTED_FEATURE: not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemContext_getMetadata(QnnSystemContext_Handle_t sysCtxHandle,
                                               const void* binaryBuffer,
                                               Qnn_ContextBinarySize_t binaryBufferSize,
                                               const QnnSystemContext_BinaryInfo_t** binaryInfo);

/**
 * @brief A function to free the instance of the System Context object.
 *        This API clears any intermediate memory allocated and associated
 *        with a valid handle.
 *
 * @param[in] sysCtxHandle Handle to the System Context object
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully freed instance of System Context
 *         - QNN_SYSTEM_CONTEXT_ERROR_INVALID_HANDLE: Invalid System Context handle to free
 *         - QNN_SYSTEM_CONTEXT_ERROR_UNSUPPORTED_FEATURE: not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemContext_free(QnnSystemContext_Handle_t sysCtxHandle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_SYSTEM_CONTEXT_H
