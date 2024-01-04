//==============================================================================
//
// Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  A header which contains the base types required by the API.
 *          Strings are expected to be UTF-8 encoded and NULL terminated.
 */

#ifndef QNN_TYPES_H
#define QNN_TYPES_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#include "QnnCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief An enum which defines various data types.
 *        FIXED_XX types are targeted for data in tensors.
 *        UINT / INT types are targeted for scalar params.
 *        4-bit types are tightly packed into one byte in
 *        little endian format.
 */
typedef enum {
  // Signed Int: 0x00XX

  /// 8-bit integer type
  QNN_DATATYPE_INT_8 = 0x0008,
  /// 16-bit integer type
  QNN_DATATYPE_INT_16 = 0x0016,
  /// 32-bit integer type
  QNN_DATATYPE_INT_32 = 0x0032,
  /// 64-bit integer type
  QNN_DATATYPE_INT_64 = 0x0064,

  // Unsigned Int: 0x01XX
  QNN_DATATYPE_UINT_8  = 0x0108,
  QNN_DATATYPE_UINT_16 = 0x0116,
  QNN_DATATYPE_UINT_32 = 0x0132,
  QNN_DATATYPE_UINT_64 = 0x0164,

  // Float: 0x02XX
  QNN_DATATYPE_FLOAT_16 = 0x0216,
  QNN_DATATYPE_FLOAT_32 = 0x0232,
  QNN_DATATYPE_FLOAT_64 = 0x0264,

  // Signed Fixed Point: 0x03XX
  QNN_DATATYPE_SFIXED_POINT_4  = 0x0304,
  QNN_DATATYPE_SFIXED_POINT_8  = 0x0308,
  QNN_DATATYPE_SFIXED_POINT_16 = 0x0316,
  QNN_DATATYPE_SFIXED_POINT_32 = 0x0332,

  // Unsigned Fixed Point: 0x04XX
  QNN_DATATYPE_UFIXED_POINT_4  = 0x0404,
  QNN_DATATYPE_UFIXED_POINT_8  = 0x0408,
  QNN_DATATYPE_UFIXED_POINT_16 = 0x0416,
  QNN_DATATYPE_UFIXED_POINT_32 = 0x0432,

  // Bool: 0x05XX
  /// 8-bit boolean type, 0 = false, any non-zero value = true
  QNN_DATATYPE_BOOL_8 = 0x0508,

  // String: 0x06xx
  QNN_DATATYPE_STRING = 0x0608,

  // Unused, present to ensure 32 bits.
  QNN_DATATYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_DataType_t;

/**
 * @brief An enum which defines the different precision modes supported by QNN backends.
 *        A precision mode may be used to express the math type used in the implementation
 *        of an operation.
 */
typedef enum {
  // FLOATING POINT REPRESENTATIONS

  /// 32-bit Floating point precision. The format of the floating point
  /// value is left to backends to choose.
  QNN_PRECISION_FLOAT32 = 0,
  /// 16-bit Floating point precision. The format of the floating point
  /// value is left to backends to choose.
  QNN_PRECISION_FLOAT16 = 1,

  // Unused, present to ensure 32 bits.
  QNN_PRECISION_UNDEFINED = 0x7FFFFFFF
} Qnn_Precision_t;

/**
 * @brief An enum to specify the tensor type, application accessible or native to QNN
 *
 */
typedef enum {
  /// Client application writeable tensor.
  QNN_TENSOR_TYPE_APP_WRITE = 0,
  /// Client application readable tensor.
  QNN_TENSOR_TYPE_APP_READ = 1,
  /// Tensor that can both be read and written by an application. Used in scenarios that may include
  /// supplying an output tensor from one graph as the input to another graph.
  QNN_TENSOR_TYPE_APP_READWRITE = 2,
  /// Tensor native to a graph which may be optimized by a backend and are not accessible by a
  /// client.
  QNN_TENSOR_TYPE_NATIVE = 3,
  /// Static data which doesn't change during execution and may be optimized by a backend.
  QNN_TENSOR_TYPE_STATIC = 4,
  /// Tensor type NULL which can be used to represent optional tensors. Other Qnn_Tensor_t metadata
  /// is ignored.
  QNN_TENSOR_TYPE_NULL = 5,
  // Unused, present to ensure 32 bits.
  QNN_TENSOR_TYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorType_t;

/**
 * @brief An enum to specify the parameter type : Scalar or Tensor
 */
typedef enum {
  QNN_PARAMTYPE_SCALAR = 0,
  QNN_PARAMTYPE_TENSOR = 1,
  // Unused, present to ensure 32 bits.
  QNN_PARAMTYPE_UNDEFINED = 0xFFFFFFFF
} Qnn_ParamType_t;

/**
 * @brief An enum to specify definition source for field(s) following this enum
 */
typedef enum {
  /// Indicates backend implementation to update or decide
  QNN_DEFINITION_IMPL_GENERATED = 0,
  /// Indicates that provided definition needs to be used
  QNN_DEFINITION_DEFINED = 1,
  // Unused, present to ensure 32 bits.
  QNN_DEFINITION_UNDEFINED = 0x7FFFFFFF
} Qnn_Definition_t;

/**
 * @brief An enum to specify a priority.
 */
typedef enum {
  /// QNN_PRIORITY_LOW is always available for use.
  QNN_PRIORITY_LOW = 0,
  /// QNN_PRIORITY_NORMAL is always available for use.
  QNN_PRIORITY_NORMAL  = 100,
  QNN_PRIORITY_DEFAULT = QNN_PRIORITY_NORMAL,
  /// QNN_PRIORITY_NORMAL_HIGH usage may be restricted and would silently be treated as
  /// QNN_PRIORITY_NORMAL
  QNN_PRIORITY_NORMAL_HIGH = 150,
  /// QNN_PRIORITY_HIGH usage may be restricted and would silently be treated as
  /// QNN_PRIORITY_NORMAL
  QNN_PRIORITY_HIGH = 200,
  // Unused, present to ensure 32 bits.
  QNN_PRIORITY_UNDEFINED = 0x7FFFFFFF
} Qnn_Priority_t;

/**
 * @brief A typedef to indicate context binary size.
 */
typedef uint64_t Qnn_ContextBinarySize_t;

/**
 * @brief An enum to describe reporting levels for the error handling API
 * QNN_ERROR_REPORTING_LEVEL_BRIEF: get basic information about an error
 * QNN_ERROR_REPORTING_LEVEL_DETAILED: get detailed information about an error
 * in memory-based object forms
 */
typedef enum {
  QNN_ERROR_REPORTING_LEVEL_BRIEF    = 0,
  QNN_ERROR_REPORTING_LEVEL_DETAILED = 1,
  // Unused, present to ensure 32 bits.
  QNN_ERROR_REPORTING_LEVEL_UNDEFINED = 0x7FFFFFFF
} Qnn_ErrorReportingLevel_t;

/**
 * @brief A typedef describing error reporting configuration
 */
typedef struct {
  /// Error reporting level
  Qnn_ErrorReportingLevel_t reportingLevel;
  /// Amount of memory to be reserved for error information. Specified in KB
  uint32_t storageLimit;
} Qnn_ErrorReportingConfig_t;

// clang-format off
/// Qnn_ErrorReportingConfig_t initializer macro
#define QNN_ERROR_REPORTING_CONFIG_INIT                     \
  {                                                         \
    QNN_ERROR_REPORTING_LEVEL_UNDEFINED, /*reportingLevel*/ \
    0u                                   /*storageLimit*/   \
  }
// clang-format on

/**
 * @brief A struct which is used to provide a version number using 3 values:
 * major, minor, patch
 */
typedef struct {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
} Qnn_Version_t;

// clang-format off
/// Qnn_Version_t initializer macro
#define QNN_VERSION_INIT \
  {                      \
    0u,    /*major*/     \
    0u,    /*minor*/     \
    0u     /*patch*/     \
  }
// clang-format on

/**
 * @brief A struct used to provide the versions of both the core QNN API
 * and any Backend Specific API
 */
typedef struct {
  /// Version of the QNN core API common to all backends
  Qnn_Version_t coreApiVersion;
  /// Version of the backend-specific API
  Qnn_Version_t backendApiVersion;
} Qnn_ApiVersion_t;

/// Qnn_ApiVersion_t initializer macro
#define QNN_API_VERSION_INIT                            \
  {                                                     \
    {                                                   \
        QNN_API_VERSION_MAJOR, /*coreApiVersion.major*/ \
        QNN_API_VERSION_MINOR, /*coreApiVersion.minor*/ \
        QNN_API_VERSION_PATCH  /*coreApiVersion.patch*/ \
    },                                                  \
        QNN_VERSION_INIT /*backendApiVersion*/          \
  }

/**
 * @brief A value representing an immutable value which configures a node.
 */
typedef struct {
  Qnn_DataType_t dataType;
  union UNNAMED {
    float floatValue;
    double doubleValue;
    uint64_t uint64Value;
    int64_t int64Value;
    uint32_t uint32Value;
    int32_t int32Value;
    uint16_t uint16Value;
    int16_t int16Value;
    uint8_t uint8Value;
    int8_t int8Value;
    uint8_t bool8Value;
    const char* stringValue;
  };
} Qnn_Scalar_t;

/// Qnn_Scalar_t initializer macro
#define QNN_SCALAR_INIT                  \
  {                                      \
    QNN_DATATYPE_UNDEFINED, /*dataType*/ \
    {                                    \
      0.0f /*floatValue*/                \
    }                                    \
  }

/**
 * @brief An enum to specify quantization encoding type structure
 *
 */
typedef enum {
  /// Indicates Qnn_ScaleOffset_t encoding type
  QNN_QUANTIZATION_ENCODING_SCALE_OFFSET = 0,
  /// Indicates Qnn_AxisScaleOffset_t encoding type
  QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET = 1,
  /// Indicates Qnn_BwScaleOffset_t encoding type
  QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET = 2,
  /// Indicates Qnn_BwAxisScaleOffset_t encoding type
  QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET = 3,
  // Unused, present to ensure 32 bits.
  QNN_QUANTIZATION_ENCODING_UNDEFINED = 0x7FFFFFFF
} Qnn_QuantizationEncoding_t;

/**
 * @brief A struct to express quantization parameters as a positive scale with a zero offset.
 *
 * float_value = (quantized_value + offset) * scale
 */
typedef struct {
  /// scale must be strictly positive
  float scale;
  int32_t offset;
} Qnn_ScaleOffset_t;

// clang-format off
/// Qnn_ScaleOffset_t initializer macro
#define QNN_SCALE_OFFSET_INIT \
  {                           \
    0.0f, /*scale*/           \
    0     /*offset*/          \
  }
// clang-format on

/**
 * @brief A struct to express quantization parameters as a positive scale with a zero offset and a
 * bitwidth.
 *
 * float_value = (quantized_value + offset) * scale
 *
 * bitwidth must be > 0, and is used to express the true number of bits used to quantize the value,
 * which may be different from the bitwidth of the tensor indicated by its data type. For example:
 * the quantization encoding for a tensor of type QNN_DATATYPE_UFIXED_POINT_8 that is quantized to
 * 4-bit precision may be expressed by setting bitwidth = 4. In such circumstances, data quantized
 * to a lower precision will still occupy the full extent of bits allotted to the tensor as per its
 * data type in unpacked form.
 *
 * The datatype used must be the smallest type which can accommodate the bitwidth. For example: a
 * tensor quantized to 4-bit precision must use an 8-bit datatype, 16-bit or larger datatypes are
 * not permitted.
 *
 * Tensor elements are expected to occupy the least significant bits of the total size alloted to
 * the datatype, and all bits above the specified bitwidth will be ignored. For example: an 8-bit
 * datatype tensor quantized to 4-bit precision will be interpreted as a 4-bit value contained in
 * the lower 4 bits of each element, and the upper 4 bits will be ignored. For signed datatypes, the
 * value will be interpreted as a two's complement integer where the signed bit is the most
 * significant bit permitted by the specified bitwidth. For example: -3 would be represented as
 * 0b11111101 as a signed 8-bit integer, but can also be represented as 0b00001101 as a signed 4-bit
 * integer stored in an 8-bit container. Either of these representations are valid to express -3 as
 * a 4-bit signed integer in an 8-bit container, and will be treated identically because the upper 4
 * bits will be ignored.
 */
typedef struct {
  /// bitwidth must be <= number of bits specified by data type of tensor
  uint32_t bitwidth;
  /// scale must be strictly positive
  float scale;
  int32_t offset;
} Qnn_BwScaleOffset_t;

// clang-format off
/// Qnn_BwScaleOffset_t initializer macro
#define QNN_BW_SCALE_OFFSET_INIT \
  {                              \
    0u,   /*bitwidth*/           \
    0.0f, /*scale*/              \
    0     /*offset*/             \
  }
// clang-format on

/**
 * @brief A struct to express per-axis quantization parameters as a scale with a zero offset
 */
typedef struct {
  int32_t axis;
  uint32_t numScaleOffsets;
  Qnn_ScaleOffset_t* scaleOffset;
} Qnn_AxisScaleOffset_t;

// clang-format off
/// Qnn_AxisScaleOffset_t initializer macro
#define QNN_AXIS_SCALE_OFFSET_INIT \
  {                                \
    0,       /*axis*/              \
    0u,      /*numScaleOffsets*/   \
    NULL     /*scaleOffset*/       \
  }                                \
// clang-format on

/**
 * @brief A struct to express per-axis quantization parameters as collection of scales, offsets
 * and bitwidth.
 *
 * bitwidth must be > 0 and applies commonly to all axes. It is used to express the true number of
 * bits used to quantize the value, which may be different from the bitwidth of the tensor indicated
 * by its data type. For example: the quantization encoding for a tensor of type
 * QNN_DATATYPE_UFIXED_POINT_8 that is quantized to 4-bit precision may be expressed by setting
 * bitwidth = 4. In such circumstances, data quantized to a lower precision will still occupy the
 * full extent of bits allotted to the tensor as per its data type in unpacked form.
 *
 * The datatype used must be the smallest type which can accommodate the bitwidth. For example: a
 * tensor quantized to 4-bit precision must use an 8-bit datatype, 16-bit or larger datatypes are
 * not permitted.
 *
 * Tensor elements are expected to occupy the least significant bits of the total size alloted to
 * the datatype, and all bits above the specified bitwidth will be ignored. For example: an 8-bit
 * datatype tensor quantized to 4-bit precision will be interpreted as a 4-bit value contained in
 * the lower 4 bits of each element, and the upper 4 bits will be ignored. For signed datatypes, the
 * value will be interpreted as a two's complement integer where the signed bit is the most
 * significant bit permitted by the specified bitwidth. For example: -3 would be represented as
 * 0b11111101 as a signed 8-bit integer, but can also be represented as 0b00001101 as a signed 4-bit
 * integer stored in an 8-bit container. Either of these representations are valid to express -3 as
 * a 4-bit signed integer in an 8-bit container, and will be treated identically because the upper 4
 * bits will be ignored.
 */
typedef struct {
  /// bitwidth must be <= number of bits specified by data type of tensor
  uint32_t bitwidth;
  int32_t axis;
  /// numElements applies to both scales and offsets and they are supposed to be a one-to-one match
  uint32_t numElements;
  /// scales must be strictly positive
  float* scales;
  /// offsets must match scales in their dimension except when it can be NULL to indicate that the
  /// value is symmetrically quantized and hence, offset = 0
  int32_t* offsets;
} Qnn_BwAxisScaleOffset_t;

// clang-format off
/// Qnn_BwAxisScaleOffset_t initializer macro
#define QNN_BW_AXIS_SCALE_OFFSET_INIT \
  {                                   \
    0u,      /*bitwidth*/             \
    0,       /*axis*/                 \
    0u,      /*numElements*/          \
    NULL,    /*scales*/               \
    NULL     /*offsets*/              \
  }
// clang-format on

/**
 * @brief A struct which defines the quantization parameters, and union of supported quantization
 * encoding structs.
 */
typedef struct {
  Qnn_Definition_t encodingDefinition;
  /// Quantization encoding type identifying quantization encoding structure to use
  Qnn_QuantizationEncoding_t quantizationEncoding;
  union UNNAMED {
    Qnn_ScaleOffset_t scaleOffsetEncoding;
    Qnn_AxisScaleOffset_t axisScaleOffsetEncoding;
    Qnn_BwScaleOffset_t bwScaleOffsetEncoding;
    Qnn_BwAxisScaleOffset_t bwAxisScaleOffsetEncoding;
  };
} Qnn_QuantizeParams_t;

// clang-format off
/// Qnn_QuantizeParams_t initializer macro
#define QNN_QUANTIZE_PARAMS_INIT                                      \
  {                                                                   \
    QNN_DEFINITION_UNDEFINED,                /*encodingDefinition*/   \
    QNN_QUANTIZATION_ENCODING_UNDEFINED,     /*quantizationEncoding*/ \
    {                                                                 \
      QNN_SCALE_OFFSET_INIT /*scaleOffsetEncoding*/                   \
    }                                                                 \
  }
// clang-format on

/**
 * @brief An n-dimensional tensor formatted in memory as flat buffer where the last dimension varies
 *        the fastest
 */
#define QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER 0

/**
 * @brief Implementation-defined data format identifier for tensors.
 *        Legal values and semantics are defined by QNN backends, the default format
 *        QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER is supported by all backends
 * @note  Data format for intermediate tensors, i.e ones of type QNN_TENSOR_TYPE_NATIVE
 *        may not be honored by a backend, because it can choose to pick a data format that is
 *        more conducive for its execution.
 */
typedef uint32_t Qnn_TensorDataFormat_t;

/**
 * @brief An enum specifying memory types of tensor data
 */
typedef enum {
  /// Raw memory pointer
  QNN_TENSORMEMTYPE_RAW = 0,
  /// Memory object, provide capability for memory sharing in between QNN accelerator backends.
  QNN_TENSORMEMTYPE_MEMHANDLE = 1,
  // Unused, present to ensure 32 bits.
  QNN_TENSORMEMTYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorMemType_t;

/**
 * @brief A struct which defines a memory buffer
 *
 */
typedef struct {
  /// app-accessible data pointer, provided by app.
  void* data;
  /// size of buffer, in bytes, pointed to by data.
  uint32_t dataSize;
} Qnn_ClientBuffer_t;

// clang-format off
/// Qnn_ClientBuffer_t initializer macro
#define QNN_CLIENT_BUFFER_INIT \
  {                            \
    NULL, /*data*/             \
    0u    /*dataSize*/         \
  }
// clang-format on

/**
 * @brief A struct which defines an opaque object
 *
 */
typedef struct {
  /// Data pointer to the opaque object
  void* data;
  /// Size of buffer, in bytes, pointed to by data
  uint64_t len;
} Qnn_OpaqueObject_t;

// clang-format off
/// Qnn_OpaqueObject_t initializer macro
#define QNN_OPAQUE_OBJECT_INIT \
  {                            \
    NULL, /*data*/             \
    0u    /*len*/              \
  }
// clang-format on

/**
 * @brief A struct which describes the properties of a V1 version of tensor.
 *
 */
typedef struct {
  /// Integer identifier for a tensor.
  uint32_t id;
  /// Tensor name.
  const char* name;
  /// Tensor type.
  Qnn_TensorType_t type;
  /// Tensor data formatting in memory (refer to definition type for info).
  Qnn_TensorDataFormat_t dataFormat;
  /// Tensor data type.
  Qnn_DataType_t dataType;
  /// Tensor quantization params.
  Qnn_QuantizeParams_t quantizeParams;
  /// Tensor rank.
  uint32_t rank;
  /// Tensor dimension array of length _rank_. For detailed behavior of dimensions field with
  /// various APIs, refer SDK documentation. Must be NULL when rank is 0.
  uint32_t* dimensions;
  /// Tensor memory type.
  Qnn_TensorMemType_t memType;
  /// Actual data contained in the tensor.
  union UNNAMED {
    /// Tensor data provided by client as a pointer to raw memory (see QNN_TENSORMEMTYPE_RAW).
    Qnn_ClientBuffer_t clientBuf;
    /// Tensor data shared via a memory handle (see QNN_TENSORMEMTYPE_MEMHANDLE).
    Qnn_MemHandle_t memHandle;
  };
} Qnn_TensorV1_t;

// clang-format off
/// Qnn_TensorV1_t initializer macro
#define QNN_TENSOR_V1_INIT                                        \
  {                                                               \
    0u,                                     /*id*/                \
    NULL,                                   /*name*/              \
    QNN_TENSOR_TYPE_UNDEFINED,              /*type*/              \
    QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,     /*dataFormat*/        \
    QNN_DATATYPE_UNDEFINED,                 /*dataType*/          \
    QNN_QUANTIZE_PARAMS_INIT,               /*quantizeParams*/    \
    0u,                                     /*rank*/              \
    NULL,                                   /*dimensions*/        \
    QNN_TENSORMEMTYPE_UNDEFINED,            /*memType*/           \
    {                                                             \
      QNN_CLIENT_BUFFER_INIT                /*clientBuf*/         \
    }                                                             \
  }
// clang-format on

/**
 * @brief Enum to distinguish various tensor versions
 */
typedef enum {
  /// Enum to choose usage of Qnn_TensorV1_t in Qnn_Tensor_t
  QNN_TENSOR_VERSION_1 = 1,
  // Unused, present to ensure 32 bits.
  QNN_TENSOR_VERSION_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorVersion_t;

/**
 * @brief A struct which provides various versions of a tensor
 */
typedef struct {
  /// Version of the QNN tensor
  Qnn_TensorVersion_t version;
  union UNNAMED {
    /// Tensor version 1 (see QNN_TENSOR_VERSION_1)
    Qnn_TensorV1_t v1;
  };
} Qnn_Tensor_t;

/// Qnn_Tensor_t initializer macro
#define QNN_TENSOR_INIT               \
  {                                   \
    QNN_TENSOR_VERSION_1, /*version*/ \
    {                                 \
      QNN_TENSOR_V1_INIT /*v1*/       \
    }                                 \
  }

/**
 * @brief A struct which defines a named scalar or tensor parameter.
 *
 */
typedef struct {
  /// Parameter type: scalar or tensor
  Qnn_ParamType_t paramType;
  /// Name of the parameter
  const char* name;

  union UNNAMED {
    /// Scalar parameter specification
    Qnn_Scalar_t scalarParam;
    /// Tensor parameter specification; tensors referred to must be STATIC.
    Qnn_Tensor_t tensorParam;
  };
} Qnn_Param_t;

// clang-format off
/// Qnn_Param_t initializer macro
#define QNN_PARAM_INIT                     \
  {                                        \
    QNN_PARAMTYPE_UNDEFINED, /*paramType*/ \
    NULL,                    /*name*/      \
    {                                      \
      QNN_SCALAR_INIT /*scalarParam*/      \
    }                                      \
  }
// clang-format on

/**
 * @brief This struct defines the configuration for a single operation.
 */
typedef struct {
  /// A human-readable name for the operation instance.
  const char* name;
  /// The name of the operation package to which this operation's type belongs.
  const char* packageName;
  /// The name of operation type (e.g. Conv2D).
  const char* typeName;
  /// The number of static parameters provided in the params array.
  uint32_t numOfParams;
  /// Array of operation parameters.
  Qnn_Param_t* params;
  /// The number of input tensors.
  uint32_t numOfInputs;
  /// Array of input tensors.
  Qnn_Tensor_t* inputTensors;
  /// The number of output tensors.
  uint32_t numOfOutputs;
  /// Array of output tensors.
  Qnn_Tensor_t* outputTensors;
} Qnn_OpConfigV1_t;

// clang-format off
/// Qnn_OpConfig_t initializer macro
#define QNN_OPCONFIG_V1_INIT    \
  {                             \
    NULL,     /*name*/          \
    NULL,     /*packageName*/   \
    NULL,     /*typeName*/      \
    0u,       /*numOfParams*/   \
    NULL,     /*params*/        \
    0u,       /*numOfInputs*/   \
    NULL,     /*inputTensors*/  \
    0u,       /*numOfOutputs*/  \
    NULL      /*outputTensors*/ \
  }
// clang-format on

/**
 * @brief Enum to distinguish various opConfig versions
 */
typedef enum {
  /// Enum to choose usage of Qnn_OpConfigV1_t in Qnn_OpConfig_t
  QNN_OPCONFIG_VERSION_1 = 1,
  // Unused, present to ensure 32 bits.
  QNN_OPCONFIG_VERSION_UNDEFINED = 0x7FFFFFFF
} Qnn_OpConfigVersion_t;

/**
 * @brief Structure which provides various versions of an opConfig
 */
typedef struct {
  /// Version of the QNN opConfig
  Qnn_OpConfigVersion_t version;
  union UNNAMED {
    /// Op config version 1 (see QNN_OPCONFIG_VERSION_1)
    Qnn_OpConfigV1_t v1;
  };
} Qnn_OpConfig_t;

// clang-format off
/// Qnn_OpConfig_t initializer macro
#define QNN_OPCONFIG_INIT               \
  {                                     \
    QNN_OPCONFIG_VERSION_1, /*version*/ \
    {                                   \
      QNN_OPCONFIG_V1_INIT /*v1*/       \
    }                                   \
  }
// clang-format on

/**
 * @brief An enum which identifies SOC models.
 *
 * @deprecated This enumeration will no longer be updated.
 */
typedef enum {
  QNN_SOC_MODEL_UNKNOWN = 0,
  QNN_SOC_MODEL_SM8350  = 30,
  QNN_SOC_MODEL_SM8325  = 34,
  QNN_SOC_MODEL_SM7350  = 32,
  QNN_SOC_MODEL_SM7325  = 35,
  QNN_SOC_MODEL_SM8450  = 36,
  QNN_SOC_MODEL_SC8280X = 37,
  QNN_SOC_MODEL_SM7315  = 38,
  QNN_SOC_MODEL_SA8295  = 39,
  QNN_SOC_MODEL_SM7450  = 41,
  QNN_SOC_MODEL_SM8475  = 42,
  QNN_SOC_MODEL_SM8550  = 43,
  QNN_SOC_MODEL_SM6450  = 50,
  QNN_SOC_MODEL_SA8255  = 52,
  QNN_SOC_MODEL_SM7475  = 54,
  QNN_SOC_MODEL_SM4450  = 59,
} Qnn_SocModel_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_TYPES_H
