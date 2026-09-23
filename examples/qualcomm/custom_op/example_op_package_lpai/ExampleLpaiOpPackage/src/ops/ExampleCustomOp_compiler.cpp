//==============================================================================
// Auto Generated Code for ExampleLpaiOpPackage
//==============================================================================

// Use the C stdio header rather than <iostream> so that this file can be built
// with toolchains that do not ship the C++ standard library headers (e.g. bare
// cross compilers targeting the DSP).
#include <stdio.h>

#include "QnnLpaiOpPackage.h"
#include "QnnTypes.h"

extern "C" QnnOpPackage_OperationInfo_t get_ExampleCustomOp_opInfo_Inference();
// ---------------------------------------------------------------------------
// Op metadata (counts)
// ---------------------------------------------------------------------------

uint32_t g_ExampleCustomOpStaticParamsNum = 0;
uint32_t g_ExampleCustomOpInputsNum = 1;
uint32_t g_ExampleCustomOpOutputsNum = 1;

// ---------------------------------------------------------------------------
// Layout support
// ---------------------------------------------------------------------------

static Qnn_ErrorHandle_t ExampleCustomOp_getLayoutSupportFlag(
    uint32_t* layoutFlag) {
  if (layoutFlag == nullptr) {
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  }
  *layoutFlag = QNN_LPAI_OP_LAYOUT_DEFAULT;
  return QNN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Supported data types (per input / per output)
// ---------------------------------------------------------------------------

// The LPAI lowering validates an op config twice: once from the
// LpaiPartitionFallbackSupport edge pass, while the graph still carries
// explicit quantize / dequantize nodes and the activations are therefore plain
// integer tensors, and once after FoldQDQ has folded the quantization
// parameters into the tensors. Both the integer and the fixed point spellings
// of the activation type are accepted so that the op passes validation at
// either stage.
//
// Only 8 bit activations are declared. The kernel itself is written for 8 and
// 16 bit, but 16 bit activations do not currently work end to end on the LPAI
// backend, so declaring them here would advertise a broken configuration:
//
//   * a 16 bit activation is a torch.int32 tensor holding the uint16 range
//     before folding (see get_16a16w_qnn_ptq_config), so it arrives as
//     QNN_DATATYPE_INT_32 rather than QNN_DATATYPE_UINT_16 - the UINT_16 that
//     used to be listed here was therefore never matched anyway;
//   * declaring INT_32 does get a 16a16w graph past validation and the kernel
//     runs, but only the first half of the output tensor comes back correct.
//     That reproduces with an identity requantization (calibration value equal
//     to the inference value, so the scale ratio is exactly 1 and the kernel is
//     a plain copy), which rules out the requantization arithmetic and points
//     at how the backend hands 16 bit custom op tensors over.
//
// Keeping the declaration to 8 bit means an unsupported graph is rejected at
// compile time and falls back cleanly instead of silently returning half a
// result. See examples/qualcomm/custom_op/README.md.
static Qnn_DataType_t g_ExampleCustomOpInput_0_DataTypes[] = {
    QNN_DATATYPE_UFIXED_POINT_8,
    QNN_DATATYPE_UINT_8,
};

QnnLpaiOpPackage_DataTypeInfo_t g_ExampleCustomOpInputDatatypes[] = {
    {sizeof(g_ExampleCustomOpInput_0_DataTypes) /
         sizeof(g_ExampleCustomOpInput_0_DataTypes[0]),
     g_ExampleCustomOpInput_0_DataTypes},
};

static Qnn_DataType_t g_ExampleCustomOpOutput_0_DataTypes[] = {
    QNN_DATATYPE_UFIXED_POINT_8,
    QNN_DATATYPE_UINT_8,
};

QnnLpaiOpPackage_DataTypeInfo_t g_ExampleCustomOpOutputDatatypes[] = {
    {sizeof(g_ExampleCustomOpOutput_0_DataTypes) /
         sizeof(g_ExampleCustomOpOutput_0_DataTypes[0]),
     g_ExampleCustomOpOutput_0_DataTypes},
};

// ---------------------------------------------------------------------------
// Compile-time validation
// ---------------------------------------------------------------------------

/**
 * @brief Number of bits held by a QNN data type.
 *
 * The lower byte of Qnn_DataType_t encodes the bit width in BCD, so
 * QNN_DATATYPE_UFIXED_POINT_8 and QNN_DATATYPE_UINT_8 share the same width.
 */
static uint32_t dataTypeBitWidth(Qnn_DataType_t dataType) {
  const uint32_t lowerByte = (uint32_t)dataType & 0xFF;
  return ((lowerByte >> 4) * 10) + (lowerByte & 0xF);
}

/**
 * @brief Validate op configuration at compilation time.
 *
 * Called by the host compiler to verify that the Qnn_OpConfig_t provided by
 * the user matches what this op package supports (tensor ranks, data types,
 * parameter values, etc.).
 *
 * @return QNN_SUCCESS on success, error code otherwise.
 */
Qnn_ErrorHandle_t ExampleCustomOp_validateOp(Qnn_OpConfig_t opConfig) {
  // The package-level ValidateOpConfig already checked the tensor/param
  // counts and that each data type is in the supported set declared above.
  // Here we add the op specific constraints:
  //   1. input and output must use the same element width
  //   2. input and output must be rank 4 with identical dimensions
  const Qnn_Tensor_t& input = opConfig.v1.inputTensors[0];
  const Qnn_Tensor_t& output = opConfig.v1.outputTensors[0];

  // Compare element widths rather than the data types themselves. The op is
  // validated both before and after the quantization parameters are folded
  // into the tensors, so the very same graph is presented once as
  // QNN_DATATYPE_UINT_8 and once as QNN_DATATYPE_UFIXED_POINT_8.
  if (dataTypeBitWidth(input.v1.dataType) !=
      dataTypeBitWidth(output.v1.dataType)) {
    fprintf(
        stderr,
        "ExampleCustomOp: input/output data type mismatch (0x%x vs 0x%x)\n",
        (unsigned)input.v1.dataType,
        (unsigned)output.v1.dataType);
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }

  constexpr uint32_t kExpectedRank = 4;
  if (input.v1.rank != kExpectedRank || output.v1.rank != kExpectedRank) {
    fprintf(
        stderr,
        "ExampleCustomOp: expected rank 4 input/output, got %u/%u\n",
        input.v1.rank,
        output.v1.rank);
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  }

  for (uint32_t i = 0; i < kExpectedRank; i++) {
    if (input.v1.dimensions[i] != output.v1.dimensions[i]) {
      fprintf(
          stderr,
          "ExampleCustomOp: input/output dimension mismatch at %u (%u != %u)\n",
          i,
          input.v1.dimensions[i],
          output.v1.dimensions[i]);
      return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
  }

  return QNN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Buffer size calculation
// ---------------------------------------------------------------------------

/**
 * @brief Compute the required temp buffer size for this op.
 *
 * The buffer is sized to hold the first input tensor, which is what the
 * generated skeleton suggests and is enough for an elementwise op.
 *
 * @param[out] bufferSize  Receives the required temp buffer size in bytes.
 * @return QNN_SUCCESS on success, error code otherwise.
 */
Qnn_ErrorHandle_t ExampleCustomOp_getTempBufferSize(
    Qnn_OpConfig_t opConfig,
    size_t* bufferSize) {
  if (!bufferSize) {
    return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  }

  const Qnn_Tensor_t* inputTensor0 = &opConfig.v1.inputTensors[0];
  const uint32_t rank = inputTensor0->v1.rank;
  const uint32_t* dims = inputTensor0->v1.dimensions;
  const uint32_t dataTypeSize = dataTypeBitWidth(inputTensor0->v1.dataType) / 8;

  // size = dataTypeSize * dim[0] * dim[1] * ... * dim[rank-1]
  size_t size = dataTypeSize;
  for (uint32_t i = 0; i < rank; i++) {
    size *= dims[i];
  }
  *bufferSize = size;

  return QNN_SUCCESS;
}

// ---------------------------------------------------------------------------
// Op info registration (compiler mode)
// ---------------------------------------------------------------------------

/**
 * @brief Return the op info struct populated with both inference and
 *        compiler-specific fields.
 *
 * Calls get_ExampleCustomOp_opInfo_Inference() to obtain the base struct
 * (opType + executeOp), then layers the compiler-only fields on top.
 */
extern "C" QnnOpPackage_OperationInfo_t get_ExampleCustomOp_opInfo_Compiler() {
  QnnOpPackage_OperationInfo_t ExampleCustomOp_opInfo =
      get_ExampleCustomOp_opInfo_Inference();

  // Metadata counts
  ExampleCustomOp_opInfo->numOfParams = g_ExampleCustomOpStaticParamsNum;
  ExampleCustomOp_opInfo->numOfInputs = g_ExampleCustomOpInputsNum;
  ExampleCustomOp_opInfo->numOfOutputs = g_ExampleCustomOpOutputsNum;

  // Compiler-side callbacks
  ExampleCustomOp_opInfo->validateOp = ExampleCustomOp_validateOp;
  ExampleCustomOp_opInfo->getTempBufferSize = ExampleCustomOp_getTempBufferSize;
  ExampleCustomOp_opInfo->getLayoutSupportFlag =
      ExampleCustomOp_getLayoutSupportFlag;

  // Supported data types
  ExampleCustomOp_opInfo->inputDatatypes = g_ExampleCustomOpInputDatatypes;
  ExampleCustomOp_opInfo->outputDatatypes = g_ExampleCustomOpOutputDatatypes;

  return ExampleCustomOp_opInfo;
}
