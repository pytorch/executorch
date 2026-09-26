//==============================================================================
// Auto Generated Code for ExampleLpaiOpPackage
//==============================================================================

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "QnnLpaiOpPackage.h"
#include "QnnLpaiOpPackageInfrastructure.h"

extern QnnLpaiOpPackage_GlobalInfrastructure_t sg_globalInfra;

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

// Every rejection below returns one of a handful of error codes, which on their
// own do not say *which* check tripped. Tracing is compiled out by default
// because this code also runs on the DSP, where printf is expensive and garbles
// long argument lists (read diagnostics back with FARF + logcat instead, see
// examples/qualcomm/custom_op/README.md). Build with
// -DEXAMPLE_CUSTOM_OP_TRACE=1 to switch it on while developing a kernel.
#if defined(EXAMPLE_CUSTOM_OP_TRACE) && EXAMPLE_CUSTOM_OP_TRACE
#define TRACE(msg) printf("ExampleCustomOp: " msg "\n")
#else
#define TRACE(msg) ((void)0)
#endif

// Report why the node was rejected and return the matching error code.
#define FAIL(code, msg)  \
    do {                 \
        TRACE(msg);      \
        return (code);   \
    } while (0)

// ---------------------------------------------------------------------------
// Op metadata
// ---------------------------------------------------------------------------

QnnLpaiOpPackage_OperationInfo_t g_ExampleCustomOp_opInfo;
char g_ExampleCustomOpOpType[] = "ExampleCustomOp";

// --- helpers ------------------------------------------------------------------

typedef struct {
    uint32_t                  shapeRank;
    uint32_t                  layoutRank;
    uint32_t                  dimSizes[QNN_LPAI_CUSTOM_OP_MAX_LAYOUT_RANK];
    QnnLpaiCustomOp_Layout_t  layout;
} TensorIterInfo;

// Populate TensorIterInfo for the given tensor.
static QnnLpaiCustomOp_Error_t getTensorIterInfo(QnnLpai_CustomOpTensor_t tensor,
                                                 TensorIterInfo*       info) {
    if (sg_globalInfra.getTensorShapeRank(tensor,  &info->shapeRank)  != LPAI_CUSTOM_OP_SUCCESS ||
        sg_globalInfra.getTensorLayoutRank(tensor, &info->layoutRank) != LPAI_CUSTOM_OP_SUCCESS ||
        sg_globalInfra.getTensorLayout(tensor,     &info->layout)     != LPAI_CUSTOM_OP_SUCCESS) {
        return LPAI_CUSTOM_OP_FAIL;
    }
    // dimSizes[] (and the coords[] array indexed by the same ranks below) hold
    // QNN_LPAI_CUSTOM_OP_MAX_LAYOUT_RANK entries, so refuse anything deeper
    // rather than writing past the end of them.
    if (info->shapeRank  > QNN_LPAI_CUSTOM_OP_MAX_LAYOUT_RANK ||
        info->layoutRank > QNN_LPAI_CUSTOM_OP_MAX_LAYOUT_RANK) {
        return LPAI_CUSTOM_OP_FAIL;
    }
    for (uint32_t i = 0; i < info->shapeRank; i++) {
        if (sg_globalInfra.getTensorShapeDimSize(tensor, i, &info->dimSizes[i]) != LPAI_CUSTOM_OP_SUCCESS) {
            return LPAI_CUSTOM_OP_FAIL;
        }
    }
    return LPAI_CUSTOM_OP_SUCCESS;
}

// Compute the byte offset of the element at coords[] using the tensor's layout.
// offset = sum_i( coords[layoutOrder[i]] * layoutStride[i] )
static uint32_t computeByteOffset(const uint32_t*                 coords,
                                  const QnnLpaiCustomOp_Layout_t* layout,
                                  uint32_t                        layoutRank) {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < layoutRank; i++) {
        offset += coords[layout->layoutOrder[i]] * layout->layoutStride[i];
    }
    return offset;
}

// Advance coords[] by one step in memory order (fastest dimension first).
// Returns after the last element (all coords wrap back to 0).
//
// layoutOrder is walked back to front on purpose: QnnLpaiOpPackageInfrastructure.h
// defines index 0 as the *slowest* moving (least contiguous) dimension and the
// last valid index as the fastest moving one, so the innermost digit of this
// odometer is layoutOrder[layoutRank - 1]. Iterating the other way round also
// visits every element, but in the least cache friendly order possible, and it
// silently breaks any kernel that is not a pure elementwise map.
static void advanceCoords(uint32_t*       coords,
                          const uint32_t* dimSizes,
                          const uint8_t*  layoutOrder,
                          uint32_t        layoutRank) {
    for (uint32_t i = layoutRank; i-- > 0;) {
        const uint8_t dim = layoutOrder[i];
        if (++coords[dim] < dimSizes[dim]) return;
        coords[dim] = 0;
    }
}

// Convert a QnnLpaiCustomOp_QuantScale_t into a float scale.
//
// The scale may be stored either as a float or as an integer approximation
// where: scale_float = scale_int / (2 ^ shift)
//
// ldexpf() is used rather than (1u << shift) on purpose: shift is routinely
// larger than 31 (the eNPU reports e.g. scale=538976320, shift=37 for 1/255),
// and shifting a 32 bit value by >= 32 is undefined behaviour. On the DSP that
// evaluates to 0, so the division silently produced +inf and every subsequent
// requantization became NaN.
//
// The scale type is matched exhaustively rather than treating "not float" as
// "integer": the enum also has an UNDEFINED value, and reading that as an
// integer scale would silently interpret uninitialized union bytes as a
// scale/shift pair.
static QnnLpaiCustomOp_Error_t getFloatScale(
    const QnnLpaiCustomOp_QuantScale_t* scale, float* out) {
    switch (scale->type) {
        case QNN_LPAI_CUSTOM_OP_QUANT_SCALE_TYPE_FLOAT:
            *out = scale->fScale;
            return LPAI_CUSTOM_OP_SUCCESS;
        case QNN_LPAI_CUSTOM_OP_QUANT_SCALE_TYPE_INT:
            *out = ldexpf((float)scale->iScale.scale, -scale->iScale.shift);
            return LPAI_CUSTOM_OP_SUCCESS;
        default:
            return LPAI_CUSTOM_OP_FAIL;
    }
}

// Round value to the nearest integer and clamp it into [minValue, maxValue].
//
// The comparisons are done in floating point, before the conversion to
// int32_t, because converting a float that is NaN or outside the range of the
// target integer type is undefined behaviour, and the clamp cannot undo that
// afterwards. Written so that a NaN input fails the first comparison and
// returns minValue rather than propagating.
static int32_t saturateCast(float value, int32_t minValue, int32_t maxValue) {
    if (!(value > (float)minValue)) {
        return minValue;
    }
    if (value >= (float)maxValue) {
        return maxValue;
    }
    return (int32_t)(value + 0.5f);
}

// Number of bytes an element of the given data type occupies. Returns 0 for
// types this kernel does not handle.
//
// Both the signed and the unsigned 8/16 bit types are accepted here. The op
// package declares QNN_DATATYPE_UFIXED_POINT_8 in its config, and the tensors
// really are unsigned on device, but the eNPU reports the 8 bit type back as
// LPAI_CUSTOM_OP_DATATYPE_INT_8 through getTensorDataType(). Only the element
// width is taken from the reported type; see the comment in
// ExampleCustomOp_executeOp() for why the reported sign is ignored.
//
// The 16 bit cases are deliberately kept, because the arithmetic below is
// written to be width agnostic and that is the interesting part to copy into a
// new kernel. Note though that they are currently *unreachable and untested*:
// the op package only declares 8 bit activations, since a 16 bit graph returns
// a half correct output tensor for reasons that sit below this kernel (it also
// reproduces when the requantization is an identity copy). See the data type
// comment in ExampleCustomOp_compiler.cpp.
static uint32_t elementSize(QnnLpaiCustomOp_DataType_t type) {
    switch (type) {
        case LPAI_CUSTOM_OP_DATATYPE_INT_8:
        case LPAI_CUSTOM_OP_DATATYPE_UINT_8:
            return 1u;
        case LPAI_CUSTOM_OP_DATATYPE_INT_16:
        case LPAI_CUSTOM_OP_DATATYPE_UINT_16:
            return 2u;
        default:
            return 0u;
    }
}

// --- op implementation --------------------------------------------------------

/**
 * @brief Execute the ExampleCustomOp operation on the given node.
 *
 * Uses the package-level sg_globalInfra (declared extern at the top of this
 * file, defined in ExampleLpaiOpPackageInterface.c) to access tensors.
 */
int32_t ExampleCustomOp_executeOp(QnnLpai_CustomOpNode_t lpaiNode) {

    // Retrieve input tensors
    QnnLpai_CustomOpTensor_t input0       = sg_globalInfra.getInputTensor(lpaiNode, 0);

    // Retrieve output tensors
    QnnLpai_CustomOpTensor_t output0      = sg_globalInfra.getOutputTensor(lpaiNode, 0);

    // Retrieve parameter tensors

    // Validate that all required tensors were obtained
    if (!input0)  FAIL(LPAI_CUSTOM_OP_FAIL, "no input tensor 0");
    if (!output0) FAIL(LPAI_CUSTOM_OP_FAIL, "no output tensor 0");

    // Query data types via out-pointer API
    QnnLpaiCustomOp_DataType_t inType0  = (QnnLpaiCustomOp_DataType_t)0;
    QnnLpaiCustomOp_DataType_t outType0 = (QnnLpaiCustomOp_DataType_t)0;
    if (sg_globalInfra.getTensorDataType(input0,  &inType0)  != LPAI_CUSTOM_OP_SUCCESS) {
        FAIL(LPAI_CUSTOM_OP_FAIL, "getTensorDataType(input) failed");
    }
    if (sg_globalInfra.getTensorDataType(output0, &outType0) != LPAI_CUSTOM_OP_SUCCESS) {
        FAIL(LPAI_CUSTOM_OP_FAIL, "getTensorDataType(output) failed");
    }
    // This op supports 8-bit and 16-bit fixed point and requires the input and
    // output to use the same data type.
    const uint32_t bytesPerElement = elementSize(inType0);
    if (inType0 != outType0 || bytesPerElement == 0u) {
        FAIL(LPAI_CUSTOM_OP_NOT_SUPPORTED, "unsupported or mismatched data type");
    }

    // Retrieve raw data pointers
    void* inData0  = sg_globalInfra.getTensorData(input0);
    void* outData0 = sg_globalInfra.getTensorData(output0);
    if (!inData0)  FAIL(LPAI_CUSTOM_OP_FAIL, "no input data pointer");
    if (!outData0) FAIL(LPAI_CUSTOM_OP_FAIL, "no output data pointer");

    // Retrieve layout and shape info needed for strided iteration
    TensorIterInfo inInfo0  = {0};
    TensorIterInfo outInfo0 = {0};
    if (getTensorIterInfo(input0,  &inInfo0)  != LPAI_CUSTOM_OP_SUCCESS) {
        FAIL(LPAI_CUSTOM_OP_FAIL, "cannot describe input tensor");
    }
    if (getTensorIterInfo(output0, &outInfo0) != LPAI_CUSTOM_OP_SUCCESS) {
        FAIL(LPAI_CUSTOM_OP_FAIL, "cannot describe output tensor");
    }

    // Input and output must describe the same logical tensor shape, and the
    // iteration below relies on both of them being consistent:
    //   * dimSizes[] is only filled in up to shapeRank, so a layoutRank beyond
    //     that would make advanceCoords() read a zero extent and never finish
    //     the odometer;
    //   * layoutOrder[] indexes coords[], so entries must stay inside the
    //     logical rank. The header documents the entries past the rank as
    //     invalid, so they are not to be trusted blindly.
    if (inInfo0.shapeRank != outInfo0.shapeRank ||
        inInfo0.layoutRank != inInfo0.shapeRank ||
        outInfo0.layoutRank != outInfo0.shapeRank) {
        FAIL(LPAI_CUSTOM_OP_NOT_SUPPORTED, "rank mismatch between shape/layout");
    }
    for (uint32_t i = 0; i < inInfo0.shapeRank; i++) {
        if (inInfo0.dimSizes[i] != outInfo0.dimSizes[i]) {
            FAIL(LPAI_CUSTOM_OP_NOT_SUPPORTED, "input/output dimension mismatch");
        }
        if (inInfo0.layout.layoutOrder[i] >= inInfo0.shapeRank ||
            outInfo0.layout.layoutOrder[i] >= outInfo0.shapeRank) {
            FAIL(LPAI_CUSTOM_OP_FAIL, "layoutOrder entry outside the tensor rank");
        }
    }
    // Only the flat layout form is understood by computeByteOffset().
    if (inInfo0.layout.layoutForm != QNN_LPAI_CUSTOM_OP_LAYOUT_FORM_FLAT ||
        outInfo0.layout.layoutForm != QNN_LPAI_CUSTOM_OP_LAYOUT_FORM_FLAT) {
        FAIL(LPAI_CUSTOM_OP_NOT_SUPPORTED, "unsupported layout form");
    }

    // A zero sized dimension yields no elements, which the loop below handles
    // by simply not running. Keep the count 64-bit so a large tensor does not
    // silently wrap while accumulating its dimensions.
    uint64_t totalElements = 1;
    for (uint32_t i = 0; i < inInfo0.shapeRank; i++) {
        totalElements *= inInfo0.dimSizes[i];
    }

    // Retrieve the per-tensor quantization parameters so the multiply can be
    // performed in real (dequantized) space and requantized on the way out.
    QnnLpaiCustomOp_PerTensorQuantInfo_t inQuant  = {0};
    QnnLpaiCustomOp_PerTensorQuantInfo_t outQuant = {0};
    if (sg_globalInfra.getPerTensorQuantParams(input0, &inQuant) != LPAI_CUSTOM_OP_SUCCESS) {
        FAIL(LPAI_CUSTOM_OP_FAIL, "no input quantization parameters");
    }
    if (sg_globalInfra.getPerTensorQuantParams(output0, &outQuant) != LPAI_CUSTOM_OP_SUCCESS) {
        FAIL(LPAI_CUSTOM_OP_FAIL, "no output quantization parameters");
    }

    float inScale  = 0.0f;
    float outScale = 0.0f;
    if (getFloatScale(&inQuant.scale, &inScale) != LPAI_CUSTOM_OP_SUCCESS ||
        getFloatScale(&outQuant.scale, &outScale) != LPAI_CUSTOM_OP_SUCCESS) {
        FAIL(LPAI_CUSTOM_OP_FAIL, "unsupported quantization scale type");
    }
    // Both scales are checked: a zero output scale would divide by zero, and a
    // zero input scale would silently map the whole tensor to code 0, which is
    // a bad quantization parameter rather than a legitimate result.
    if (inScale == 0.0f || outScale == 0.0f) {
        FAIL(LPAI_CUSTOM_OP_FAIL, "quantization scale is zero");
    }

    // ExampleCustomOp computes: output = input * 3
    //
    // Quantization convention on the eNPU
    // -----------------------------------
    // getPerTensorQuantParams() reports an *offset* that biases the values as
    // they sit in memory. The relation between a byte in the buffer and the
    // quantized code that the rest of the graph works with is
    //
    //     code = stored - offset          (offset is -128 on the 8a8w graphs
    //     stored = code + offset           produced here, so stored = code-128)
    //
    // and the real value is  v = scale * code.  Note that this is *not* the
    // usual v = scale * (q - zero_point) with q taken straight from memory: the
    // surrounding quantize/dequantize nodes in the ExecuTorch graph use a zero
    // point of 0, so the codes they produce arrive here already biased by the
    // offset, and the kernel is responsible for removing that bias on the way
    // in and re-applying it on the way out. Both directions are needed; doing
    // only one of them shifts the whole tensor by 128 codes. Measured on
    // SM8850: input code 255 is handed to the kernel as the stored byte 127.
    //
    // Both directions have to wrap modulo the storage width. With offset -128
    // the codes 0..127 are stored as the bytes 128..255, so reading the byte
    // back and subtracting the offset without wrapping yields 256..383 instead
    // of 0..127 - i.e. small values would decode as large ones and then
    // saturate. On the way out the narrowing store already wraps, on the way in
    // the mask below does it explicitly.
    //
    // Putting it together, the multiply becomes a pure operation on codes:
    //
    //     code_in  = (stored_in - in_offset) & code_mask
    //     code_out = round(3 * in_scale / out_scale * code_in)
    //     stored_out = code_out + out_offset
    const float kMultiplier = 3.0f;
    const float requantScale = kMultiplier * inScale / outScale;
    // The scales come from the runtime, so the ratio is not guaranteed to be a
    // usable number. Checking it once here keeps the inner loop free of
    // non-finite values, which saturateCast() would otherwise have to absorb
    // for every element.
    if (!isfinite(requantScale)) {
        FAIL(LPAI_CUSTOM_OP_FAIL, "requantization scale is not finite");
    }

    // Saturation bounds for a *code*, not for the biased byte that is written to
    // memory. Clamping the biased byte instead looks harmless but is wrong: for
    // input 1.0 the code is 255 and the byte is 127, so a clamp on the byte
    // never triggers, yet a code of 256 (one past the maximum) would be stored
    // as the perfectly innocent looking byte 128 and read back by the consumer
    // as code 256-256 = 0. Overflow has to be caught while the value is still a
    // code.
    //
    // For this particular op the clamp cannot actually fire: the output range
    // observed during calibration is exactly three times the input range, so
    // requantScale is 1 and code_out == code_in. It is kept because a kernel
    // whose output range is not a multiple of its input range does overflow
    // here, and because clamping the wrong quantity is a silent error.
    // Derived from the storage width so that the two stay in step: for an
    // n bit code the largest value and the wraparound mask are both 2^n - 1.
    const uint32_t codeBits = bytesPerElement * 8u;
    const int32_t codeMin = 0;
    const int32_t codeMax = (int32_t)((1u << codeBits) - 1u);
    // Wraps the un-biased input back into the code range, see above.
    const int32_t codeMask = codeMax;

    uint32_t coords[QNN_LPAI_CUSTOM_OP_MAX_LAYOUT_RANK] = {0};
    for (uint64_t i = 0; i < totalElements; i++) {
        // Byte offsets are derived from the same logical coordinates, so the
        // input and output are free to use different memory layouts.
        const uint32_t inOffset =
            computeByteOffset(coords, &inInfo0.layout, inInfo0.layoutRank);
        const uint32_t outOffset =
            computeByteOffset(coords, &outInfo0.layout, outInfo0.layoutRank);

        // Read the stored value, widening to int32_t so the same arithmetic
        // covers both supported widths. The 16 bit access goes through memcpy
        // because nothing guarantees that a stride, and therefore the byte
        // offset computed from it, is even; an unaligned uint16_t load is
        // undefined behaviour and traps on some Hexagon configurations.
        const uint8_t* inPtr = (const uint8_t*)inData0 + inOffset;
        int32_t storedIn;
        if (bytesPerElement == 1u) {
            storedIn = (int32_t)*inPtr;
        } else {
            uint16_t raw;
            memcpy(&raw, inPtr, sizeof(raw));
            storedIn = (int32_t)raw;
        }

        // Remove the input bias to get a code, wrapping modulo the storage
        // width so that a biased byte >= 128 maps back to a small code instead
        // of running past codeMax, then scale and saturate in the code domain.
        const int32_t codeIn = (storedIn - (int32_t)inQuant.offset) & codeMask;
        const int32_t code =
            saturateCast(requantScale * (float)codeIn, codeMin, codeMax);

        // Re-apply the output bias. The narrowing cast below wraps modulo the
        // storage width, mirroring the mask applied on the way in, and is
        // precisely the inverse of the consumer's (stored - offset), so the
        // round trip is exact.
        const int32_t storedOut = code + (int32_t)outQuant.offset;

        uint8_t* outPtr = (uint8_t*)outData0 + outOffset;
        if (bytesPerElement == 1u) {
            *outPtr = (uint8_t)storedOut;
        } else {
            const uint16_t raw = (uint16_t)storedOut;
            memcpy(outPtr, &raw, sizeof(raw));
        }

        advanceCoords(
            coords, inInfo0.dimSizes, inInfo0.layout.layoutOrder, inInfo0.layoutRank);
    }

    return LPAI_CUSTOM_OP_SUCCESS;
}

// ---------------------------------------------------------------------------
// Op info registration (inference mode)
// ---------------------------------------------------------------------------

// Report the memory layouts this kernel can consume. The kernel walks tensors
// through layoutOrder/layoutStride, so the default layout is what it expects.
//
// This lives on the inference side on purpose: the eAI runtime on the DSP
// queries the layout support flag while setting up a custom op node, so the
// callback must be populated in the inference build too, not only in the
// compiler build. Leaving it NULL makes the on-device query fail with
// AEE_EBADSTATE (0x8000040D), surfacing on the host as
// "Failed to register custom op package through platform".
static Qnn_ErrorHandle_t ExampleCustomOp_getLayoutSupportFlagInference(
    uint32_t* layoutFlag) {
    if (layoutFlag == NULL) {
        return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    }
    *layoutFlag = QNN_LPAI_OP_LAYOUT_DEFAULT;
    return QNN_SUCCESS;
}

extern QnnOpPackage_OperationInfo_t get_ExampleCustomOp_opInfo_Inference() {
    memset(&g_ExampleCustomOp_opInfo, 0, sizeof(QnnLpaiOpPackage_OperationInfo_t));
    g_ExampleCustomOp_opInfo.opType    = g_ExampleCustomOpOpType;
    g_ExampleCustomOp_opInfo.executeOp = ExampleCustomOp_executeOp;

    // The tensor counts and the layout support callback are needed by the
    // runtime as well as by the compiler, so populate them here. The compiler
    // build layers its own callbacks on top of this struct.
    g_ExampleCustomOp_opInfo.numOfInputs  = 1;
    g_ExampleCustomOp_opInfo.numOfOutputs = 1;
    g_ExampleCustomOp_opInfo.numOfParams  = 0;
    g_ExampleCustomOp_opInfo.getLayoutSupportFlag =
        ExampleCustomOp_getLayoutSupportFlagInference;

    return &g_ExampleCustomOp_opInfo;
}
