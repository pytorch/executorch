#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/core/portable_type/tensor.h>  // for torch::executor::Tensor
#include <executorch/runtime/core/portable_type/scalar.h>  // for torch::executor::Scalar

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" {
#include "Include/arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {

using Tensor = torch::executor::Tensor;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

// Determine quantization scale from fp32 data
float determine_input_scale(const float* data, int size) {
    float min_val = *std::min_element(data, data + size);
    float max_val = *std::max_element(data, data + size);
    return (max_val - min_val) / 255.0f; // For int8 range [-128, 127]
}
// Quantize fp32 to int8
void quantize_tensor(const float* input, int8_t* output, int size,
                    float scale, int32_t zero_point) {
    for (int i = 0; i < size; i++) {
        int32_t quantized = std::round(input[i] / scale) + zero_point;
        // This ensures that the value quantized stays within the specified bounds — in this case, between -128 and 127, 
        // which are the limits of int8_t.
        output[i] = std::clamp(quantized, static_cast<int32_t>(-128), static_cast<int32_t>(127));
    }
}
// Dequantize int8 to fp32
void dequantize_tensor(const int8_t* input, float* output, int size,
                      float scale, int32_t zero_point) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - zero_point) * scale;
    }
}

// Converts a floating-point scale to CMSIS-NN fixed-point multiplier and shift
// scale: the floating-point scale factor from ExecuTorch quantization
// multiplier: output fixed-point multiplier (Q31 format)
// shift: output left shift amount (positive means left shift)
// diff_min: output minimum difference threshold (usually -128 for int8)
void convert_scale_to_cmsis_params(float scale, int32_t* multiplier, int32_t* shift, int32_t* diff_min) {
    if (scale == 0.0f) {
        *multiplier = 0;
        *shift = 0;
        *diff_min = -128;
        return;
    }
    // Decompose scale into mantissa and exponent: scale = mantissa * 2^exponent
    int exponent;
    float mantissa = std::frexp(scale, &exponent); // mantissa in [0.5, 1)
    // Convert mantissa to Q31 fixed-point format
    int64_t q_fixed = static_cast<int64_t>(std::round(mantissa * (1ll << 31)));
    // Adjust multiplier and shift for CMSIS-NN
    *multiplier = static_cast<int32_t>(q_fixed);
    // CMSIS-NN expects a left shift, so negate exponent to get shift
    *shift = -exponent;
    // Typical diff_min for int8 softmax
    *diff_min = -128;
}

torch::executor::Tensor& aten_softmax(
    KernelRuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {

    ET_LOG(Info, "CMSIS-NN quantized softmax kernel called");
    
    // Step 1: Extract fp32 data
    const float* input_data_fp32 = self.data_ptr<float>();
    float* output_data_fp32 = out.data_ptr<float>();
    
    // Step 2: Get tensor dimensions
    int rows = self.sizes()[0];
    int cols = self.sizes()[1];
    
    // Step 3: Quantize input (fp32 -> int8)
    // Determine appropriate scale/zero_point
    float input_scale = determine_input_scale(input_data_fp32, rows * cols);

    // '0' a reasonable default for symmetric quantization in int8, 
    // especially if the input data is centered around zero else TBD
    int32_t input_zero_point = 0;
    
    std::vector<int8_t> input_quantized(rows * cols);
    quantize_tensor(input_data_fp32, input_quantized.data(), 
                   rows * cols, input_scale, input_zero_point);
    
    // Step 4: Convert to CMSIS-NN parameters
    int32_t input_mult, input_shift, diff_min;
    convert_scale_to_cmsis_params(input_scale, &input_mult, &input_shift, &diff_min);
    
    // Step 5: Call CMSIS-NN kernel
    std::vector<int8_t> output_quantized(rows * cols);
    arm_softmax_s8(input_quantized.data(), rows, cols, 
                   input_mult, input_shift, diff_min,
                   output_quantized.data());
    
    // Step 6: Dequantize output (int8 -> fp32)
    dequantize_tensor(output_quantized.data(), output_data_fp32,
                     rows * cols, input_scale, input_zero_point);
    
    return out;
}

} // namespace native
} // namespace cortex_m
