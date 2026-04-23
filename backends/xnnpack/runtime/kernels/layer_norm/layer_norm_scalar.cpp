#include <executorch/backends/xnnpack/runtime/kernels/layer_norm/layer_norm_scalar.h>

#include <cmath>

namespace executorch::backends::xnnpack::kernels {

void layer_norm_f32_scalar(
    const float* input, float* output,
    const float* weight, const float* bias,
    size_t outer_size, size_t inner_size, float eps) {
    for (size_t i = 0; i < outer_size; i++) {
        const float* in_row = input + i * inner_size;
        float* out_row = output + i * inner_size;

        float sum = 0.0f;
        for (size_t j = 0; j < inner_size; j++) {
            sum += in_row[j];
        }
        float mean = sum / static_cast<float>(inner_size);

        float var_sum = 0.0f;
        for (size_t j = 0; j < inner_size; j++) {
            float diff = in_row[j] - mean;
            var_sum += diff * diff;
        }
        float inv_std = 1.0f / std::sqrt(var_sum / static_cast<float>(inner_size) + eps);

        for (size_t j = 0; j < inner_size; j++) {
            float normalized = (in_row[j] - mean) * inv_std;
            if (weight) { normalized *= weight[j]; }
            if (bias) { normalized += bias[j]; }
            out_row[j] = normalized;
        }
    }
}

}
