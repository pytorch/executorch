#pragma once

#include <cstddef>

namespace executorch::backends::xnnpack::kernels {

void layer_norm_f32_neon(
    const float* input, float* output,
    const float* weight, const float* bias,
    size_t outer_size, size_t inner_size, float eps);

}
