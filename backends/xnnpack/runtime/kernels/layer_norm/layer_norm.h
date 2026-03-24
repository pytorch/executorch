#pragma once

#include <cstddef>

namespace executorch::backends::xnnpack::kernels {

using LayerNormF32Fn = void(*)(
    const float* input, float* output,
    const float* weight, const float* bias,
    size_t outer_size, size_t inner_size, float eps);

LayerNormF32Fn select_layer_norm_f32_kernel();

}
