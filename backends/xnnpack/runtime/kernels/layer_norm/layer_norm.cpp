#include <executorch/backends/xnnpack/runtime/kernels/layer_norm/layer_norm.h>
#include <executorch/backends/xnnpack/runtime/kernels/layer_norm/layer_norm_scalar.h>
#ifdef __aarch64__
#include <executorch/backends/xnnpack/runtime/kernels/layer_norm/layer_norm_neon.h>
#endif

#include <cpuinfo.h>

namespace executorch::backends::xnnpack::kernels {

LayerNormF32Fn select_layer_norm_f32_kernel() {
#ifdef __aarch64__
    if (cpuinfo_initialize() && cpuinfo_has_arm_neon()) {
        return layer_norm_f32_neon;
    }
#endif
    return layer_norm_f32_scalar;
}

}
