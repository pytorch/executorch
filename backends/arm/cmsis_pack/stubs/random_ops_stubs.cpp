/*
 * Copyright (c) 2025 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stub implementations for random operations on bare-metal targets.
 *
 * ARM bare-metal libc++ disables std::random_device for bare-metal targets
 * (_LIBCPP_HAS_RANDOM_DEVICE=0). These stubs satisfy the linker for the
 * operator registrations.
 *
 * By default, these stubs abort at runtime if called. Define
 * EXECUTORCH_RANDOM_OPS_USE_LCG to enable a deterministic LCG fallback
 * for testing (NOT for production use).
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstdlib>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::IntArrayRef;
using Tensor = executorch::aten::Tensor;

#ifdef EXECUTORCH_RANDOM_OPS_USE_LCG
static uint32_t lcg_state = 0x12345678;
static float lcg_uniform() {
    lcg_state = (lcg_state * 1103515245U + 12345U) & 0x7FFFFFFF;
    return static_cast<float>(lcg_state) / static_cast<float>(0x7FFFFFFF);
}
#endif

Tensor& rand_out(
    KernelRuntimeContext& ctx,
    const IntArrayRef sizes,
    Tensor& out) {
    (void)ctx;
    (void)sizes;
#ifdef EXECUTORCH_RANDOM_OPS_USE_LCG
    float* data = out.mutable_data_ptr<float>();
    size_t numel = out.numel();
    for (size_t i = 0; i < numel; i++) {
        data[i] = lcg_uniform();
    }
    return out;
#else
    ET_LOG(Error, "rand_out: Random operations not supported on this platform");
    ctx.fail(executorch::runtime::Error::NotSupported);
    return out;
#endif
}

Tensor& randn_out(
    KernelRuntimeContext& ctx,
    const IntArrayRef sizes,
    Tensor& out) {
    (void)ctx;
    (void)sizes;
#ifdef EXECUTORCH_RANDOM_OPS_USE_LCG
    float* data = out.mutable_data_ptr<float>();
    size_t numel = out.numel();
    for (size_t i = 0; i < numel; i += 2) {
        float u1 = lcg_uniform();
        float u2 = lcg_uniform();
        if (u1 < 1e-10f) u1 = 1e-10f;
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.14159265f * u2;
        data[i] = r * cosf(theta);
        if (i + 1 < numel) {
            data[i + 1] = r * sinf(theta);
        }
    }
    return out;
#else
    ET_LOG(Error, "randn_out: Random operations not supported on this platform");
    ctx.fail(executorch::runtime::Error::NotSupported);
    return out;
#endif
}

::std::tuple<Tensor&, Tensor&> native_dropout_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double p,
    ::std::optional<bool> train,
    Tensor& out,
    Tensor& mask) {
    (void)ctx;
    (void)input;
    (void)p;
    (void)train;
#ifdef EXECUTORCH_RANDOM_OPS_USE_LCG
    memcpy(out.mutable_data_ptr(), input.const_data_ptr(),
           input.numel() * input.element_size());
    uint8_t* mask_data = mask.mutable_data_ptr<uint8_t>();
    float keep_prob = 1.0f - static_cast<float>(p);
    for (size_t i = 0; i < mask.numel(); i++) {
        mask_data[i] = (lcg_uniform() < keep_prob) ? 1 : 0;
    }
    float* out_data = out.mutable_data_ptr<float>();
    float scale = (keep_prob > 0) ? 1.0f / keep_prob : 0.0f;
    for (size_t i = 0; i < out.numel(); i++) {
        out_data[i] *= mask_data[i] * scale;
    }
    return ::std::tie(out, mask);
#else
    ET_LOG(Error, "native_dropout_out: Random operations not supported on this platform");
    ctx.fail(executorch::runtime::Error::NotSupported);
    return ::std::tie(out, mask);
#endif
}

} // namespace native
} // namespace executor
} // namespace torch
