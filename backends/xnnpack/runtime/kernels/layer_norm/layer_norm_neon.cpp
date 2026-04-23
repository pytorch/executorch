#ifdef __aarch64__

#include <executorch/backends/xnnpack/runtime/kernels/layer_norm/layer_norm_neon.h>

#include <arm_neon.h>
#include <cassert>
#include <cmath>

namespace executorch::backends::xnnpack::kernels {

namespace {
float sum_f32_neon(const float* data, size_t len) {
    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0);
    float32x4_t acc3 = vdupq_n_f32(0);
    float32x4_t acc4 = vdupq_n_f32(0);
    float32x4_t acc5 = vdupq_n_f32(0);
    float32x4_t acc6 = vdupq_n_f32(0);
    float32x4_t acc7 = vdupq_n_f32(0);

    size_t i = len;
    for (; i >= 32; i -= 32) {
        float32x4x2_t in01 = vld1q_f32_x2(data);
        float32x4x2_t in23 = vld1q_f32_x2(data + 8);
        float32x4x2_t in45 = vld1q_f32_x2(data + 16);
        float32x4x2_t in67 = vld1q_f32_x2(data + 24);

        acc0 = vaddq_f32(acc0, in01.val[0]);
        acc1 = vaddq_f32(acc1, in01.val[1]);
        acc2 = vaddq_f32(acc2, in23.val[0]);
        acc3 = vaddq_f32(acc3, in23.val[1]);
        acc4 = vaddq_f32(acc4, in45.val[0]);
        acc5 = vaddq_f32(acc5, in45.val[1]);
        acc6 = vaddq_f32(acc6, in67.val[0]);
        acc7 = vaddq_f32(acc7, in67.val[1]);

        data += 32;
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc4 = vaddq_f32(acc4, acc5);
    acc6 = vaddq_f32(acc6, acc7);

    acc0 = vaddq_f32(acc0, acc2);
    acc4 = vaddq_f32(acc4, acc6);

    acc0 = vaddq_f32(acc0, acc4);

    for (; i >= 4; i -= 4) {
        float32x4_t in = vld1q_f32(data);
        acc0 = vaddq_f32(acc0, in);
        data += 4;
    }

    float acc = vaddvq_f32(acc0);

    for (; i > 0; i--) {
        acc += *data;
        data++;
    }

    return acc;
}

float var_sum_f32_neon(const float* data, float mean, size_t len) {
    float32x4_t vmean = vdupq_n_f32(mean);

    float32x4_t acc0 = vdupq_n_f32(0);
    float32x4_t acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0);
    float32x4_t acc3 = vdupq_n_f32(0);

    size_t i = len;
    for (; i >= 16; i -= 16) {
        float32x4x2_t in01 = vld1q_f32_x2(data);
        float32x4x2_t in23 = vld1q_f32_x2(data + 8);

        float32x4_t delta0 = vsubq_f32(in01.val[0], vmean);
        float32x4_t delta1 = vsubq_f32(in01.val[1], vmean);
        float32x4_t delta2 = vsubq_f32(in23.val[0], vmean);
        float32x4_t delta3 = vsubq_f32(in23.val[1], vmean);

        float32x4_t delta_sq0 = vmulq_f32(delta0, delta0);
        float32x4_t delta_sq1 = vmulq_f32(delta1, delta1);
        float32x4_t delta_sq2 = vmulq_f32(delta2, delta2);
        float32x4_t delta_sq3 = vmulq_f32(delta3, delta3);

        acc0 = vaddq_f32(acc0, delta_sq0);
        acc1 = vaddq_f32(acc1, delta_sq1);
        acc2 = vaddq_f32(acc2, delta_sq2);
        acc3 = vaddq_f32(acc3, delta_sq3);

        data += 16;
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    for (; i >= 4; i -= 4) {
        float32x4_t in = vld1q_f32(data);
        float32x4_t delta = vsubq_f32(in, vmean);
        float32x4_t delta_sq = vmulq_f32(delta, delta);
        acc0 = vaddq_f32(acc0, delta_sq);
        data += 4;
    }

    float acc = vaddvq_f32(acc0);

    for (; i > 0; i--) {
        float in = *data;
        float delta = in - mean;
        float delta_sq = delta * delta;
        acc += delta_sq;
        data++;
    }

    return acc;
}

template <bool UseWeightBias>
void normalize_f32_neon(
    const float* input,
    float mean,
    float inv_std,
    const float* weight,
    const float* bias,
    float* out,
    size_t len)
{
    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vinv_std = vdupq_n_f32(inv_std);

    size_t i = len;
    for (; i >= 16; i -= 16) {
        float32x4x2_t in01 = vld1q_f32_x2(input);
        float32x4x2_t in23 = vld1q_f32_x2(input + 8);

        float32x4_t norm0 = vmulq_f32(vsubq_f32(in01.val[0], vmean), vinv_std);
        float32x4_t norm1 = vmulq_f32(vsubq_f32(in01.val[1], vmean), vinv_std);
        float32x4_t norm2 = vmulq_f32(vsubq_f32(in23.val[0], vmean), vinv_std);
        float32x4_t norm3 = vmulq_f32(vsubq_f32(in23.val[1], vmean), vinv_std);

        if constexpr (UseWeightBias) {
            float32x4x2_t w01 = vld1q_f32_x2(weight);
            float32x4x2_t w23 = vld1q_f32_x2(weight + 8);

            float32x4x2_t b01 = vld1q_f32_x2(bias);
            float32x4x2_t b23 = vld1q_f32_x2(bias + 8);

            norm0 = vmlaq_f32(b01.val[0], norm0, w01.val[0]);
            norm1 = vmlaq_f32(b01.val[1], norm1, w01.val[1]);
            norm2 = vmlaq_f32(b23.val[0], norm2, w23.val[0]);
            norm3 = vmlaq_f32(b23.val[1], norm3, w23.val[1]);

            weight += 16;
            bias += 16;
        }

        vst1q_f32(out, norm0);
        vst1q_f32(out + 4, norm1);
        vst1q_f32(out + 8, norm2);
        vst1q_f32(out + 12, norm3);

        input += 16;
        out += 16;
    }

    for (; i > 0; i--) {
        float in = *input;
        float norm = (in - mean) * inv_std;

        if constexpr (UseWeightBias) {
            auto w = *weight;
            auto b = *bias;

            norm = (norm * w) + b;

            weight++;
            bias++;
        }

        *out = norm;

        input++;
        out++;
    }
}
} // anonymous namespace

void layer_norm_f32_neon(
    const float* input, float* output,
    const float* weight, const float* bias,
    size_t outer_size, size_t inner_size, float eps) {
    for (size_t i = 0; i < outer_size; i++) {
        const float* in_row = input + i * inner_size;
        float* out_row = output + i * inner_size;

        float sum = sum_f32_neon(in_row, inner_size);
        float mean = sum / static_cast<float>(inner_size);

        float var_sum = var_sum_f32_neon(in_row, mean, inner_size);
        float inv_std = 1.0f / std::sqrt(var_sum / static_cast<float>(inner_size) + eps);

        if (weight != nullptr) {
            assert(bias != nullptr);
            normalize_f32_neon<true>(
                in_row,
                mean,
                inv_std,
                weight,
                bias,
                out_row,
                inner_size);
       } else {
            assert(bias == nullptr);
            normalize_f32_neon<false>(
                in_row,
                mean,
                inv_std,
                nullptr,
                nullptr,
                out_row,
                inner_size);
       }
    }
}

}

#endif
