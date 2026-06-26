/*
 * Copyright (c) 2026 iote.ai
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * AXON op extensions — sigmoid and tanh CPU callbacks.
 *
 * These are CPU-side implementations of the activation functions that
 * the AXON command buffer dispatches to when it encounters op extension
 * segments (op code 101=sigmoid, 102=tanh). The preceding AXON layer
 * outputs INT16 q3.12 data, and these functions convert it to the
 * final INT8 output.
 *
 * These replace Nordic's nrf_axon_nn_op_extension_sigmoid/_tanh from
 * sdk-edge-ai/drivers/axon/nrf_axon_nn_op_extensions.c. The AXON
 * backend's codegen step rewrites the generated model headers to
 * reference axon_op_extension_sigmoid/_tanh instead.
 *
 * Why custom implementations
 * --------------------------
 * Nordic's stock sigmoid uses double-precision libm exp() per element,
 * which on the Cortex-M33's single-precision FPU is software-emulated
 * (~2,800 cycles per element). Using single-precision expf() instead
 * gives ~1.5x speedup with identical quantized output.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_nn_op_extensions.h"

/* Quantise a sigmoid result in [0,1] to int8. */
static inline int8_t axon_quantize_sigmoid(float v)
{
    float q = roundf(v * 256.0f) - 128.0f;
    if (q > 127.0f)  return 127;
    if (q < -128.0f) return -128;
    return (int8_t)q;
}

/* Quantise a tanh result in [-1,1] to int8. */
static inline int8_t axon_quantize_tanh(float v)
{
    float q = roundf(v * 128.0f);
    if (q > 127.0f)  return 127;
    if (q < -128.0f) return -128;
    return (int8_t)q;
}

nrf_axon_result_e axon_op_extension_sigmoid(
    uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
    if (args == NULL ||
        (argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE))
            < sizeof(nrf_axon_nn_op_extension_base1_args_s)) {
        return NRF_AXON_RESULT_FAILURE;
    }
    nrf_axon_nn_op_extension_base1_args_s *base1_args =
        (nrf_axon_nn_op_extension_base1_args_s *)args;

    if (base1_args->remaining_args.output_bytewidth != 1) {
        return NRF_AXON_RESULT_FAILURE;
    }

    const uint8_t input_extra_stride =
        (!base1_args->remaining_args.input_is_packed
         && (base1_args->remaining_args.width & 1))
            ? 1 : 0;

    int16_t *input_ptr  = (int16_t *)base1_args->ptr_args.input;
    int8_t  *output_ptr = (int8_t *)base1_args->ptr_args.output;
    const uint16_t channels = base1_args->remaining_args.channel_cnt;
    const uint16_t height   = base1_args->remaining_args.height;
    const uint16_t width    = base1_args->remaining_args.width;

    for (uint16_t ch = 0; ch < channels; ch++) {
        for (uint16_t row = 0; row < height; row++) {
            for (uint16_t col = 0; col < width; col++, input_ptr++, output_ptr++) {
                /* Input is q3.12 INT16; convert to float. */
                float x = (float)*input_ptr * (1.0f / 4096.0f);
                /* Single-precision expf instead of double-precision exp. */
                float e = expf(-x);
                *output_ptr = axon_quantize_sigmoid(1.0f / (1.0f + e));
            }
            input_ptr += input_extra_stride;
        }
    }
    return NRF_AXON_RESULT_SUCCESS;
}

nrf_axon_result_e axon_op_extension_tanh(
    uint16_t argc, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *args)
{
    if (args == NULL ||
        (argc * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE))
            < sizeof(nrf_axon_nn_op_extension_base1_args_s)) {
        return NRF_AXON_RESULT_FAILURE;
    }
    nrf_axon_nn_op_extension_base1_args_s *base1_args =
        (nrf_axon_nn_op_extension_base1_args_s *)args;

    if (base1_args->remaining_args.output_bytewidth != 1) {
        return NRF_AXON_RESULT_FAILURE;
    }

    const uint8_t input_extra_stride =
        (!base1_args->remaining_args.input_is_packed
         && (base1_args->remaining_args.width & 1))
            ? 1 : 0;

    int16_t *input_ptr  = (int16_t *)base1_args->ptr_args.input;
    int8_t  *output_ptr = (int8_t *)base1_args->ptr_args.output;
    const uint16_t channels = base1_args->remaining_args.channel_cnt;
    const uint16_t height   = base1_args->remaining_args.height;
    const uint16_t width    = base1_args->remaining_args.width;

    for (uint16_t ch = 0; ch < channels; ch++) {
        for (uint16_t row = 0; row < height; row++) {
            for (uint16_t col = 0; col < width; col++, input_ptr++, output_ptr++) {
                /* q3.12 with 2x factor folded into divisor (1<<11). */
                float two_x = (float)*input_ptr * (1.0f / 2048.0f);
                float e = expf(two_x);
                *output_ptr = axon_quantize_tanh((e - 1.0f) / (e + 1.0f));
            }
            input_ptr += input_extra_stride;
        }
    }
    return NRF_AXON_RESULT_SUCCESS;
}
