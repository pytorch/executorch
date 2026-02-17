/*
 * kernel_executors.h
 *
 *  Created on: Dec 8, 2025
 *      Author: Suraj Raut
 *
 *  Description:
 *      Header file declaring kernel-specific executor functions.
 *      Each kernel (7x7j2d1, 3x3j1d1, 3x3j2d1, 1x1j2d1, 1x1j1d1) has its own
 *      executor with exact DMA formulas matching convIdma.c reference.
 *      
 *      Non-VQ versions use per-tensor quantization (no outScale_ptr parameter).
 */

#ifndef KERNEL_EXECUTORS_H_
#define KERNEL_EXECUTORS_H_

#include "conv_layer_configs.h"

/* 
 * XAI error type: Use actual library type if available, otherwise define locally.
 * The actual xai_cnn_api.h should be included by implementation files.
 */
#ifndef XAI_ERR_TYPE
typedef int XAI_ERR_TYPE;
#define XAI_ERR_OK 0
#define XAI_ERR_BADARG 4
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execute 7x7 stride-2 convolution with DMA (per-tensor output scaling)
 */
XAI_ERR_TYPE conv_exec_7x7j2d1(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

/**
 * Execute 3x3 stride-1 convolution (standard ResNet 3x3 layers)
 */
XAI_ERR_TYPE conv_exec_3x3j1d1(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

/**
 * Execute 3x3 stride-2 convolution (downsampling layers)
 */
XAI_ERR_TYPE conv_exec_3x3j2d1(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

/**
 * Execute 1x1 stride-2 convolution (projection layers for downsampling)
 */
XAI_ERR_TYPE conv_exec_1x1j2d1(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

/**
 * Execute 1x1 stride-1 convolution (bottleneck layers)
 */
XAI_ERR_TYPE conv_exec_1x1j1d1(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

/*============================================================================
 * Cache-based executors (no DMA, uses processor cache)
 *============================================================================*/

XAI_ERR_TYPE conv_exec_7x7j2d1_cache(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

XAI_ERR_TYPE conv_exec_3x3j1d1_cache(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

XAI_ERR_TYPE conv_exec_3x3j2d1_cache(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

XAI_ERR_TYPE conv_exec_1x1j2d1_cache(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

XAI_ERR_TYPE conv_exec_1x1j1d1_cache(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

/**
 * Dispatch to appropriate kernel executor based on config->kernel_name
 */
XAI_ERR_TYPE conv_execute_kernel(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config);

#ifdef __cplusplus
}
#endif

#endif /* KERNEL_EXECUTORS_H_ */
