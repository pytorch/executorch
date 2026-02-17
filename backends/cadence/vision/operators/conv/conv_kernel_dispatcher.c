/*
 * conv_kernel_dispatcher.c
 *
 *  Created on: Dec 8, 2025
 *      Author: Suraj Raut
 *
 *  Description:
 *      Dispatcher that routes convolution execution to kernel-specific executors.
 *      Each kernel type has its own source file with exact DMA formulas from convIdma.c.
 */

#include "kernel_executors.h"
#include <stdio.h>
#include <string.h>

/**
 * Dispatch to appropriate kernel executor based on config->kernel_name
 */
XAI_ERR_TYPE conv_execute_kernel(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config)
{
    printf("  Dispatching to kernel: %s\n", config->kernel_name);

    // Dispatch to kernel-specific executor
    if (strcmp(config->kernel_name, "7x7j2d1") == 0) {
        return conv_exec_7x7j2d1(src, dst, coeff_ptr, bias_ptr, config);
    } else if(strcmp(config->kernel_name, "7x7j2d1_cache") == 0) {
        return conv_exec_7x7j2d1_cache(src, dst, coeff_ptr, bias_ptr, config);
    } else if (strcmp(config->kernel_name, "3x3j1d1") == 0) {
        return conv_exec_3x3j1d1(src, dst, coeff_ptr, bias_ptr, config);
    } else if (strcmp(config->kernel_name, "3x3j1d1_cache") == 0) {
        return conv_exec_3x3j1d1_cache(src, dst, coeff_ptr, bias_ptr, config);
    } else if (strcmp(config->kernel_name, "3x3j2d1") == 0) {
        return conv_exec_3x3j2d1(src, dst, coeff_ptr, bias_ptr, config);
    } else if (strcmp(config->kernel_name, "3x3j2d1_cache") == 0) {
        return conv_exec_3x3j2d1_cache(src, dst, coeff_ptr, bias_ptr, config);
    } else if (strcmp(config->kernel_name, "1x1j2d1") == 0) {
        return conv_exec_1x1j2d1(src, dst, coeff_ptr, bias_ptr, config);
    } else if (strcmp(config->kernel_name, "1x1j2d1_cache") == 0) {
        return conv_exec_1x1j2d1_cache(src, dst, coeff_ptr, bias_ptr, config);
    } else if (strcmp(config->kernel_name, "1x1j1d1") == 0) {
        return conv_exec_1x1j1d1(src, dst, coeff_ptr, bias_ptr, config);
    } else if (strcmp(config->kernel_name, "1x1j1d1_cache") == 0) {
        return conv_exec_1x1j1d1_cache(src, dst, coeff_ptr, bias_ptr, config);
    } else {
        printf("ERROR: Unknown kernel type: %s\n should fall back to generic executor" , config->kernel_name);
        return XAI_ERR_BADARG;
    }
}

