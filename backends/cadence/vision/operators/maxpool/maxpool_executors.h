/*
 * maxpool_executors.h
 *
 *  Created on: Apr 21, 2026
 *      Author: Suraj Raut
 *
 *  Description:
 *      Function declarations for DMA-tiled maxpool executors.
 *      Parallels conv/kernel_executors.h for the maxpool operator.
 */

#ifndef MAXPOOL_EXECUTORS_H_
#define MAXPOOL_EXECUTORS_H_

#include "../layer_configs.h"

#ifndef XAI_ERR_TYPE
typedef int XAI_ERR_TYPE;
#define XAI_ERR_OK 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execute MxN stride-2 maxpool with DMA tiling.
 *
 * Operates on float32 data in NCHW layout (one batch at a time).
 * Uses ping-pong DMA transfers on local DRAM for overlap of
 * DMA and computation.
 *
 * @param src   System-memory pointer to input  [C x H x W] float32
 * @param dst   System-memory pointer to output [C x OH x OW] float32
 * @param config  Pre-computed layer configuration (buffer sizes, tiling, etc.)
 * @return XAI_ERR_OK on success
 */
XAI_ERR_TYPE maxpool_exec_mxnj2(
    float* src,
    float* dst,
    const maxpool_layer_config_t* config);

/**
 * Execute MxN stride-2 maxpool without DMA (data accessed via processor cache).
 * Fallback path when DRAM buffers are not available.
 *
 * @param src   System-memory pointer to input  [C x H x W] float32
 * @param dst   System-memory pointer to output [C x OH x OW] float32
 * @param config  Layer configuration (only dimension fields used)
 * @return XAI_ERR_OK on success
 */
XAI_ERR_TYPE maxpool_exec_mxnj2_no_dma(
    float* src,
    float* dst,
    const maxpool_layer_config_t* config);

#ifdef __cplusplus
}
#endif

#endif /* MAXPOOL_EXECUTORS_H_ */
