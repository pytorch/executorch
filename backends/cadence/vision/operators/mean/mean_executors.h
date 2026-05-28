/*
 * mean_executors.h
 *
 *  Created on: Apr 22, 2026
 *      Author: Suraj Raut
 *
 *  Description:
 *      Function declarations for DMA-tiled mean (adaptive_avg_pool2d) executors.
 *      Parallels maxpool/maxpool_executors.h.
 */

#ifndef MEAN_EXECUTORS_H_
#define MEAN_EXECUTORS_H_

#ifndef XAI_ERR_TYPE
typedef int XAI_ERR_TYPE;
#define XAI_ERR_OK 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execute mean pooling (adaptive_avg_pool2d) with DMA ping-pong tiling.
 *
 * Reduces [C x H x W] float32 to [C] by averaging all spatial elements.
 * Currently optimized for H=2, W=2 (calls simd_mean_pool_2x2_to_1x1_float32).
 *
 * Uses ping-pong DMA: prefetches next input chunk while computing on current.
 * Channel tiles are rounded to 16 for SIMD alignment.
 *
 * @param src        System-memory pointer to input  [C x H x W] float32
 * @param dst        System-memory pointer to output [C] float32
 * @param channels   Number of channels
 * @param spatial_h  Spatial height (must be 2 for optimized path)
 * @param spatial_w  Spatial width  (must be 2 for optimized path)
 * @return XAI_ERR_OK on success, -1 if buffers unavailable
 */
XAI_ERR_TYPE mean_exec_dma(
    const float* src,
    float* dst,
    int channels,
    int spatial_h,
    int spatial_w);

#ifdef __cplusplus
}
#endif

#endif /* MEAN_EXECUTORS_H_ */
