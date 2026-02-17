/*
 * dma.h
 *
 *  Created on: Oct 30, 2025
 *      Author: sraut
 */

#ifndef __DMA_H__
#define __DMA_H__

// Enable DMA for cache-mode input copy (instead of xaiCopyTile3D)
// NOTE: Requires AXI-to-AXI DMA support on the target core
// Uncomment to use DMA 3D transfer in cache executors
// #define USE_DMA_FOR_CACHE_COPY

#define IDMA_USE_INTR 0
#define IDMA_USE_MULTICHANNEL 1
#define CHL_MAX 2
#include <xtensa/hal.h>
#include <xtensa/idma.h>

#ifdef __cplusplus
extern "C" {
#endif

// DMA initialization functions
 void dma_2dm_init(int ch);
 void dma_3dm_init(int ch);

//
//// DMA transfer functions
//void dma_1dm(int ch, void *_psrc, void *_pdst, int num_bytes);
  void dma_1dm(int ch,void *_psrc,void *_pdst, int num_bytes);
void dma_2dm(int ch, void *_psrc, void *_pdst, int src_stride, int dst_stride,
             int num_bytes, short num_lines);
  void dma_3dm(int ch, void *src, void *dst, int src_row_pitch, int dst_row_pitch,
                              int src_tile_pitch, int dst_tile_pitch, int row_sz,
                              int nrows, int ntiles) ;
////
//// DMA scheduling
//void dma_2dm_schd(int ch, int cnt);

#ifdef __cplusplus
}
#endif

#endif /* __DMA_H__ */
