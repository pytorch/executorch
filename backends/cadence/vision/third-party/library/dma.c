/*
 * dma.c
 *
 *  Created on: Oct 30, 2025
 *      Author: sraut
 */

#include "lib.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>


#if MEASURE_DMA_CYCLES
#include <xtensa/config/system.h>
long long dma_start[2], dma_stop[2], dma_cycles[2];
double dma_size[2];
#endif

#ifdef IDMA_DEBUG
void xlog(const char *str) { DPRINT("**iDMAlib**: %s", str); }
#endif

IDMA_BUFFER_DEFINE(buffer_idma_ch0, 2 * CHL_MAX, IDMA_2D_DESC);
IDMA_BUFFER_DEFINE(buffer_idma_ch1, 2 * CHL_MAX, IDMA_2D_DESC);
IDMA_BUFFER_DEFINE(buffer_idma_ch3, 2 * CHL_MAX, IDMA_64B_DESC);

idma_buffer_t *  descbuf[] = {
    buffer_idma_ch0,
    buffer_idma_ch1,
};


// Pointers to DRAM buffers used by softmax
void *ptr_dram0 = (void *)dram0_pool;
void *ptr_dram1 = (void *)dram1_pool;



void err_cb_func(const idma_error_details_t *error) {
  printf(
      "ERROR CALLBACK: iDMA in Error, Error %d at desc:%p, PIF src/dst=%x/%x\n",
      error->err_type, (void *)error->currDesc, error->srcAddr, error->dstAddr);
}

////////////////////////////////////////////////////////////////////

#if (defined IDMA_USE_MULTICHANNEL)

void dma_3dm_init(int ch) {

#ifdef IDMA_DEBUG
  idma_log_handler(xlog);
#endif
  idma_init(ch, 0, MAX_BLOCK_16, 16, TICK_CYCLES_8, 100000, err_cb_func);
  idma_init_loop(ch, buffer_idma_ch3, IDMA_64B_DESC, CHL_MAX, NULL, NULL);
}
#endif

void dma_2dm_init(int ch) {
#ifdef IDMA_DEBUG
  idma_log_handler(xlog);
#endif
#if (defined IDMA_USE_MULTICHANNEL)
  idma_init(ch, 0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, err_cb_func);
  idma_init_loop(ch, descbuf[ch], IDMA_2D_DESC, CHL_MAX, NULL, NULL);
#else
  idma_init(0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, err_cb_func);
  idma_init_loop(buffer_idma_ch0, IDMA_2D_DESC, CHL_MAX, NULL, NULL);
#endif
}

void dma_3dm(int ch, void *src, void *dst, int src_row_pitch, int dst_row_pitch,
                            int src_tile_pitch, int dst_tile_pitch, int row_sz,
                            int nrows, int ntiles) {

#if MEASURE_DMA_CYCLES
  if (((int)src) > XSHAL_RAM_VADDR) {
    dma_start[0] = clock();
  } else {
    dma_start[1] = clock();
  }
#endif

  // printf("  DMA 3D Transfer: Src=%p, Dst=%p, RowSize=%d bytes, Rows=%d, "
	// 	 "Tiles=%d, SrcRowPitch=%d, DstRowPitch=%d, SrcTilePitch=%d, DstTilePitch=%d\n",
	// 	 src, dst, row_sz, nrows, ntiles, src_row_pitch, dst_row_pitch,
	// 	 src_tile_pitch, dst_tile_pitch);
  idma_copy_3d_desc64(ch, &dst, &src, DESC_IDMA_PRIOR_L /*Default*/, row_sz,
                      nrows, ntiles, src_row_pitch, dst_row_pitch,
                      src_tile_pitch, dst_tile_pitch);

#if MEASURE_DMA_CYCLES
  IDMA_WAIT();
  if (((int)src[0]) > XSHAL_RAM_VADDR) {
    dma_stop[0] = clock();
    dma_cycles[0] += (dma_stop[0] - dma_start[0]);
    dma_size[0] += nrows * row_sz * ntiles;
  } else {
    dma_stop[1] = clock();
    dma_cycles[1] += (dma_stop[1] - dma_start[1]);
    dma_size[1] += nrows * row_sz * ntiles;
  }
#endif
}


void dma_2dm(int ch,void *_psrc,void *_pdst, int src_stride, int dst_stride,
                            int num_bytes, short num_lines
                            ) {
//  int i;
//  dst_stride *= 4;
//  src_stride *= 4;
//  num_bytes  *= 4;

#if MEASURE_DMA_CYCLES
  if (((int)ParamS->psrc[0]) > XSHAL_RAM_VADDR) {
    dma_start[0] = clock();
  } else {
    dma_start[1] = clock();
  }
#endif

//  void *_psrc, *_pdst;
//  for (i = 0; i < numTransfers; i++) {
//    _psrc = ParamS->psrc[i];
//    _pdst = ParamS->pdst[i];
    // printf("  DMA 2D Transfer: Src=%p, Dst=%p, Size=%d bytes, Lines=%d, "
		//    "SrcStride=%d, DstStride=%d\n",
		//    _psrc, _pdst, num_bytes, num_lines, src_stride, dst_stride);
#if (defined IDMA_USE_MULTICHANNEL)
    (void)idma_copy_2d_desc(ch, _pdst, _psrc, num_bytes,
                      DESC_IDMA_PRIOR_L /*Default*/, num_lines, src_stride,
                      dst_stride);
#else
    idma_copy_2d_desc(_pdst, _psrc, num_bytes, DESC_IDMA_PRIOR_L /*Default*/,
                      num_lines, src_stride, dst_stride);
#endif


#if MEASURE_DMA_CYCLES
  IDMA_WAIT();
  if (((int)ParamS->psrc[0]) > XSHAL_RAM_VADDR) {
    dma_stop[0] = clock();
    dma_cycles[0] += (dma_stop[0] - dma_start[0]);
    dma_size[0] += numTransfers * num_lines * (double)num_bytes;
  } else {
    dma_stop[1] = clock();
    dma_cycles[1] 	+= (dma_stop[1] - dma_start[1]);
    dma_size[1] 	+= numTransfers * num_lines * (double)num_bytes;
  }
#endif
}

void dma_1dm(int ch,void *_psrc,void *_pdst, int num_bytes) {
//  void *_psrc, *_pdst;
//  _psrc = ParamS->psrc[0];
//  _pdst = ParamS->pdst[0];
//  num_bytes *= 4;

#if MEASURE_DMA_CYCLES
  if (((int)_psrc) > XSHAL_RAM_VADDR) {
    dma_start[0] = clock();
  } else {
    dma_start[1] = clock();
  }
#endif
  // printf("  DMA 1D Transfer: Src=%p, Dst=%p, Size=%d bytes\n", _psrc, _pdst,
	// 	 num_bytes);
#if (defined IDMA_USE_MULTICHANNEL)
  idma_copy_2d_desc(ch, _pdst, _psrc, num_bytes, DESC_IDMA_PRIOR_L /*Default*/,
                    1, 0, 0);
#else
  idma_copy_2d_desc(_pdst, _psrc, num_bytes, DESC_IDMA_PRIOR_L /*Default*/, 1,
                    0, 0);
#endif

#if MEASURE_DMA_CYCLES
  IDMA_WAIT();
  if (((int)_psrc) > XSHAL_RAM_VADDR) {
    dma_stop[0] = clock();
    dma_cycles[0] += (dma_stop[0] - dma_start[0]);
    dma_size[0] += (double)num_bytes;
  } else {
    dma_stop[1] = clock();
    dma_cycles[1] += (dma_stop[1] - dma_start[1]);
    dma_size[1] += (double)num_bytes;
  }
#endif
}

void dma_2dm_schd(int ch, int cnt) {
  printf("  Schedule DMA Channel %d\n", ch);
#if (defined IDMA_USE_MULTICHANNEL)
  idma_schedule_desc(ch, cnt);
#else
  idma_schedule_desc(cnt);
#endif
}



