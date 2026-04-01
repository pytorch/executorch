/*
 * dma.c
 *
 *  Created on: Oct 30, 2025
 *      Author: sraut
 */

#include "lib.h"

// We assume that the DSP uses multichannel IDMA with 2 channels available for 2D transfers (e.g., ping-pong buffers)
// and 1 channel for 3D transfers.

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
  (void) error;
}

void dma_3dm_init(int ch) {
  idma_init(ch, 0, MAX_BLOCK_16, 16, TICK_CYCLES_8, 100000, err_cb_func);
  idma_init_loop(ch, buffer_idma_ch3, IDMA_64B_DESC, CHL_MAX, NULL, NULL);
}

void dma_2dm_init(int ch) {
  idma_init(ch, 0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, err_cb_func);
  idma_init_loop(ch, descbuf[ch], IDMA_2D_DESC, CHL_MAX, NULL, NULL);
}

void dma_3dm(int ch, void *src, void *dst, int src_row_pitch, int dst_row_pitch,
            int src_tile_pitch, int dst_tile_pitch, int row_sz,
            int nrows, int ntiles) {
  (void) idma_copy_3d_desc64(ch, &dst, &src, DESC_IDMA_PRIOR_L /*Default*/, row_sz,
                          nrows, ntiles, src_row_pitch, dst_row_pitch,
                          src_tile_pitch, dst_tile_pitch);
}


void dma_2dm(int ch,void *_psrc,void *_pdst, int src_stride, int dst_stride,
            int num_bytes, short num_lines) {
  (void) idma_copy_2d_desc(ch, _pdst, _psrc, num_bytes,
                          DESC_IDMA_PRIOR_L /*Default*/, num_lines, src_stride,
                          dst_stride);
}

void dma_1dm(int ch,void *_psrc,void *_pdst, int num_bytes) {
  (void) idma_copy_2d_desc(ch, _pdst, _psrc, num_bytes, DESC_IDMA_PRIOR_L /*Default*/,
                          1, 0, 0);
}



