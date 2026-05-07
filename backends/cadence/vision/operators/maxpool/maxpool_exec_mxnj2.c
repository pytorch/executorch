/*
 * maxpool_exec_mxnj2.c
 *
 *  Created on: Apr 21, 2026
 *      Author: Suraj Raut
 *
 *  Description:
 *      DMA-tiled maxpool executor for float32 NCHW data.
 *      Supports arbitrary kernel sizes (2x2, 3x3, …), strides, and padding.
 *
 *      Architecture mirrors conv_exec_3x3j2d1.c:
 *        - Ping-pong input/output buffers in local DRAM
 *        - 3D DMA for input prefetch (ch1), 2D DMA for output writeback (ch0)
 *        - C-tile x H-tile nested loop
 *
 *      Key difference from conv:  maxpool has NO coefficients, bias, or
 *      outscale.  Channels are independent, so C-tiling replaces N-tiling.
 *
 *      DMA tiling with overlap handling:
 *        When kernel_h > stride_h (e.g. 3x3/s2), consecutive height tiles
 *        share (kernel_h - stride_h) rows of input.  The source-row start
 *        for tile h is:  h * output_rows * stride_h - pad_h  (clamped to 0).
 *        Top/bottom padding rows are supplied by a MIN_FLT32 pre-fill so
 *        the kernel's ky loop reads through them naturally.
 *
 *      The optimized SIMD kernel maxpool2d_j2x2_f32() is called per-channel.
 */

#include "maxpool_executors.h"
#include "memory_manager.h"
#include "dma.h"
#include <string.h>
#include <xtensa/hal.h>

/* Minimal float definitions to avoid pulling in full math.h */
#ifndef MIN_FLT32
#define MIN_FLT32 (-3.402823466e+38F)
#endif

/* HW-optimised maxpool kernel (in library) */
extern void maxpool2d_j2x2_f32(
    float* restrict ptr_out,
    const float* restrict ptr_inp,
    int inp_height, int inp_width,
    int out_height, int out_width,
    int in_pitch_width, int in_pitch_height,
    int out_pitch_width, int out_pitch_height,
    unsigned char kernel_height,
    unsigned char kernel_width);

/* ---------------------------------------------------------------------- */
/* Helper: fill a float buffer with a constant value (e.g. MIN_FLT32)     */
/* ---------------------------------------------------------------------- */
static void fill_buffer_f32(float* buf, float val, int count)
{
    for (int i = 0; i < count; i++) {
        buf[i] = val;
    }
}

/* ---------------------------------------------------------------------- */
/* Helper: swap two pointers                                               */
/* ---------------------------------------------------------------------- */
static inline void swap_f32_ptrs(float** a, float** b)
{
    float* t = *a;
    *a = *b;
    *b = t;
}

/* ====================================================================== */
/* DMA-tiled executor                                                      */
/* ====================================================================== */
XAI_ERR_TYPE maxpool_exec_mxnj2(
    float* src,
    float* dst,
    const maxpool_layer_config_t* config)
{
    /* ================================================================== */
    /* SECTION 1: DRAM Buffer Allocation                                   */
    /* ================================================================== */
    int dram0_used = 0;
    int dram1_used = 0;

    int8_t* raw_in0 = allocate_dram_buffer(config->input_buffer_size,
                                            config->input_ping_dram,
                                            &dram0_used, &dram1_used);
    int8_t* raw_in1 = allocate_dram_buffer(config->input_buffer_size,
                                            config->input_pong_dram,
                                            &dram0_used, &dram1_used);
    int8_t* raw_out0 = allocate_dram_buffer(config->output_buffer_size,
                                             config->output_ping_dram,
                                             &dram0_used, &dram1_used);
    int8_t* raw_out1 = allocate_dram_buffer(config->output_buffer_size,
                                             config->output_pong_dram,
                                             &dram0_used, &dram1_used);

    if (!raw_in0 || !raw_in1 || !raw_out0 || !raw_out1) {
        return (-1);
    }

    /* Cast to float pointers for kernel calls */
    float* p_input0  = (float*)raw_in0;
    float* p_input1  = (float*)raw_in1;
    float* p_output0 = (float*)raw_out0;
    float* p_output1 = (float*)raw_out1;

    /* ================================================================== */
    /* SECTION 2: Initialise DMA engines                                   */
    /* ================================================================== */
    dma_3dm_init(1);   /* ch1: 3D input prefetch  */
    dma_2dm_init(0);   /* ch0: 2D output writeback */

    /* ================================================================== */
    /* SECTION 3: Load first input tile                                    */
    /* ================================================================== */
    /*
     * The first tile starts at source row 0.  For kernels with pad_h > 0
     * the buffer is pre-filled with MIN_FLT32 (identity for max) and
     * data is placed at in_data_offset = pad_h*in_tile_w + pad_w, so the
     * leading MIN_FLT32 rows/columns act as top/left padding.
     *
     * For subsequent tiles the DMA offset is recomputed per-tile to
     * account for kernel overlap (kernel_h > stride_h).
     */
    fill_buffer_f32(p_input0, MIN_FLT32,
                    config->c_tile_size * config->in_tile_plane);

    /*
     * Compute actual source rows for tile 0.
     * Conceptual first input row = 0*stride_h - pad_h = -pad_h.
     * top_pad rows are supplied by the MIN_FLT32 fill.
     */
    int first_in_end = (config->output_rows - 1) * config->stride_h
                       - config->pad_h + config->kernel_h - 1;
    int first_load_rows = (first_in_end >= config->src_height
                           ? config->src_height - 1 : first_in_end)
                          - 0 + 1;  /* src starts at row 0 */

    /* First DMA: c_tile_size planes, first_load_rows rows each */
    dma_3dm(1,
            /* src */            (void*)src,
            /* dst */            (void*)&p_input0[config->in_data_offset],
            /* src_row_pitch */  config->src_width  * (int)sizeof(float),
            /* dst_row_pitch */  config->in_tile_w  * (int)sizeof(float),
            /* src_tile_pitch */ config->src_plane_pitch * (int)sizeof(float),
            /* dst_tile_pitch */ config->in_tile_plane   * (int)sizeof(float),
            /* row_sz */         config->src_width * (int)sizeof(float),
            /* nrows */          first_load_rows,
            /* ntiles */         config->c_tile_size);

    idma_hw_wait_all(1);  /* input ready */

    /* ================================================================== */
    /* SECTION 4: Tiled Execution Loop  (C-tiles x H-tiles)                */
    /* ================================================================== */
    int last_tile = 1;

    for (int idx_c = 0; idx_c < config->c_tiles; idx_c++) {
        int last_c_tile = (last_tile) && (idx_c == config->c_tiles - 1);
        int current_c = (idx_c < config->c_tiles - 1)
                         ? config->c_tile_size
                         : config->c_tile_size_last;

        for (int idx_h = 0; idx_h < config->height_tiles; idx_h++) {
            int last_h_tile = (last_c_tile) &&
                              (idx_h == config->height_tiles - 1);

            /* Output rows for this tile (last tile may be shorter) */
            int cur_out_rows = (idx_h < config->height_tiles - 1)
                               ? config->output_rows
                               : (config->dst_height -
                                  config->output_rows * idx_h);
            int cur_in_rows  = cur_out_rows * config->stride_h;

            /* ========================================================== */
            /* Prefetch next input tile into pong buffer                    */
            /* ========================================================== */
            if (!last_h_tile) {
                /* Determine next (c, h) indices */
                int next_c = idx_c;
                int next_h = idx_h + 1;
                if (next_h >= config->height_tiles) {
                    next_h = 0;
                    next_c = idx_c + 1;
                }
                int next_c_start = config->c_tile_size * next_c;
                int next_c_size  = (next_c < config->c_tiles - 1)
                                    ? config->c_tile_size
                                    : config->c_tile_size_last;

                /*
                 * Compute source-row start, load count, and DMA
                 * destination offset for the next height tile.
                 *
                 * For kernel_h > stride_h (e.g. 3x3/s2) consecutive
                 * tiles overlap in the source by (kernel_h - stride_h)
                 * rows, so the stride between tiles in source space is
                 * output_rows * stride_h, NOT input_rows.
                 */
                int next_out_start = config->output_rows * next_h;
                int next_in_first  = next_out_start * config->stride_h
                                     - config->pad_h;
                int next_top_pad   = (next_in_first < 0)
                                     ? -next_in_first : 0;
                int next_src_row   = next_in_first + next_top_pad;

                int next_actual_out =
                    (next_h < config->height_tiles - 1)
                    ? config->output_rows
                    : (config->dst_height - next_out_start);
                int next_in_last =
                    (next_out_start + next_actual_out - 1)
                    * config->stride_h
                    - config->pad_h + config->kernel_h - 1;
                int next_in_end_clamped =
                    (next_in_last >= config->src_height)
                    ? config->src_height - 1
                    : next_in_last;
                int next_load_rows = next_in_end_clamped
                                     - next_src_row + 1;

                /* DMA offset: top_pad rows of MIN_FLT32 + left pad */
                int next_dma_offset = next_top_pad * config->in_tile_w
                                      + config->pad_w;

                fill_buffer_f32(p_input1, MIN_FLT32,
                                next_c_size * config->in_tile_plane);

                dma_3dm(1,
                    /* src */
                    (void*)&src[next_c_start * config->src_plane_pitch +
                                next_src_row * config->src_width],
                    /* dst */
                    (void*)&p_input1[next_dma_offset],
                    /* src_row_pitch */
                    config->src_width * (int)sizeof(float),
                    /* dst_row_pitch */
                    config->in_tile_w * (int)sizeof(float),
                    /* src_tile_pitch */
                    config->src_plane_pitch * (int)sizeof(float),
                    /* dst_tile_pitch */
                    config->in_tile_plane * (int)sizeof(float),
                    /* row_sz */
                    config->src_width * (int)sizeof(float),
                    /* nrows */
                    next_load_rows,
                    /* ntiles */
                    next_c_size);
            }

            /* ========================================================== */
            /* Execute maxpool on current input tile                        */
            /* ========================================================== */
            for (int c = 0; c < current_c; c++) {
                /*
                 * Pass the kernel a pointer to the START of the
                 * padded tile (row 0, col 0 of the buffer).  The
                 * MIN_FLT32 fill provides top/left/right/bottom
                 * padding; the kernel reads through them naturally
                 * via its ky/kx loops.
                 *
                 * NOTE: the old code added in_data_offset here,
                 * which skipped past the padding and produced wrong
                 * results for any kernel with pad_h or pad_w > 0.
                 */
                float* in_plane  = &p_input0[c * config->in_tile_plane];
                float* out_plane = &p_output1[c * config->out_tile_plane];

                maxpool2d_j2x2_f32(
                    out_plane,
                    in_plane,
                    cur_in_rows,            /* inp_height    */
                    config->src_width,      /* inp_width     */
                    cur_out_rows,           /* out_height    */
                    config->dst_width,      /* out_width     */
                    config->in_tile_w,      /* in_pitch_width  */
                    config->in_tile_plane,  /* in_pitch_height */
                    config->dst_width,      /* out_pitch_width */
                    config->out_tile_plane, /* out_pitch_height*/
                    (unsigned char)config->kernel_h,
                    (unsigned char)config->kernel_w);
            }

            /* ========================================================== */
            /* Write output tile back to system memory via 2D DMA          */
            /* ========================================================== */
            {
                int c_start  = config->c_tile_size * idx_c;
                int h_out_start = config->output_rows * idx_h;
                int row_bytes = config->dst_width * cur_out_rows
                                * (int)sizeof(float);

                dma_2dm(0,
                    /* src */        (void*)p_output1,
                    /* dst */        (void*)&dst[c_start *
                                        config->dst_plane_pitch +
                                        h_out_start * config->dst_width],
                    /* src_stride */ config->out_tile_plane *
                                        (int)sizeof(float),
                    /* dst_stride */ config->dst_plane_pitch *
                                        (int)sizeof(float),
                    /* row_size */   row_bytes,
                    /* num_lines */  (short)current_c);
            }

            /* Swap ping-pong buffers */
            swap_f32_ptrs(&p_output0, &p_output1);
            swap_f32_ptrs(&p_input0,  &p_input1);
        }
    }

    /* Wait for last output DMA before returning */
    idma_hw_wait_all(0);

    return XAI_ERR_OK;
}

/* ====================================================================== */
/* Cache-mode fallback (no DMA, data accessed via processor cache)         */
/* ====================================================================== */
XAI_ERR_TYPE maxpool_exec_mxnj2_no_dma(
    float* src,
    float* dst,
    const maxpool_layer_config_t* config)
{
    int padded_w = config->src_width  + 2 * config->pad_w;
    int padded_h = config->src_height + 2 * config->pad_h;
    int plane_size = padded_w * padded_h;
    int total_size = plane_size * config->channels;

    /* Use shared padded-input scratch buffer from memory manager */
    int8_t* raw_buf = get_cache_padded_input();
    if (total_size * (int)sizeof(float) > (int)get_cache_padded_input_size()) {
        return (-1);   /* buffer too small */
    }
    float* padded = (float*)raw_buf;

    /* Fill with MIN_FLT32 (identity for max) */
    fill_buffer_f32(padded, MIN_FLT32, total_size);

    /* Copy source data into padded buffer at correct offset */
    int data_off = config->pad_h * padded_w + config->pad_w;

    for (int c = 0; c < config->channels; c++) {
        for (int h = 0; h < config->src_height; h++) {
            memcpy(&padded[c * plane_size + data_off + h * padded_w],
                   &src[c * config->src_plane_pitch + h * config->src_width],
                   config->src_width * sizeof(float));
        }
    }

    /* Run maxpool per channel plane.
     * Pass the pointer at the START of the padded buffer (row 0, col 0)
     * so the kernel's ky/kx loops read through the MIN_FLT32 padding. */
    for (int c = 0; c < config->channels; c++) {
        float* in_plane  = &padded[c * plane_size];
        float* out_plane = &dst[c * config->dst_plane_pitch];

        maxpool2d_j2x2_f32(
            out_plane,
            in_plane,
            config->src_height,
            config->src_width,
            config->dst_height,
            config->dst_width,
            padded_w,
            plane_size,
            config->dst_width,
            config->dst_plane_pitch,
            (unsigned char)config->kernel_h,
            (unsigned char)config->kernel_w);
    }

    /* Writeback output from cache */
    xthal_dcache_region_writeback(dst,
        config->dst_plane_pitch * config->channels * (int)sizeof(float));

    return XAI_ERR_OK;
}
