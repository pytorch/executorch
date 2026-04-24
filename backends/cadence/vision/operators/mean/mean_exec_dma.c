/*
 * mean_exec_dma.c
 *
 *  Created on: Apr 22, 2026
 *      Author: Suraj Raut
 *
 *  Description:
 *      DMA-tiled mean pooling (adaptive_avg_pool2d) executor for float32.
 *
 *      Reduces [C x H x W] -> [C] by averaging all spatial elements.
 *      Currently optimized for H=2, W=2 via simd_mean_pool_2x2_to_1x1_float32.
 *
 *      Architecture mirrors maxpool_exec_mxnj2.c:
 *        - Ping-pong input/output buffers split across DRAM0 and DRAM1
 *        - Prefetch next input chunk via DMA while computing on current
 *        - Output DMA overlaps with next iteration's prefetch
 *
 *      Buffer layout (per DRAM bank):
 *        [  input chunk  |  output chunk  ]
 *        80/20 split: input = chunk_ch * spatial * 4B,
 *                     output = chunk_ch * 4B
 *
 *      Channel tile size is rounded to 16 for SIMD alignment.
 */

#include "mean_executors.h"
#include "memory_manager.h"
#include "dma.h"
#include <xtensa/hal.h>

/* SIMD mean kernel (in library) */
extern void simd_mean_pool_2x2_to_1x1_float32(
    float* restrict output,
    const float* restrict input,
    int N);

/* ---------------------------------------------------------------------- */
/* Helper: swap two float pointers                                         */
/* ---------------------------------------------------------------------- */
static inline void swap_ptrs(float** a, float** b)
{
    float* t = *a; *a = *b; *b = t;
}

/* ====================================================================== */
/* DMA-tiled mean executor with ping-pong                                  */
/* ====================================================================== */
XAI_ERR_TYPE mean_exec_dma(
    const float* src,
    float* dst,
    int channels,
    int spatial_h,
    int spatial_w)
{
    int spatial = spatial_h * spatial_w;  /* e.g. 4 for 2x2 */

    /* ================================================================== */
    /* Compute tiling: how many channels per chunk?                         */
    /*                                                                      */
    /* Each DRAM bank holds one ping or pong set:                           */
    /*   input_chunk  = chunk_ch * spatial * sizeof(float)                  */
    /*   output_chunk = chunk_ch * sizeof(float)                            */
    /*   total = chunk_ch * (spatial + 1) * 4                               */
    /*                                                                      */
    /* chunk_ch must be a multiple of 16 (SIMD processes 16 ch/iteration).  */
    /* ================================================================== */
    int bytes_per_ch = (spatial + 1) * (int)sizeof(float);
    int chunk_ch = IDMA_BUFFER_SIZE_DRAM0 / bytes_per_ch;
    chunk_ch = (chunk_ch / 16) * 16;  /* round down to SIMD multiple */

    if (chunk_ch < 16) {
        return (-1);  /* DRAM too small */
    }

    /* Cap to actual channel count (round up to multiple of 16 for last tile) */
    if (chunk_ch > channels) {
        chunk_ch = ((channels + 15) / 16) * 16;
    }

    int inp_chunk_bytes = chunk_ch * spatial * (int)sizeof(float);
    int out_chunk_bytes = chunk_ch * (int)sizeof(float);

    /* ================================================================== */
    /* Buffer allocation: ping in DRAM0, pong in DRAM1                      */
    /* Each bank: [ input_chunk | output_chunk ]                            */
    /* ================================================================== */
    float* inp_ping = (float*)dram0_pool;
    float* out_ping = (float*)(dram0_pool + inp_chunk_bytes);
    float* inp_pong = (float*)dram1_pool;
    float* out_pong = (float*)(dram1_pool + inp_chunk_bytes);

    /* ================================================================== */
    /* Initialise DMA engines                                               */
    /* ================================================================== */
    dma_2dm_init(0);   /* ch0: output writeback */
    dma_2dm_init(1);   /* ch1: input prefetch   */

    /* ================================================================== */
    /* Load first input chunk (serial — no overlap possible)                */
    /* ================================================================== */
    int ch_done = 0;
    int cur_ch = (channels - ch_done > chunk_ch)
                 ? chunk_ch
                 : channels - ch_done;
    int cur_inp_bytes = cur_ch * spatial * (int)sizeof(float);

    dma_1dm(1, (void*)&src[ch_done * spatial], (void*)inp_ping, cur_inp_bytes);
    idma_hw_wait_all(1);

    /* ================================================================== */
    /* Tiled execution loop with ping-pong                                  */
    /* ================================================================== */
    float* p_inp_cur  = inp_ping;
    float* p_out_cur  = out_ping;
    float* p_inp_next = inp_pong;
    float* p_out_next = out_pong;

    while (ch_done < channels) {
        int this_ch = cur_ch;
        int next_ch_start = ch_done + this_ch;
        int have_next = (next_ch_start < channels);

        /* ============================================================== */
        /* Prefetch next input chunk into pong buffer (async)               */
        /* ============================================================== */
        int next_ch = 0;
        if (have_next) {
            next_ch = (channels - next_ch_start > chunk_ch)
                      ? chunk_ch
                      : channels - next_ch_start;
            int next_inp_bytes = next_ch * spatial * (int)sizeof(float);

            dma_1dm(1, (void*)&src[next_ch_start * spatial],
                    (void*)p_inp_next, next_inp_bytes);
            /* DMA runs in background while we compute below */
        }

        /* ============================================================== */
        /* Execute SIMD mean on current chunk                               */
        /* ============================================================== */
        simd_mean_pool_2x2_to_1x1_float32(
            p_out_cur,
            p_inp_cur,
            this_ch * spatial);

        /* ============================================================== */
        /* Write output chunk to system memory (async)                      */
        /* ============================================================== */
        int cur_out_bytes = this_ch * (int)sizeof(float);
        dma_1dm(0, (void*)p_out_cur,
                (void*)&dst[ch_done], cur_out_bytes);

        /* ============================================================== */
        /* Wait for input DMA of next tile to finish                        */
        /* (In a well-tuned pipeline, DMA finishes during compute above)    */
        /* ============================================================== */
        if (have_next) {
            idma_hw_wait_all(1);
        }

        /* Wait for output DMA before reusing this buffer as next pong */
        idma_hw_wait_all(0);

        /* Advance */
        ch_done = next_ch_start;
        cur_ch  = next_ch;

        /* Swap ping-pong: current pong becomes next ping */
        swap_ptrs(&p_inp_cur,  &p_inp_next);
        swap_ptrs(&p_out_cur,  &p_out_next);
    }

    return XAI_ERR_OK;
}
