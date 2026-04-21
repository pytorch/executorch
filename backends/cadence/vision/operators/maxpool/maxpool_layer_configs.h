/*
 * maxpool_layer_configs.h
 *
 * Auto-generated maxpool layer configurations
 * Generated from model layer extraction
 *
 * DO NOT EDIT MANUALLY - Regenerate with generate_maxpool_configs.py
 */

#ifndef MAXPOOL_LAYER_CONFIGS_H
#define MAXPOOL_LAYER_CONFIGS_H

#include <stdint.h>
#include <stddef.h>  /* for NULL */

/**
 * Runtime configuration for a DMA-tiled maxpool layer.
 *
 * Data format : float32, NCHW layout (batch processed externally).
 * Maxpool is channel-wise independent: no weights, bias, or outscale.
 *
 * Tiling strategy (parallel to conv tiling):
 *   C-tiles : group channels together
 *   H-tiles : group output rows together
 *   Ping-pong on both input and output buffers.
 *
 * Buffer sizes are in BYTES (float32 = 4 bytes per element).
 * Dimension/pitch fields are in float32 ELEMENTS unless noted.
 */
typedef struct {
    /* Layer identification */
    int layer_id;
    const char* layer_name;
    const char* config_key;     /* "C_H_W_kh_kw_sh_sw_ph_pw" */

    /* Source dimensions (system memory, float32 elements, per batch) */
    int src_width;              /* W  */
    int src_height;             /* H  */
    int channels;               /* C  (= output channels) */

    /* Destination dimensions */
    int dst_width;              /* out W = (W + 2*pad_w - kernel_w) / stride_w + 1 */
    int dst_height;             /* out H = (H + 2*pad_h - kernel_h) / stride_h + 1 */

    /* System memory pitches (float32 elements) */
    int src_row_pitch;          /* = src_width  (contiguous rows)  */
    int src_plane_pitch;        /* = src_height * src_width        */
    int dst_row_pitch;          /* = dst_width                     */
    int dst_plane_pitch;        /* = dst_height * dst_width        */

    /* Maxpool parameters */
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;

    /* Local-memory tile dimensions (float32 elements) */
    int in_tile_w;              /* padded width  = src_width + 2*pad_w      */
    int in_tile_rows;           /* padded height = input_rows + 2*pad_h     */
    int in_tile_plane;          /* in_tile_w * in_tile_rows                 */
    int in_data_offset;         /* pad_h * in_tile_w + pad_w  (elements)    */
    int out_tile_w;             /* = dst_width                              */
    int out_tile_rows;          /* = output_rows                            */
    int out_tile_plane;         /* dst_width * output_rows                  */

    /* Tiling parameters */
    int c_tile_size;            /* Channels per C-tile                      */
    int c_tiles;                /* Number of C-tiles                        */
    int c_tile_size_last;       /* Channels in last C-tile                  */
    int height_tiles;           /* Number of H-tiles                        */
    int output_rows;            /* Output rows per H-tile                   */
    int input_rows;             /* Input rows per H-tile (before padding)   */

    /* Buffer sizes (bytes) */
    int input_buffer_size;      /* c_tile_size * in_tile_plane * 4          */
    int output_buffer_size;     /* c_tile_size * out_tile_plane * 4         */

    /* DRAM placement (0 = DRAM0, 1 = DRAM1) */
    int input_ping_dram;
    int input_pong_dram;
    int output_ping_dram;
    int output_pong_dram;

} maxpool_layer_config_t;

/* ======================================================================== */
/* Generated configurations                                                  */
/* ======================================================================== */

/* ResNet-18 (64x64 input): maxpool 32x32x64 -> 16x16x64, kernel=2, stride=2, pad=0 */
#define NUM_MAXPOOL_LAYERS 1

static const maxpool_layer_config_t MAXPOOL_LAYER_CONFIGS[] = {
    {
        .layer_id = 0,
        .layer_name = "maxpool_2x2s2_c64",
        .config_key = "64_32_32_2_2_2_2_0_0",

        /* Source: 32x32x64 */
        .src_width      = 32,
        .src_height     = 32,
        .channels       = 64,

        /* Destination: 16x16x64 */
        .dst_width      = 16,
        .dst_height     = 16,

        /* System memory pitches */
        .src_row_pitch   = 32,
        .src_plane_pitch = 1024,   /* 32*32 */
        .dst_row_pitch   = 16,
        .dst_plane_pitch = 256,    /* 16*16 */

        /* Maxpool params */
        .kernel_h = 2,
        .kernel_w = 2,
        .stride_h = 2,
        .stride_w = 2,
        .pad_h    = 0,
        .pad_w    = 0,

        /* Local tile dims (float elements) */
        .in_tile_w      = 32,     /* src_width + 2*pad_w */
        .in_tile_rows   = 6,      /* input_rows + 2*pad_h = 6 */
        .in_tile_plane  = 192,    /* 32 * 6 */
        .in_data_offset = 0,      /* pad=0 */
        .out_tile_w     = 16,
        .out_tile_rows  = 3,
        .out_tile_plane = 48,     /* 16 * 3 */

        /* Tiling: 64 ch/tile x 1 tile, 3 rows/tile x 6 tiles */
        .c_tile_size      = 64,
        .c_tiles          = 1,
        .c_tile_size_last = 64,
        .height_tiles     = 6,    /* ceil(16/3) */
        .output_rows      = 3,
        .input_rows       = 6,    /* 3 * 2 */

        /* Buffer sizes (bytes, float32) */
        .input_buffer_size  = 49152,   /* 64 * 192 * 4 */
        .output_buffer_size = 12288,   /* 64 * 48 * 4  */

        /* DRAM placement: cross-bank ping-pong */
        .input_ping_dram  = 0,
        .input_pong_dram  = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 0,
    },
};

/* ======================================================================== */
/* Accessor helpers                                                          */
/* ======================================================================== */

static inline int get_num_maxpool_layers(void) {
    return NUM_MAXPOOL_LAYERS;
}

static inline const maxpool_layer_config_t* get_maxpool_config(int layer_id) {
    if (layer_id < 0 || layer_id >= NUM_MAXPOOL_LAYERS) return NULL;
    return &MAXPOOL_LAYER_CONFIGS[layer_id];
}

/**
 * Look up a maxpool config by runtime parameters.
 *
 * @return Pointer to matching config, or NULL if not found.
 */
static inline const maxpool_layer_config_t* get_maxpool_config_by_params(
    int channels, int src_height, int src_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    for (int i = 0; i < NUM_MAXPOOL_LAYERS; i++) {
        const maxpool_layer_config_t* c = &MAXPOOL_LAYER_CONFIGS[i];
        if (c->channels   == channels   &&
            c->src_height == src_height &&
            c->src_width  == src_width  &&
            c->kernel_h   == kernel_h   &&
            c->kernel_w   == kernel_w   &&
            c->stride_h   == stride_h   &&
            c->stride_w   == stride_w   &&
            c->pad_h      == pad_h      &&
            c->pad_w      == pad_w)
        {
            return c;
        }
    }
    return NULL;
}

#endif /* MAXPOOL_LAYER_CONFIGS_H */
