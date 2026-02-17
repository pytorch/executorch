/*
 * conv_layer_configs.h
 *
 * Auto-generated convolution layer configurations
 * Generated from model layer extraction
 *
 * DO NOT EDIT MANUALLY - Regenerate with generate_layer_configs.py
 */

#ifndef CONV_LAYER_CONFIGS_H
#define CONV_LAYER_CONFIGS_H

#include <stdint.h>
#include <stddef.h>  // for NULL

/**
 * Runtime configuration for a single convolution layer
 * Contains all parameters needed to execute the layer
 * Matches convIdma_buffers.h schema
 */
typedef struct {
    // Layer identification
    int layer_id;
    const char* layer_name;
    const char* kernel_name;
    const char* config_key;     // Unique key: ic_ih_iw_oc_kh_kw_oh_ow_sy_sx_pad_dil
    
    // Source (DRAM) dimensions
    int src_dim1_size;      // Input width in DRAM
    int src_dim2_size;      // Input height in DRAM
    int src_dim3_size;      // Input channels in DRAM
    int src_dim1_pitch;     // DRAM row pitch
    int src_dim2_pitch;     // DRAM plane pitch
    
    // Destination (DRAM) dimensions
    int dst_dim1_size;      // Output width in DRAM
    int dst_dim2_size;      // Output height in DRAM
    int dst_dim3_size;      // Output channels in DRAM
    int dst_dim1_pitch;     // DRAM row pitch
    int dst_dim2_pitch;     // DRAM plane pitch
    
    // Input tile (local memory) dimensions
    int in_dim1_size;       // Tile width (with padding)
    int in_dim1_pitch;      // Tile row pitch
    int in_dim2_size;       // Tile height (rows per iteration)
    int in_dim2_pitch;      // Tile plane pitch
    int in_dim1_edge1;      // Left padding
    int in_dim1_edge2;      // Right padding
    int in_dim2_edge1;      // Top padding
    int in_dim2_edge2;      // Bottom padding
    int in_dim3_edge1;      // Channel padding (usually 0)
    int in_dim3_edge2;      // Channel padding (usually 0)
    int in_data_offset;     // Offset to actual data in buffer
    int in_rows_firstdma;   // Rows to transfer in first DMA
    
    // Output tile (local memory) dimensions
    int out_dim1_size;      // Output width
    int out_dim1_pitch;     // Output row pitch
    int out_dim2_size;      // Output rows per iteration
    int out_dim2_pitch;     // Output plane pitch
    int out_dim3_size;      // Output channels per N-tile
    
    // Coefficient tile dimensions
    int coeff_dim1_size;    // Kernel width
    int coeff_dim2_size;    // Kernel height
    int coeff_dim3_size;    // Input channels
    int coeff_dim4_size;    // Output channels (total)
    int coeff_dim1_pitch;   // Kernel row pitch
    int coeff_dim2_pitch;   // Kernel plane pitch (W*H)
    int coeff_dim3_pitch;   // Kernel 3D pitch (W*H*D)
    
    // Bias array dimensions
    int bias_dim1_size;     // Number of bias values
    int bias_dim2_size;     // Always 1
    
    // Output scale array dimensions
    int outscale_dim1_size; // Number of scale values
    int outscale_dim2_size; // Always 1
    
    // Buffer sizes (bytes)
    int input_buffer_size;
    int coeff_buffer_size;
    int output_buffer_size;
    int bias_buffer_size;
    int outscale_buffer_size;
    
    // Buffer DRAM placement (0 = DRAM0, 1 = DRAM1)
    int input_ping_dram;
    int input_pong_dram;
    int coeff_dram;
    int output_ping_dram;
    int output_pong_dram;
    int bias_dram;
    int outscale_dram;
    
    // Tiling parameters
    int n_tile_size;        // Output channels per N-tile
    int n_tiles;            // Total number of N-tiles
    int n_tile_size_last;   // Channels in last N-tile
    int height_tiles;       // Total number of H-tiles
    int output_rows;        // Output rows per H-tile
    int input_rows;         // Input rows needed per H-tile
    
    // Convolution parameters
    int kernel_w;
    int kernel_h;
    int stride_x;
    int stride_y;
    int padding;            // Symmetric padding
    int dilation;
    int accum_shift;        // Accumulator shift
    int relu_max;           // ReLU clamp maximum
    int relu_min;           // ReLU clamp minimum
    int output_shift;       // Output quantization shift
    int output_scale;       // Output scale factor
    int flags;              // Convolution flags
    int input_zero_point;   // Input zero-point for padding fill (set at runtime)
    
} conv_layer_config_t;

// Total number of convolution layers
#define NUM_CONV_LAYERS 29

 #define IDMA_BUFFER_SIZE_DRAM0 (32768) // 32 KB for DRAM0
 #define IDMA_BUFFER_SIZE_DRAM1 (32768) // 32 KB for DRAM1

static const conv_layer_config_t CONV_LAYER_CONFIGS[] = {
    {
        .layer_id = 0,
        .layer_name = "conv1",
        .kernel_name = "7x7j2d1",
        .config_key = "3_64_64_64_7_7_32_32_2_2_3_1",
        
        // Source (DRAM): 64×64×3
        .src_dim1_size = 64,
        .src_dim2_size = 64,
        .src_dim3_size = 3,
        .src_dim1_pitch = 64,
        .src_dim2_pitch = 4096,
        
        // Destination (DRAM): 32×32×64
        .dst_dim1_size = 32,
        .dst_dim2_size = 32,
        .dst_dim3_size = 64,
        .dst_dim1_pitch = 32,
        .dst_dim2_pitch = 1024,
        
        // Input tile: 64×21 (edges: 3,3,3,3)
        .in_dim1_size = 64,
        .in_dim1_pitch = 70,
        .in_dim2_size = 21,
        .in_dim2_pitch = 1470,
        .in_dim1_edge1 = 3,
        .in_dim1_edge2 = 3,
        .in_dim2_edge1 = 3,
        .in_dim2_edge2 = 3,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 213,
        .in_rows_firstdma = 18,
        
        // Output tile: 32×8×64
        .out_dim1_size = 32,
        .out_dim1_pitch = 32,
        .out_dim2_size = 8,
        .out_dim2_pitch = 256,
        .out_dim3_size = 64,
        
        // Coefficients: 7×7×3×64
        .coeff_dim1_size = 7,
        .coeff_dim2_size = 7,
        .coeff_dim3_size = 3,
        .coeff_dim4_size = 64,
        .coeff_dim1_pitch = 7,
        .coeff_dim2_pitch = 49,
        .coeff_dim3_pitch = 147,
        
        // Bias/Outscale: 64
        .bias_dim1_size = 64,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 64,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 4410,
        .coeff_buffer_size = 9408,
        .output_buffer_size = 16384,
        .bias_buffer_size = 256,
        .outscale_buffer_size = 128,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 0,
        .outscale_dram = 0,
        
        // Tiling: 64 ch/tile × 1 tiles, 8 rows/tile × 4 tiles
        .n_tile_size = 64,
        .n_tiles = 1,
        .n_tile_size_last = 64,
        .height_tiles = 4,
        .output_rows = 8,
        .input_rows = 21,
        
        // Conv params: 7×7, stride 2×2, pad 3
        .kernel_w = 7,
        .kernel_h = 7,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 3,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 1,
        .layer_name = "conv2.1",
        .kernel_name = "3x3j1d1",
        .config_key = "64_16_16_64_3_3_16_16_1_1_1_1",
        
        // Source (DRAM): 16×16×64
        .src_dim1_size = 16,
        .src_dim2_size = 16,
        .src_dim3_size = 64,
        .src_dim1_pitch = 16,
        .src_dim2_pitch = 256,
        
        // Destination (DRAM): 16×16×64
        .dst_dim1_size = 16,
        .dst_dim2_size = 16,
        .dst_dim3_size = 64,
        .dst_dim1_pitch = 16,
        .dst_dim2_pitch = 256,
        
        // Input tile: 16×4 (edges: 1,1,1,1)
        .in_dim1_size = 16,
        .in_dim1_pitch = 18,
        .in_dim2_size = 4,
        .in_dim2_pitch = 72,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 19,
        .in_rows_firstdma = 3,
        
        // Output tile: 16×2×32
        .out_dim1_size = 16,
        .out_dim1_pitch = 16,
        .out_dim2_size = 2,
        .out_dim2_pitch = 32,
        .out_dim3_size = 32,
        
        // Coefficients: 3×3×64×64
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 64,
        .coeff_dim4_size = 64,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 576,
        
        // Bias/Outscale: 64
        .bias_dim1_size = 64,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 64,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 4608,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 1024,
        .bias_buffer_size = 256,
        .outscale_buffer_size = 128,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 32 ch/tile × 2 tiles, 2 rows/tile × 8 tiles
        .n_tile_size = 32,
        .n_tiles = 2,
        .n_tile_size_last = 32,
        .height_tiles = 8,
        .output_rows = 2,
        .input_rows = 4,
        
        // Conv params: 3×3, stride 1×1, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 2,
        .layer_name = "conv4b.1",
        .kernel_name = "3x3j2d1",
        .config_key = "64_16_16_128_3_3_8_8_2_2_1_1",
        
        // Source (DRAM): 16×16×64
        .src_dim1_size = 16,
        .src_dim2_size = 16,
        .src_dim3_size = 64,
        .src_dim1_pitch = 16,
        .src_dim2_pitch = 256,
        
        // Destination (DRAM): 8×8×128
        .dst_dim1_size = 8,
        .dst_dim2_size = 8,
        .dst_dim3_size = 128,
        .dst_dim1_pitch = 8,
        .dst_dim2_pitch = 64,
        
        // Input tile: 16×5 (edges: 1,1,1,1)
        .in_dim1_size = 16,
        .in_dim1_pitch = 18,
        .in_dim2_size = 5,
        .in_dim2_pitch = 90,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 19,
        .in_rows_firstdma = 4,
        
        // Output tile: 8×2×32
        .out_dim1_size = 8,
        .out_dim1_pitch = 8,
        .out_dim2_size = 2,
        .out_dim2_pitch = 16,
        .out_dim3_size = 32,
        
        // Coefficients: 3×3×64×128
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 64,
        .coeff_dim4_size = 128,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 576,
        
        // Bias/Outscale: 128
        .bias_dim1_size = 128,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 128,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 5760,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 512,
        .bias_buffer_size = 512,
        .outscale_buffer_size = 256,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 32 ch/tile × 4 tiles, 2 rows/tile × 4 tiles
        .n_tile_size = 32,
        .n_tiles = 4,
        .n_tile_size_last = 32,
        .height_tiles = 4,
        .output_rows = 2,
        .input_rows = 5,
        
        // Conv params: 3×3, stride 2×2, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 3,
        .layer_name = "conv4b.2",
        .kernel_name = "3x3j1d1",
        .config_key = "128_8_8_128_3_3_8_8_1_1_1_1",
        
        // Source (DRAM): 8×8×128
        .src_dim1_size = 8,
        .src_dim2_size = 8,
        .src_dim3_size = 128,
        .src_dim1_pitch = 8,
        .src_dim2_pitch = 64,
        
        // Destination (DRAM): 8×8×128
        .dst_dim1_size = 8,
        .dst_dim2_size = 8,
        .dst_dim3_size = 128,
        .dst_dim1_pitch = 8,
        .dst_dim2_pitch = 64,
        
        // Input tile: 8×4 (edges: 1,1,1,1)
        .in_dim1_size = 8,
        .in_dim1_pitch = 10,
        .in_dim2_size = 4,
        .in_dim2_pitch = 40,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 11,
        .in_rows_firstdma = 3,
        
        // Output tile: 8×2×16
        .out_dim1_size = 8,
        .out_dim1_pitch = 8,
        .out_dim2_size = 2,
        .out_dim2_pitch = 16,
        .out_dim3_size = 16,
        
        // Coefficients: 3×3×128×128
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 128,
        .coeff_dim4_size = 128,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 1152,
        
        // Bias/Outscale: 128
        .bias_dim1_size = 128,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 128,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 5120,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 256,
        .bias_buffer_size = 512,
        .outscale_buffer_size = 256,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 16 ch/tile × 8 tiles, 2 rows/tile × 4 tiles
        .n_tile_size = 16,
        .n_tiles = 8,
        .n_tile_size_last = 16,
        .height_tiles = 4,
        .output_rows = 2,
        .input_rows = 4,
        
        // Conv params: 3×3, stride 1×1, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 4,
        .layer_name = "conv4a.1",
        .kernel_name = "1x1j2d1",
        .config_key = "64_16_16_128_1_1_8_8_2_2_0_1",
        
        // Source (DRAM): 16×16×64
        .src_dim1_size = 16,
        .src_dim2_size = 16,
        .src_dim3_size = 64,
        .src_dim1_pitch = 16,
        .src_dim2_pitch = 256,
        
        // Destination (DRAM): 8×8×128
        .dst_dim1_size = 8,
        .dst_dim2_size = 8,
        .dst_dim3_size = 128,
        .dst_dim1_pitch = 8,
        .dst_dim2_pitch = 64,
        
        // Input tile: 16×15 (edges: 0,0,0,0)
        .in_dim1_size = 16,
        .in_dim1_pitch = 16,
        .in_dim2_size = 15,
        .in_dim2_pitch = 240,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 15,
        
        // Output tile: 8×8×128
        .out_dim1_size = 8,
        .out_dim1_pitch = 8,
        .out_dim2_size = 8,
        .out_dim2_pitch = 64,
        .out_dim3_size = 128,
        
        // Coefficients: 1×1×64×128
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 64,
        .coeff_dim4_size = 128,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 64,
        
        // Bias/Outscale: 128
        .bias_dim1_size = 128,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 128,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 15360,
        .coeff_buffer_size = 8192,
        .output_buffer_size = 8192,
        .bias_buffer_size = 512,
        .outscale_buffer_size = 256,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 128 ch/tile × 1 tiles, 8 rows/tile × 1 tiles
        .n_tile_size = 128,
        .n_tiles = 1,
        .n_tile_size_last = 128,
        .height_tiles = 1,
        .output_rows = 8,
        .input_rows = 15,
        
        // Conv params: 1×1, stride 2×2, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 5,
        .layer_name = "conv6b.1",
        .kernel_name = "3x3j2d1",
        .config_key = "128_8_8_256_3_3_4_4_2_2_1_1",
        
        // Source (DRAM): 8×8×128
        .src_dim1_size = 8,
        .src_dim2_size = 8,
        .src_dim3_size = 128,
        .src_dim1_pitch = 8,
        .src_dim2_pitch = 64,
        
        // Destination (DRAM): 4×4×256
        .dst_dim1_size = 4,
        .dst_dim2_size = 4,
        .dst_dim3_size = 256,
        .dst_dim1_pitch = 4,
        .dst_dim2_pitch = 16,
        
        // Input tile: 8×5 (edges: 1,1,1,1)
        .in_dim1_size = 8,
        .in_dim1_pitch = 10,
        .in_dim2_size = 5,
        .in_dim2_pitch = 50,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 11,
        .in_rows_firstdma = 4,
        
        // Output tile: 4×2×16
        .out_dim1_size = 4,
        .out_dim1_pitch = 4,
        .out_dim2_size = 2,
        .out_dim2_pitch = 8,
        .out_dim3_size = 16,
        
        // Coefficients: 3×3×128×256
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 128,
        .coeff_dim4_size = 256,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 1152,
        
        // Bias/Outscale: 256
        .bias_dim1_size = 256,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 256,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 6400,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 128,
        .bias_buffer_size = 1024,
        .outscale_buffer_size = 512,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 16 ch/tile × 16 tiles, 2 rows/tile × 2 tiles
        .n_tile_size = 16,
        .n_tiles = 16,
        .n_tile_size_last = 16,
        .height_tiles = 2,
        .output_rows = 2,
        .input_rows = 5,
        
        // Conv params: 3×3, stride 2×2, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 6,
        .layer_name = "conv6b.2",
        .kernel_name = "3x3j1d1",
        .config_key = "256_4_4_256_3_3_4_4_1_1_1_1",
        
        // Source (DRAM): 4×4×256
        .src_dim1_size = 4,
        .src_dim2_size = 4,
        .src_dim3_size = 256,
        .src_dim1_pitch = 4,
        .src_dim2_pitch = 16,
        
        // Destination (DRAM): 4×4×256
        .dst_dim1_size = 4,
        .dst_dim2_size = 4,
        .dst_dim3_size = 256,
        .dst_dim1_pitch = 4,
        .dst_dim2_pitch = 16,
        
        // Input tile: 4×4 (edges: 1,1,1,1)
        .in_dim1_size = 4,
        .in_dim1_pitch = 6,
        .in_dim2_size = 4,
        .in_dim2_pitch = 24,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 7,
        .in_rows_firstdma = 3,
        
        // Output tile: 4×2×8
        .out_dim1_size = 4,
        .out_dim1_pitch = 4,
        .out_dim2_size = 2,
        .out_dim2_pitch = 8,
        .out_dim3_size = 8,
        
        // Coefficients: 3×3×256×256
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 256,
        .coeff_dim4_size = 256,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 2304,
        
        // Bias/Outscale: 256
        .bias_dim1_size = 256,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 256,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 6144,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 64,
        .bias_buffer_size = 1024,
        .outscale_buffer_size = 512,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 8 ch/tile × 32 tiles, 2 rows/tile × 2 tiles
        .n_tile_size = 8,
        .n_tiles = 32,
        .n_tile_size_last = 8,
        .height_tiles = 2,
        .output_rows = 2,
        .input_rows = 4,
        
        // Conv params: 3×3, stride 1×1, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 7,
        .layer_name = "conv6a.1",
        .kernel_name = "1x1j2d1",
        .config_key = "128_8_8_256_1_1_4_4_2_2_0_1",
        
        // Source (DRAM): 8×8×128
        .src_dim1_size = 8,
        .src_dim2_size = 8,
        .src_dim3_size = 128,
        .src_dim1_pitch = 8,
        .src_dim2_pitch = 64,
        
        // Destination (DRAM): 4×4×256
        .dst_dim1_size = 4,
        .dst_dim2_size = 4,
        .dst_dim3_size = 256,
        .dst_dim1_pitch = 4,
        .dst_dim2_pitch = 16,
        
        // Input tile: 8×3 (edges: 0,0,0,0)
        .in_dim1_size = 8,
        .in_dim1_pitch = 8,
        .in_dim2_size = 3,
        .in_dim2_pitch = 24,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 3,
        
        // Output tile: 4×2×128
        .out_dim1_size = 4,
        .out_dim1_pitch = 4,
        .out_dim2_size = 2,
        .out_dim2_pitch = 8,
        .out_dim3_size = 128,
        
        // Coefficients: 1×1×128×256
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 128,
        .coeff_dim4_size = 256,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 128,
        
        // Bias/Outscale: 256
        .bias_dim1_size = 256,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 256,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 3072,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 1024,
        .bias_buffer_size = 1024,
        .outscale_buffer_size = 512,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 128 ch/tile × 2 tiles, 2 rows/tile × 2 tiles
        .n_tile_size = 128,
        .n_tiles = 2,
        .n_tile_size_last = 128,
        .height_tiles = 2,
        .output_rows = 2,
        .input_rows = 3,
        
        // Conv params: 1×1, stride 2×2, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 8,
        .layer_name = "conv8b.1",
        .kernel_name = "3x3j2d1",
        .config_key = "256_4_4_512_3_3_2_2_2_2_1_1",
        
        // Source (DRAM): 4×4×256
        .src_dim1_size = 4,
        .src_dim2_size = 4,
        .src_dim3_size = 256,
        .src_dim1_pitch = 4,
        .src_dim2_pitch = 16,
        
        // Destination (DRAM): 2×2×512
        .dst_dim1_size = 2,
        .dst_dim2_size = 2,
        .dst_dim3_size = 512,
        .dst_dim1_pitch = 2,
        .dst_dim2_pitch = 4,
        
        // Input tile: 4×5 (edges: 1,1,1,1)
        .in_dim1_size = 4,
        .in_dim1_pitch = 6,
        .in_dim2_size = 5,
        .in_dim2_pitch = 30,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 7,
        .in_rows_firstdma = 4,
        
        // Output tile: 2×2×8
        .out_dim1_size = 2,
        .out_dim1_pitch = 2,
        .out_dim2_size = 2,
        .out_dim2_pitch = 4,
        .out_dim3_size = 8,
        
        // Coefficients: 3×3×256×512
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 256,
        .coeff_dim4_size = 512,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 2304,
        
        // Bias/Outscale: 512
        .bias_dim1_size = 512,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 512,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 7680,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 32,
        .bias_buffer_size = 2048,
        .outscale_buffer_size = 1024,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 8 ch/tile × 64 tiles, 2 rows/tile × 1 tiles
        .n_tile_size = 8,
        .n_tiles = 64,
        .n_tile_size_last = 8,
        .height_tiles = 1,
        .output_rows = 2,
        .input_rows = 5,
        
        // Conv params: 3×3, stride 2×2, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 9,
        .layer_name = "conv8b.2",
        .kernel_name = "3x3j1d1",
        .config_key = "512_2_2_512_3_3_2_2_1_1_1_1",
        
        // Source (DRAM): 2×2×512
        .src_dim1_size = 2,
        .src_dim2_size = 2,
        .src_dim3_size = 512,
        .src_dim1_pitch = 2,
        .src_dim2_pitch = 4,
        
        // Destination (DRAM): 2×2×512
        .dst_dim1_size = 2,
        .dst_dim2_size = 2,
        .dst_dim3_size = 512,
        .dst_dim1_pitch = 2,
        .dst_dim2_pitch = 4,
        
        // Input tile: 2×4 (edges: 1,1,1,1)
        .in_dim1_size = 2,
        .in_dim1_pitch = 4,
        .in_dim2_size = 4,
        .in_dim2_pitch = 16,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 5,
        .in_rows_firstdma = 3,
        
        // Output tile: 2×2×4
        .out_dim1_size = 2,
        .out_dim1_pitch = 2,
        .out_dim2_size = 2,
        .out_dim2_pitch = 4,
        .out_dim3_size = 4,
        
        // Coefficients: 3×3×512×512
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 512,
        .coeff_dim4_size = 512,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 4608,
        
        // Bias/Outscale: 512
        .bias_dim1_size = 512,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 512,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 8192,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 16,
        .bias_buffer_size = 2048,
        .outscale_buffer_size = 1024,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 4 ch/tile × 128 tiles, 2 rows/tile × 1 tiles
        .n_tile_size = 4,
        .n_tiles = 128,
        .n_tile_size_last = 4,
        .height_tiles = 1,
        .output_rows = 2,
        .input_rows = 4,
        
        // Conv params: 3×3, stride 1×1, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 10,
        .layer_name = "conv8a.1",
        .kernel_name = "1x1j2d1",
        .config_key = "256_4_4_512_1_1_2_2_2_2_0_1",
        
        // Source (DRAM): 4×4×256
        .src_dim1_size = 4,
        .src_dim2_size = 4,
        .src_dim3_size = 256,
        .src_dim1_pitch = 4,
        .src_dim2_pitch = 16,
        
        // Destination (DRAM): 2×2×512
        .dst_dim1_size = 2,
        .dst_dim2_size = 2,
        .dst_dim3_size = 512,
        .dst_dim1_pitch = 2,
        .dst_dim2_pitch = 4,
        
        // Input tile: 4×3 (edges: 0,0,0,0)
        .in_dim1_size = 4,
        .in_dim1_pitch = 4,
        .in_dim2_size = 3,
        .in_dim2_pitch = 12,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 3,
        
        // Output tile: 2×2×64
        .out_dim1_size = 2,
        .out_dim1_pitch = 2,
        .out_dim2_size = 2,
        .out_dim2_pitch = 4,
        .out_dim3_size = 64,
        
        // Coefficients: 1×1×256×512
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 256,
        .coeff_dim4_size = 512,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 256,
        
        // Bias/Outscale: 512
        .bias_dim1_size = 512,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 512,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 3072,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 256,
        .bias_buffer_size = 2048,
        .outscale_buffer_size = 1024,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 64 ch/tile × 8 tiles, 2 rows/tile × 1 tiles
        .n_tile_size = 64,
        .n_tiles = 8,
        .n_tile_size_last = 64,
        .height_tiles = 1,
        .output_rows = 2,
        .input_rows = 3,
        
        // Conv params: 1×1, stride 2×2, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 11,
        .layer_name = "conv2b.1",
        .kernel_name = "1x1j1d1",
        .config_key = "64_16_16_64_1_1_16_16_1_1_0_1",
        
        // Source (DRAM): 16×16×64
        .src_dim1_size = 16,
        .src_dim2_size = 16,
        .src_dim3_size = 64,
        .src_dim1_pitch = 16,
        .src_dim2_pitch = 256,
        
        // Destination (DRAM): 16×16×64
        .dst_dim1_size = 16,
        .dst_dim2_size = 16,
        .dst_dim3_size = 64,
        .dst_dim1_pitch = 16,
        .dst_dim2_pitch = 256,
        
        // Input tile: 16×14 (edges: 0,0,0,0)
        .in_dim1_size = 16,
        .in_dim1_pitch = 16,
        .in_dim2_size = 14,
        .in_dim2_pitch = 224,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 14,
        
        // Output tile: 16×14×64
        .out_dim1_size = 16,
        .out_dim1_pitch = 16,
        .out_dim2_size = 14,
        .out_dim2_pitch = 224,
        .out_dim3_size = 64,
        
        // Coefficients: 1×1×64×64
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 64,
        .coeff_dim4_size = 64,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 64,
        
        // Bias/Outscale: 64
        .bias_dim1_size = 64,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 64,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 14336,
        .coeff_buffer_size = 4096,
        .output_buffer_size = 14336,
        .bias_buffer_size = 256,
        .outscale_buffer_size = 128,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 64 ch/tile × 1 tiles, 14 rows/tile × 2 tiles
        .n_tile_size = 64,
        .n_tiles = 1,
        .n_tile_size_last = 64,
        .height_tiles = 2,
        .output_rows = 14,
        .input_rows = 14,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 12,
        .layer_name = "conv2b.3",
        .kernel_name = "1x1j1d1",
        .config_key = "64_16_16_256_1_1_16_16_1_1_0_1",
        
        // Source (DRAM): 16×16×64
        .src_dim1_size = 16,
        .src_dim2_size = 16,
        .src_dim3_size = 64,
        .src_dim1_pitch = 16,
        .src_dim2_pitch = 256,
        
        // Destination (DRAM): 16×16×256
        .dst_dim1_size = 16,
        .dst_dim2_size = 16,
        .dst_dim3_size = 256,
        .dst_dim1_pitch = 16,
        .dst_dim2_pitch = 256,
        
        // Input tile: 16×4 (edges: 0,0,0,0)
        .in_dim1_size = 16,
        .in_dim1_pitch = 16,
        .in_dim2_size = 4,
        .in_dim2_pitch = 64,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 4,
        
        // Output tile: 16×4×256
        .out_dim1_size = 16,
        .out_dim1_pitch = 16,
        .out_dim2_size = 4,
        .out_dim2_pitch = 64,
        .out_dim3_size = 256,
        
        // Coefficients: 1×1×64×256
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 64,
        .coeff_dim4_size = 256,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 64,
        
        // Bias/Outscale: 256
        .bias_dim1_size = 256,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 256,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 4096,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 16384,
        .bias_buffer_size = 1024,
        .outscale_buffer_size = 512,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 0,
        .outscale_dram = 0,
        
        // Tiling: 256 ch/tile × 1 tiles, 4 rows/tile × 4 tiles
        .n_tile_size = 256,
        .n_tiles = 1,
        .n_tile_size_last = 256,
        .height_tiles = 4,
        .output_rows = 4,
        .input_rows = 4,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 13,
        .layer_name = "conv3.1",
        .kernel_name = "1x1j1d1",
        .config_key = "256_16_16_64_1_1_16_16_1_1_0_1",
        
        // Source (DRAM): 16×16×256
        .src_dim1_size = 16,
        .src_dim2_size = 16,
        .src_dim3_size = 256,
        .src_dim1_pitch = 16,
        .src_dim2_pitch = 256,
        
        // Destination (DRAM): 16×16×64
        .dst_dim1_size = 16,
        .dst_dim2_size = 16,
        .dst_dim3_size = 64,
        .dst_dim1_pitch = 16,
        .dst_dim2_pitch = 256,
        
        // Input tile: 16×4 (edges: 0,0,0,0)
        .in_dim1_size = 16,
        .in_dim1_pitch = 16,
        .in_dim2_size = 4,
        .in_dim2_pitch = 64,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 4,
        
        // Output tile: 16×4×64
        .out_dim1_size = 16,
        .out_dim1_pitch = 16,
        .out_dim2_size = 4,
        .out_dim2_pitch = 64,
        .out_dim3_size = 64,
        
        // Coefficients: 1×1×256×64
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 256,
        .coeff_dim4_size = 64,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 256,
        
        // Bias/Outscale: 64
        .bias_dim1_size = 64,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 64,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 16384,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 4096,
        .bias_buffer_size = 256,
        .outscale_buffer_size = 128,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 64 ch/tile × 1 tiles, 4 rows/tile × 4 tiles
        .n_tile_size = 64,
        .n_tiles = 1,
        .n_tile_size_last = 64,
        .height_tiles = 4,
        .output_rows = 4,
        .input_rows = 4,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 14,
        .layer_name = "conv5b.1",
        .kernel_name = "1x1j1d1",
        .config_key = "256_16_16_128_1_1_16_16_1_1_0_1",
        
        // Source (DRAM): 16×16×256
        .src_dim1_size = 16,
        .src_dim2_size = 16,
        .src_dim3_size = 256,
        .src_dim1_pitch = 16,
        .src_dim2_pitch = 256,
        
        // Destination (DRAM): 16×16×128
        .dst_dim1_size = 16,
        .dst_dim2_size = 16,
        .dst_dim3_size = 128,
        .dst_dim1_pitch = 16,
        .dst_dim2_pitch = 256,
        
        // Input tile: 16×2 (edges: 0,0,0,0)
        .in_dim1_size = 16,
        .in_dim1_pitch = 16,
        .in_dim2_size = 2,
        .in_dim2_pitch = 32,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 2,
        
        // Output tile: 16×2×64
        .out_dim1_size = 16,
        .out_dim1_pitch = 16,
        .out_dim2_size = 2,
        .out_dim2_pitch = 32,
        .out_dim3_size = 64,
        
        // Coefficients: 1×1×256×128
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 256,
        .coeff_dim4_size = 128,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 256,
        
        // Bias/Outscale: 128
        .bias_dim1_size = 128,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 128,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 8192,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 2048,
        .bias_buffer_size = 512,
        .outscale_buffer_size = 256,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 64 ch/tile × 2 tiles, 2 rows/tile × 8 tiles
        .n_tile_size = 64,
        .n_tiles = 2,
        .n_tile_size_last = 64,
        .height_tiles = 8,
        .output_rows = 2,
        .input_rows = 2,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 15,
        .layer_name = "conv5b.2",
        .kernel_name = "3x3j2d1",
        .config_key = "128_16_16_128_3_3_8_8_2_2_1_1",
        
        // Source (DRAM): 16×16×128
        .src_dim1_size = 16,
        .src_dim2_size = 16,
        .src_dim3_size = 128,
        .src_dim1_pitch = 16,
        .src_dim2_pitch = 256,
        
        // Destination (DRAM): 8×8×128
        .dst_dim1_size = 8,
        .dst_dim2_size = 8,
        .dst_dim3_size = 128,
        .dst_dim1_pitch = 8,
        .dst_dim2_pitch = 64,
        
        // Input tile: 16×5 (edges: 1,1,1,1)
        .in_dim1_size = 16,
        .in_dim1_pitch = 18,
        .in_dim2_size = 5,
        .in_dim2_pitch = 90,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 19,
        .in_rows_firstdma = 4,
        
        // Output tile: 8×2×16
        .out_dim1_size = 8,
        .out_dim1_pitch = 8,
        .out_dim2_size = 2,
        .out_dim2_pitch = 16,
        .out_dim3_size = 16,
        
        // Coefficients: 3×3×128×128
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 128,
        .coeff_dim4_size = 128,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 1152,
        
        // Bias/Outscale: 128
        .bias_dim1_size = 128,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 128,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 11520,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 256,
        .bias_buffer_size = 512,
        .outscale_buffer_size = 256,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 16 ch/tile × 8 tiles, 2 rows/tile × 4 tiles
        .n_tile_size = 16,
        .n_tiles = 8,
        .n_tile_size_last = 16,
        .height_tiles = 4,
        .output_rows = 2,
        .input_rows = 5,
        
        // Conv params: 3×3, stride 2×2, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 16,
        .layer_name = "conv5b.3",
        .kernel_name = "1x1j1d1",
        .config_key = "128_8_8_512_1_1_8_8_1_1_0_1",
        
        // Source (DRAM): 8×8×128
        .src_dim1_size = 8,
        .src_dim2_size = 8,
        .src_dim3_size = 128,
        .src_dim1_pitch = 8,
        .src_dim2_pitch = 64,
        
        // Destination (DRAM): 8×8×512
        .dst_dim1_size = 8,
        .dst_dim2_size = 8,
        .dst_dim3_size = 512,
        .dst_dim1_pitch = 8,
        .dst_dim2_pitch = 64,
        
        // Input tile: 8×2 (edges: 0,0,0,0)
        .in_dim1_size = 8,
        .in_dim1_pitch = 8,
        .in_dim2_size = 2,
        .in_dim2_pitch = 16,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 2,
        
        // Output tile: 8×2×128
        .out_dim1_size = 8,
        .out_dim1_pitch = 8,
        .out_dim2_size = 2,
        .out_dim2_pitch = 16,
        .out_dim3_size = 128,
        
        // Coefficients: 1×1×128×512
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 128,
        .coeff_dim4_size = 512,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 128,
        
        // Bias/Outscale: 512
        .bias_dim1_size = 512,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 512,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 2048,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 2048,
        .bias_buffer_size = 2048,
        .outscale_buffer_size = 1024,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 128 ch/tile × 4 tiles, 2 rows/tile × 4 tiles
        .n_tile_size = 128,
        .n_tiles = 4,
        .n_tile_size_last = 128,
        .height_tiles = 4,
        .output_rows = 2,
        .input_rows = 2,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 17,
        .layer_name = "conv5a.1",
        .kernel_name = "1x1j2d1",
        .config_key = "256_16_16_512_1_1_8_8_2_2_0_1",
        
        // Source (DRAM): 16×16×256
        .src_dim1_size = 16,
        .src_dim2_size = 16,
        .src_dim3_size = 256,
        .src_dim1_pitch = 16,
        .src_dim2_pitch = 256,
        
        // Destination (DRAM): 8×8×512
        .dst_dim1_size = 8,
        .dst_dim2_size = 8,
        .dst_dim3_size = 512,
        .dst_dim1_pitch = 8,
        .dst_dim2_pitch = 64,
        
        // Input tile: 16×3 (edges: 0,0,0,0)
        .in_dim1_size = 16,
        .in_dim1_pitch = 16,
        .in_dim2_size = 3,
        .in_dim2_pitch = 48,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 3,
        
        // Output tile: 8×2×64
        .out_dim1_size = 8,
        .out_dim1_pitch = 8,
        .out_dim2_size = 2,
        .out_dim2_pitch = 16,
        .out_dim3_size = 64,
        
        // Coefficients: 1×1×256×512
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 256,
        .coeff_dim4_size = 512,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 256,
        
        // Bias/Outscale: 512
        .bias_dim1_size = 512,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 512,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 12288,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 1024,
        .bias_buffer_size = 2048,
        .outscale_buffer_size = 1024,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 64 ch/tile × 8 tiles, 2 rows/tile × 4 tiles
        .n_tile_size = 64,
        .n_tiles = 8,
        .n_tile_size_last = 64,
        .height_tiles = 4,
        .output_rows = 2,
        .input_rows = 3,
        
        // Conv params: 1×1, stride 2×2, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 18,
        .layer_name = "conv6.1",
        .kernel_name = "1x1j1d1",
        .config_key = "512_8_8_128_1_1_8_8_1_1_0_1",
        
        // Source (DRAM): 8×8×512
        .src_dim1_size = 8,
        .src_dim2_size = 8,
        .src_dim3_size = 512,
        .src_dim1_pitch = 8,
        .src_dim2_pitch = 64,
        
        // Destination (DRAM): 8×8×128
        .dst_dim1_size = 8,
        .dst_dim2_size = 8,
        .dst_dim3_size = 128,
        .dst_dim1_pitch = 8,
        .dst_dim2_pitch = 64,
        
        // Input tile: 8×2 (edges: 0,0,0,0)
        .in_dim1_size = 8,
        .in_dim1_pitch = 8,
        .in_dim2_size = 2,
        .in_dim2_pitch = 16,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 2,
        
        // Output tile: 8×2×32
        .out_dim1_size = 8,
        .out_dim1_pitch = 8,
        .out_dim2_size = 2,
        .out_dim2_pitch = 16,
        .out_dim3_size = 32,
        
        // Coefficients: 1×1×512×128
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 512,
        .coeff_dim4_size = 128,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 512,
        
        // Bias/Outscale: 128
        .bias_dim1_size = 128,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 128,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 8192,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 512,
        .bias_buffer_size = 512,
        .outscale_buffer_size = 256,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 32 ch/tile × 4 tiles, 2 rows/tile × 4 tiles
        .n_tile_size = 32,
        .n_tiles = 4,
        .n_tile_size_last = 32,
        .height_tiles = 4,
        .output_rows = 2,
        .input_rows = 2,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 19,
        .layer_name = "conv9b.1",
        .kernel_name = "1x1j1d1",
        .config_key = "512_8_8_256_1_1_8_8_1_1_0_1",
        
        // Source (DRAM): 8×8×512
        .src_dim1_size = 8,
        .src_dim2_size = 8,
        .src_dim3_size = 512,
        .src_dim1_pitch = 8,
        .src_dim2_pitch = 64,
        
        // Destination (DRAM): 8×8×256
        .dst_dim1_size = 8,
        .dst_dim2_size = 8,
        .dst_dim3_size = 256,
        .dst_dim1_pitch = 8,
        .dst_dim2_pitch = 64,
        
        // Input tile: 8×2 (edges: 0,0,0,0)
        .in_dim1_size = 8,
        .in_dim1_pitch = 8,
        .in_dim2_size = 2,
        .in_dim2_pitch = 16,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 2,
        
        // Output tile: 8×2×32
        .out_dim1_size = 8,
        .out_dim1_pitch = 8,
        .out_dim2_size = 2,
        .out_dim2_pitch = 16,
        .out_dim3_size = 32,
        
        // Coefficients: 1×1×512×256
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 512,
        .coeff_dim4_size = 256,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 512,
        
        // Bias/Outscale: 256
        .bias_dim1_size = 256,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 256,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 8192,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 512,
        .bias_buffer_size = 1024,
        .outscale_buffer_size = 512,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 32 ch/tile × 8 tiles, 2 rows/tile × 4 tiles
        .n_tile_size = 32,
        .n_tiles = 8,
        .n_tile_size_last = 32,
        .height_tiles = 4,
        .output_rows = 2,
        .input_rows = 2,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 20,
        .layer_name = "conv9b.2",
        .kernel_name = "3x3j2d1",
        .config_key = "256_8_8_256_3_3_4_4_2_2_1_1",
        
        // Source (DRAM): 8×8×256
        .src_dim1_size = 8,
        .src_dim2_size = 8,
        .src_dim3_size = 256,
        .src_dim1_pitch = 8,
        .src_dim2_pitch = 64,
        
        // Destination (DRAM): 4×4×256
        .dst_dim1_size = 4,
        .dst_dim2_size = 4,
        .dst_dim3_size = 256,
        .dst_dim1_pitch = 4,
        .dst_dim2_pitch = 16,
        
        // Input tile: 8×5 (edges: 1,1,1,1)
        .in_dim1_size = 8,
        .in_dim1_pitch = 10,
        .in_dim2_size = 5,
        .in_dim2_pitch = 50,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 11,
        .in_rows_firstdma = 4,
        
        // Output tile: 4×2×8
        .out_dim1_size = 4,
        .out_dim1_pitch = 4,
        .out_dim2_size = 2,
        .out_dim2_pitch = 8,
        .out_dim3_size = 8,
        
        // Coefficients: 3×3×256×256
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 256,
        .coeff_dim4_size = 256,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 2304,
        
        // Bias/Outscale: 256
        .bias_dim1_size = 256,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 256,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 12800,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 64,
        .bias_buffer_size = 1024,
        .outscale_buffer_size = 512,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 8 ch/tile × 32 tiles, 2 rows/tile × 2 tiles
        .n_tile_size = 8,
        .n_tiles = 32,
        .n_tile_size_last = 8,
        .height_tiles = 2,
        .output_rows = 2,
        .input_rows = 5,
        
        // Conv params: 3×3, stride 2×2, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 21,
        .layer_name = "conv9b.3",
        .kernel_name = "1x1j1d1",
        .config_key = "256_4_4_1024_1_1_4_4_1_1_0_1",
        
        // Source (DRAM): 4×4×256
        .src_dim1_size = 4,
        .src_dim2_size = 4,
        .src_dim3_size = 256,
        .src_dim1_pitch = 4,
        .src_dim2_pitch = 16,
        
        // Destination (DRAM): 4×4×1024
        .dst_dim1_size = 4,
        .dst_dim2_size = 4,
        .dst_dim3_size = 1024,
        .dst_dim1_pitch = 4,
        .dst_dim2_pitch = 16,
        
        // Input tile: 4×2 (edges: 0,0,0,0)
        .in_dim1_size = 4,
        .in_dim1_pitch = 4,
        .in_dim2_size = 2,
        .in_dim2_pitch = 8,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 2,
        
        // Output tile: 4×2×64
        .out_dim1_size = 4,
        .out_dim1_pitch = 4,
        .out_dim2_size = 2,
        .out_dim2_pitch = 8,
        .out_dim3_size = 64,
        
        // Coefficients: 1×1×256×1024
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 256,
        .coeff_dim4_size = 1024,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 256,
        
        // Bias/Outscale: 1024
        .bias_dim1_size = 1024,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 1024,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 2048,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 512,
        .bias_buffer_size = 4096,
        .outscale_buffer_size = 2048,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 64 ch/tile × 16 tiles, 2 rows/tile × 2 tiles
        .n_tile_size = 64,
        .n_tiles = 16,
        .n_tile_size_last = 64,
        .height_tiles = 2,
        .output_rows = 2,
        .input_rows = 2,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 22,
        .layer_name = "conv9a.1",
        .kernel_name = "1x1j2d1",
        .config_key = "512_8_8_1024_1_1_4_4_2_2_0_1",
        
        // Source (DRAM): 8×8×512
        .src_dim1_size = 8,
        .src_dim2_size = 8,
        .src_dim3_size = 512,
        .src_dim1_pitch = 8,
        .src_dim2_pitch = 64,
        
        // Destination (DRAM): 4×4×1024
        .dst_dim1_size = 4,
        .dst_dim2_size = 4,
        .dst_dim3_size = 1024,
        .dst_dim1_pitch = 4,
        .dst_dim2_pitch = 16,
        
        // Input tile: 8×3 (edges: 0,0,0,0)
        .in_dim1_size = 8,
        .in_dim1_pitch = 8,
        .in_dim2_size = 3,
        .in_dim2_pitch = 24,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 3,
        
        // Output tile: 4×2×32
        .out_dim1_size = 4,
        .out_dim1_pitch = 4,
        .out_dim2_size = 2,
        .out_dim2_pitch = 8,
        .out_dim3_size = 32,
        
        // Coefficients: 1×1×512×1024
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 512,
        .coeff_dim4_size = 1024,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 512,
        
        // Bias/Outscale: 1024
        .bias_dim1_size = 1024,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 1024,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 12288,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 256,
        .bias_buffer_size = 4096,
        .outscale_buffer_size = 2048,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 32 ch/tile × 32 tiles, 2 rows/tile × 2 tiles
        .n_tile_size = 32,
        .n_tiles = 32,
        .n_tile_size_last = 32,
        .height_tiles = 2,
        .output_rows = 2,
        .input_rows = 3,
        
        // Conv params: 1×1, stride 2×2, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 23,
        .layer_name = "conv10.1",
        .kernel_name = "1x1j1d1",
        .config_key = "1024_4_4_256_1_1_4_4_1_1_0_1",
        
        // Source (DRAM): 4×4×1024
        .src_dim1_size = 4,
        .src_dim2_size = 4,
        .src_dim3_size = 1024,
        .src_dim1_pitch = 4,
        .src_dim2_pitch = 16,
        
        // Destination (DRAM): 4×4×256
        .dst_dim1_size = 4,
        .dst_dim2_size = 4,
        .dst_dim3_size = 256,
        .dst_dim1_pitch = 4,
        .dst_dim2_pitch = 16,
        
        // Input tile: 4×2 (edges: 0,0,0,0)
        .in_dim1_size = 4,
        .in_dim1_pitch = 4,
        .in_dim2_size = 2,
        .in_dim2_pitch = 8,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 2,
        
        // Output tile: 4×2×16
        .out_dim1_size = 4,
        .out_dim1_pitch = 4,
        .out_dim2_size = 2,
        .out_dim2_pitch = 8,
        .out_dim3_size = 16,
        
        // Coefficients: 1×1×1024×256
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 1024,
        .coeff_dim4_size = 256,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 1024,
        
        // Bias/Outscale: 256
        .bias_dim1_size = 256,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 256,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 8192,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 128,
        .bias_buffer_size = 1024,
        .outscale_buffer_size = 512,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 16 ch/tile × 16 tiles, 2 rows/tile × 2 tiles
        .n_tile_size = 16,
        .n_tiles = 16,
        .n_tile_size_last = 16,
        .height_tiles = 2,
        .output_rows = 2,
        .input_rows = 2,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 24,
        .layer_name = "conv15b.1",
        .kernel_name = "1x1j1d1",
        .config_key = "1024_4_4_512_1_1_4_4_1_1_0_1",
        
        // Source (DRAM): 4×4×1024
        .src_dim1_size = 4,
        .src_dim2_size = 4,
        .src_dim3_size = 1024,
        .src_dim1_pitch = 4,
        .src_dim2_pitch = 16,
        
        // Destination (DRAM): 4×4×512
        .dst_dim1_size = 4,
        .dst_dim2_size = 4,
        .dst_dim3_size = 512,
        .dst_dim1_pitch = 4,
        .dst_dim2_pitch = 16,
        
        // Input tile: 4×2 (edges: 0,0,0,0)
        .in_dim1_size = 4,
        .in_dim1_pitch = 4,
        .in_dim2_size = 2,
        .in_dim2_pitch = 8,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 2,
        
        // Output tile: 4×2×16
        .out_dim1_size = 4,
        .out_dim1_pitch = 4,
        .out_dim2_size = 2,
        .out_dim2_pitch = 8,
        .out_dim3_size = 16,
        
        // Coefficients: 1×1×1024×512
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 1024,
        .coeff_dim4_size = 512,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 1024,
        
        // Bias/Outscale: 512
        .bias_dim1_size = 512,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 512,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 8192,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 128,
        .bias_buffer_size = 2048,
        .outscale_buffer_size = 1024,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 16 ch/tile × 32 tiles, 2 rows/tile × 2 tiles
        .n_tile_size = 16,
        .n_tiles = 32,
        .n_tile_size_last = 16,
        .height_tiles = 2,
        .output_rows = 2,
        .input_rows = 2,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 25,
        .layer_name = "conv15b.2",
        .kernel_name = "3x3j2d1",
        .config_key = "512_4_4_512_3_3_2_2_2_2_1_1",
        
        // Source (DRAM): 4×4×512
        .src_dim1_size = 4,
        .src_dim2_size = 4,
        .src_dim3_size = 512,
        .src_dim1_pitch = 4,
        .src_dim2_pitch = 16,
        
        // Destination (DRAM): 2×2×512
        .dst_dim1_size = 2,
        .dst_dim2_size = 2,
        .dst_dim3_size = 512,
        .dst_dim1_pitch = 2,
        .dst_dim2_pitch = 4,
        
        // Input tile: 4×5 (edges: 1,1,1,1)
        .in_dim1_size = 4,
        .in_dim1_pitch = 6,
        .in_dim2_size = 5,
        .in_dim2_pitch = 30,
        .in_dim1_edge1 = 1,
        .in_dim1_edge2 = 1,
        .in_dim2_edge1 = 1,
        .in_dim2_edge2 = 1,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 7,
        .in_rows_firstdma = 4,
        
        // Output tile: 2×2×4
        .out_dim1_size = 2,
        .out_dim1_pitch = 2,
        .out_dim2_size = 2,
        .out_dim2_pitch = 4,
        .out_dim3_size = 4,
        
        // Coefficients: 3×3×512×512
        .coeff_dim1_size = 3,
        .coeff_dim2_size = 3,
        .coeff_dim3_size = 512,
        .coeff_dim4_size = 512,
        .coeff_dim1_pitch = 3,
        .coeff_dim2_pitch = 9,
        .coeff_dim3_pitch = 4608,
        
        // Bias/Outscale: 512
        .bias_dim1_size = 512,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 512,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 15360,
        .coeff_buffer_size = 18432,
        .output_buffer_size = 16,
        .bias_buffer_size = 2048,
        .outscale_buffer_size = 1024,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 4 ch/tile × 128 tiles, 2 rows/tile × 1 tiles
        .n_tile_size = 4,
        .n_tiles = 128,
        .n_tile_size_last = 4,
        .height_tiles = 1,
        .output_rows = 2,
        .input_rows = 5,
        
        // Conv params: 3×3, stride 2×2, pad 1
        .kernel_w = 3,
        .kernel_h = 3,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 1,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 26,
        .layer_name = "conv15b.3",
        .kernel_name = "1x1j1d1",
        .config_key = "512_2_2_2048_1_1_2_2_1_1_0_1",
        
        // Source (DRAM): 2×2×512
        .src_dim1_size = 2,
        .src_dim2_size = 2,
        .src_dim3_size = 512,
        .src_dim1_pitch = 2,
        .src_dim2_pitch = 4,
        
        // Destination (DRAM): 2×2×2048
        .dst_dim1_size = 2,
        .dst_dim2_size = 2,
        .dst_dim3_size = 2048,
        .dst_dim1_pitch = 2,
        .dst_dim2_pitch = 4,
        
        // Input tile: 2×2 (edges: 0,0,0,0)
        .in_dim1_size = 2,
        .in_dim1_pitch = 2,
        .in_dim2_size = 2,
        .in_dim2_pitch = 4,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 2,
        
        // Output tile: 2×2×32
        .out_dim1_size = 2,
        .out_dim1_pitch = 2,
        .out_dim2_size = 2,
        .out_dim2_pitch = 4,
        .out_dim3_size = 32,
        
        // Coefficients: 1×1×512×2048
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 512,
        .coeff_dim4_size = 2048,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 512,
        
        // Bias/Outscale: 2048
        .bias_dim1_size = 2048,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 2048,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 2048,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 128,
        .bias_buffer_size = 8192,
        .outscale_buffer_size = 4096,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 32 ch/tile × 64 tiles, 2 rows/tile × 1 tiles
        .n_tile_size = 32,
        .n_tiles = 64,
        .n_tile_size_last = 32,
        .height_tiles = 1,
        .output_rows = 2,
        .input_rows = 2,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 27,
        .layer_name = "conv15a.1",
        .kernel_name = "1x1j2d1",
        .config_key = "1024_4_4_2048_1_1_2_2_2_2_0_1",
        
        // Source (DRAM): 4×4×1024
        .src_dim1_size = 4,
        .src_dim2_size = 4,
        .src_dim3_size = 1024,
        .src_dim1_pitch = 4,
        .src_dim2_pitch = 16,
        
        // Destination (DRAM): 2×2×2048
        .dst_dim1_size = 2,
        .dst_dim2_size = 2,
        .dst_dim3_size = 2048,
        .dst_dim1_pitch = 2,
        .dst_dim2_pitch = 4,
        
        // Input tile: 4×3 (edges: 0,0,0,0)
        .in_dim1_size = 4,
        .in_dim1_pitch = 4,
        .in_dim2_size = 3,
        .in_dim2_pitch = 12,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 3,
        
        // Output tile: 2×2×16
        .out_dim1_size = 2,
        .out_dim1_pitch = 2,
        .out_dim2_size = 2,
        .out_dim2_pitch = 4,
        .out_dim3_size = 16,
        
        // Coefficients: 1×1×1024×2048
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 1024,
        .coeff_dim4_size = 2048,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 1024,
        
        // Bias/Outscale: 2048
        .bias_dim1_size = 2048,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 2048,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 12288,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 64,
        .bias_buffer_size = 8192,
        .outscale_buffer_size = 4096,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 1,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 16 ch/tile × 128 tiles, 2 rows/tile × 1 tiles
        .n_tile_size = 16,
        .n_tiles = 128,
        .n_tile_size_last = 16,
        .height_tiles = 1,
        .output_rows = 2,
        .input_rows = 3,
        
        // Conv params: 1×1, stride 2×2, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 2,
        .stride_y = 2,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
    {
        .layer_id = 28,
        .layer_name = "conv16.1",
        .kernel_name = "1x1j1d1",
        .config_key = "2048_2_2_512_1_1_2_2_1_1_0_1",
        
        // Source (DRAM): 2×2×2048
        .src_dim1_size = 2,
        .src_dim2_size = 2,
        .src_dim3_size = 2048,
        .src_dim1_pitch = 2,
        .src_dim2_pitch = 4,
        
        // Destination (DRAM): 2×2×512
        .dst_dim1_size = 2,
        .dst_dim2_size = 2,
        .dst_dim3_size = 512,
        .dst_dim1_pitch = 2,
        .dst_dim2_pitch = 4,
        
        // Input tile: 2×2 (edges: 0,0,0,0)
        .in_dim1_size = 2,
        .in_dim1_pitch = 2,
        .in_dim2_size = 2,
        .in_dim2_pitch = 4,
        .in_dim1_edge1 = 0,
        .in_dim1_edge2 = 0,
        .in_dim2_edge1 = 0,
        .in_dim2_edge2 = 0,
        .in_dim3_edge1 = 0,
        .in_dim3_edge2 = 0,
        .in_data_offset = 0,
        .in_rows_firstdma = 2,
        
        // Output tile: 2×2×8
        .out_dim1_size = 2,
        .out_dim1_pitch = 2,
        .out_dim2_size = 2,
        .out_dim2_pitch = 4,
        .out_dim3_size = 8,
        
        // Coefficients: 1×1×2048×512
        .coeff_dim1_size = 1,
        .coeff_dim2_size = 1,
        .coeff_dim3_size = 2048,
        .coeff_dim4_size = 512,
        .coeff_dim1_pitch = 1,
        .coeff_dim2_pitch = 1,
        .coeff_dim3_pitch = 2048,
        
        // Bias/Outscale: 512
        .bias_dim1_size = 512,
        .bias_dim2_size = 1,
        .outscale_dim1_size = 512,
        .outscale_dim2_size = 1,
        
        // Buffer sizes (bytes)
        .input_buffer_size = 8192,
        .coeff_buffer_size = 16384,
        .output_buffer_size = 32,
        .bias_buffer_size = 2048,
        .outscale_buffer_size = 1024,
        
        // DRAM placement
        .input_ping_dram = 0,
        .input_pong_dram = 0,
        .coeff_dram = 0,
        .output_ping_dram = 1,
        .output_pong_dram = 1,
        .bias_dram = 1,
        .outscale_dram = 1,
        
        // Tiling: 8 ch/tile × 64 tiles, 2 rows/tile × 1 tiles
        .n_tile_size = 8,
        .n_tiles = 64,
        .n_tile_size_last = 8,
        .height_tiles = 1,
        .output_rows = 2,
        .input_rows = 2,
        
        // Conv params: 1×1, stride 1×1, pad 0
        .kernel_w = 1,
        .kernel_h = 1,
        .stride_x = 1,
        .stride_y = 1,
        .padding = 0,
        .dilation = 1,
        .accum_shift = 8,
        .relu_max = 4000,
        .relu_min = 0,
        .output_shift = 11,
        .output_scale = 0,
        .flags = 0,
    },
};


/**
 * Get total number of convolution layers
 */
static inline int get_num_conv_layers(void) {
    return NUM_CONV_LAYERS;
}

/**
 * Get configuration for a specific layer by layer_id
 * 
 * @param layer_id Layer index (0 to NUM_CONV_LAYERS-1)
 * @return Pointer to configuration, or NULL if invalid layer_id
 */
static inline const conv_layer_config_t* get_layer_config(int layer_id) {
    if (layer_id < 0 || layer_id >= NUM_CONV_LAYERS) {
        return NULL;
    }
    return &CONV_LAYER_CONFIGS[layer_id];
}

/**
 * Get configuration for a layer by its parameters
 * Searches for a layer matching the given convolution parameters
 *
 * @param ic   Input channels
 * @param ih   Input height
 * @param iw   Input width
 * @param oc   Output channels
 * @param kh   Kernel height
 * @param kw   Kernel width
 * @param oh   Output height
 * @param ow   Output width
 * @param sy   Stride Y
 * @param sx   Stride X
 * @param pad  Padding (symmetric)
 * @param dil  Dilation
 * @return Pointer to configuration, or NULL if not found
 */
static inline const conv_layer_config_t* get_layer_config_by_params(
    int ic, int ih, int iw,
    int oc, int kh, int kw,
    int oh, int ow,
    int sy, int sx,
    int pad, int dil) {
    
    for (int i = 0; i < NUM_CONV_LAYERS; i++) {
        const conv_layer_config_t* cfg = &CONV_LAYER_CONFIGS[i];
        if (cfg->src_dim3_size == ic &&
            cfg->src_dim2_size == ih &&
            cfg->src_dim1_size == iw &&
            cfg->dst_dim3_size == oc &&
            cfg->coeff_dim2_size == kh &&
            cfg->coeff_dim1_size == kw &&
            cfg->dst_dim2_size == oh &&
            cfg->dst_dim1_size == ow &&
            cfg->stride_y == sy &&
            cfg->stride_x == sx &&
            cfg->padding == pad &&
            cfg->dilation == dil) {
            return cfg;
        }
    }
    return NULL;
}

/**
 * Get configuration for a layer by config key string
 * Key format: "ic_ih_iw_oc_kh_kw_oh_ow_sy_sx_pad_dil"
 *
 * @param config_key The unique configuration key string
 * @return Pointer to configuration, or NULL if not found
 */
static inline const conv_layer_config_t* get_layer_config_by_key(const char* config_key) {
    if (config_key == NULL) return NULL;
    
    for (int i = 0; i < NUM_CONV_LAYERS; i++) {
        const conv_layer_config_t* cfg = &CONV_LAYER_CONFIGS[i];
        // Simple string comparison
        const char* a = cfg->config_key;
        const char* b = config_key;
        int match = 1;
        while (*a && *b) {
            if (*a++ != *b++) { match = 0; break; }
        }
        if (match && *a == *b) return cfg;
    }
    return NULL;
}

#endif // CONV_LAYER_CONFIGS_H
