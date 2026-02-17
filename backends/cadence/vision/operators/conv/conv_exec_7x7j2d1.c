/*
 * conv_exec_7x7j2d1.c
 *
 *  Created on: Dec 8, 2025
 *      Author: Suraj Raut
 *
 *  Description:
 *      7x7 stride-2 convolution executor matching convIdma.c exactly.
 *      Layer 0: 224x224x3 -> 112x112x64
 *      
 *      Key DMA formulas for 7x7j2d1:
 *      - Source offset: max(((edge+1)*idx - edge) * src_dim1_size, 0)
 *      - Dest offset: data_offset - min((edge+1)*idx*in_dim1_pitch, data_offset - edge)
 *      - Rows: (idx < height_tiles-1) ? 
 *              min((edge+1)*idx + (in_dim2_size - edge), 
 *                  min(-(edge+1)*idx + src_dim1_size + edge2, in_dim2_size))
 *              : coeff_dim1_size (kernel height = 7)
 *      - DIM2_COORD: stride_y * out_dim2_size * idx_h
 *      - DIM2: in_rows_firstdma - edge (constant throughout loop)
 */

#include "kernel_executors.h"
#include "memory_manager.h"
#include "dma.h"
#include "utils.h"
#include <xai_cnn_api.h>
#include <stdio.h>
#include <string.h>

// conv 7x7j2d1 VQ executor with DMA (per-channel output scaling)
XAI_ERR_TYPE conv_exec_7x7j2d1VQ(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    int8_t* outScale_ptr,
    const conv_layer_config_t* config)
{
    // ========================================================================
    // SECTION 1: DRAM Buffer Allocation
    // ========================================================================
    int dram0_used = 0;
    int dram1_used = 0;
    
    int8_t* p_input0 = allocate_dram_buffer(config->input_buffer_size, 
                                             config->input_ping_dram, 
                                             &dram0_used, &dram1_used);
    int8_t* p_input1 = allocate_dram_buffer(config->input_buffer_size, 
                                             config->input_pong_dram, 
                                             &dram0_used, &dram1_used);
    int8_t* p_coeff = allocate_dram_buffer(config->coeff_buffer_size, 
                                            config->coeff_dram, 
                                            &dram0_used, &dram1_used);
    int8_t* p_output0 = allocate_dram_buffer(config->output_buffer_size, 
                                              config->output_ping_dram, 
                                              &dram0_used, &dram1_used);
    int8_t* p_output1 = allocate_dram_buffer(config->output_buffer_size, 
                                              config->output_pong_dram, 
                                              &dram0_used, &dram1_used);
    int8_t* p_bias = allocate_dram_buffer(config->bias_buffer_size, 
                                           config->bias_dram, 
                                           &dram0_used, &dram1_used);
    int8_t* p_outscale = allocate_dram_buffer(config->outscale_buffer_size, 
                                               config->outscale_dram, 
                                               &dram0_used, &dram1_used);
    
    if (!p_input0 || !p_input1 || !p_coeff || 
        !p_output0 || !p_output1 || !p_bias || !p_outscale) {
        printf("ERROR: Buffer allocation failed in conv_exec_7x7j2d1VQ\n");
        return (-1);
    }
    
    printf("  [7x7j2d1VQ] DRAM usage: dram0=%d, dram1=%d\n", dram0_used, dram1_used);
    
    // ========================================================================
    // SECTION 2: Initialize XAI Tile Descriptors
    // ========================================================================
    xai_tile3D tile_input;
    xai_size3D frame_size_input;
    xai_tile4D tile_coeff;
    xai_array tile_bias;
    xai_array tile_outscale;
    xai_tile3D tile_output;
    xai_cnn_conv_params params;
    
    /* Initialize DMA engines */
    dma_3dm_init(1);
    dma_2dm_init(0);

    // Transfer constant data
    dma_1dm(0, coeff_ptr, p_coeff, config->coeff_buffer_size);
    dma_1dm(0, bias_ptr, p_bias, config->bias_buffer_size);
    dma_1dm(0, outScale_ptr, p_outscale, config->outscale_buffer_size);
    
    // Initialize input buffer and load first tile
    _proto_FillBuffer_I8(p_input0, config->input_zero_point, config->input_buffer_size);
    
    // First DMA: load IN_ROWS_FIRSTDMA rows at data offset
    dma_3dm(1,
            (void*)src,
            (void*)&p_input0[config->in_data_offset],
            config->src_dim1_pitch,
            config->in_dim1_pitch,
            config->src_dim2_pitch,
            config->in_dim2_pitch,
            config->src_dim1_size,
            config->in_rows_firstdma,
            config->src_dim3_size);
    
    // ========================================================================
    // Configure Input Tile Descriptor
    // ========================================================================
    XAI_TILE3D_SET_BUFF_PTR(&tile_input, p_input0);
    XAI_TILE3D_SET_BUFF_SIZE(&tile_input, config->input_buffer_size);
    XAI_TILE3D_SET_DATA_PTR(&tile_input, &p_input0[config->in_data_offset]);
    XAI_TILE3D_SET_DATA_ORDER(&tile_input, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&tile_input, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&tile_input, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&tile_input, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&tile_input, config->in_dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&tile_input, config->in_dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM1(&tile_input, config->src_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&tile_input, config->in_dim1_edge1);
    XAI_TILE3D_SET_DIM1_EDGE2(&tile_input, config->in_dim1_edge2);
    XAI_TILE3D_SET_DIM2(&tile_input, (config->in_rows_firstdma - config->in_dim2_edge1));
    XAI_TILE3D_SET_DIM2_EDGE1(&tile_input, config->in_dim2_edge1);
    XAI_TILE3D_SET_DIM2_EDGE2(&tile_input, config->in_dim2_edge2);
    XAI_TILE3D_SET_DIM3_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM3(&tile_input, config->src_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&tile_input, config->in_dim3_edge1);
    XAI_TILE3D_SET_DIM3_EDGE2(&tile_input, config->in_dim3_edge2);
    
    // Frame size for edge extension
    frame_size_input.dim1Size = config->src_dim1_size;
    frame_size_input.dim2Size = config->src_dim2_size;
    frame_size_input.dim3Size = config->src_dim3_size;
    
    // ========================================================================
    // Configure Coefficient Tile Descriptor (4D: W×H×C×N)
    // ========================================================================
    XAI_TILE4D_SET_BUFF_PTR(&tile_coeff, p_coeff);
    XAI_TILE4D_SET_BUFF_SIZE(&tile_coeff, config->coeff_buffer_size);
    XAI_TILE4D_SET_DATA_PTR(&tile_coeff, p_coeff);
    XAI_TILE4D_SET_DATA_ORDER(&tile_coeff, XAI_WHDN);
    XAI_TILE4D_SET_TYPE(&tile_coeff, XAI_TILE4D_S8);
    XAI_TILE4D_SET_FRAME_PTR(&tile_coeff, 0);
    XAI_TILE4D_SET_STATUS_FLAGS(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1_PITCH(&tile_coeff, config->coeff_dim1_pitch);
    XAI_TILE4D_SET_DIM2_PITCH(&tile_coeff, config->coeff_dim2_pitch);
    XAI_TILE4D_SET_DIM3_PITCH(&tile_coeff, config->coeff_dim3_pitch);
    XAI_TILE4D_SET_DIM1_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1(&tile_coeff, config->coeff_dim1_size);
    XAI_TILE4D_SET_DIM1_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2(&tile_coeff, config->coeff_dim2_size);
    XAI_TILE4D_SET_DIM2_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3(&tile_coeff, config->coeff_dim3_size);
    XAI_TILE4D_SET_DIM3_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM4_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM4(&tile_coeff, config->n_tile_size);
    
    // ========================================================================
    // Configure Bias Array
    // ========================================================================
    XAI_ARRAY_SET_BUFF_PTR(&tile_bias, p_bias);
    XAI_ARRAY_SET_BUFF_SIZE(&tile_bias, config->bias_buffer_size);
    XAI_ARRAY_SET_DATA_PTR(&tile_bias, p_bias);
    XAI_ARRAY_SET_WIDTH(&tile_bias, config->n_tile_size);
    XAI_ARRAY_SET_HEIGHT(&tile_bias, 1);
    XAI_ARRAY_SET_TYPE(&tile_bias, XAI_ARRAY_S32);
    XAI_ARRAY_SET_CAPACITY(&tile_bias, config->n_tile_size);
    
    // ========================================================================
    // Configure Output Scale Array
    // ========================================================================
    XAI_ARRAY_SET_BUFF_PTR(&tile_outscale, p_outscale);
    XAI_ARRAY_SET_BUFF_SIZE(&tile_outscale, config->outscale_buffer_size);
    XAI_ARRAY_SET_DATA_PTR(&tile_outscale, p_outscale);
    XAI_ARRAY_SET_WIDTH(&tile_outscale, config->n_tile_size);
    XAI_ARRAY_SET_HEIGHT(&tile_outscale, 1);
    XAI_ARRAY_SET_TYPE(&tile_outscale, XAI_ARRAY_U16);
    XAI_ARRAY_SET_CAPACITY(&tile_outscale, config->n_tile_size);
    
    // ========================================================================
    // Configure Output Tile Descriptor (matches convIdma.c)
    // ========================================================================
    XAI_TILE3D_SET_BUFF_SIZE(&tile_output, config->output_buffer_size);
    XAI_TILE3D_SET_DATA_ORDER(&tile_output, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&tile_output, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&tile_output, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&tile_output, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&tile_output, config->out_dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&tile_output, config->out_dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM1(&tile_output, config->out_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM1_EDGE2(&tile_output, 0);
    XAI_TILE3D_SET_DIM2(&tile_output, config->out_dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM2_EDGE2(&tile_output, 0);
    XAI_TILE3D_SET_DIM3_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM3(&tile_output, config->out_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM3_EDGE2(&tile_output, 0);
    
    // ========================================================================
    // Configure Convolution Parameters
    // ========================================================================
    XAI_CNN_CONV_SET_ACCUM_SHIFT(&params, config->accum_shift);
    XAI_CNN_CONV_SET_DILATION(&params, config->dilation);
    XAI_CNN_CONV_SET_FLAGS(&params, config->flags);
    XAI_CNN_CONV_SET_OUTPUT_SCALE(&params, config->output_scale);
    XAI_CNN_CONV_SET_OUTPUT_SHIFT(&params, config->output_shift);
    XAI_CNN_CONV_SET_RELU_MAX(&params, config->relu_max);
    XAI_CNN_CONV_SET_RELU_MIN(&params, config->relu_min);
    XAI_CNN_CONV_SET_STRIDEX(&params, config->stride_x);
    XAI_CNN_CONV_SET_STRIDEY(&params, config->stride_y);
    
    // ========================================================================
    // SECTION 3: Tiled Execution Loop (N-tiles × H-tiles)
    // ========================================================================
    int last_tile = 1;
    
    for (int idx_n = 0; idx_n < config->n_tiles; idx_n++) {
        int last_n_tile = (last_tile) && (idx_n == config->n_tiles - 1);
        int current_n_size = (idx_n < config->n_tiles - 1) ? 
                             config->n_tile_size : config->n_tile_size_last;
        
        // Update coefficient/bias/outscale for N-tile
        XAI_TILE4D_SET_DIM4_COORD(&tile_coeff, config->n_tile_size * idx_n);
        XAI_TILE4D_SET_DIM4(&tile_coeff, current_n_size);
        
        XAI_ARRAY_SET_DATA_PTR(&tile_bias, &p_bias[config->n_tile_size * 4 * idx_n]);
        XAI_ARRAY_SET_WIDTH(&tile_bias, current_n_size);
        XAI_ARRAY_SET_CAPACITY(&tile_bias, current_n_size);
        
        XAI_ARRAY_SET_DATA_PTR(&tile_outscale, &p_outscale[config->n_tile_size * 2 * idx_n]);
        XAI_ARRAY_SET_WIDTH(&tile_outscale, current_n_size);
        XAI_ARRAY_SET_CAPACITY(&tile_outscale, current_n_size);
        
        XAI_TILE3D_SET_DIM3_COORD(&tile_output, config->n_tile_size * idx_n);
        XAI_TILE3D_SET_DIM3(&tile_output, current_n_size);
        
        // Process vertical tiles
        for (int idx_h = 0; idx_h < config->height_tiles; idx_h++) {
            int last_h_tile = (last_n_tile) && (idx_h == config->height_tiles - 1);
            
            // ================================================================
            // Prefetch Next Input Tile (Ping-Pong Buffering)
            // ================================================================
            if (!last_h_tile) {
                int temp_idx_h;
                inc_iter_to_temp(&temp_idx_h, idx_h, config->height_tiles, 1);
                _proto_FillBuffer_I8(p_input1, config->input_zero_point, config->input_buffer_size);

                // Transfer next input tile from system memory to DRAM
                // Generalized formulas for variable output_rows
                // Key insight: (EDGE+1) in original = stride_y * output_rows
                int stride_output = config->stride_y * config->output_rows;
                
                // Calculate row count for this tile
                int row_count;
                if (temp_idx_h < (config->height_tiles - 1)) {
                    // Non-last tiles: min of (progress + buffer_size - edge, remaining_from_end, buffer_size)
                    int prog_rows = (stride_output * temp_idx_h) + (config->in_dim2_size - config->in_dim2_edge1);
                    int rem_rows = (config->src_dim2_size + config->in_dim1_edge2) - (stride_output * temp_idx_h);
                    row_count = min(prog_rows, min(rem_rows, config->in_dim2_size));
                } else {
                    // Last tile: transfer remaining rows from source (accounting for source starting offset)
                    int src_start_row = max(stride_output * temp_idx_h - config->in_dim2_edge1, 0);
                    row_count = config->src_dim2_size - src_start_row;
                }
                
                dma_3dm(1,
                    (uint64_t *)&(src[max(((stride_output * temp_idx_h - config->in_dim2_edge1) * config->src_dim1_size),0)]),
                    (uint64_t *)&(p_input1[config->in_data_offset - min(stride_output * temp_idx_h * config->in_dim1_pitch, config->in_data_offset - config->in_dim2_edge1)]),
                    config->src_dim1_pitch,
                    config->in_dim1_pitch,
                    config->src_dim2_pitch,
                    config->in_dim2_pitch,
                    config->src_dim1_size,
                    row_count,
                    config->src_dim3_size);
            }
            
            // ================================================================
            // Update Tile Descriptors for Current Iteration
            // ================================================================
            XAI_TILE3D_SET_BUFF_PTR(&tile_input, p_input0);
            XAI_TILE3D_SET_DATA_PTR(&tile_input, &p_input0[config->in_data_offset]);
            // Input vertical coordinate: stride * tile_size * tile_index (matches convIdma.c)
            XAI_TILE3D_SET_DIM2_COORD(&tile_input, (config->stride_y * config->out_dim2_size)*(idx_h));
            XAI_TILE3D_SET_BUFF_PTR(&tile_output, p_output1);
            XAI_TILE3D_SET_DATA_PTR(&tile_output, &(p_output1[0]));
            XAI_TILE3D_SET_DIM2_COORD(&tile_output, (config->out_dim2_size)*(idx_h));
            
            // ================================================================
            // Perform Edge Extension and Convolution
            // ================================================================
            xaiExtendEdgesConst3D_I8(&tile_input, config->input_zero_point, frame_size_input);

            XAI_ERR_TYPE status = xaiConvolvedVQ3D_S_7x7j2d1_S8S8IX_MOW_WHD(
                                        &(tile_input),
                                        &(tile_coeff),
                                        &(tile_bias),
                                        &(tile_outscale),
                                        &(tile_output),
                                        &(params));
            
            if (status != XAI_ERR_OK) {
                return status;
            }

            // ================================================================
            // Prefetch next coefficient tile (if needed)
            // ================================================================
            if ((!(last_h_tile)) && ((idx_h) == (config->height_tiles - 1))) {
                int temp_idx_n;
                int temp_idx_h;
                inc_iter_to_temp(&(temp_idx_n), idx_n, config->n_tiles, inc_iter_to_temp(&(temp_idx_h), idx_h, config->height_tiles, 1));
                dma_1dm(0, (coeff_ptr + (config->coeff_buffer_size * temp_idx_n)), &(p_coeff[0]), (((temp_idx_n) < (config->n_tiles - 1))?(config->coeff_buffer_size):(config->coeff_dim1_size * config->coeff_dim2_size * config->coeff_dim3_size * config->n_tile_size_last)));
            }

            // ================================================================
            // Write Output Tile to System Memory (matches convIdma.c formula)
            // ================================================================
            dma_2dm(0,
                    &(p_output1[0]),
                    &dst[((config->dst_dim2_pitch * config->n_tile_size)*(idx_n)) + ((config->out_dim2_pitch)*(idx_h))],
                    config->out_dim2_pitch,
                    config->dst_dim2_pitch,
                    config->out_dim2_pitch,
                    current_n_size);

            // Swap ping-pong buffers for next iteration
            swap_buffers(&(p_output0), &(p_output1));
            swap_buffers(&(p_input0), &(p_input1));
        }
    }
    
    return XAI_ERR_OK;
}

// conv 7x7j2d1 executor with DMA (per-tensor output scaling)
XAI_ERR_TYPE conv_exec_7x7j2d1(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config)
{
    // ========================================================================
    // SECTION 1: DRAM Buffer Allocation
    // ========================================================================
    int dram0_used = 0;
    int dram1_used = 0;
    
    int8_t* p_input0 = allocate_dram_buffer(config->input_buffer_size, 
                                             config->input_ping_dram, 
                                             &dram0_used, &dram1_used);
    int8_t* p_input1 = allocate_dram_buffer(config->input_buffer_size, 
                                             config->input_pong_dram, 
                                             &dram0_used, &dram1_used);
    int8_t* p_coeff = allocate_dram_buffer(config->coeff_buffer_size, 
                                            config->coeff_dram, 
                                            &dram0_used, &dram1_used);
    int8_t* p_output0 = allocate_dram_buffer(config->output_buffer_size, 
                                              config->output_ping_dram, 
                                              &dram0_used, &dram1_used);
    int8_t* p_output1 = allocate_dram_buffer(config->output_buffer_size, 
                                              config->output_pong_dram, 
                                              &dram0_used, &dram1_used);
    int8_t* p_bias = allocate_dram_buffer(config->bias_buffer_size, 
                                           config->bias_dram, 
                                           &dram0_used, &dram1_used);
    
    if (!p_input0 || !p_input1 || !p_coeff || 
        !p_output0 || !p_output1 || !p_bias) {
        printf("ERROR: Buffer allocation failed in conv_exec_7x7j2d1\n");
        return (-1);
    }
    
    printf("  [7x7j2d1] DRAM usage: dram0=%d, dram1=%d\n", dram0_used, dram1_used);
    
    // ========================================================================
    // SECTION 2: Initialize XAI Tile Descriptors
    // ========================================================================
    xai_tile3D tile_input;
    xai_size3D frame_size_input;
    xai_tile4D tile_coeff;
    xai_array tile_bias;
    xai_tile3D tile_output;
    xai_cnn_conv_params params;
    
    /* Initialize DMA engines */
    dma_3dm_init(1);
    dma_2dm_init(0);

    // Transfer constant data
    dma_1dm(0, coeff_ptr, p_coeff, config->coeff_buffer_size);
    dma_1dm(0, bias_ptr, p_bias, config->bias_buffer_size);
    
    // Initialize input buffer and load first tile
    _proto_FillBuffer_I8(p_input0, config->input_zero_point, config->input_buffer_size);
    
    // First DMA: load IN_ROWS_FIRSTDMA rows at data offset
    dma_3dm(1,
            (void*)src,
            (void*)&p_input0[config->in_data_offset],
            config->src_dim1_pitch,
            config->in_dim1_pitch,
            config->src_dim2_pitch,
            config->in_dim2_pitch,
            config->src_dim1_size,
            config->in_rows_firstdma,
            config->src_dim3_size);
    
    // ========================================================================
    // Configure Input Tile Descriptor
    // ========================================================================
    XAI_TILE3D_SET_BUFF_PTR(&tile_input, p_input0);
    XAI_TILE3D_SET_BUFF_SIZE(&tile_input, config->input_buffer_size);
    XAI_TILE3D_SET_DATA_PTR(&tile_input, &p_input0[config->in_data_offset]);
    XAI_TILE3D_SET_DATA_ORDER(&tile_input, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&tile_input, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&tile_input, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&tile_input, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&tile_input, config->in_dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&tile_input, config->in_dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM1(&tile_input, config->src_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&tile_input, config->in_dim1_edge1);
    XAI_TILE3D_SET_DIM1_EDGE2(&tile_input, config->in_dim1_edge2);
    XAI_TILE3D_SET_DIM2(&tile_input, (config->in_rows_firstdma - config->in_dim2_edge1));
    XAI_TILE3D_SET_DIM2_EDGE1(&tile_input, config->in_dim2_edge1);
    XAI_TILE3D_SET_DIM2_EDGE2(&tile_input, config->in_dim2_edge2);
    XAI_TILE3D_SET_DIM3_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM3(&tile_input, config->src_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&tile_input, config->in_dim3_edge1);
    XAI_TILE3D_SET_DIM3_EDGE2(&tile_input, config->in_dim3_edge2);
    
    // Frame size for edge extension
    frame_size_input.dim1Size = config->src_dim1_size;
    frame_size_input.dim2Size = config->src_dim2_size;
    frame_size_input.dim3Size = config->src_dim3_size;
    
    // ========================================================================
    // Configure Coefficient Tile Descriptor (4D: W×H×C×N)
    // ========================================================================
    XAI_TILE4D_SET_BUFF_PTR(&tile_coeff, p_coeff);
    XAI_TILE4D_SET_BUFF_SIZE(&tile_coeff, config->coeff_buffer_size);
    XAI_TILE4D_SET_DATA_PTR(&tile_coeff, p_coeff);
    XAI_TILE4D_SET_DATA_ORDER(&tile_coeff, XAI_WHDN);
    XAI_TILE4D_SET_TYPE(&tile_coeff, XAI_TILE4D_S8);
    XAI_TILE4D_SET_FRAME_PTR(&tile_coeff, 0);
    XAI_TILE4D_SET_STATUS_FLAGS(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1_PITCH(&tile_coeff, config->coeff_dim1_pitch);
    XAI_TILE4D_SET_DIM2_PITCH(&tile_coeff, config->coeff_dim2_pitch);
    XAI_TILE4D_SET_DIM3_PITCH(&tile_coeff, config->coeff_dim3_pitch);
    XAI_TILE4D_SET_DIM1_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1(&tile_coeff, config->coeff_dim1_size);
    XAI_TILE4D_SET_DIM1_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2(&tile_coeff, config->coeff_dim2_size);
    XAI_TILE4D_SET_DIM2_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3(&tile_coeff, config->coeff_dim3_size);
    XAI_TILE4D_SET_DIM3_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM4_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM4(&tile_coeff, config->n_tile_size);
    
    // ========================================================================
    // Configure Bias Array
    // ========================================================================
    XAI_ARRAY_SET_BUFF_PTR(&tile_bias, p_bias);
    XAI_ARRAY_SET_BUFF_SIZE(&tile_bias, config->bias_buffer_size);
    XAI_ARRAY_SET_DATA_PTR(&tile_bias, p_bias);
    XAI_ARRAY_SET_WIDTH(&tile_bias, config->n_tile_size);
    XAI_ARRAY_SET_HEIGHT(&tile_bias, 1);
    XAI_ARRAY_SET_TYPE(&tile_bias, XAI_ARRAY_S32);
    XAI_ARRAY_SET_CAPACITY(&tile_bias, config->n_tile_size);
    
    // ========================================================================
    // Configure Output Tile Descriptor (matches convIdma.c)
    // ========================================================================
    XAI_TILE3D_SET_BUFF_SIZE(&tile_output, config->output_buffer_size);
    XAI_TILE3D_SET_DATA_ORDER(&tile_output, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&tile_output, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&tile_output, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&tile_output, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&tile_output, config->out_dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&tile_output, config->out_dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM1(&tile_output, config->out_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM1_EDGE2(&tile_output, 0);
    XAI_TILE3D_SET_DIM2(&tile_output, config->out_dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM2_EDGE2(&tile_output, 0);
    XAI_TILE3D_SET_DIM3_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM3(&tile_output, config->out_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM3_EDGE2(&tile_output, 0);
    
    // ========================================================================
    // Configure Convolution Parameters
    // ========================================================================
    XAI_CNN_CONV_SET_ACCUM_SHIFT(&params, config->accum_shift);
    XAI_CNN_CONV_SET_DILATION(&params, config->dilation);
    XAI_CNN_CONV_SET_FLAGS(&params, config->flags);
    XAI_CNN_CONV_SET_OUTPUT_SCALE(&params, config->output_scale);
    XAI_CNN_CONV_SET_OUTPUT_SHIFT(&params, config->output_shift);
    XAI_CNN_CONV_SET_RELU_MAX(&params, config->relu_max);
    XAI_CNN_CONV_SET_RELU_MIN(&params, config->relu_min);
    XAI_CNN_CONV_SET_STRIDEX(&params, config->stride_x);
    XAI_CNN_CONV_SET_STRIDEY(&params, config->stride_y);


    //print config eg . config->accum_shift, config->dilation, config->flags, config->output_scale, config->output_shift, config->relu_max, config->relu_min, config->stride_x, config->stride_y
    printf("  [7x7j2d1] Convolution parameters: accum_shift=%d, dilation=%d, flags=%d, output_scale=%d, output_shift=%d, relu_max=%d, relu_min=%d, stride_x=%d, stride_y=%d\n",
            config->accum_shift, config->dilation, config->flags, config->output_scale, config->output_shift, config->relu_max, config->relu_min, config->stride_x, config->stride_y);
    
    // ========================================================================
    // SECTION 3: Tiled Execution Loop (N-tiles × H-tiles)
    // ========================================================================
    int last_tile = 1;
    
    for (int idx_n = 0; idx_n < config->n_tiles; idx_n++) {
        int last_n_tile = (last_tile) && (idx_n == config->n_tiles - 1);
        int current_n_size = (idx_n < config->n_tiles - 1) ? 
                             config->n_tile_size : config->n_tile_size_last;
        
        // Update coefficient/bias for N-tile
        XAI_TILE4D_SET_DIM4_COORD(&tile_coeff, config->n_tile_size * idx_n);
        XAI_TILE4D_SET_DIM4(&tile_coeff, current_n_size);
        
        XAI_ARRAY_SET_DATA_PTR(&tile_bias, &p_bias[config->n_tile_size * 4 * idx_n]);
        XAI_ARRAY_SET_WIDTH(&tile_bias, current_n_size);
        XAI_ARRAY_SET_CAPACITY(&tile_bias, current_n_size);
        
        XAI_TILE3D_SET_DIM3_COORD(&tile_output, config->n_tile_size * idx_n);
        XAI_TILE3D_SET_DIM3(&tile_output, current_n_size);
        
        // Process vertical tiles
        for (int idx_h = 0; idx_h < config->height_tiles; idx_h++) {
            int last_h_tile = (last_n_tile) && (idx_h == config->height_tiles - 1);
            
            // ================================================================
            // Prefetch Next Input Tile (Ping-Pong Buffering)
            // ================================================================
            if (!last_h_tile) {
                int temp_idx_h;
                inc_iter_to_temp(&temp_idx_h, idx_h, config->height_tiles, 1);
                _proto_FillBuffer_I8(p_input1, config->input_zero_point, config->input_buffer_size);

                // Transfer next input tile from system memory to DRAM
                // Generalized formulas for variable output_rows
                // Key insight: (EDGE+1) in original = stride_y * output_rows
                int stride_output = config->stride_y * config->output_rows;
                
                // Calculate row count for this tile
                int row_count;
                if (temp_idx_h < (config->height_tiles - 1)) {
                    // Non-last tiles: min of (progress + buffer_size - edge, remaining_from_end, buffer_size)
                    int prog_rows = (stride_output * temp_idx_h) + (config->in_dim2_size - config->in_dim2_edge1);
                    int rem_rows = (config->src_dim2_size + config->in_dim1_edge2) - (stride_output * temp_idx_h);
                    row_count = min(prog_rows, min(rem_rows, config->in_dim2_size));
                } else {
                    // Last tile: transfer remaining rows from source (accounting for source starting offset)
                    int src_start_row = max(stride_output * temp_idx_h - config->in_dim2_edge1, 0);
                    row_count = config->src_dim2_size - src_start_row;
                }
                
                dma_3dm(1,
                    (uint64_t *)&(src[max(((stride_output * temp_idx_h - config->in_dim2_edge1) * config->src_dim1_size),0)]),
                    (uint64_t *)&(p_input1[config->in_data_offset - min(stride_output * temp_idx_h * config->in_dim1_pitch, config->in_data_offset - config->in_dim2_edge1)]),
                    config->src_dim1_pitch,
                    config->in_dim1_pitch,
                    config->src_dim2_pitch,
                    config->in_dim2_pitch,
                    config->src_dim1_size,
                    row_count,
                    config->src_dim3_size);
            }
            
            // ================================================================
            // Update Tile Descriptors for Current Iteration
            // ================================================================
            XAI_TILE3D_SET_BUFF_PTR(&tile_input, p_input0);
            XAI_TILE3D_SET_DATA_PTR(&tile_input, &p_input0[config->in_data_offset]);
            // Input vertical coordinate: stride * tile_size * tile_index (matches convIdma.c)
            XAI_TILE3D_SET_DIM2_COORD(&tile_input, (config->stride_y * config->out_dim2_size)*(idx_h));
            XAI_TILE3D_SET_BUFF_PTR(&tile_output, p_output1);
            XAI_TILE3D_SET_DATA_PTR(&tile_output, &(p_output1[0]));
            XAI_TILE3D_SET_DIM2_COORD(&tile_output, (config->out_dim2_size)*(idx_h));
            
            // ================================================================
            // Perform Edge Extension and Convolution
            // ================================================================
            xaiExtendEdgesConst3D_I8(&tile_input, config->input_zero_point, frame_size_input);

            XAI_ERR_TYPE status = xaiConvolved3D_S_7x7j2d1_S8S8IX_MOW_WHD(
                                        &(tile_input),
                                        &(tile_coeff),
                                        &(tile_bias),
                                        &(tile_output),
                                        &(params));
            
            printf("    [7x7j2d1] N-tile %d, H-tile %d: Convolution status = %d\n", idx_n, idx_h, status);
            if (status != XAI_ERR_OK) {
                return status;
            }

            // ================================================================
            // Prefetch next coefficient tile (if needed)
            // ================================================================
            if ((!(last_h_tile)) && ((idx_h) == (config->height_tiles - 1))) {
                int temp_idx_n;
                int temp_idx_h;
                inc_iter_to_temp(&(temp_idx_n), idx_n, config->n_tiles, inc_iter_to_temp(&(temp_idx_h), idx_h, config->height_tiles, 1));
                dma_1dm(0, (coeff_ptr + (config->coeff_buffer_size * temp_idx_n)), &(p_coeff[0]), (((temp_idx_n) < (config->n_tiles - 1))?(config->coeff_buffer_size):(config->coeff_dim1_size * config->coeff_dim2_size * config->coeff_dim3_size * config->n_tile_size_last)));
            }

            // ================================================================
            // Write Output Tile to System Memory (matches convIdma.c formula)
            // ================================================================
            dma_2dm(0,
                    &(p_output1[0]),
                    &dst[((config->dst_dim2_pitch * config->n_tile_size)*(idx_n)) + ((config->out_dim2_pitch)*(idx_h))],
                    config->out_dim2_pitch,
                    config->dst_dim2_pitch,
                    config->out_dim2_pitch,
                    current_n_size);

            // Swap ping-pong buffers for next iteration
            swap_buffers(&(p_output0), &(p_output1));
            swap_buffers(&(p_input0), &(p_input1));
        }
    }
    
    return XAI_ERR_OK;
}

// conv 7x7j2d1 executor with caching (no DMA)
// All data stays in system memory and is accessed through processor cache
XAI_ERR_TYPE conv_exec_7x7j2d1_cache(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    const conv_layer_config_t* config)
{
    // ========================================================================
    // Setup source raw tile descriptor (points to raw input without padding)
    // ========================================================================
    xai_tile3D src_raw;
    XAI_TILE3D_SET_BUFF_PTR(&src_raw, src);
    XAI_TILE3D_SET_BUFF_SIZE(&src_raw, config->src_dim2_pitch * config->src_dim3_size);
    XAI_TILE3D_SET_DATA_PTR(&src_raw, src);
    XAI_TILE3D_SET_DATA_ORDER(&src_raw, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&src_raw, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&src_raw, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&src_raw, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&src_raw, config->src_dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&src_raw, config->src_dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&src_raw, 0);
    XAI_TILE3D_SET_DIM1(&src_raw, config->src_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&src_raw, 0);
    XAI_TILE3D_SET_DIM1_EDGE2(&src_raw, 0);
    XAI_TILE3D_SET_DIM2_COORD(&src_raw, 0);
    XAI_TILE3D_SET_DIM2(&src_raw, config->src_dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(&src_raw, 0);
    XAI_TILE3D_SET_DIM2_EDGE2(&src_raw, 0);
    XAI_TILE3D_SET_DIM3_COORD(&src_raw, 0);
    XAI_TILE3D_SET_DIM3(&src_raw, config->src_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&src_raw, 0);
    XAI_TILE3D_SET_DIM3_EDGE2(&src_raw, 0);

    // ========================================================================
    // Get padded input buffer from allocator (shared across cache kernels)
    // ========================================================================
    int padded_dim1 = config->src_dim1_size + config->in_dim1_edge1 + config->in_dim1_edge2;
    int dim1_pitch = (padded_dim1 + 2*XCHAL_IVPN_SIMD_WIDTH - 1) & ~(2*XCHAL_IVPN_SIMD_WIDTH - 1);
    int padded_dim2 = config->src_dim2_size + config->in_dim2_edge1 + config->in_dim2_edge2;
    int dim2_pitch = dim1_pitch * padded_dim2;
    int input_buffer_size = dim2_pitch * config->src_dim3_size;
    
    // Get shared padded input buffer from allocator
    int8_t* padded_input = get_cache_padded_input();
    
    if (input_buffer_size > (int)get_cache_padded_input_size()) {
        printf("ERROR: Input buffer size %d exceeds max %d\n", 
               input_buffer_size, (int)get_cache_padded_input_size());
        return XAI_ERR_DATASIZE;
    }
    
    // Zero-fill the padded buffer
    memset(padded_input, config->input_zero_point, input_buffer_size);

    // ========================================================================
    // Setup padded input tile descriptor
    // ========================================================================
    int data_offset = (config->in_dim2_edge1 * dim1_pitch) + config->in_dim1_edge1;
    
    xai_tile3D tile_input;
    XAI_TILE3D_SET_BUFF_PTR(&tile_input, padded_input);
    XAI_TILE3D_SET_BUFF_SIZE(&tile_input, input_buffer_size);
    XAI_TILE3D_SET_DATA_PTR(&tile_input, &padded_input[data_offset]);
    XAI_TILE3D_SET_DATA_ORDER(&tile_input, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&tile_input, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&tile_input, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&tile_input, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&tile_input, dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&tile_input, dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM1(&tile_input, config->src_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&tile_input, config->in_dim1_edge1);
    XAI_TILE3D_SET_DIM1_EDGE2(&tile_input, config->in_dim1_edge2);
    XAI_TILE3D_SET_DIM2_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM2(&tile_input, config->src_dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(&tile_input, config->in_dim2_edge1);
    XAI_TILE3D_SET_DIM2_EDGE2(&tile_input, config->in_dim2_edge2);
    XAI_TILE3D_SET_DIM3_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM3(&tile_input, config->src_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&tile_input, 0);
    XAI_TILE3D_SET_DIM3_EDGE2(&tile_input, 0);

    // ========================================================================
    // Copy raw input to padded buffer and extend edges
    // ========================================================================
#ifdef USE_DMA_FOR_CACHE_COPY
    // Use DMA 3D transfer to copy input data into padded buffer at data_offset
    dma_3dm(0,
            /* src */           src,
            /* dst */           &padded_input[data_offset],
            /* src_row_pitch */ config->src_dim1_pitch,
            /* dst_row_pitch */ dim1_pitch,
            /* src_tile_pitch */ config->src_dim2_pitch,
            /* dst_tile_pitch */ dim2_pitch,
            /* row_sz */        config->src_dim1_size,
            /* nrows */         config->src_dim2_size,
            /* ntiles */        config->src_dim3_size);
#else
    // Use library tile copy function (no DMA required)
    xaiCopyTile3D(&src_raw, &tile_input, true);
#endif
    
    xai_size3D frame_size;
    frame_size.dim1Size = config->dst_dim1_size * config->stride_x;
    frame_size.dim2Size = config->dst_dim2_size * config->stride_y;
    frame_size.dim3Size = config->src_dim3_size;
    
    xaiExtendEdgesConst3D_I8(&tile_input, config->input_zero_point, frame_size);

    // ========================================================================
    // Configure Coefficient Tile Descriptor (4D: W×H×C×N)  
    // ========================================================================
    xai_tile4D tile_coeff;
    XAI_TILE4D_SET_BUFF_PTR(&tile_coeff, coeff_ptr);    
    XAI_TILE4D_SET_BUFF_SIZE(&tile_coeff, config->coeff_buffer_size);
    XAI_TILE4D_SET_DATA_PTR(&tile_coeff, coeff_ptr);
    XAI_TILE4D_SET_DATA_ORDER(&tile_coeff, XAI_WHDN);
    XAI_TILE4D_SET_TYPE(&tile_coeff, XAI_TILE4D_S8);
    XAI_TILE4D_SET_FRAME_PTR(&tile_coeff, 0);   
    XAI_TILE4D_SET_STATUS_FLAGS(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1_PITCH(&tile_coeff, config->coeff_dim1_pitch);
    XAI_TILE4D_SET_DIM2_PITCH(&tile_coeff, config->coeff_dim2_pitch);
    XAI_TILE4D_SET_DIM3_PITCH(&tile_coeff, config->coeff_dim3_pitch);
    XAI_TILE4D_SET_DIM1_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1(&tile_coeff, config->coeff_dim1_size);
    XAI_TILE4D_SET_DIM1_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2(&tile_coeff, config->coeff_dim2_size);
    XAI_TILE4D_SET_DIM2_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3(&tile_coeff, config->coeff_dim3_size);
    XAI_TILE4D_SET_DIM3_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM4_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM4(&tile_coeff, config->dst_dim3_size);

    // ========================================================================
    // Configure Bias Array 
    // ========================================================================
    xai_array tile_bias;
    XAI_ARRAY_SET_BUFF_PTR(&tile_bias, bias_ptr);   
    XAI_ARRAY_SET_BUFF_SIZE(&tile_bias, config->bias_buffer_size);
    XAI_ARRAY_SET_DATA_PTR(&tile_bias, bias_ptr);
    XAI_ARRAY_SET_WIDTH(&tile_bias, config->dst_dim3_size);
    XAI_ARRAY_SET_HEIGHT(&tile_bias, 1);
    XAI_ARRAY_SET_TYPE(&tile_bias, XAI_ARRAY_S32);
    XAI_ARRAY_SET_CAPACITY(&tile_bias, config->dst_dim3_size);


    // ========================================================================
    // Configure Output Tile Descriptor (points to system memory)
    // ========================================================================
    xai_tile3D tile_output;
    XAI_TILE3D_SET_BUFF_PTR(&tile_output, dst);
    XAI_TILE3D_SET_BUFF_SIZE(&tile_output, config->dst_dim2_pitch * config->dst_dim3_size);
    XAI_TILE3D_SET_DATA_PTR(&tile_output, dst);
    XAI_TILE3D_SET_DATA_ORDER(&tile_output, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&tile_output, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&tile_output, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&tile_output, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&tile_output, config->dst_dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&tile_output, config->dst_dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM1(&tile_output, config->dst_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM1_EDGE2(&tile_output, 0);
    XAI_TILE3D_SET_DIM2_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM2(&tile_output, config->dst_dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM2_EDGE2(&tile_output, 0);
    XAI_TILE3D_SET_DIM3_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM3(&tile_output, config->dst_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM3_EDGE2(&tile_output, 0);

    // ========================================================================
    // Configure Convolution Parameters
    // ========================================================================
    xai_cnn_conv_params params;
    XAI_CNN_CONV_SET_ACCUM_SHIFT(&params, config->accum_shift);
    XAI_CNN_CONV_SET_DILATION(&params, config->dilation);
    XAI_CNN_CONV_SET_FLAGS(&params, config->flags);
    XAI_CNN_CONV_SET_OUTPUT_SCALE(&params, config->output_scale);
    XAI_CNN_CONV_SET_OUTPUT_SHIFT(&params, config->output_shift);
    XAI_CNN_CONV_SET_RELU_MAX(&params, config->relu_max);
    XAI_CNN_CONV_SET_RELU_MIN(&params, config->relu_min);
    XAI_CNN_CONV_SET_STRIDEX(&params, config->stride_x);
    XAI_CNN_CONV_SET_STRIDEY(&params, config->stride_y);

    // ========================================================================
    // Execute convolution using generic system-memory API
    // This version accesses data through the processor cache
    // ========================================================================
    XAI_ERR_TYPE status = xaiConvolved3D(&tile_input, &tile_coeff, &tile_bias, 
                                            &tile_output, &params);
    
    return status;
}


XAI_ERR_TYPE conv_exec_7x7j2d1VQ_cache(
    int8_t* src,
    int8_t* dst,
    int8_t* coeff_ptr,
    int8_t* bias_ptr,
    int8_t* outScale_ptr,
    const conv_layer_config_t* config)
{
    // ========================================================================
    // Setup source raw tile descriptor (points to raw input without padding)
    // ========================================================================
    xai_tile3D src_raw;
    XAI_TILE3D_SET_BUFF_PTR(&src_raw, src);
    XAI_TILE3D_SET_BUFF_SIZE(&src_raw, config->src_dim2_pitch * config->src_dim3_size);
    XAI_TILE3D_SET_DATA_PTR(&src_raw, src);
    XAI_TILE3D_SET_DATA_ORDER(&src_raw, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&src_raw, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&src_raw, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&src_raw, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&src_raw, config->src_dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&src_raw, config->src_dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&src_raw, 0);
    XAI_TILE3D_SET_DIM1(&src_raw, config->src_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&src_raw, 0);
    XAI_TILE3D_SET_DIM1_EDGE2(&src_raw, 0);
    XAI_TILE3D_SET_DIM2_COORD(&src_raw, 0);
    XAI_TILE3D_SET_DIM2(&src_raw, config->src_dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(&src_raw, 0);
    XAI_TILE3D_SET_DIM2_EDGE2(&src_raw, 0);
    XAI_TILE3D_SET_DIM3_COORD(&src_raw, 0);
    XAI_TILE3D_SET_DIM3(&src_raw, config->src_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&src_raw, 0);
    XAI_TILE3D_SET_DIM3_EDGE2(&src_raw, 0);

    // ========================================================================
    // Get padded input buffer from allocator (shared across cache kernels)
    // ========================================================================
    int padded_dim1 = config->src_dim1_size + config->in_dim1_edge1 + config->in_dim1_edge2;
    int dim1_pitch = (padded_dim1 + 2*XCHAL_IVPN_SIMD_WIDTH - 1) & ~(2*XCHAL_IVPN_SIMD_WIDTH - 1);
    int padded_dim2 = config->src_dim2_size + config->in_dim2_edge1 + config->in_dim2_edge2;
    int dim2_pitch = dim1_pitch * padded_dim2;
    int input_buffer_size = dim2_pitch * config->src_dim3_size;
    
    // Get shared padded input buffer from allocator
    int8_t* padded_input = get_cache_padded_input();
    
    if (input_buffer_size > (int)get_cache_padded_input_size()) {
        printf("ERROR: Input buffer size %d exceeds max %d\n", 
               input_buffer_size, (int)get_cache_padded_input_size());
        return XAI_ERR_DATASIZE;
    }
    
    // Zero-fill the padded buffer
    memset(padded_input, config->input_zero_point, input_buffer_size);

    // ========================================================================
    // Setup padded input tile descriptor
    // ========================================================================
    int data_offset = (config->in_dim2_edge1 * dim1_pitch) + config->in_dim1_edge1;
    
    xai_tile3D tile_input;
    XAI_TILE3D_SET_BUFF_PTR(&tile_input, padded_input);
    XAI_TILE3D_SET_BUFF_SIZE(&tile_input, input_buffer_size);
    XAI_TILE3D_SET_DATA_PTR(&tile_input, &padded_input[data_offset]);
    XAI_TILE3D_SET_DATA_ORDER(&tile_input, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&tile_input, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&tile_input, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&tile_input, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&tile_input, dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&tile_input, dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM1(&tile_input, config->src_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&tile_input, config->in_dim1_edge1);
    XAI_TILE3D_SET_DIM1_EDGE2(&tile_input, config->in_dim1_edge2);
    XAI_TILE3D_SET_DIM2_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM2(&tile_input, config->src_dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(&tile_input, config->in_dim2_edge1);
    XAI_TILE3D_SET_DIM2_EDGE2(&tile_input, config->in_dim2_edge2);
    XAI_TILE3D_SET_DIM3_COORD(&tile_input, 0);
    XAI_TILE3D_SET_DIM3(&tile_input, config->src_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&tile_input, 0);
    XAI_TILE3D_SET_DIM3_EDGE2(&tile_input, 0);

    // ========================================================================
    // Copy raw input to padded buffer and extend edges
    // ========================================================================
#ifdef USE_DMA_FOR_CACHE_COPY
    // Use DMA 3D transfer to copy input data into padded buffer at data_offset
    dma_3dm(0,
            /* src */           src,
            /* dst */           &padded_input[data_offset],
            /* src_row_pitch */ config->src_dim1_pitch,
            /* dst_row_pitch */ dim1_pitch,
            /* src_tile_pitch */ config->src_dim2_pitch,
            /* dst_tile_pitch */ dim2_pitch,
            /* row_sz */        config->src_dim1_size,
            /* nrows */         config->src_dim2_size,
            /* ntiles */        config->src_dim3_size);
#else
    // Use library tile copy function (no DMA required)
    xaiCopyTile3D(&src_raw, &tile_input, true);
#endif
    
    xai_size3D frame_size;
    frame_size.dim1Size = config->dst_dim1_size * config->stride_x;
    frame_size.dim2Size = config->dst_dim2_size * config->stride_y;
    frame_size.dim3Size = config->src_dim3_size;
    
    xaiExtendEdgesConst3D_I8(&tile_input, config->input_zero_point, frame_size);

    // ========================================================================
    // Configure Coefficient Tile Descriptor (4D: W×H×C×N)  
    // ========================================================================
    xai_tile4D tile_coeff;
    XAI_TILE4D_SET_BUFF_PTR(&tile_coeff, coeff_ptr);    
    XAI_TILE4D_SET_BUFF_SIZE(&tile_coeff, config->coeff_buffer_size);
    XAI_TILE4D_SET_DATA_PTR(&tile_coeff, coeff_ptr);
    XAI_TILE4D_SET_DATA_ORDER(&tile_coeff, XAI_WHDN);
    XAI_TILE4D_SET_TYPE(&tile_coeff, XAI_TILE4D_S8);
    XAI_TILE4D_SET_FRAME_PTR(&tile_coeff, 0);   
    XAI_TILE4D_SET_STATUS_FLAGS(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1_PITCH(&tile_coeff, config->coeff_dim1_pitch);
    XAI_TILE4D_SET_DIM2_PITCH(&tile_coeff, config->coeff_dim2_pitch);
    XAI_TILE4D_SET_DIM3_PITCH(&tile_coeff, config->coeff_dim3_pitch);
    XAI_TILE4D_SET_DIM1_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1(&tile_coeff, config->coeff_dim1_size);
    XAI_TILE4D_SET_DIM1_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM1_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2(&tile_coeff, config->coeff_dim2_size);
    XAI_TILE4D_SET_DIM2_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM2_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3(&tile_coeff, config->coeff_dim3_size);
    XAI_TILE4D_SET_DIM3_EDGE1(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM3_EDGE2(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM4_COORD(&tile_coeff, 0);
    XAI_TILE4D_SET_DIM4(&tile_coeff, config->dst_dim3_size);

    // ========================================================================
    // Configure Bias Array 
    // ========================================================================
    xai_array tile_bias;
    XAI_ARRAY_SET_BUFF_PTR(&tile_bias, bias_ptr);   
    XAI_ARRAY_SET_BUFF_SIZE(&tile_bias, config->bias_buffer_size);
    XAI_ARRAY_SET_DATA_PTR(&tile_bias, bias_ptr);
    XAI_ARRAY_SET_WIDTH(&tile_bias, config->dst_dim3_size);
    XAI_ARRAY_SET_HEIGHT(&tile_bias, 1);
    XAI_ARRAY_SET_TYPE(&tile_bias, XAI_ARRAY_S32);
    XAI_ARRAY_SET_CAPACITY(&tile_bias, config->dst_dim3_size);

    // ========================================================================
    // Configure Output Scale Array
    // ========================================================================
    xai_array tile_outscale;
    XAI_ARRAY_SET_BUFF_PTR(&tile_outscale, outScale_ptr);
    XAI_ARRAY_SET_BUFF_SIZE(&tile_outscale, config->outscale_buffer_size);
    XAI_ARRAY_SET_DATA_PTR(&tile_outscale, outScale_ptr);
    XAI_ARRAY_SET_WIDTH(&tile_outscale, config->dst_dim3_size);
    XAI_ARRAY_SET_HEIGHT(&tile_outscale, 1);
    XAI_ARRAY_SET_TYPE(&tile_outscale, XAI_ARRAY_U16);
    XAI_ARRAY_SET_CAPACITY(&tile_outscale, config->dst_dim3_size);

    // ========================================================================
    // Configure Output Tile Descriptor (points to system memory)
    // ========================================================================
    xai_tile3D tile_output;
    XAI_TILE3D_SET_BUFF_PTR(&tile_output, dst);
    XAI_TILE3D_SET_BUFF_SIZE(&tile_output, config->dst_dim2_pitch * config->dst_dim3_size);
    XAI_TILE3D_SET_DATA_PTR(&tile_output, dst);
    XAI_TILE3D_SET_DATA_ORDER(&tile_output, XAI_WHD);
    XAI_TILE3D_SET_TYPE(&tile_output, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(&tile_output, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(&tile_output, 0);
    XAI_TILE3D_SET_DIM1_PITCH(&tile_output, config->dst_dim1_pitch);
    XAI_TILE3D_SET_DIM2_PITCH(&tile_output, config->dst_dim2_pitch);
    XAI_TILE3D_SET_DIM1_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM1(&tile_output, config->dst_dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM1_EDGE2(&tile_output, 0);
    XAI_TILE3D_SET_DIM2_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM2(&tile_output, config->dst_dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM2_EDGE2(&tile_output, 0);
    XAI_TILE3D_SET_DIM3_COORD(&tile_output, 0);
    XAI_TILE3D_SET_DIM3(&tile_output, config->dst_dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(&tile_output, 0);
    XAI_TILE3D_SET_DIM3_EDGE2(&tile_output, 0);

    // ========================================================================
    // Configure Convolution Parameters
    // ========================================================================
    xai_cnn_conv_params params;
    XAI_CNN_CONV_SET_ACCUM_SHIFT(&params, config->accum_shift);
    XAI_CNN_CONV_SET_DILATION(&params, config->dilation);
    XAI_CNN_CONV_SET_FLAGS(&params, config->flags);
    XAI_CNN_CONV_SET_OUTPUT_SCALE(&params, config->output_scale);
    XAI_CNN_CONV_SET_OUTPUT_SHIFT(&params, config->output_shift);
    XAI_CNN_CONV_SET_RELU_MAX(&params, config->relu_max);
    XAI_CNN_CONV_SET_RELU_MIN(&params, config->relu_min);
    XAI_CNN_CONV_SET_STRIDEX(&params, config->stride_x);
    XAI_CNN_CONV_SET_STRIDEY(&params, config->stride_y);

    // ========================================================================
    // Execute convolution using generic system-memory API
    // This version accesses data through the processor cache
    // ========================================================================
    XAI_ERR_TYPE status = xaiConvolvedVQ3D(&tile_input, &tile_coeff, &tile_bias, 
                                            &tile_outscale, &tile_output, &params);
    
    return status;
}
