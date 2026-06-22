/*
 * utils.h
 *
 *  Created on: Nov 4, 2025
 *      Author: sraut
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <stdint.h>
#include <xtensa/tie/xt_ivpn.h>
#include "../libxai_common/include/xai_tile_manager.h"


/**
 * @brief Increment iterator to temp with carry
 * @param temp Pointer to temporary variable
 * @param var Current value
 * @param bound Upper bound
 * @param carry Carry value
 * @return New carry value
 */

// required for windows
#undef min
#undef max
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }


static inline int inc_iter_to_temp(int *temp, int var, int bound, int carry) {
    int new_val = var + carry;
    carry = new_val == bound;
    *temp = carry ? 0 : new_val;
    return carry;
}

/**
 * @brief Swap two uint8_t buffer pointers
 * @param a Pointer to first buffer pointer
 * @param b Pointer to second buffer pointer
 */
static inline void swap_buffers(int8_t **a, int8_t **b) {
    int8_t *t = *a;
    *a = *b;
    *b = t;
}

static inline void _proto_FillBuffer_I8(void *buff, int val, unsigned size) {

  unsigned its = size / (2 * XCHAL_IVPN_SIMD_WIDTH);
  unsigned rem = size % (2 * XCHAL_IVPN_SIMD_WIDTH);
  xb_vec2Nx8 *pDst = (xb_vec2Nx8 *)buff;
  valign vaDst = IVP_ZALIGN();
  xb_vec2Nx8 pattern = IVP_MOVVA8(val);
  for (unsigned i = 0; i < its; i++) {
    IVP_SAV2NX8_XP(pattern, vaDst, pDst, 2 * XCHAL_IVPN_SIMD_WIDTH);
  }
  IVP_SAV2NX8_XP(pattern, vaDst, pDst, rem);
  IVP_SAPOS2NX8_FP(vaDst, pDst);
}

/**
 * @brief Setup a tile3D descriptor for cache-mode input tile
 * 
 * This initializes a tile3D structure pointing to a local buffer with
 * proper dimensions, edges, and pitches for convolution operations.
 * Used by cache-mode executors where input is copied to SRAM scratch buffer.
 * 
 * @param tile Pointer to tile3D descriptor to initialize
 * @param buffer Pointer to data buffer (in SRAM)
 * @param dim1_size Width without padding
 * @param dim2_size Height without padding
 * @param dim3_size Channels
 * @param edge1 Edge padding on left/top
 * @param edge2 Edge padding on right/bottom
 * @param stride_alignment Pitch alignment (typically 2*XCHAL_IVPN_SIMD_WIDTH)
 */
static inline void setup_tile3d_cache_input(
    xai_tile3D* tile,
    int8_t* buffer,
    int dim1_size,        // Width (W)
    int dim2_size,        // Height (H)
    int dim3_size,        // Channels (D)
    int dim1_edge1,       // Left edge
    int dim1_edge2,       // Right edge
    int dim2_edge1,       // Top edge
    int dim2_edge2,       // Bottom edge
    int dim3_edge1,       // Channel edge start
    int dim3_edge2,       // Channel edge end
    int stride_alignment  // Pitch alignment
) {
    // Calculate padded dimensions
    int padded_dim1 = dim1_size + dim1_edge1 + dim1_edge2;
    int padded_dim2 = dim2_size + dim2_edge1 + dim2_edge2;
    int padded_dim3 = dim3_size + dim3_edge1 + dim3_edge2;
    
    // Calculate aligned pitch for dim1
    int dim1_pitch = padded_dim1;
    if (stride_alignment > 0) {
        dim1_pitch = (padded_dim1 + stride_alignment - 1) & ~(stride_alignment - 1);
    }
    
    // Calculate pitch for dim2
    int dim2_pitch = dim1_pitch * padded_dim2;
    
    // Calculate total buffer size
    int buffer_size = dim2_pitch * padded_dim3;
    
    // Initialize tile descriptor
    XAI_TILE3D_SET_BUFF_PTR(tile, buffer);
    XAI_TILE3D_SET_BUFF_SIZE(tile, buffer_size);
    XAI_TILE3D_SET_DATA_PTR(tile, buffer + (dim3_edge1 * dim2_pitch) + 
                                           (dim2_edge1 * dim1_pitch) + 
                                           dim1_edge1);
    XAI_TILE3D_SET_DATA_ORDER(tile, XAI_WHD);
    XAI_TILE3D_SET_TYPE(tile, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(tile, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(tile, 0);
    
    // Set dimensions
    XAI_TILE3D_SET_DIM1(tile, dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(tile, dim1_edge1);
    XAI_TILE3D_SET_DIM1_EDGE2(tile, dim1_edge2);
    XAI_TILE3D_SET_DIM1_PITCH(tile, dim1_pitch);
    XAI_TILE3D_SET_DIM1_COORD(tile, 0);
    
    XAI_TILE3D_SET_DIM2(tile, dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(tile, dim2_edge1);
    XAI_TILE3D_SET_DIM2_EDGE2(tile, dim2_edge2);
    XAI_TILE3D_SET_DIM2_PITCH(tile, dim2_pitch);
    XAI_TILE3D_SET_DIM2_COORD(tile, 0);
    
    XAI_TILE3D_SET_DIM3(tile, dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(tile, dim3_edge1);
    XAI_TILE3D_SET_DIM3_EDGE2(tile, dim3_edge2);
    XAI_TILE3D_SET_DIM3_COORD(tile, 0);
}

/**
 * @brief Setup source tile descriptor for raw input data (before copy)
 * 
 * Used to describe the source input data in system memory before
 * copying to the padded SRAM tile.
 */
static inline void setup_tile3d_source(
    xai_tile3D* tile,
    int8_t* buffer,
    int dim1_size,
    int dim2_size,
    int dim3_size,
    int dim1_pitch,
    int dim2_pitch
) {
    XAI_TILE3D_SET_BUFF_PTR(tile, buffer);
    XAI_TILE3D_SET_BUFF_SIZE(tile, dim2_pitch * dim3_size);
    XAI_TILE3D_SET_DATA_PTR(tile, buffer);
    XAI_TILE3D_SET_DATA_ORDER(tile, XAI_WHD);
    XAI_TILE3D_SET_TYPE(tile, XAI_TILE3D_S8);
    XAI_TILE3D_SET_FRAME_PTR(tile, 0);
    XAI_TILE3D_SET_STATUS_FLAGS(tile, 0);
    
    XAI_TILE3D_SET_DIM1(tile, dim1_size);
    XAI_TILE3D_SET_DIM1_EDGE1(tile, 0);
    XAI_TILE3D_SET_DIM1_EDGE2(tile, 0);
    XAI_TILE3D_SET_DIM1_PITCH(tile, dim1_pitch);
    XAI_TILE3D_SET_DIM1_COORD(tile, 0);
    
    XAI_TILE3D_SET_DIM2(tile, dim2_size);
    XAI_TILE3D_SET_DIM2_EDGE1(tile, 0);
    XAI_TILE3D_SET_DIM2_EDGE2(tile, 0);
    XAI_TILE3D_SET_DIM2_PITCH(tile, dim2_pitch);
    XAI_TILE3D_SET_DIM2_COORD(tile, 0);
    
    XAI_TILE3D_SET_DIM3(tile, dim3_size);
    XAI_TILE3D_SET_DIM3_EDGE1(tile, 0);
    XAI_TILE3D_SET_DIM3_EDGE2(tile, 0);
    XAI_TILE3D_SET_DIM3_COORD(tile, 0);
}

#endif /* UTILS_H_ */
