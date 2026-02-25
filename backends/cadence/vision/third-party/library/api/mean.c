/* ------------------------------------------------------------------------ */
/* Copyright (c) 2024 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.  */
/* These coded instructions, statements, and computer programs ('Cadence    */
/* Libraries') are the copyrighted works of Cadence Design Systems Inc.     */
/* Cadence IP is licensed for use with Cadence processor cores only and     */
/* must not be used for any other processors and platforms. Your use of the */
/* Cadence Libraries is subject to the terms of the license agreement you   */
/* have entered into with Cadence Design Systems, or a sublicense granted   */
/* to you by a direct Cadence licensee.                                     */
/* ------------------------------------------------------------------------ */

/*-------------------------------------------------------------------------
  SIMD Mean Pooling Operations
  
  This module implements optimized mean pooling operations using Xtensa
  Vision DSP SIMD intrinsics for float32 data.
-------------------------------------------------------------------------*/

#include <xtensa/tie/xt_ivpn.h>
#include <stdint.h>
#include "api.h"
#include "common.h"
typedef float float32_t;

#ifndef IVP_SIMD_WIDTH
#define IVP_SIMD_WIDTH XCHAL_IVPN_SIMD_WIDTH
#endif



/*-------------------------------------------------------------------------
  SIMD Mean Pooling 2x2 -> 1x1
  
  Description: 
  This function implements mean pooling across 2x2 spatial dimensions for
  float32 data using Xtensa SIMD intrinsics.
  
  Input shape:  1 x C x 2 x 2 (batch=1, channels=C, height=2, width=2)
  Output shape: 1 x C x 1 x 1 (batch=1, channels=C, height=1, width=1)
  
  Algorithm:
  - Load 16 float32 elements at a time (4 channels x 2x2 spatial) in ONE vector
  - For each channel, compute mean of 4 spatial values (2x2)
  - Use SIMD vector operations for efficient computation
  
  With SIMD width N=32, xb_vecN_2xf32 holds 16 float32 values.
  Single load gets all 16 values: ch0[0,0], ch0[0,1], ch0[1,0], ch0[1,1], 
    ch1[0,0], ch1[0,1], ch1[1,0], ch1[1,1], ch2[0,0], ch2[0,1], ch2[1,0], ch2[1,1],
    ch3[0,0], ch3[0,1], ch3[1,0], ch3[1,1]
  
  Then shuffle to group elements from same channel together,
  sum them, and divide by 4 to get the mean.
  
  Parameters:
  Input:
    input[num_channels*4]   Input tensor in CHW format (channels, 2x2 spatial)
    num_channels            Number of input channels
  Output:
    output[num_channels]    Output tensor (channels, 1x1 spatial)
  
  Restrictions:
    - num_channels must be a multiple of 4
    - input and output must be aligned to 64-byte boundary
    - input and output must not overlap
    
-------------------------------------------------------------------------*/
void simd_mean_pool_2x2_to_1x1_float32(float32_t* restrict output, 
                                       const float32_t* restrict input,
                                       int N) 
{
    int n;
    xb_vecN_2xf32 vec0, vec1, vec2, vec3;
    xb_vecN_2xf32 vec0_0, vec0_1, vec1_0, vec1_1;
    xb_vecN_2xf32 v0, v1, v2, v3, sum_all, result;
    const xb_vecN_2xf32* restrict pInput = (const xb_vecN_2xf32*)input;
    xb_vecN_2xf32* restrict pOutput = (xb_vecN_2xf32*)output;
    
    if (N <= 0) return;
   
    __Pragma("no_reorder");
    __Pragma("loop_count min=1");
    
    for (n = 0; n < (N >> (LOG2_IVP_SIMD_WIDTH + 1)); n++) {
        // Load 64 float32 values (4 vectors) - 16 channels × 4 values each
        IVP_LVN_2XF32_IP(vec0, pInput, 2 * IVP_SIMD_WIDTH);  // 0-15
        IVP_LVN_2XF32_IP(vec1, pInput, 2 * IVP_SIMD_WIDTH);  // 16-31
        IVP_LVN_2XF32_IP(vec2, pInput, 2 * IVP_SIMD_WIDTH);  // 32-47
        IVP_LVN_2XF32_IP(vec3, pInput, 2 * IVP_SIMD_WIDTH);  // 48-63
        
        // First level: Deinterleave vec0-vec1 and vec2-vec3 pairs (independent)
        IVP_DSELN_2XF32I(vec0_0, vec0_1, vec1, vec0, IVP_DSELI_DEINTERLEAVE_2);
        IVP_DSELN_2XF32I(vec1_0, vec1_1, vec3, vec2, IVP_DSELI_DEINTERLEAVE_2);
        
        // Second level: Cross-deinterleave directly to final vectors
        IVP_DSELN_2XF32I(v2, v0, vec1_0, vec0_0, IVP_DSELI_DEINTERLEAVE_2);
        IVP_DSELN_2XF32I(v3, v1, vec1_1, vec0_1, IVP_DSELI_DEINTERLEAVE_2);
        
        // v0=vec3_1 (stride-4, mod 0), v1=vec2_1 (stride-4, mod 1)
        // v2=vec3_0 (stride-4, mod 2), v3=vec2_0 (stride-4, mod 3)
        
        // Fused add: ((v0 + v1) + (v2 + v3)) for better pipelining
        sum_all = IVP_ADDN_2XF32(IVP_ADDN_2XF32(v0, v1), IVP_ADDN_2XF32(v2, v3));
        
        // Multiply by 0.25 to get mean
        result = IVP_MULN_2XF32(sum_all, 0.25f);
        
        // Store result
        IVP_SVN_2XF32_IP(result, pOutput, 2 * IVP_SIMD_WIDTH);
    }
}
