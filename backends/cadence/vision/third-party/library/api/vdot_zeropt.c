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

#include <stdio.h>
#include "api.h"
#include "common.h"

// Macro to emulate reduction of N 32-bit elements from Nx48
#define IVP_RADDNX32W_EMULATED(vecNx48) ({ \
  xb_vecN_2x32v q0 = IVP_CVT32SNX48H(vecNx48); \
  xb_vecN_2x32v q1 = IVP_CVT32SNX48L(vecNx48); \
  xb_int32v s0 = IVP_RADDN_2X32(q0); \
  xb_int32v s1 = IVP_RADDN_2X32(q1); \
  s0 + s1; \
})

/*-------------------------------------------------------------------------
  Vector Dot Product with Zero-Point Subtraction

  Description: This routine performs dot product of two quantized int8 vectors
  with zero-point subtraction applied before multiplication:
    result = init_acc + sum((x[i] - x_zp) * (y[i] - y_zp)) for i=0..N-1

  This is commonly used in quantized neural network operations where
  zero-point offset needs to be removed before computation.

  Representation:
  rvdot_zeropt   Signed fixed-point format. 8-bit inputs, 32-bit result

  Parameters:
  Input:
  init_acc  Initial accumulator value (int32)
  x[N]      Input vector (int8)
  y[N]      Input vector (int8)
  x_zp      Zero-point for x vector (int8)
  y_zp      Zero-point for y vector (int8)
  N         Length of vectors

  Output:
            Returns 32-bit accumulated dot product result

  Restrictions:
  x,y       Aligned on 2*BBE_SIMD_WIDTH-byte boundary preferred
  N         Any positive value (tail handling included)
-------------------------------------------------------------------------*/
int32_t rvdot_zeropt(
    int32_t init_acc,
    const int8_t *restrict x,
    const int8_t *restrict y,
    int8_t x_zp,
    int8_t y_zp,
    int N) {
  
  const xb_vecNx8 *restrict pX = (const xb_vecNx8 *)x;
  const xb_vecNx8 *restrict pY = (const xb_vecNx8 *)y;
  
  xb_vecNx48 acc = 0;  // Initialize accumulator to zero
  xb_vecNx16 vx, vy;
  xb_vecNx16 vx_shifted, vy_shifted;
  
  int k;
  
  if (N <= 0)
    return init_acc;
  
  // Process in chunks of IVP_SIMD_WIDTH (typically 32 elements) using Nx16
  for (k = 0; k < (N >> LOG2_IVP_SIMD_WIDTH); k++) {
    // Load vectors as Nx8 with sign-extension to Nx16 (loads N int8 elements)
    IVP_LVNX8S_IP(vx, pX, IVP_SIMD_WIDTH);
    IVP_LVNX8S_IP(vy, pY, IVP_SIMD_WIDTH);
    
    // Subtract zero-points in 16-bit: (x - x_zp), (y - y_zp)
    vx_shifted = IVP_SUBNX16(vx, (int16_t)x_zp);
    vy_shifted = IVP_SUBNX16(vy, (int16_t)y_zp);
    
    // Multiply-accumulate: acc += (x - x_zp) * (y - y_zp)
    IVP_MULANX16(acc, vx_shifted, vy_shifted);
  }
  
  // Handle remaining elements with SIMD
  int processed = k << LOG2_IVP_SIMD_WIDTH;
  int remaining = N - processed;
  
  if (remaining > 0) {
    valign vaX = IVP_LANX8S_PP((const xb_vecNx8 *)pX);
    valign vaY = IVP_LANX8S_PP((const xb_vecNx8 *)pY);
    
    // Load remaining elements with variable alignment
    IVP_LAVNX8S_XP(vx, vaX, (const xb_vecNx8 *)pX, remaining);
    IVP_LAVNX8S_XP(vy, vaY, (const xb_vecNx8 *)pY, remaining);
    
    // Subtract zero-points in 16-bit
    vx_shifted = IVP_SUBNX16(vx, (int16_t)x_zp);
    vy_shifted = IVP_SUBNX16(vy, (int16_t)y_zp);
    
    // Create mask for valid elements (true for indices < remaining)
    vboolN mask = IVP_LTNX16(IVP_SEQNX16(), remaining);
    
    // Zero out invalid positions: keep valid values, replace invalid with 0
    vx_shifted = IVP_MOVNX16T(vx_shifted, IVP_ZERONX16(), mask);
    vy_shifted = IVP_MOVNX16T(vy_shifted, IVP_ZERONX16(), mask);
    
    // Multiply-accumulate for tail (accumulate into same acc)
    // Invalid positions are 0*0 = 0, so they don't contribute
    IVP_MULANX16(acc, vx_shifted, vy_shifted);
  }
  
  // Reduce accumulator to single int32 (after all elements processed)
  int32_t result = IVP_RADDNX32W_EMULATED(acc);
  
  // Add initial accumulator value
  result += init_acc;
  
  return result;
}
