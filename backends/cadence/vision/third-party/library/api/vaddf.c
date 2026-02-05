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
/*  IntegrIT, Ltd.   www.integrIT.com, info@integrIT.com                    */
/*                                                                          */
/* NatureDSP_Baseband Library                                               */
/*                                                                          */
/* This library contains copyrighted materials, trade secrets and other     */
/* proprietary information of IntegrIT, Ltd. This software is licensed for  */
/* use with Cadence processor cores only and must not be used for any other */
/* processors and platforms. The license to use these sources was given to  */
/* Cadence, Inc. under Terms and Condition of a Software License Agreement  */
/* between Cadence, Inc. and IntegrIT, Ltd.                                 */
/* ------------------------------------------------------------------------ */
/*          Copyright (C) 2009-2022 IntegrIT, Limited.                      */
/*                      All Rights Reserved.                                */
/* ------------------------------------------------------------------------ */
/*
  NatureDSP_Baseband library. Vector Operations
    Real Vectors Sum
*/

/* Cross-platform data type definitions. */
/* Common helper macros. */
#include "api.h"
#include "common.h"
#define IVP_SIMD_WIDTH XCHAL_IVPN_SIMD_WIDTH
/* Vector Operations. */


/*-------------------------------------------------------------------------
Real Vectors Sum

Description: These routines perform pairwise summation of real vectors.

Representation:
rvadd        Signed fixed-point format. 16-bit inputs, 16-bit saturated results
rvadd_32b    Signed fixed-point format. 32-bit inputs, 32-bit saturated results
rvadd_fp16   IEEE-754 Std. half precision floating-point format for
             input/output data
rvaddf       IEEE-754 Std. single precision floating-point format for
             input/output data
rvadd_f64    IEEE-754 Std. double precision floating-point format for
             input/output data

Parameters:
Input:
x[N]   Input vector
y[N]   Input vector
N      Length of vectors
Output:
z[N]   Sum of input vectirs

Restrictions:
z,x,y  Must not overlap
z,x,y  Aligned on 2*BBE_SIMD_WIDTH-byte boundary
N      Multiple of BBE_SIMD_WIDTH (rvadd,rvadd_fp16)
       Multiple of BBE_SIMD_WIDTH/2 (rvadd_32b, rvaddf)
       Multiple of BBE_SIMD_WIDTH/4 (rvadd_f64)
-------------------------------------------------------------------------*/
void rvaddf(float32_t *restrict z, const float32_t *restrict x,
            const float32_t *restrict y, int N) {
#if (1)
  int n;
  xb_vecN_2xf32 x0, y0, z0;
  xb_vecN_2xf32 x1, y1, z1;
  const xb_vecN_2xf32 *restrict pX = (const xb_vecN_2xf32 *)x;
  const xb_vecN_2xf32 *restrict pY = (const xb_vecN_2xf32 *)y;
  xb_vecN_2xf32 *restrict pZ = (xb_vecN_2xf32 *)z;
  NASSERT_ALIGN(x, (2 * IVP_SIMD_WIDTH));
  NASSERT_ALIGN(y, (2 * IVP_SIMD_WIDTH));
  NASSERT_ALIGN(z, (2 * IVP_SIMD_WIDTH));
  NASSERT(N % (IVP_SIMD_WIDTH / 2) == 0);
  if (N <= 0)
    return;
  __Pragma("no_reorder");
  __Pragma("no_reorder");

  for (n = 0; n < (N >> (LOG2_IVP_SIMD_WIDTH-1)); n++) {
    IVP_LVN_2XF32_IP(x0, pX, 2 * IVP_SIMD_WIDTH);
    IVP_LVN_2XF32_IP(y0, pY, 2 * IVP_SIMD_WIDTH);
    z0 = IVP_ADDN_2XF32(x0, y0);
    IVP_SVN_2XF32_IP(z0, pZ, 2 * IVP_SIMD_WIDTH);
  }

  if (N & ((IVP_SIMD_WIDTH>>1) - 1)) {
	  valign vx0 = IVP_LAN_2XF32_PP(pX);
	  valign vy0 = IVP_LAN_2XF32_PP(pY);
	  valign vz0 = IVP_ZALIGN();

	  IVP_LAVN_2XF32_XP(x0, vx0, pX, 2 * ((IVP_SIMD_WIDTH>>1) - 1));
	  IVP_LAVN_2XF32_XP(y0, vy0, pY, 2 * ((IVP_SIMD_WIDTH>>1) - 1));
	  z0 = IVP_ADDN_2XF32(x0, y0);
	  IVP_SAVN_2XF32_XP(z0, vz0, pZ, 2 * ((IVP_SIMD_WIDTH>>1) - 1));
	  IVP_SAPOSN_2XF32_FP(vz0, pZ);
  }
#else
  int n;
  xtfloat x0, y0, z0;
  const xtfloat *restrict pX = (const xtfloat *)x;
  const xtfloat *restrict pY = (const xtfloat *)y;
  xtfloat *restrict pZ = (xtfloat *)z;
  NASSERT_ALIGN(x, (2 * IVP_SIMD_WIDTH));
  NASSERT_ALIGN(y, (2 * IVP_SIMD_WIDTH));
  NASSERT_ALIGN(z, (2 * IVP_SIMD_WIDTH));
  NASSERT(N % (IVP_SIMD_WIDTH / 2) == 0);
  if (N <= 0)
    return;

  for (n = 0; n < (N); n++) {
    XT_LSIP(x0, pX, sizeof(xtfloat));
    XT_LSIP(y0, pY, sizeof(xtfloat));
    z0 = XT_ADD_S(x0, y0);
    XT_SSIP(z0, pZ, sizeof(xtfloat));
  }
#endif
}
