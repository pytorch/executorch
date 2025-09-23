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
  NatureDSP_Baseband library. Vector Mathematics.
    Softmax, floating-point data
*/
#include "api.h"
#include "common.h"
#include "expf_tbl.h"
#include "inff_tbl.h"
#include "nanf_tbl.h"

/*-------------------------------------------------------------------------
Softmax

Description: The function computes the softmax (normalized exponential
function) of input data. 16-bit fixed-point functions accept inputs in
Q3.12 and form outputs in Q7.8 format.

vsoftmax          16-bit
vsoftmax_fp16     IEEE-754 Std. half precision floating-point.
vsoftmaxf         IEEE-754 Std. single precision floating-point.

Accuracy:
2 LSB for fixed point API
2 ULP for floating point API
NOTE: Accuracy of function may depend on amount of data and their
distribution. Given accuracy is achieved for N=2 for any pair of
data from input domain.


Parameters:
Input
:
x[N]   input data, Q3.12 floating point
N      Length of input/output data vectors
Output:
y[N]   result, Q7.8 or floating point

Restrictions:
x,y    Must not overlap
-------------------------------------------------------------------------*/

#define IVP_ADDSN_2X32(b_, c_)                                                 \
  ({                                                                           \
    xb_vecN_2x32v a_;                                                          \
    xb_vecN_2x64w tmp_a_;                                                      \
    tmp_a_ = IVP_MULN_2X32(b_, 1);                                             \
    IVP_MULAN_2X32(tmp_a_, c_, 1);                                             \
    a_ = IVP_PACKVRN_2X64W(tmp_a_, 0);                                         \
    a_;                                                                        \
  })

#if !HAVE_VFPU
DISCARD_FUN(void, vsoftmaxf, (float32_t * y, const float32_t *x, int N))
#else
void vsoftmaxf(float32_t *y, const float32_t *x, int N) {
#if !defined(IVP_MULN_2X32)
#else
  const int *pTbl = (const int *)expftbl_Q30;
#endif
  const xb_vecN_2xf32 *restrict pX;
  xb_vecN_2xf32 *restrict pY;
  xb_vecN_2xf32 norm, ysum, xmax;
  int n;
  valign al_X, al_R, al_Y;
  if (N < 0)
    return;
  xmax = minusInff.f;
  pX = (const xb_vecN_2xf32 *)x;
  al_X = IVP_LAN_2XF32_PP(pX);
  al_Y = IVP_ZALIGN();
  for (n = 0; n < (N >> (LOG2_IVP_SIMD_WIDTH - 1)); n++) {
    xb_vecN_2xf32 x;
    IVP_LAN_2XF32_IP(x, al_X, pX);
    xmax = IVP_MAXNUMN_2XF32(xmax, x);
  }
  if (N & (IVP_SIMD_WIDTH / 2 - 1)) {
    xb_vecN_2xf32 x;
    IVP_LAVN_2XF32_XP(x, al_X, pX,
                      sizeof(float32_t) * (N & (IVP_SIMD_WIDTH / 2 - 1)));
    IVP_MAXNUMN_2XF32T(xmax, xmax, x,
                       IVP_LTRSN_2((N & (IVP_SIMD_WIDTH / 2 - 1))));
  }

  xmax = IVP_REPN_2XF32(IVP_RMAXNUMN_2XF32(xmax), 0);
  __Pragma("no_reorder");
  ysum = 0.f;
  pX = (const xb_vecN_2xf32 *)x;
  pY = (xb_vecN_2xf32 *)y;
  al_X = IVP_LAN_2XF32_PP(pX);
  {
    vboolN_2 bnan;
    bnan = IVP_LTRN_2I(0);
    for (n = 0; n < (N >> (LOG2_IVP_SIMD_WIDTH - 1)); n++) {
      xb_vecN_2xf32 x;
      IVP_LAN_2XF32_IP(x, al_X, pX);
      x = IVP_SUBN_2XF32(x, xmax);
      bnan |= IVP_UNN_2XF32(x, x);
      {
        xb_vecN_2xf32 gf, zout;
        xb_vecN_2x32v xin_i, fr, exp, t;
        xb_vecN_2x32v y, y1, y2, c1, c2, f2;
        xb_vecN_2x64w w;
        xin_i = IVP_TRUNCN_2XF32(x, 24);
        /* Multiply by 1/ln2, extract the integer and fractional (Q32)
         * components.     */
        /* Q54 <- Q24*Q30 */
        w = IVP_MULN_2X32(xin_i, invln2_Q30);
        exp = IVP_PACKVRNRN_2X64W(w, 54);
        fr = IVP_SRLN_2X32(IVP_PACKVRNRN_2X64W(w, 22), 1);
        /* polynomial for 2^x */
        f2 = IVP_PACKVRN_2X64W(IVP_MULN_2X32(fr, fr), 31);
        y1 = IVP_LSRN_2X32_I(pTbl, 0 * sizeof(int32_t));
        y2 = IVP_LSRN_2X32_I(pTbl, 1 * sizeof(int32_t));
        c1 = IVP_LSRN_2X32_I(pTbl, 2 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y1), 31);
        y1 = IVP_ADDSN_2X32(c1, t);
        c2 = IVP_LSRN_2X32_I(pTbl, 3 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y2), 31);
        y2 = IVP_ADDSN_2X32(c2, t);
        c1 = IVP_LSRN_2X32_I(pTbl, 4 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y1), 31);
        y1 = IVP_ADDSN_2X32(c1, t);
        c2 = IVP_LSRN_2X32_I(pTbl, 5 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y2), 31);
        y2 = IVP_ADDSN_2X32(c2, t);
        c1 = IVP_LSRN_2X32_I(pTbl, 6 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y1), 31);
        y1 = IVP_ADDSN_2X32(c1, t);
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(fr, y2), 31);
        y = IVP_ADDSN_2X32(y1, t);
        /* scale result to original exponent ignoring very low items */
        gf = IVP_FLOATN_2X32(y, 30);
        exp = IVP_SLLIN_2X32(IVP_MAXN_2X32(IVP_ADDN_2X32(127, exp), 0), 23);
        zout = IVP_MULN_2XF32(gf, IVP_MOVN_2XF32_FROMN_2X32(exp));
        x = zout;
      }
      ysum = IVP_ADDN_2XF32(ysum, x);
      IVP_SAN_2XF32_IP(x, al_Y, pY);
    }
    if (N & (IVP_SIMD_WIDTH / 2 - 1)) {
      xb_vecN_2xf32 x;
      IVP_LAVN_2XF32_XP(x, al_X, pX,
                        sizeof(float32_t) * (N & (IVP_SIMD_WIDTH / 2 - 1)));
      x = IVP_SUBN_2XF32(x, xmax);
      bnan |= IVP_UNN_2XF32(x, x);
      {
        xb_vecN_2xf32 gf, zout;
        xb_vecN_2x32v xin_i, fr, exp, t;
        xb_vecN_2x32v y, y1, y2, c1, c2, f2;
        xb_vecN_2x64w w;
        xin_i = IVP_TRUNCN_2XF32(x, 24);
        /* Multiply by 1/ln2, extract the integer and fractional (Q32)
         * components.     */
        /* Q54 <- Q24*Q30 */
        w = IVP_MULN_2X32(xin_i, invln2_Q30);
        exp = IVP_PACKVRNRN_2X64W(w, 54);
        fr = IVP_SRLN_2X32(IVP_PACKVRNRN_2X64W(w, 22), 1);
        /* polynomial for 2^x */
        f2 = IVP_PACKVRN_2X64W(IVP_MULN_2X32(fr, fr), 31);
        y1 = IVP_LSRN_2X32_I(pTbl, 0 * sizeof(int32_t));
        y2 = IVP_LSRN_2X32_I(pTbl, 1 * sizeof(int32_t));
        c1 = IVP_LSRN_2X32_I(pTbl, 2 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y1), 31);
        y1 = IVP_ADDSN_2X32(c1, t);
        c2 = IVP_LSRN_2X32_I(pTbl, 3 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y2), 31);
        y2 = IVP_ADDSN_2X32(c2, t);
        c1 = IVP_LSRN_2X32_I(pTbl, 4 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y1), 31);
        y1 = IVP_ADDSN_2X32(c1, t);
        c2 = IVP_LSRN_2X32_I(pTbl, 5 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y2), 31);
        y2 = IVP_ADDSN_2X32(c2, t);
        c1 = IVP_LSRN_2X32_I(pTbl, 6 * sizeof(int32_t));
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(f2, y1), 31);
        y1 = IVP_ADDSN_2X32(c1, t);
        t = IVP_PACKVRN_2X64W(IVP_MULN_2X32(fr, y2), 31);
        y = IVP_ADDSN_2X32(y1, t);
        /* scale result to original exponent ignoring very low items */
        gf = IVP_FLOATN_2X32(y, 30);
        exp = IVP_SLLIN_2X32(IVP_MAXN_2X32(IVP_ADDN_2X32(127, exp), 0), 23);
        zout = IVP_MULN_2XF32(gf, IVP_MOVN_2XF32_FROMN_2X32(exp));
        x = zout;
      }
      IVP_ADDN_2XF32T(ysum, ysum, x,
                      IVP_LTRSN_2((N & (IVP_SIMD_WIDTH / 2 - 1))));
      IVP_SAVN_2XF32_XP(x, al_Y, pY,
                        sizeof(float32_t) * (N & (IVP_SIMD_WIDTH / 2 - 1)));
    }
    IVP_SAPOSN_2XF32_FP(al_Y, pY);
    ysum = IVP_MOVN_2XF32T(qNaNf.f, ysum, bnan);
  }
  norm = XT_RECIP_S(IVP_RADDN_2XF32(ysum));
  __Pragma("no_reorder");
  pX = (const xb_vecN_2xf32 *)y;
  pY = (xb_vecN_2xf32 *)y;

  al_R = IVP_LAN_2XF32_PP(pX);

  for (n = 0; n < (N >> (LOG2_IVP_SIMD_WIDTH - 1)); n++) {
    xb_vecN_2xf32 x;
    IVP_LAN_2XF32_IP(x, al_R, pX);
    x = IVP_MULN_2XF32(x, norm);
    IVP_SAN_2XF32_IP(x, al_Y, pY);
  }
  if (N & (IVP_SIMD_WIDTH / 2 - 1)) {
    xb_vecN_2xf32 x;
    IVP_LAVN_2XF32_XP(x, al_R, pX,
                      sizeof(float32_t) * (N & (IVP_SIMD_WIDTH / 2 - 1)));
    x = IVP_MULN_2XF32(x, norm);
    IVP_SAVN_2XF32_XP(x, al_Y, pY,
                      sizeof(float32_t) * (N & (IVP_SIMD_WIDTH / 2 - 1)));
  }
  IVP_SAPOSN_2XF32_FP(al_Y, pY);

} /* vsoftmaxf() */
#endif
