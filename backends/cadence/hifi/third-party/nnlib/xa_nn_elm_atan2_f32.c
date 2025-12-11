/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
#include <float.h>

#include "NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "xa_nn_common.h"

/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"

#include "xa_nnlib_common.h"

const union ufloat32uint32 xa_nnlib_plusInff_local = {0x7f800000};
const union ufloat32uint32 xa_nnlib_qNaNf = {0x7fc00000};
const union ufloat32uint32 pif = {0x40490fdb}; /* pi */
const union ufloat32uint32 pi2f = {0x3fc90fdb}; /* pi/2 */

const union ufloat32uint32 ALIGN(8) xa_nnlib_atanftbl1[8] = {
    {0x3dbc14c0}, /* 9.183645248413086e-002 */
    {0xbe30c39c}, /*-1.726211905479431e-001 */
    {0x3b2791e4}, /* 2.556913532316685e-003 */
    {0x3e4dac9d}, /* 2.008537799119949e-001 */
    {0xb97d9a57}, /*-2.418545627733693e-004 */
    {0xbeaaa7b5}, /*-3.333107531070709e-001 */
    {0xb54f34c8}, /*-7.719031600572635e-007 */
    {0x31cf3fa2} /* 6.031727117772334e-009 */
};

const union ufloat32uint32 ALIGN(8) xa_nnlib_atanftbl2[8] = {
    {0xbcccc037}, /*-2.499399892985821e-002 */
    {0x3e217c35}, /* 1.577003747224808e-001 */
    {0xbecf4163}, /*-4.047957360744476e-001 */
    {0x3ef7b762}, /* 4.838209748268127e-001 */
    {0xbdf35059}, /*-1.188055947422981e-001 */
    {0xbe9b8b75}, /*-3.037983477115631e-001 */
    {0xbb80ed5c}, /*-3.934545442461968e-003 */
    {0x3956fc52} /* 2.050262701231986e-004 */
};

#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN(
    void,
    xa_nn_elm_atan2_f32,
    (FLOAT32 * z, const FLOAT32* y, const FLOAT32* x, int N))
#elif HAVE_VFPU
#define sz_f32 (int)sizeof(FLOAT32)

/*===========================================================================
  Vector matematics:
  vec_atan2          full quadrant Arctangent
===========================================================================*/

/*-------------------------------------------------------------------------
  Full-Quadrant Arc Tangent
  The functions compute the arc tangent of the ratios y[N]/x[N] and store the
  result to output vector z[N].
  Floating point functions output is in radians. Fixed point functions
  scale its output by pi.

  NOTE:
  1. Scalar floating point function is compatible with standard ANSI C routines
and set errno and exception flags accordingly
  2. Scalar floating point function assigns EDOM to errno whenever y==0 and
x==0.

  Accuracy:
  24 bit version: 768 (3.57e-7)
  floating point: 2 ULP

  Special cases:
       y    |   x   |  result   |  extra conditions
    --------|-------|-----------|---------------------
     +/-0   | -0    | +/-pi     |
     +/-0   | +0    | +/-0      |
     +/-0   |  x    | +/-pi     | x<0
     +/-0   |  x    | +/-0      | x>0
     y      | +/-0  | -pi/2     | y<0
     y      | +/-0  |  pi/2     | y>0
     +/-y   | -inf  | +/-pi     | finite y>0
     +/-y   | +inf  | +/-0      | finite y>0
     +/-inf | x     | +/-pi/2   | finite x
     +/-inf | -inf  | +/-3*pi/4 |
     +/-inf | +inf  | +/-pi/4   |

  Input:
    y[N]  vector of numerator values, Q31 or floating point
    x[N]  vector of denominator values, Q31 or floating point
    N     length of vectors
  Output:
    z[N]  results, Q31 or floating point

---------------------------------------------------------------------------*/

void xa_nn_elm_atan2_f32(
    FLOAT32* z,
    const FLOAT32* y,
    const FLOAT32* x,
    WORD32 N) {
  /*
    const union ufloat32uint32* p;
    int sx,sy,big;
    sx=takesignf(x);
    sy=takesignf(y);
    x=fabs(x);
    y=fabs(y);
    if(x==0.f && y==0.f)
    {
      // The actual result depends on input signs.
      x = 1.f;
      y = 0.f;
    }

    big=x>y;
    if(big)
    {
        x=y/x;
    }
    else
    {
      // compare x==y is necessary to support (+/-Inf, +/-Inf) cases
      x = (x == y) ? 1.0f : x / y;
    }
    p = (x<0.5f) ? atanftbl1 : atanftbl2;
    // approximate atan(x)/x-1
    y = p[0].f;
    y = x*y + p[1].f;
    y = x*y + p[2].f;
    y = x*y + p[3].f;
    y = x*y + p[4].f;
    y = x*y + p[5].f;
    y = x*y + p[6].f;
    y = x*y + p[7].f;
    // convert result to true atan(x)
    y = x*y + x;

    if (!big) y = pi2f.f - y;
    if (sx)   y = pif.f - y;
    if (sy)   y = -y;
    return   y;
  */

  const xtfloatx2* X;
  const xtfloatx2* Y;
  xtfloatx2* restrict Z;
  const xtfloatx2* S_rd;
  xtfloatx2* restrict S_wr;

  ae_valign X_va, Y_va, Z_va;

  /* Current block index; overall number of blocks; number of values in the
   * current block */
  int blkIx, blkNum, blkLen;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ / sz_f32;
  /* Allocate a fixed-size scratch area on the stack. */
  FLOAT32 ALIGN(8) scr[blkSize];

  int n;

  if (N <= 0)
    return;

  NASSERT_ALIGN8(scr);

  /*
   * Data are processed in blocks of scratch area size. Further, the algorithm
   * implementation is splitted in order to feed the optimizing compiler with a
   * few loops of managable size.
   */

  blkNum = (N + blkSize - 1) / blkSize;

  for (blkIx = 0; blkIx < blkNum; blkIx++) {
    blkLen = XT_MIN(N - blkIx * blkSize, blkSize);

    /*
     * Part I, reduction to [0,pi/4]. Reference C code:
     *
     *   {
     *     float32_t x0, y0, p0;
     *
     *     for ( n=0; n<blkLen; n++ )
     *     {
     *       x0 = fabsf( x[blkIx*blkSize+n] );
     *       y0 = fabsf( y[blkIx*blkSize+n] );
     *
     *       // The actual result depends on input signs.
     *       if ( x0==0.f && y0==0.f ) { x0 = 1.f; y0 = 0.f; };
     *
     *       if ( x0>y0 ) p0 = y0/x0;
     *       // Special case of x==y is necessary to support (+/-Inf, +/-Inf)
     * cases. else p0 = ( x0==y0 ? 1.f : x0/y0 );
     *
     *       scr[n] = p0;
     *     }
     *   }
     */

    {
      /* Input values */
      xtfloatx2 x0, y0;
      /* Numerator; denominator; reciprocal; quotient */
      xtfloatx2 num, den, rcp, quo;
      /* Scaling factor; error term */
      xtfloatx2 scl, eps;
      /* Is NaN; Inf/Inf; x/Inf; 0/0; x and y are subnormal */
      xtbool2 b_nan, b_num_inf, b_den_inf, b_eqz, b_subn;

      X = (xtfloatx2*)((uintptr_t)x + blkIx * blkSize * sz_f32);
      Y = (xtfloatx2*)((uintptr_t)y + blkIx * blkSize * sz_f32);
      S_wr = (xtfloatx2*)scr;

      X_va = XT_LASX2PP(X);
      Y_va = XT_LASX2PP(Y);

      __Pragma("loop_count min=1");
      for (n = 0; n < (blkLen + 1) / 2; n++) {
        XT_LASX2IP(x0, X_va, X);
        XT_LASX2IP(y0, Y_va, Y);

        /* Replicate NaNs in both x and y to ensure NaN propagation. */
        b_nan = XT_UN_SX2(x0, y0);
        XT_MOVT_SX2(x0, xa_nnlib_qNaNf.f, b_nan);
        XT_MOVT_SX2(y0, xa_nnlib_qNaNf.f, b_nan);

        x0 = XT_ABS_SX2(x0);
        y0 = XT_ABS_SX2(y0);

        /* num <= den */
        num = XT_MIN_SX2(x0, y0);
        den = XT_MAX_SX2(y0, x0);

        /* Scale up numerator and denominator if BOTH are subnormal. */
        b_subn = XT_OLT_SX2(num, FLT_MIN);
        scl = (xtfloatx2)8388608.f;
        XT_MOVF_SX2(scl, (xtfloatx2)1.0f, b_subn);
        num = XT_MUL_SX2(num, scl);
        den = XT_MUL_SX2(den, scl);

        /* Classify numerator and denominator. */
        b_num_inf = XT_OEQ_SX2(num, xa_nnlib_plusInff_local.f); /* Inf/Inf */
        b_den_inf = XT_OEQ_SX2(den, xa_nnlib_plusInff_local.f); /* x/Inf   */
        b_eqz = XT_OEQ_SX2(den, (xtfloatx2)(xtfloatx2)(0.0f)); /* 0/0     */

        /* Initial appromimation for 1/den. */
        rcp = XT_RECIP0_SX2(den);
        /* Newton-Raphson iteration for 1/den. */
        eps = (xtfloatx2)1.0f;
        XT_MSUB_SX2(eps, rcp, den);
        XT_MADD_SX2(rcp, rcp, eps);
        /* Approximation for the quotient num/den. */
        quo = XT_MUL_SX2(num, rcp);
        /* Refine the quotient by a modified Newton-Raphson iteration. */
        eps = num;
        XT_MSUB_SX2(eps, quo, den);
        XT_MADD_SX2(quo, rcp, eps);

        /* Force conventional results for special cases. */
        XT_MOVT_SX2(quo, (xtfloatx2)(0.0f), b_den_inf); /* x/Inf -> 0   */
        XT_MOVT_SX2(quo, (xtfloatx2)1.0f, b_num_inf); /* Inf/Inf -> 1 */
        XT_MOVT_SX2(quo, (xtfloatx2)(0.0f), b_eqz); /* 0/0 -> 0     */

        XT_SSX2IP(quo, S_wr, +2 * sz_f32);
      }
    }

    __Pragma("no_reorder");

    /*
     * Part II, polynomial approximation and full quadrant restoration.
     * Reference C code:
     *
     *   {
     *     const union ufloat32uint32 * ptbl;
     *     float32_t x0, y0, z0, p0;
     *     int sx, sy;
     *
     *     for ( n=0; n<blkLen; n++ )
     *     {
     *       x0 = x[blkIx*blkSize+n];
     *       y0 = y[blkIx*blkSize+n];
     *       p0 = scr[n];
     *
     *       sx = takesignf( x0 ); x0 = fabsf( x0 );
     *       sy = takesignf( y0 ); y0 = fabsf( y0 );
     *
     *       ptbl = ( p0<0.5f ? atanftbl1 : atanftbl2 );
     *
     *       // Approximate atan(p)/p-1
     *       z0 = ptbl[0].f;
     *       z0 = ptbl[1].f + p0*z0;
     *       z0 = ptbl[2].f + p0*z0;
     *       z0 = ptbl[3].f + p0*z0;
     *       z0 = ptbl[4].f + p0*z0;
     *       z0 = ptbl[5].f + p0*z0;
     *       z0 = ptbl[6].f + p0*z0;
     *       z0 = ptbl[7].f + p0*z0;
     *       z0 =        p0 + p0*z0;
     *
     *       if ( x0<y0 ) z0 = pi2f.f - z0;
     *       if ( sx    ) z0 = pif.f - z0;
     *       if ( sy    ) z0 = -z0;
     *
     *       z[blkIx*blkSize+n] = z0;
     *     }
     *   }
     */

    {
      /* Input values; output value; reducted input value and its 2nd power. */
      xtfloatx2 x0, y0, z0, z1, p0, p1;
      /* Temporary; input value signs */
      ae_int32x2 t, sx, sy;
      /* Polynomial coeffs for 0.f<=p<0.5f (#1) and 0.5f<=p<=1.f (#2). */
      xtfloatx2 cf1_0, cf1_1, cf1_2, cf1_3, cf1_4, cf1_5, cf1_6, cf1_7;
      xtfloatx2 cf2_0, cf2_1, cf2_2, cf2_3, cf2_4, cf2_5, cf2_6, cf2_7;
      /* Selected polynomial coeffs. */
      xtfloatx2 cf0, cf1, cf2, cf3, cf4, cf5, cf6, cf7;
      /* x less than y; x is negative; p is less than 0.5f. */
      xtbool2 b_xlty, b_sx, b_lt05;

      X = (xtfloatx2*)((uintptr_t)x + blkIx * blkSize * sz_f32);
      Y = (xtfloatx2*)((uintptr_t)y + blkIx * blkSize * sz_f32);
      Z = (xtfloatx2*)((uintptr_t)z + blkIx * blkSize * sz_f32);

      S_rd = (xtfloatx2*)scr;

      X_va = XT_LASX2PP(X);
      Y_va = XT_LASX2PP(Y);
      Z_va = AE_ZALIGN64();

      for (n = 0; n < blkLen / 2; n++) {
        XT_LASX2IP(x0, X_va, X);
        XT_LASX2IP(y0, Y_va, Y);

        /* Keep sign of x as a boolean. */
        sx = XT_AE_MOVINT32X2_FROMXTFLOATX2(x0);
        b_sx = AE_LT32(sx, AE_ZERO32());

        /* Keep y sign as a binary value. */
        sy = XT_AE_MOVINT32X2_FROMXTFLOATX2(y0);
        sy = AE_SRLI32(sy, 31);
        sy = AE_SLLI32(sy, 31);

        x0 = XT_ABS_SX2(x0);
        y0 = XT_ABS_SX2(y0);
        b_xlty = XT_OLT_SX2(x0, y0);

        XT_LSX2IP(p0, S_rd, +2 * sz_f32);

        b_lt05 = XT_OLT_SX2(p0, (xtfloatx2)0.5f);

        /* Reload coeff sets on each iteration. */
        cf1_0 = XT_LSI((xtfloat*)xa_nnlib_atanftbl1, 0 * sz_f32);
        cf1_1 = XT_LSI((xtfloat*)xa_nnlib_atanftbl1, 1 * sz_f32);
        cf1_2 = XT_LSI((xtfloat*)xa_nnlib_atanftbl1, 2 * sz_f32);
        cf1_3 = XT_LSI((xtfloat*)xa_nnlib_atanftbl1, 3 * sz_f32);
        cf1_4 = XT_LSI((xtfloat*)xa_nnlib_atanftbl1, 4 * sz_f32);
        cf1_5 = XT_LSI((xtfloat*)xa_nnlib_atanftbl1, 5 * sz_f32);
        cf1_6 = XT_LSI((xtfloat*)xa_nnlib_atanftbl1, 6 * sz_f32);
        cf1_7 = XT_LSI((xtfloat*)xa_nnlib_atanftbl1, 7 * sz_f32);

        cf2_0 = XT_LSI((xtfloat*)xa_nnlib_atanftbl2, 0 * sz_f32);
        cf2_1 = XT_LSI((xtfloat*)xa_nnlib_atanftbl2, 1 * sz_f32);
        cf2_2 = XT_LSI((xtfloat*)xa_nnlib_atanftbl2, 2 * sz_f32);
        cf2_3 = XT_LSI((xtfloat*)xa_nnlib_atanftbl2, 3 * sz_f32);
        cf2_4 = XT_LSI((xtfloat*)xa_nnlib_atanftbl2, 4 * sz_f32);
        cf2_5 = XT_LSI((xtfloat*)xa_nnlib_atanftbl2, 5 * sz_f32);
        cf2_6 = XT_LSI((xtfloat*)xa_nnlib_atanftbl2, 6 * sz_f32);
        cf2_7 = XT_LSI((xtfloat*)xa_nnlib_atanftbl2, 7 * sz_f32);

        /* Select coeffs from sets #1, #2 by reducted input value. */
        cf0 = cf1_0;
        XT_MOVF_SX2(cf0, cf2_0, b_lt05);
        cf1 = cf1_1;
        XT_MOVF_SX2(cf1, cf2_1, b_lt05);
        cf2 = cf1_2;
        XT_MOVF_SX2(cf2, cf2_2, b_lt05);
        cf3 = cf1_3;
        XT_MOVF_SX2(cf3, cf2_3, b_lt05);
        cf4 = cf1_4;
        XT_MOVF_SX2(cf4, cf2_4, b_lt05);
        cf5 = cf1_5;
        XT_MOVF_SX2(cf5, cf2_5, b_lt05);
        cf6 = cf1_6;
        XT_MOVF_SX2(cf6, cf2_6, b_lt05);
        cf7 = cf1_7;
        XT_MOVF_SX2(cf7, cf2_7, b_lt05);

        /*
         * Compute the approximation to z(p) = atan(p)/p-1. Here we use a
         * combination of Estrin's rule and Horner's method of polynomial
         * evaluation to shorten the dependency path at the cost of additional
         * multiplication.
         */

        XT_MADD_SX2(cf1, cf0, p0);
        cf0 = cf1;
        XT_MADD_SX2(cf3, cf2, p0);
        cf1 = cf3;
        XT_MADD_SX2(cf5, cf4, p0);
        cf2 = cf5;
        XT_MADD_SX2(cf7, cf6, p0);
        cf3 = cf7;

        p1 = XT_MUL_SX2(p0, p0);

        z0 = cf0;
        XT_MADD_SX2(cf1, z0, p1);
        z0 = cf1;
        XT_MADD_SX2(cf2, z0, p1);
        z0 = cf2;
        XT_MADD_SX2(cf3, z0, p1);
        z0 = cf3;

        XT_MADD_SX2(p0, p0, z0);
        z0 = p0;

        /* if ( x0<y0 ) z0 = pi2f.f - z0; */
        z1 = XT_SUB_SX2(pi2f.f, z0);
        XT_MOVT_SX2(z0, z1, b_xlty);
        /* if ( sx ) z0 = pif.f - z0; */
        z1 = XT_SUB_SX2(pif.f, z0);
        XT_MOVT_SX2(z0, z1, b_sx);
        /* if ( sy ) z0 = -z0;*/
        t = XT_AE_MOVINT32X2_FROMXTFLOATX2(z0);
        t = AE_XOR32(t, sy);
        z0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(t);

        XT_SASX2IP(z0, Z_va, Z);
      }

      XT_SASX2POSFP(Z_va, Z);

      /* Deliberately process the last input value if it's even-numbered. */
      if (blkLen & 1) {
        x0 = XT_LSI((xtfloat*)X, 0);
        y0 = XT_LSI((xtfloat*)Y, 0);

        /* Keep sign of x as a boolean. */
        sx = XT_AE_MOVINT32X2_FROMXTFLOATX2(x0);
        b_sx = AE_LT32(sx, AE_ZERO32());

        /* Keep y sign as a binary value. */
        sy = XT_AE_MOVINT32X2_FROMXTFLOATX2(y0);
        sy = AE_SRLI32(sy, 31);
        sy = AE_SLLI32(sy, 31);

        x0 = XT_ABS_SX2(x0);
        y0 = XT_ABS_SX2(y0);
        b_xlty = XT_OLT_SX2(x0, y0);

        p0 = XT_LSI((xtfloat*)S_rd, 0);

        b_lt05 = XT_OLT_SX2(p0, (xtfloatx2)0.5f);

        /* Select coeffs from sets #1, #2 by reducted input value. */
        cf0 = (xtfloat)xa_nnlib_atanftbl1[0].f;
        XT_MOVF_SX2(cf0, xa_nnlib_atanftbl2[0].f, b_lt05);
        cf1 = (xtfloat)xa_nnlib_atanftbl1[1].f;
        XT_MOVF_SX2(cf1, xa_nnlib_atanftbl2[1].f, b_lt05);
        cf2 = (xtfloat)xa_nnlib_atanftbl1[2].f;
        XT_MOVF_SX2(cf2, xa_nnlib_atanftbl2[2].f, b_lt05);
        cf3 = (xtfloat)xa_nnlib_atanftbl1[3].f;
        XT_MOVF_SX2(cf3, xa_nnlib_atanftbl2[3].f, b_lt05);
        cf4 = (xtfloat)xa_nnlib_atanftbl1[4].f;
        XT_MOVF_SX2(cf4, xa_nnlib_atanftbl2[4].f, b_lt05);
        cf5 = (xtfloat)xa_nnlib_atanftbl1[5].f;
        XT_MOVF_SX2(cf5, xa_nnlib_atanftbl2[5].f, b_lt05);
        cf6 = (xtfloat)xa_nnlib_atanftbl1[6].f;
        XT_MOVF_SX2(cf6, xa_nnlib_atanftbl2[6].f, b_lt05);
        cf7 = (xtfloat)xa_nnlib_atanftbl1[7].f;
        XT_MOVF_SX2(cf7, xa_nnlib_atanftbl2[7].f, b_lt05);

        /*
         * Compute the approximation to z(p) = atan(p)/p-1.
         */

        XT_MADD_SX2(cf1, cf0, p0);
        cf0 = cf1;
        XT_MADD_SX2(cf3, cf2, p0);
        cf1 = cf3;
        XT_MADD_SX2(cf5, cf4, p0);
        cf2 = cf5;
        XT_MADD_SX2(cf7, cf6, p0);
        cf3 = cf7;

        p1 = XT_MUL_SX2(p0, p0);

        z0 = cf0;
        XT_MADD_SX2(cf1, z0, p1);
        z0 = cf1;
        XT_MADD_SX2(cf2, z0, p1);
        z0 = cf2;
        XT_MADD_SX2(cf3, z0, p1);
        z0 = cf3;

        XT_MADD_SX2(p0, p0, z0);
        z0 = p0;

        /* if ( x0<y0 ) z0 = pi2f.f - z0; */
        z1 = XT_SUB_SX2(pi2f.f, z0);
        XT_MOVT_SX2(z0, z1, b_xlty);
        /* if ( sx ) z0 = pif.f - z0; */
        z1 = XT_SUB_SX2(pif.f, z0);
        XT_MOVT_SX2(z0, z1, b_sx);
        /* if ( sy ) z0 = -z0; */
        t = XT_AE_MOVINT32X2_FROMXTFLOATX2(z0);
        t = AE_XOR32(t, sy);
        z0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(t);

        XT_SSI(z0, (xtfloat*)Z, 0);
      }
    }

  } /* for ( blkIx=0; blkIx<blkNum; blkIx++ ) */

} /* vec_atan2f() */

#elif HAVE_FPU
#define sz_f32 (int)sizeof(float32_t)

/*===========================================================================
  Scalar matematics:
  scl_atan2          full quadrant Arctangent
===========================================================================*/
/*-------------------------------------------------------------------------
Floating-Point Full-Quadrant Arc Tangent
The functions compute the full quadrant arc tangent of the ratio y/x.
Floating point functions output is in radians. Fixed point functions
scale its output by pi.

NOTE:
1. Scalar function is compatible with standard ANSI C routines and set
   errno and exception flags accordingly
2. Scalar function assigns EDOM to errno whenever y==0 and x==0.

Special cases:
     y    |   x   |  result   |  extra conditions
  --------|-------|-----------|---------------------
   +/-0   | -0    | +/-pi     |
   +/-0   | +0    | +/-0      |
   +/-0   |  x    | +/-pi     | x<0
   +/-0   |  x    | +/-0      | x>0
   y      | +/-0  | -pi/2     | y<0
   y      | +/-0  |  pi/2     | y>0
   +/-y   | -inf  | +/-pi     | finite y>0
   +/-y   | +inf  | +/-0      | finite y>0
   +/-inf | x     | +/-pi/2   | finite x
   +/-inf | -inf  | +/-3*pi/4 |
   +/-inf | +inf  | +/-pi/4   |

Input:
  y[N]  input data, Q15 or floating point
  x[N]  input data, Q15 or floating point
  N     length of vectors
Output:
  z[N]  result, Q15 or floating point

Restrictions:
x, y, z should not overlap
---------------------------------------------------------------------------*/

// Taken from Fusion
void xa_nn_elm_atan2_f32(
    FLOAT32* z,
    const FLOAT32* y,
    const FLOAT32* x,
    WORD32 N) {
  /*
   * const union ufloat32uint32* p;
   * int sx,sy,big;
   * sx=takesignf(x);
   * sy=takesignf(y);
   * x=fabs(x);
   * y=fabs(y);
   * if(x==0.f && y==0.f)
   * {
   * // The actual result depends on input signs.
   * x = 1.f;
   * y = 0.f;
   * }
   *
   * big=x>y;
   * if(big)
   * {
   * x=y/x;
   * }
   * else
   * {
   * // compare x==y is necessary to support (+/-Inf, +/-Inf) cases
   * x = (x == y) ? 1.0f : x / y;
   * }
   * p = (x<0.5f) ? atanftbl1 : atanftbl2;
   * // approximate atan(x)/x-1
   * y = p[0].f;
   * y = x*y + p[1].f;
   * y = x*y + p[2].f;
   * y = x*y + p[3].f;
   * y = x*y + p[4].f;
   * y = x*y + p[5].f;
   * y = x*y + p[6].f;
   * y = x*y + p[7].f;
   * // convert result to true atan(x)
   * y = x*y + x;
   *
   * if (!big) y = pi2f.f - y;
   * if (sx)   y = pif.f - y;
   * if (sy)   y = -y;
   * return   y;
   */
  const xtfloat* restrict X;
  const xtfloat* restrict Y;
  int32_t* restrict Z;
  const xtfloat* restrict S_rd;
  xtfloat* restrict S_wr;
  const xtfloat* restrict POLY_TBL1;
  const xtfloat* restrict POLY_TBL2;

  /* Current block index; overall number of blocks; number of values in the
   * current block */
  int blkIx, blkNum, blkLen;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ / sz_f32;
  /* Allocate a fixed-size scratch area on the stack. */
  float32_t ALIGN(8) scr[blkSize];

  int n;

  if (N <= 0)
    return;

  NASSERT_ALIGN8(scr);

  /*
   * Data are processed in blocks of scratch area size. Further, the algorithm
   * implementation is splitted in order to feed the optimizing compiler with a
   * few loops of managable size.
   */

  blkNum = (N + blkSize - 1) / blkSize;
  POLY_TBL1 = (xtfloat*)xa_nnlib_atanftbl1;
  POLY_TBL2 = (xtfloat*)xa_nnlib_atanftbl2;
  for (blkIx = 0; blkIx < blkNum; blkIx++) {
    blkLen = XT_MIN(N - blkIx * blkSize, blkSize);

    /*
     * Part I, reduction to [0,pi/4]. Reference C code:
     *
     *   {
     *     float32_t x0, y0, p0;
     *
     *     for ( n=0; n<blkLen; n++ )
     *     {
     *       y0 = fabsf( y[blkIx*blkSize+n] );
     *       x0 = fabsf( x[blkIx*blkSize+n] );
     *
     *       // The actual result depends on input signs.
     *       if ( x0==0.f && y0==0.f ) { x0 = 1.f; y0 = 0.f; };
     *
     *       if ( x0>y0 ) p0 = y0/x0;
     *       // Special case of x==y is necessary to support (+/-Inf, +/-Inf)
     * cases. else p0 = ( x0==y0 ? 1.f : x0/y0 );
     *
     *       scr[n] = p0;
     *     }
     *   }
     */

    {
      /* Input values */
      xtfloat x0, y0, i0;
      /* Numerator; denominator; reciprocal; quotient */
      xtfloat num, den, rcp, quo;
      /* Auxiliary vars */
      xtfloat s, eps;
      /* Is NaN; Inf/Inf; x/Inf; 0/0; x and y are subnormal */
      xtbool b_nan, b_num_inf, b_den_inf, b_eqz, b_subn;
      const xtfloat* pT;

      X = (xtfloat*)((uintptr_t)x + blkIx * blkSize * sz_f32);
      Y = (xtfloat*)((uintptr_t)y + blkIx * blkSize * sz_f32);
      S_wr = (xtfloat*)scr;

      static const uint32_t TAB[4] = {
          0x7fc00000, 0x00800000, 0x4b000000, 0x7f800000};
      pT = (xtfloat*)TAB;
      __Pragma("loop_count min=1");
      for (n = 0; n < blkLen; n++) {
        XT_LSIP(x0, X, sz_f32);
        XT_LSIP(y0, Y, sz_f32);

        /* Reproduce NaN in both x and y to ensure NaN propagation. */
        b_nan = XT_UN_S(x0, y0);
        i0 = pT[0];

        XT_MOVT_S(x0, i0, b_nan);

        x0 = XT_ABS_S(x0);
        y0 = XT_ABS_S(y0);

        /* num <= den */
        num = XT_MIN_S(y0, x0);
        den = XT_MAX_S(y0, x0);

        /* Classify numerator and denominator. */
        i0 = pT[1];
        b_subn = XT_OLT_S(num, i0);

        /* Scale up numerator and denominator if BOTH are subnormal. */
        i0 = pT[2];
        s = XT_MUL_S(num, i0);
        XT_MOVT_S(num, s, b_subn);
        s = XT_MUL_S(den, i0);
        XT_MOVT_S(den, s, b_subn);

        /* Initial appromimation for 1/den. */
        rcp = XT_RECIP0_S(den);
        /* Newton-Raphson iteration for 1/den. */
        eps = XT_CONST_S(1);
        XT_MSUB_S(eps, rcp, den);
        XT_MADD_S(rcp, rcp, eps);
        /* Approximation for the quotient num/den. */
        quo = XT_MUL_S(num, rcp);
        /* Refine the quotient by a modified Newton-Raphson iteration. */
        eps = num;
        XT_MSUB_S(eps, quo, den);
        XT_MADD_S(quo, rcp, eps);

        i0 = pT[3];
        b_num_inf = XT_OEQ_S(num, i0); /* Inf/Inf! */
        b_den_inf = XT_OEQ_S(den, i0);
        b_eqz = XT_OEQ_S(den, XT_CONST_S(0)); /* 0/0! */
        b_eqz = XT_ORB(b_eqz, b_den_inf);

        XT_MOVT_S(quo, XT_CONST_S(0), b_eqz); /* 0/0 -> 0 or x/Inf -> 0*/
        XT_MOVT_S(quo, XT_CONST_S(1), b_num_inf); /* Inf/Inf -> 1 */

        XT_SSIP(quo, S_wr, sz_f32);
      }
    }
    __Pragma("no_reorder");

    /*
     * Part II, polynomial approximation and full quadrant restoration.
     * Reference C code:
     *
     *   {
     *     const union ufloat32uint32 * ptbl;
     *     float32_t x0, y0, z0, p0;
     *     int sx, sy;
     *
     *     for ( n=0; n<blkLen; n++ )
     *     {
     *       y0 = y[blkIx*blkSize+n];
     *       x0 = x[blkIx*blkSize+n];
     *       p0 = scr[n];
     *
     *       sy = takesignf( y0 ); y0 = fabsf( y0 );
     *       sx = takesignf( x0 ); x0 = fabsf( x0 );
     *
     *       ptbl = ( p0<0.5f ? atanftbl1 : atanftbl2 );
     *
     *       // Approximate atan(p)/p-1
     *       z0 = ptbl[0].f;
     *       z0 = ptbl[1].f + p0*z0;
     *       z0 = ptbl[2].f + p0*z0;
     *       z0 = ptbl[3].f + p0*z0;
     *       z0 = ptbl[4].f + p0*z0;
     *       z0 = ptbl[5].f + p0*z0;
     *       z0 = ptbl[6].f + p0*z0;
     *       z0 = ptbl[7].f + p0*z0;
     *       z0 =        p0 + p0*z0;
     *
     *       if ( x0<y0 ) z0 = pi2f.f - z0;
     *       if ( sx    ) z0 = pif.f - z0;
     *       if ( sy    ) z0 = -z0;
     *
     *       z[blkIx*blkSize+n] = z0;
     *     }
     *   }
     */
    {
      const xtfloat* pT;
      /* Input values; output value; reducted input value*/
      xtfloat x0, y0, z0, z1, p0;
      /* Temporary; input values' sign */
      int32_t sx, sy;
      /* Polynomial coeffs for 0.f<=p<0.5f (#1) and 0.5f<=p<=1.f (#2). */
      xtfloat cf1_0, cf1_1, cf1_2, cf1_3, cf1_4, cf1_5, cf1_6, cf1_7;
      xtfloat cf2_0, cf2_1, cf2_2, cf2_3, cf2_4, cf2_5, cf2_6, cf2_7;
      /* Selected polynomial coeffs. */
      xtfloat cf0, cf1, cf2, cf3, cf4, cf5, cf6, cf7;
      /* x less than y; x is negative; num/den is less than 0.5f. */
      xtbool b_xlty, b_sx, b_lt05;

      X = (xtfloat*)((uintptr_t)x + blkIx * blkSize * sz_f32);
      Y = (xtfloat*)((uintptr_t)y + blkIx * blkSize * sz_f32);
      Z = (int32_t*)((uintptr_t)z + blkIx * blkSize * sz_f32);

      S_rd = (xtfloat*)scr;
      /* pi/2, pi */
      static const uint32_t TAB[2] = {0x3fc90fdb, 0x40490fdb};
      pT = (xtfloat*)TAB;
      __Pragma("loop_count min=1");
      for (n = 0; n < blkLen; n++) {
        xtfloat i0;
        XT_LSIP(x0, X, 0 * sz_f32);
        XT_LSIP(y0, Y, 0 * sz_f32);

        x0 = XT_ABS_S(x0);
        y0 = XT_ABS_S(y0);
        b_xlty = XT_OLT_S(x0, y0);

        XT_LSIP(p0, S_rd, sz_f32);

        b_lt05 = XT_OLT_S(p0, XT_CONST_S(3));

        /*Reload polynomial coeff sets. */

        cf1_0 = XT_LSI(POLY_TBL1, 0 * sz_f32);
        cf2_0 = XT_LSI(POLY_TBL2, 0 * sz_f32);
        cf1_1 = XT_LSI(POLY_TBL1, 1 * sz_f32);
        cf2_1 = XT_LSI(POLY_TBL2, 1 * sz_f32);
        cf1_2 = XT_LSI(POLY_TBL1, 2 * sz_f32);
        cf2_2 = XT_LSI(POLY_TBL2, 2 * sz_f32);
        cf1_3 = XT_LSI(POLY_TBL1, 3 * sz_f32);
        cf2_3 = XT_LSI(POLY_TBL2, 3 * sz_f32);
        cf1_4 = XT_LSI(POLY_TBL1, 4 * sz_f32);
        cf2_4 = XT_LSI(POLY_TBL2, 4 * sz_f32);
        cf1_5 = XT_LSI(POLY_TBL1, 5 * sz_f32);
        cf2_5 = XT_LSI(POLY_TBL2, 5 * sz_f32);
        cf1_6 = XT_LSI(POLY_TBL1, 6 * sz_f32);
        cf2_6 = XT_LSI(POLY_TBL2, 6 * sz_f32);
        cf1_7 = XT_LSI(POLY_TBL1, 7 * sz_f32);
        cf2_7 = XT_LSI(POLY_TBL2, 7 * sz_f32);

        /* Select coeffs from sets #1, #2 by reducted value's magnitude. */
        {
          xtfloat p0, p1;
          p0 = cf1_0;
          p1 = cf2_0;
          XT_MOVF_S(p0, p1, b_lt05);
          cf0 = p0;
          p0 = cf1_1;
          p1 = cf2_1;
          XT_MOVF_S(p0, p1, b_lt05);
          cf1 = p0;
          p0 = cf1_2;
          p1 = cf2_2;
          XT_MOVF_S(p0, p1, b_lt05);
          cf2 = p0;
          p0 = cf1_3;
          p1 = cf2_3;
          XT_MOVF_S(p0, p1, b_lt05);
          cf3 = p0;
          p0 = cf1_4;
          p1 = cf2_4;
          XT_MOVF_S(p0, p1, b_lt05);
          cf4 = p0;
          p0 = cf1_5;
          p1 = cf2_5;
          XT_MOVF_S(p0, p1, b_lt05);
          cf5 = p0;
          p0 = cf1_6;
          p1 = cf2_6;
          XT_MOVF_S(p0, p1, b_lt05);
          cf6 = p0;
          p0 = cf1_7;
          p1 = cf2_7;
          XT_MOVF_S(p0, p1, b_lt05);
          cf7 = p0;
        }

        /* Compute the approximation to z(p) = tan(p)/p-1. We use
         * Horner's method for better pipelining of a few iterations. */
        z0 = cf0;
        XT_MADD_S(cf1, p0, z0);
        z0 = cf1;
        XT_MADD_S(cf2, p0, z0);
        z0 = cf2;
        XT_MADD_S(cf3, p0, z0);
        z0 = cf3;
        XT_MADD_S(cf4, p0, z0);
        z0 = cf4;
        XT_MADD_S(cf5, p0, z0);
        z0 = cf5;
        XT_MADD_S(cf6, p0, z0);
        z0 = cf6;
        XT_MADD_S(cf7, p0, z0);
        z0 = cf7;

        XT_MADD_S(p0, p0, z0);
        z0 = p0;

        /* Keep signs of x and y. */
        sx = (int32_t)((int*)X)[0];
        X++;
        sy = (int32_t)((int*)Y)[0];
        Y++;

        sy = sy & 0x80000000;

        b_sx = AE_INT64_LT(AE_MOVINT64_FROMINT32(sx), AE_ZERO64());

        /* if ( x0<y0 ) z0 = pi2f.f - z0; */
        i0 = XT_LSI(pT, 0 * sz_f32);
        z1 = XT_SUB_S(i0, z0);
        XT_MOVT_S(z0, z1, b_xlty);
        /* if ( sx ) z0 = pif.f - z0; */
        i0 = XT_LSI(pT, 1 * sz_f32);
        z1 = XT_SUB_S(i0, z0);
        XT_MOVT_S(z0, z1, b_sx);
        /* if ( sy ) z0 = -z0; */
        sx = XT_RFR(z0);
        sx = sx ^ sy;
        *Z++ = sx;
      }
    }
  }
} /* vec_atan2f() */
#endif
