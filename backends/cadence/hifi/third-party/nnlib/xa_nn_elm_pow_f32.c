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

#include "NatureDSP_Signal_math.h"
#include "NatureDSP_types.h"
#include "xa_nn_common.h"

/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"

#include "xa_nnlib_common.h"
/* Constant tables. */

const union ufloat32uint32 ALIGN(8) xa_nnlib_pow2f_coef[] =
{
  { 0x39222a65 },
  { 0x3aaf931c },
  { 0x3c1d94fc },
  { 0x3d63578a },
  { 0x3e75fdf0 },
  { 0x3f317218 },
  { 0x3f800000 }

 //{ 0x3aaf931b },
 //{ 0x3c1e7220 },
 //{ 0x3d63578a },
 //{ 0x3e75fcc9 },
 //{ 0x3f317218 },
 //{ 0x3f800000 }

};

const union ufloat32uint32 ALIGN(8) xa_nnlib_log2f_coef[] =
{
  { 0x3d726a49 },
  { 0x3dd91c88 },
  { 0x3ddde76c },
  { 0x3de21e63 },
  { 0x3dfe600b },
  { 0x3e124679 },
  { 0x3e2ab2f1 },
  { 0x3e4ccd1b },
  { 0x3e7fffde },
  { 0x3eaaaaaa },
  { 0x3f000000 },
  { 0x3f800000 },
  /* log2(e) */
  { 0x3fb8aa3b }, /* 1.4426950216      */
  { 0x32a57060 }  /* 1.9259629891e-008 */
};

const union ufloat32uint32 xa_nnlib_pow_plusInff  ={0x7f800000};

const union ufloat32uint32 xa_nnlib_pow_qNaNf       = { 0x7fc00000 };

#define MIN(a,b)   ( (a)<(b) ? (a) : (b) )
#define MAX(a,b)   ( (a)>(b) ? (a) : (b) )

/*-------------------------------------------------------------------------
  Power function
  These routines calculate power function for 32-bit fixed-point numbers or 
  floating point numbers. 
  For the fixed point API, The  base is represented in Q31, the exponent 
  is represented in Q6.25. Results are represented as normalized fixed point
  number with separate mantissa in Q31 and exponent.

  Precision:
  32x32  32-bit inputs, 32-bit outputs
  f      floating point input, floating point output

  Accuracy: 
  2 ULP for fixed point API
  2 ULP under condition that |y|<=100

  Notes:
1. Scalar floating point raise  to a power functions conform to ANSI C requirements on 
   standard math library functions in respect to treatment of errno and floating-
   point exceptions. Vectorized function does not touch errno and may raise or not raise 
   floating point exceptions.
2. For floating point API, If x<0 is finite, y is finite and not an integer value, 
   then the respective result z is set to NaN
3. For fixed point API, function returns zero for all non-positive x. Fixed point 
   functions never touch errno

    Special cases:
          x   |   y    | Result |  Extra Conditions    
      --------+--------+--------+---------------------
      floating point API
      --------+--------+--------+---------------------
        +/-0  | y      | +/-inf | odd y<0
        +/-0  | y      | +inf   | even y<0
        +/-0  | y      | +/-0   | odd y>0
        +/-0  | y      | 0      | even y>0
        +/-1  | +/-inf | 1      | 
        1     | y      | 1      | any y including NaN
        x     | +/-0   | 1      | any x including NaN
        x     | y      | NaN    | finite x<0 and finite 
              |        |        | non-integer y (see 
              |        |        | note 2)
        x     | -inf   | +inf   | |x|<1
        x     | -inf   | 0      | |x|>1
        x     | +inf   | 0      | |x|<1
        x     | +inf   | +inf   | |x|>1
        -inf  | y      | -0     | y an odd integer <0
        -inf  | y      | 0      | y<0 and not an odd 
              |        |        | integer
        -inf  | y      | -inf   | y an odd integer >0
        -inf  | y      | +inf   | y>0 and not an odd 
              |        |        | integer
        +inf  | y      | 0      | y<0
        +inf  | y      | +inf   | y>0
      --------+--------+--------+---------------------
      fixed point API
      --------+--------+--------+---------------------
         x    | y      | 0      | x<=0
      --------+--------+--------+---------------------

  Input:
  x[N]  input data,Q0.31 or floating point
  y[N]  input data,Q6.25 or floating point
  N     length of vectors
  Output (fixed point API):
  m[N]  mantissa of output, Q31 
  e[N]  exponent of output  
  Output (floating point API):
  z[N]  results: floating point

  Restriction:
  z,x,y,m should not overlap
-------------------------------------------------------------------------*/

#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN(void, xa_nn_elm_pow_f32, (FLOAT32 * restrict z, const FLOAT32 * restrict y, const FLOAT32 * restrict x, WORD32 N))
#elif HAVE_VFPU
#define sz_f32    (int)sizeof(FLOAT32)
static void mypowf(FLOAT32 * scr,
                  FLOAT32 * restrict z, 
            const FLOAT32 * restrict x, 
            const FLOAT32 * restrict y, 
            WORD32 N )
{
  /* Table of different constants used in computations */
  static const int32_t c_tbl[] =
  {
    -126,
    -150,
    (int32_t)0x007FFFFF,/* max denormalized floating-point number / mantissa mask */
    (int32_t)0x4B800000,/* 2^24 */
    (int32_t)0x3F3504F3,/* sqrt(0.5) */
    (int32_t)0x3F000000,/*  0.5 */
    (int32_t)0xBF000000,/* -0.5 */
    -252,
    254
  };
  int n;
  const xtfloatx2     *          pX;
  const xtfloatx2     *          pY;

  const xtfloatx2     * restrict S_rd;
        xtfloatx2     * restrict S_wr;
        xtfloatx2     * restrict pZ;
  const ae_int32      * restrict TBL;
  const  xtfloat      * restrict TBL_LOG2;
  const  xtfloat      * restrict TBL_POW2;
  xtfloatx2 x0, y0, z0, t0, t1, ef0;
  xtfloatx2 c2f, c3f, c4f;
  xtfloatx2 _0, _1, half;
  ae_int32x2 c0i, c1i, c5i, c7i, c8i;
  ae_int32x2 e0, xi0, yi0, ex0;
  xtbool2 bsx, bsy, bdenorm, bsmall;
  ae_valign aX, aY, aZ;

  /* overall number of blocks; number of values in the current block */
  int blkLen;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ / (3*sz_f32);


  if (N <= 0) return;

  NASSERT(N % 2 == 0);
  NASSERT_ALIGN16(scr);

  /*
  * Data are processed in blocks of scratch area size. Further, the algorithm
  * implementation is splitted in order to feed the optimizing compiler with a
  * few loops of managable size.
  */


  blkLen = 0;
  TBL = (const ae_int32 *)c_tbl;
  for (; N>0; N -= blkLen, x += blkSize, y += blkSize, z += blkSize)
  {
    blkLen = XT_MIN(N, blkSize);
    _0 = 0.0f;
    _1 = (1.0f);
    half = (0.5f);
    {
      pX = (const xtfloatx2*)x;
      S_wr = (xtfloatx2*)scr;
      aX = AE_LA64_PP(pX);
      for (n = 0; n<(blkLen >> 1); n++)
      {
        XT_LASX2IP(x0, aX, pX);

        x0 = XT_ABS_SX2(x0);
        c0i = AE_L32_I(TBL, 0 * 4); /*-126*/
        c1i = AE_L32_I(TBL, 1 * 4); /*-150*/
        c2f = XT_LSI((xtfloat*)TBL, 2 * 4);
        c3f = XT_LSI((xtfloat*)TBL, 3 * 4);
        /* process denormalized values */
        bdenorm = XT_OLE_SX2(x0, c2f);
        t0 = XT_MUL_SX2(x0, c3f);
        XT_MOVT_SX2(x0, t0, bdenorm);
        e0 = c0i;
        AE_MOVT32X2(e0, c1i, bdenorm);
        /* extract exponent */
        xi0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(x0);
        ex0 = AE_SRLI32(xi0, 23);
        e0 = AE_ADD32(e0, ex0);
        /* extract mantissa */
        ex0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(c2f);/* load mantissa mask */ //!!!!!!!!!!!!!
        c5i = AE_L32_I(TBL, 5 * 4);/*  0.5 */
        xi0 = AE_AND32(xi0, ex0);
        xi0 = AE_OR32(xi0, c5i);
        x0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(xi0);
        /* adjust the mantissa to range [ sqrt(0.5) ; sqrt(2.0) ) */
        c4f = XT_LSI((xtfloat*)TBL, 4 * 4);
        bsmall = XT_OLT_SX2(x0, c4f);
        t0 = XT_ADD_SX2(x0, x0);
        ex0 = AE_SUB32(e0, 1);
        XT_MOVT_SX2(x0, t0, bsmall);
        AE_MOVT32X2(e0, ex0, bsmall);
        x0 = XT_SUB_SX2(_1, x0); //!!!
        ef0 = XT_FLOAT_SX2(e0, 0); //!!!
        XT_SSX2IP(x0, S_wr, 2 * sz_f32);
        XT_SSX2IP(ef0, S_wr, 2*2 * sz_f32);
      }
    }
    __Pragma("no_reorder");
    /* */
    {
      xtfloatx2 p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
      xtfloatx2 p10, p11, p12, p13;
      xtfloatx2 t2, w0, w1;
      S_wr = (      xtfloatx2*)scr+2;
      S_rd = (const xtfloatx2*)scr;
      TBL_LOG2 = (const xtfloat *)xa_nnlib_log2f_coef;
      for (n = 0; n<(blkLen >> 1); n++)
      {
        XT_LSX2IP(x0, S_rd, 3*2 * sz_f32);
        //XT_LSX2IP(ef0, S_rd, 2 * sz_f32);

        /* evaluate polynomial approximation */
        /* Load table of coefficients */

        p0 = XT_LSI(TBL_LOG2, 0 * 4);
        p1 = XT_LSI(TBL_LOG2, 1 * 4);
        p2 = XT_LSI(TBL_LOG2, 2 * 4);
        p3 = XT_LSI(TBL_LOG2, 3 * 4);
        p4 = XT_LSI(TBL_LOG2, 4 * 4);
        p5 = XT_LSI(TBL_LOG2, 5 * 4);
        p6 = XT_LSI(TBL_LOG2, 6 * 4);
        p7 = XT_LSI(TBL_LOG2, 7 * 4);
        p8 = XT_LSX(TBL_LOG2, 8 * 4);
        p9 = XT_LSX(TBL_LOG2, 9 * 4);
        
        XT_MADD_SX2(p1, x0, p0);
        XT_MADD_SX2(p2, x0, p1);
        XT_MADD_SX2(p3, x0, p2);
        XT_MADD_SX2(p4, x0, p3);
        XT_MADD_SX2(p5, x0, p4);
        XT_MADD_SX2(p6, x0, p5);
        XT_MADD_SX2(p7, x0, p6);
        XT_MADD_SX2(p8, x0, p7);
        XT_MADD_SX2(p9, x0, p8);
        t2 = p9;
        XT_SSX2IP(t2, S_wr, 3*2 * sz_f32);
      }
      S_wr = (xtfloatx2*)scr;
      S_rd = (const xtfloatx2*)scr;
      for (n = 0; n<(blkLen >> 1); n++)
      {
        p10 = XT_LSX(TBL_LOG2, 10 * 4);
        p11 = XT_LSX(TBL_LOG2, 11 * 4);
        p12 = XT_LSX(TBL_LOG2, 12 * 4);
        p13 = XT_LSX(TBL_LOG2, 13 * 4);

        XT_LSX2IP(x0, S_rd, 2 * sz_f32);
        XT_LSX2IP(ef0, S_rd, 2 * sz_f32);
        XT_LSX2IP(t2, S_rd, 2 * sz_f32);
        /* next coefficients are computed in extended precision */
        t0 = XT_MUL_SX2(x0, t2); t1 = t0;
        XT_MSUB_SX2(t1, x0, t2);
        w0 = XT_ADD_SX2(t0, p10);
        w1 = XT_SUB_SX2(w0, p10);
        w1 = XT_SUB_SX2(t0, w1);
        w1 = XT_SUB_SX2(w1, t1);
        t0 = w0; t1 = w1;
        w0 = XT_MUL_SX2(x0, t0); w1 = w0;
        XT_MSUB_SX2(w1, x0, t0); t0 = w0;
        XT_MSUB_SX2(w1, x0, t1); t1 = w1;
        w0 = XT_ADD_SX2(t0, p11);
        w1 = XT_SUB_SX2(w0, p11);
        w1 = XT_SUB_SX2(t0, w1);
        w1 = XT_SUB_SX2(w1, t1);
        t0 = w0; t1 = w1;
        x0 = XT_NEG_SX2(x0);
        w0 = XT_MUL_SX2(x0, t0); w1 = w0;
        XT_MSUB_SX2(w1, x0, t0); t0 = w0;
        XT_MSUB_SX2(w1, x0, t1); t1 = w1;
        /* multiply by log2(e) */
        w0 = XT_MUL_SX2(t0, p12); w1 = w0;
        XT_MSUB_SX2(w1, t0, p12);
        XT_MADD_SX2(w1, t1, p12);
        XT_MSUB_SX2(w1, t0, p13);
        t0 = w0; t1 = w1;
        /* add exponent */
        w0 = XT_ADD_SX2(t0, ef0);
        w1 = XT_SUB_SX2(w0, ef0);
        w1 = XT_SUB_SX2(t0, w1);
        t1 = XT_SUB_SX2(w1, t1);//!!!!
        t0 = w0; // !!!!!
        XT_SSX2IP(t0, S_wr, 2 * sz_f32);
        XT_SSX2IP(t1, S_wr, 2*2 * sz_f32);
      }    
    }
    __Pragma("no_reorder");
    /* */
    {
      xtfloatx2 xy, dxy, c0, c1;
      xtfloatx2 p0, p1, p2, p3, p4, p5, p6;
      S_wr = (      xtfloatx2*)scr+2;
      S_rd = (const xtfloatx2*)scr;
      TBL_POW2 = (const xtfloat *)xa_nnlib_pow2f_coef;
      pY = (const xtfloatx2*)y;
      aY = AE_LA64_PP(pY);
      for (n = 0; n<(blkLen >> 1); n++)
      {
        XT_LSX2IP(t0, S_rd, 2 * sz_f32);
        XT_LSX2IP(t1, S_rd, 2*2 * sz_f32);

        XT_LASX2IP(y0, aY, pY);
        /* compute y*log2(x) and separate result into integer and fractional parts */
        xy = XT_FIROUND_SX2(XT_MUL_SX2(y0, t0));
        dxy = XT_NEG_SX2(xy);
        XT_MADD_SX2(dxy, y0, t0);
        XT_MADD_SX2(dxy, y0, t1);
        dxy = XT_MIN_SX2(dxy, (xtfloatx2)1.0f);
        dxy = XT_MAX_SX2(dxy, (xtfloatx2)-1.0f);
        /* compute 2^fract */
        p0 = XT_LSI(TBL_POW2, 0 * 4);
        p1 = XT_LSI(TBL_POW2, 1 * 4);
        p2 = XT_LSI(TBL_POW2, 2 * 4);
        p3 = XT_LSI(TBL_POW2, 3 * 4);
        p4 = XT_LSI(TBL_POW2, 4 * 4);
        
        /* NOTE: do not change the order of computations and way of polynomial decomposition ! */
        XT_MADD_SX2(p1, dxy, p0);
        XT_MADD_SX2(p2, dxy, p1);
        XT_MADD_SX2(p3, dxy, p2);
        XT_MADD_SX2(p4, dxy, p3);
        XT_SSX2IP(p4, S_wr, 3*2 * sz_f32);
      }
      __Pragma("no_reorder");
      S_wr = (xtfloatx2*)scr;
      S_rd = (const xtfloatx2*)scr;
      TBL_POW2 = (const xtfloat *)xa_nnlib_pow2f_coef;
      pY = (const xtfloatx2*)y;
      aY = AE_LA64_PP(pY);
      for (n = 0; n<(blkLen >> 1); n++)
      {

        XT_LSX2IP(t0, S_rd, 2 * sz_f32);
        XT_LSX2IP(t1, S_rd, 2 * sz_f32);
        XT_LSX2IP(p4, S_rd, 2 * sz_f32);       
        p5 = XT_LSI(TBL_POW2, 5 * 4);
        p6 = XT_LSI(TBL_POW2, 6 * 4);
        XT_LASX2IP(y0, aY, pY);
        /* compute y*log2(x) and separate result into integer and fractional parts */
        xy = XT_FIROUND_SX2(XT_MUL_SX2(y0, t0));
        dxy = XT_NEG_SX2(xy);
        XT_MADD_SX2(dxy, y0, t0);
        XT_MADD_SX2(dxy, y0, t1);
        dxy = XT_MIN_SX2(dxy, (xtfloatx2)1.0f);
        dxy = XT_MAX_SX2(dxy, (xtfloatx2)-1.0f);
        XT_MADD_SX2(p5, dxy, p4);
        XT_MADD_SX2(p6, dxy, p5);
        z0 = p6;
        /* apply integer part */
        e0 = XT_TRUNC_SX2(xy, 0);
        c7i = AE_L32_I(TBL, 7 * 4);/* -252 */
        c8i = AE_L32_X(TBL, 8 * 4);/* 254 */
        e0 = AE_MAX32(e0, c7i);
        e0 = AE_MIN32(e0, c8i);
        e0 = AE_ADD32(e0, c8i);
        ex0 = AE_SRAI32(e0, 1);
        e0 = AE_SUB32(e0, ex0);
        ex0 = AE_SLLI32(ex0, 23);
        e0 = AE_SLLI32(e0, 23);
        c0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(e0);
        c1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(ex0);
        z0 = XT_MUL_SX2(z0, c1);
        z0 = XT_MUL_SX2(z0, c0); //!!!!!!!!!!!!
        XT_SSX2IP(z0, S_wr, 2 * sz_f32);
      }
    }
    __Pragma("no_reorder");
    /* */
    {
      xtbool2 b_yint, b_e0, b0, b_notspec;
      xtbool2 b_yeqz, b_yinf, b_xeqz, b_xeq1, b_xinf;
      xtbool2 b_NaN1, b_NaN2, b_one, b_Inf, b_zero;
      uint32_t b0i, b1i;
      uint32_t yeqz, yinf, xeqz, xeq1, xinf, sx, sy, yint;
      uint32_t one, NaN1, Inf, zero;
      xtfloatx2 xabs, spec;
      ae_int32x2 sgn, zi0;

      S_rd = (const xtfloatx2*)scr;
      pY = (const xtfloatx2*)y;
      pX = (const xtfloatx2*)x;
      pZ = (      xtfloatx2*)z;
      aY = AE_LA64_PP(pY);
      aX = AE_LA64_PP(pX);
      aZ = AE_ZALIGN64();
      for (n = 0; n<(blkLen >> 1); n++)
      {
        XT_LSX2IP(z0, S_rd, 2 * sz_f32);
        XT_LASX2IP(x0, aX, pX);
        XT_LASX2IP(y0, aY, pY);
        /* Take sign of x and y */
        xi0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(x0);
        yi0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(y0);
        bsx = XT_OLT_SX2(xi0, (xtfloatx2)0.0f);
        bsy = XT_OLT_SX2(yi0, (xtfloatx2)0.0f);

        xabs = XT_ABS_SX2(x0);
        /* check if y is integer */
        t0 = XT_FITRUNC_SX2(y0);
        b_yint = XT_OEQ_SX2(t0, y0);

        /* check if y is odd */
        e0 = XT_TRUNC_SX2(y0, 0); //temp0
        b_e0 = AE_EQ32(e0, MAX_INT32);//~b_tmp0
        b0i = AE_MOVAB2(b_e0);
        b1i = AE_MOVAB2(b_yint);
        b0i = b1i&(~b0i);
        b0 = AE_MOVBA2(b0i);
        AE_MOVF32X2(e0, AE_ZERO32(), b0);
        e0 = AE_SLLI32(e0, 31);
        sgn = AE_AND32(e0, xi0);
        /* process special numbers */
        b_yeqz = XT_OEQ_SX2((xtfloatx2)0.0f, y0);            /*  y ==0      */
        b_yinf = XT_OEQ_SX2(XT_ABS_SX2(y0), xa_nnlib_pow_plusInff.f);     /* |y|==Inf    */
        b_xeqz = XT_OEQ_SX2(x0, (xtfloatx2)0.0f);            /*  x ==0      */
        b_xeq1 = XT_OEQ_SX2(xabs, (xtfloatx2)1.0f);          /* |x|==1      */
        b_xinf = XT_OEQ_SX2(xabs, xa_nnlib_pow_plusInff.f);               /* |x|==INF    */

        yint = AE_MOVAB2(b_yint);
        yeqz = AE_MOVAB2(b_yeqz);
        yinf = AE_MOVAB2(b_yinf);
        xeqz = AE_MOVAB2(b_xeqz);
        xeq1 = AE_MOVAB2(b_xeq1);
        xinf = AE_MOVAB2(b_xinf);
        sx = AE_MOVAB2(bsx);
        sy = AE_MOVAB2(bsy);
        one = xeq1 & (yinf | (~sx));  /* |x|==1 && ( |y|==Inf || x>0 )                       */
        one = one | yeqz;           /* ( |x|==1 && ( |y|==Inf || x>0 ) ) || y==0 --> z=1.0 */
        NaN1 = sx&(~yint);          /* x<0 && y is not an integer --> z=NaN                */
        Inf = xinf&(~sy);          /* x==INF && y>0 --> z=INF */
        Inf = Inf | (xeqz & sy);    /* x==0   && y<0 --> z=INF */
        zero = xeqz &(~sy);         /* x==0   && y>0 --> z=0.0 */
        zero = zero | (xinf & sy);  /* x==INF && y<0 --> z=0.0 */

        b_NaN1 = AE_MOVBA2(NaN1);
        b_NaN2 = XT_UN_SX2(x0, y0);         /* isnan(x) || isnan(y) --> z=NaN                      */
        b_one = AE_MOVBA2(one);
        b_Inf = AE_MOVBA2(Inf);
        b_zero = AE_MOVBA2(zero);

        /* Save special numbers and mask for special numbers */
        spec = (xtfloatx2)xa_nnlib_pow_qNaNf.f;
        XT_MOVF_SX2(spec, half, b_NaN1);
        XT_MOVT_SX2(spec, _0, b_zero);
        XT_MOVT_SX2(spec, xa_nnlib_pow_plusInff.f, b_Inf);
        XT_MOVT_SX2(spec, xa_nnlib_pow_qNaNf.f, b_NaN2);
        XT_MOVT_SX2(spec, _1, b_one);

        b_notspec = XT_OEQ_SX2(spec, half);
        /* Replace result with special numbers if needed */
        XT_MOVF_SX2(z0, spec, b_notspec);
        /* Restore sign and store result */
        zi0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(z0);
        zi0 = AE_XOR32(zi0, sgn);
        z0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(zi0);
        XT_SASX2IP(z0, aZ, pZ);
      }    
    }
    XT_SASX2POSFP(aZ, pZ);
  }
} /* mypowf() */
void xa_nn_elm_pow_f32(   FLOAT32 * restrict z, 
            const FLOAT32 * restrict x, 
            const FLOAT32 * restrict y, 
            int N )
{
  const int blkSize = MAX_ALLOCA_SZ/sz_f32;
  /* Allocate a fixed-size scratch area on the stack. */
  FLOAT32 ALIGN(16) scr[blkSize];
  int M;
  if ( N<=0 ) return;
  M=N&~1;
  if ( M )
  {
    mypowf(scr,z,x,y,M); 
    y += M;
    x += M;
    z += M;
    N&=1;
  }
  if (N) 
  {     // processing the tail
    static const int32_t c_tbl[] =
    {
      -126,
      -150,
      (int32_t)0x007FFFFF,/* max denormalized floating-point number / mantissa mask */
      (int32_t)0x4B800000,/* 2^24 */
      (int32_t)0x3F3504F3,/* sqrt(0.5) */
      (int32_t)0x3F000000,/*  0.5 */
      (int32_t)0xBF000000,/* -0.5 */
      -252,
      254
    };
    xtfloat x0, y0, t0, ef0, t1, t2;
    xtfloat xy, dxy, z0, c0, c1;
    xtfloat p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
    xtfloat p10, p11, p12, p13, w0, w1;
    xtbool bdenorm, bsmall;
    ae_int32 e0, xi0, ex0;
    x0=XT_LSI((const xtfloat*)x,0);

    x0 = XT_ABS_S(x0);

    /* process denormalized values */
    bdenorm = xtbool2_extract_0(XT_OLE_S(x0, XT_LSI((xtfloat*)c_tbl, 2 * 4)));
    t0 = XT_MUL_S(x0, XT_LSI((xtfloat*)c_tbl, 3 * 4));
    XT_MOVT_S(x0, t0, (bdenorm));
    e0 = AE_L32_I((ae_int32 *)c_tbl, 0 * 4);;
    AE_MOVT_32(e0, AE_L32_I((ae_int32 *)c_tbl, 1 * 4), (bdenorm));
    /* extract exponent */
    xi0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(x0);
    ex0 = AE_SRLI32(xi0, 23);
    e0 = AE_ADD32(e0, ex0);
    /* extract mantissa */
    ex0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(XT_LSI((xtfloat*)c_tbl, 2 * 4));/* load mantissa mask */ //!!!!!!!!!!!!!
    xi0 = AE_AND32(xi0, ex0);
    xi0 = AE_OR32(xi0, AE_L32_I((ae_int32 *)c_tbl, 5 * 4));
    x0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(xi0);
    /* adjust the mantissa to range [ sqrt(0.5) ; sqrt(2.0) ) */
    
    bsmall = xtbool2_extract_0(XT_OLT_S(x0, XT_LSI((xtfloat*)c_tbl, 4 * 4)));


    t0 = XT_ADD_S(x0, x0);
    ex0 = AE_SUB32(e0, 1);
    XT_MOVT_S(x0, t0, bsmall);
    AE_MOVT_32(e0, ex0, bsmall);
    x0 = XT_SUB_S(1.0f, x0); //!!!
    ef0 = XT_FLOAT_S(e0, 0); //!!!

    /* evaluate polynomial approximation */
    /* Load table of coefficients */

    p0 = XT_LSI((const xtfloat *)xa_nnlib_log2f_coef, 0 * 4);
    p1 = XT_LSI((const xtfloat *)xa_nnlib_log2f_coef, 1 * 4);
    p2 = XT_LSI((const xtfloat *)xa_nnlib_log2f_coef, 2 * 4);
    p3 = XT_LSI((const xtfloat *)xa_nnlib_log2f_coef, 3 * 4);
    p4 = XT_LSI((const xtfloat *)xa_nnlib_log2f_coef, 4 * 4);
    p5 = XT_LSI((const xtfloat *)xa_nnlib_log2f_coef, 5 * 4);
    p6 = XT_LSI((const xtfloat *)xa_nnlib_log2f_coef, 6 * 4);
    p7 = XT_LSI((const xtfloat *)xa_nnlib_log2f_coef, 7 * 4);
    p8 = XT_LSX((const xtfloat *)xa_nnlib_log2f_coef, 8 * 4);
    p9 = XT_LSX((const xtfloat *)xa_nnlib_log2f_coef, 9 * 4);
    

    XT_MADD_S(p1, x0, p0);
    XT_MADD_S(p2, x0, p1);
    XT_MADD_S(p3, x0, p2);
    XT_MADD_S(p4, x0, p3);
    XT_MADD_S(p5, x0, p4);
    XT_MADD_S(p6, x0, p5);
    XT_MADD_S(p7, x0, p6);
    XT_MADD_S(p8, x0, p7);
    XT_MADD_S(p9, x0, p8);
    t2 = p9;


    p10 = XT_LSX((const xtfloat *)xa_nnlib_log2f_coef, 10 * 4);
    p11 = XT_LSX((const xtfloat *)xa_nnlib_log2f_coef, 11 * 4);
    p12 = XT_LSX((const xtfloat *)xa_nnlib_log2f_coef, 12 * 4);
    p13 = XT_LSX((const xtfloat *)xa_nnlib_log2f_coef, 13 * 4);

    /* next coefficients are computed in extended precision */
    t0 = XT_MUL_S(x0, t2); t1 = t0;
    XT_MSUB_S(t1, x0, t2);
    w0 = XT_ADD_S(t0, p10);
    w1 = XT_SUB_S(w0, p10);
    w1 = XT_SUB_S(t0, w1);
    w1 = XT_SUB_S(w1, t1);
    t0 = w0; t1 = w1;
    w0 = XT_MUL_S(x0, t0); w1 = w0;
    XT_MSUB_S(w1, x0, t0); t0 = w0;
    XT_MSUB_S(w1, x0, t1); t1 = w1;
    w0 = XT_ADD_S(t0, p11);
    w1 = XT_SUB_S(w0, p11);
    w1 = XT_SUB_S(t0, w1);
    w1 = XT_SUB_S(w1, t1);
    t0 = w0; t1 = w1;
    x0 = XT_NEG_S(x0);
    w0 = XT_MUL_S(x0, t0); w1 = w0;
    XT_MSUB_S(w1, x0, t0); t0 = w0;
    XT_MSUB_S(w1, x0, t1); t1 = w1;
    /* multiply by log2(e) */
    w0 = XT_MUL_S(t0, p12); w1 = w0;
    XT_MSUB_S(w1, t0, p12);
    XT_MADD_S(w1, t1, p12);
    XT_MSUB_S(w1, t0, p13);
    t0 = w0; t1 = w1;
    /* add exponent */
    w0 = XT_ADD_S(t0, ef0);
    w1 = XT_SUB_S(w0, ef0);
    w1 = XT_SUB_S(t0, w1);
    t1 = XT_SUB_S(w1, t1);//!!!!
    t0 = w0; // !!!!!

    /* compute y*log2(x) and separate result into integer and fractional parts */
    y0 = XT_LSI((const xtfloat*)y, 0);
    xy = XT_FIROUND_S(XT_MUL_S(y0, t0));
    dxy = XT_NEG_S(xy);
    XT_MADD_S(dxy, y0, t0);
    XT_MADD_S(dxy, y0, t1);
    dxy = XT_MIN_S(dxy, (xtfloatx2)1.0f);
    dxy = XT_MAX_S(dxy, (xtfloatx2)-1.0f);
    /* compute 2^fract */
    p0 = XT_LSI( (const xtfloat *)xa_nnlib_pow2f_coef, 0 * 4);
    p1 = XT_LSI( (const xtfloat *)xa_nnlib_pow2f_coef, 1 * 4);
    p2 = XT_LSI( (const xtfloat *)xa_nnlib_pow2f_coef, 2 * 4);
    p3 = XT_LSI( (const xtfloat *)xa_nnlib_pow2f_coef, 3 * 4);
    p4 = XT_LSI( (const xtfloat *)xa_nnlib_pow2f_coef, 4 * 4);
    p5 = XT_LSI( (const xtfloat *)xa_nnlib_pow2f_coef, 5 * 4);
    p6 = XT_LSI( (const xtfloat *)xa_nnlib_pow2f_coef, 6 * 4);
    /* NOTE: do not change the order of computations and way of polynomial decomposition ! */
    XT_MADD_S(p1, dxy, p0);
    XT_MADD_S(p2, dxy, p1);
    XT_MADD_S(p3, dxy, p2);
    XT_MADD_S(p4, dxy, p3);
    XT_MADD_S(p5, dxy, p4);
    XT_MADD_S(p6, dxy, p5);
    z0 = p6;
    /* apply integer part */
    e0 = XT_TRUNC_SX2(xy, 0);
    e0 = AE_MAX32(e0, AE_L32_I((ae_int32 *)c_tbl, 7 * 4));
    e0 = AE_MIN32(e0, AE_L32_X((ae_int32 *)c_tbl, 8 * 4));
    e0 = AE_ADD32(e0, AE_L32_X((ae_int32 *)c_tbl, 8 * 4));
    ex0 = AE_SRAI32(e0, 1);
    e0 = AE_SUB32(e0, ex0);
    ex0 = AE_SLLI32(ex0, 23);
    e0 = AE_SLLI32(e0, 23);
    c0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(e0);
    c1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(ex0);
    z0 = XT_MUL_S(z0, c1);
    z0 = XT_MUL_S(z0, c0); //!!!!!!!!!!!!


    /* Take sign of x and y */
    {
      xtbool2 bsx, bsy, b_yint, b_e0, b0, b_notspec;

      xtbool2 b_yeqz, b_yinf, b_xeqz, b_xeq1, b_xinf;
      xtbool2 b_NaN1, b_NaN2, b_one, b_Inf, b_zero;
      uint32_t b0i, b1i;
      uint32_t yeqz, yinf, xeqz, xeq1, xinf, sx, sy, yint;
      uint32_t one, NaN1, Inf, zero;
      xtfloat xabs, spec;
      ae_int32 sgn, zi0;

      x0 = XT_LSI((const xtfloat*)x, 0);
      y0 = XT_LSI((const xtfloat*)y, 0);
      xi0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(x0);
      bsx = (XT_OLT_S(x0, (xtfloat)0.0f));
      bsy = (XT_OLT_S(y0, (xtfloat)0.0f));

      xabs = XT_ABS_S(x0);
      /* check if y is integer */
      t0 = XT_FITRUNC_S(y0);
      b_yint = (XT_OEQ_S(t0, y0));
  
      /* check if y is odd */
      e0 = XT_TRUNC_S(y0, 0); //temp0
      b_e0 = (AE_EQ32(e0, MAX_INT32));//~b_tmp0
      b0i = AE_MOVAB2(b_e0);
      b1i = AE_MOVAB2(b_yint);
      b0i = b1i&(~b0i);
      b0 = AE_MOVBA2(b0i);
      AE_MOVF_32(e0, AE_ZERO32(), xtbool2_extract_0(b0));
      e0 = AE_SLLI32(e0, 31);
      sgn = AE_AND32(e0, xi0);
      /* process special numbers */
      b_yeqz = (XT_OEQ_S((xtfloatx2)0.0f, y0));            /*  y ==0      */
      b_yinf = (XT_OEQ_S(XT_ABS_SX2(y0), xa_nnlib_pow_plusInff.f));     /* |y|==Inf    */
      b_xeqz = (XT_OEQ_S(x0, (xtfloatx2)0.0f));            /*  x ==0      */
      b_xeq1 = (XT_OEQ_S(xabs, (xtfloatx2)1.0f));          /* |x|==1      */
      b_xinf = (XT_OEQ_S(xabs, xa_nnlib_pow_plusInff.f));               /* |x|==INF    */
  
      yint = AE_MOVAB2 (b_yint);
      yeqz = AE_MOVAB2 (b_yeqz);
      yinf = AE_MOVAB2 (b_yinf);
      xeqz = AE_MOVAB2 (b_xeqz);
      xeq1 = AE_MOVAB2 (b_xeq1);
      xinf = AE_MOVAB2 (b_xinf);
      sx = AE_MOVAB2 (bsx);
      sy = AE_MOVAB2 (bsy);
      
      one = xeq1 & (yinf | (~sx));  /* |x|==1 && ( |y|==Inf || x>0 )                       */
      one = one | yeqz;           /* ( |x|==1 && ( |y|==Inf || x>0 ) ) || y==0 --> z=1.0 */
      NaN1 = sx&(~yint);          /* x<0 && y is not an integer --> z=NaN                */
      Inf = xinf&(~sy);          /* x==INF && y>0 --> z=INF */
      Inf = Inf | (xeqz & sy);    /* x==0   && y<0 --> z=INF */
      zero = xeqz &(~sy);         /* x==0   && y>0 --> z=0.0 */
      zero = zero | (xinf & sy);  /* x==INF && y<0 --> z=0.0 */
  
      b_NaN1 = AE_MOVBA2(NaN1);
      b_NaN2 = XT_UN_SX2(x0, y0);         /* isnan(x) || isnan(y) --> z=NaN                      */
      b_one = AE_MOVBA2(one);
      b_Inf = AE_MOVBA2(Inf);
      b_zero = AE_MOVBA2(zero);
  
      /* Save special numbers and mask for special numbers */
      spec = (xtfloat)xa_nnlib_pow_qNaNf.f;
      XT_MOVF_S(spec, 0.5f, xtbool2_extract_0(b_NaN1));
      XT_MOVT_S(spec, 0.0f, xtbool2_extract_0(b_zero));
      XT_MOVT_S(spec, xa_nnlib_pow_plusInff.f, xtbool2_extract_0(b_Inf));
      XT_MOVT_S(spec, xa_nnlib_pow_qNaNf.f, xtbool2_extract_0(b_NaN2));
      XT_MOVT_S(spec, 1.0f, xtbool2_extract_0(b_one));
  
      b_notspec = XT_OEQ_S(spec, 0.5f);
      /* Replace result with special numbers if needed */
      XT_MOVF_S(z0, spec, xtbool2_extract_0(b_notspec));
      /* Restore sign and store result */
      zi0 = XT_AE_MOVINT32X2_FROMXTFLOATX2(z0);
      zi0 = AE_XOR32(zi0, sgn);
      z0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(zi0);

      XT_SSI(z0,(xtfloat*)z,0);
    
    }
  }

} /* vec_powf() */
#else
#define sz_f32    (int)sizeof(FLOAT32)
void xa_nn_elm_pow_f32(FLOAT32 * restrict z,
  const FLOAT32 * restrict x,
  const FLOAT32 * restrict y,
  int N)
{

  const int blkSizef = MAX_ALLOCA_SZ / sz_f32;
  /* Allocate a fixed-size scratch area on the stack. */
  float ALIGN(16) scr[blkSizef];
  /* Table of different constants used in computations */
  static const int32_t c_tbl[] =
  {
    -126,
    -150,
    (int32_t)0x007FFFFF,/* max denormalized floating-point number / mantissa mask */
    (int32_t)0x4B800000,/* 2^24 */
    (int32_t)0x3F3504F3,/* sqrt(0.5) */
    (int32_t)0x3F000000,/*  0.5 */
    (int32_t)0xBF000000,/* -0.5 */
    -252,
    254
  };
  int n;
  const xtfloat     *          pX;
  const xtfloat     *          pY;

  const xtfloat     * restrict S_rd;
  xtfloat     * restrict S_wr;
  xtfloat     * restrict pZ;
  const ae_int32      * restrict TBL;
  const  xtfloat      * restrict TBL_LOG2;
  const  xtfloat      * restrict TBL_POW2;
  xtfloat x0, y0, z0, t0, t1, ef0;
  xtfloat c2f, c3f, c4f;
  xtfloat _0, _1, half;
  ae_int32x2 c0i, c1i, c5i, c6i, c7i, c8i;
  ae_int32 e0, xi0, yi0, ex0;
  xtbool bsx, bsy, bdenorm, bsmall;

  /* overall number of blocks; number of values in the current block */
  int blkLen;
  /* Block size, blkLen <= blkSize */
  const int blkSize = MAX_ALLOCA_SZ / (3 * sz_f32);


  if (N <= 0) return;

  NASSERT_ALIGN16(scr);

  /*
  * Data are processed in blocks of scratch area size. Further, the algorithm
  * implementation is splitted in order to feed the optimizing compiler with a
  * few loops of managable size.
  */

  blkLen = 0;
  TBL = (const ae_int32 *)c_tbl;
  for (; N>0; N -= blkLen, x += blkSize, y += blkSize, z += blkSize)
  {
    blkLen = XT_MIN(N, blkSize);
    _0 = 0.0f;
    _1 = (1.0f);
    half = (0.5f);
    {
      pX   = (const xtfloat*)x;
      S_wr = (      xtfloat*)scr;
     
      for (n = 0; n<(blkLen); n++)
      {
        XT_LSIP(x0, pX, sz_f32);
       
        x0 = XT_ABS_S(x0);
        c0i = AE_L32_I(TBL, 0 * 4); /* -126 */
        c1i = AE_L32_I(TBL, 1 * 4); /* -150 */
        c2f = XT_LSI((xtfloat*)TBL, 2 * 4);
        c3f = XT_LSI((xtfloat*)TBL, 3 * 4);
        /* process denormalized values */
        bdenorm = XT_OLE_S(x0, c2f);
        t0 = XT_MUL_S(x0, c3f);
        XT_MOVT_S(x0, t0, bdenorm);
        e0 = c0i;
        
        AE_MOVT_32(e0, c1i, bdenorm);
        /* extract exponent */
        xi0 = XT_RFR(x0);
        ex0 = AE_SRLI32(xi0, 23);
        e0 = AE_ADD32(e0, ex0);
        /* extract mantissa */
        ex0 = XT_RFR(c2f);/* load mantissa mask */ //!!!!!!!!!!!!!
        c5i = AE_L32_I(TBL, 5 * 4);/*  0.5 */
        xi0 = AE_AND32(xi0, ex0);
        xi0 = AE_OR32(xi0, c5i);
        x0 = XT_WFR(xi0);
        /* adjust the mantissa to range [ sqrt(0.5) ; sqrt(2.0) ) */
        c4f = XT_LSI((xtfloat*)TBL, 4 * 4);
        bsmall = XT_OLT_S(x0, c4f);
        t0 = XT_ADD_S(x0, x0);
        ex0 = AE_SUB32(e0, 1);
        XT_MOVT_S(x0, t0, bsmall);
        AE_MOVT_32(e0, ex0, bsmall);
        x0 = XT_SUB_S(_1, x0); //!!!
        ef0 = XT_FLOAT_S(e0, 0); //!!!
        XT_SSIP(x0, S_wr, sz_f32);
        XT_SSIP(ef0, S_wr, 2 * sz_f32);

      }
    }
    __Pragma("no_reorder");
    /* */
    {
      xtfloat p0, p1, p2, p3, p4, p5, p6, p7, p8, p9;
      xtfloat p10, p11, p12, p13;
      xtfloat t2, w0, w1;
      S_wr = (      xtfloat*)scr + 2;
      S_rd = (const xtfloat*)scr;
      TBL_LOG2 = (const xtfloat *)xa_nnlib_log2f_coef;
   
      for (n = 0; n<(blkLen); n++)
      {
        XT_LSIP(x0, S_rd, 3*sz_f32);

        /* evaluate polynomial approximation */
        /* Load table of coefficients */

         p0 = XT_LSI(TBL_LOG2, 0 * 4);
         p1 = XT_LSI(TBL_LOG2, 1 * 4);
         p2 = XT_LSI(TBL_LOG2, 2 * 4);
         p3 = XT_LSI(TBL_LOG2, 3 * 4);
         p4 = XT_LSI(TBL_LOG2, 4 * 4);
         p5 = XT_LSI(TBL_LOG2, 5 * 4);
         p6 = XT_LSI(TBL_LOG2, 6 * 4);
         p7 = XT_LSI(TBL_LOG2, 7 * 4);
         p8 = XT_LSX(TBL_LOG2, 8 * 4);
         p9 = XT_LSX(TBL_LOG2, 9 * 4);
       
         XT_MADD_S(p1, x0, p0);
         XT_MADD_S(p2, x0, p1);
         XT_MADD_S(p3, x0, p2);
         XT_MADD_S(p4, x0, p3);
         XT_MADD_S(p5, x0, p4);
         XT_MADD_S(p6, x0, p5);
         XT_MADD_S(p7, x0, p6);
         XT_MADD_S(p8, x0, p7);
         XT_MADD_S(p9, x0, p8);
         t2 = p9;
         XT_SSIP(t2, S_wr, 3 * sz_f32);
      }
      S_wr = (      xtfloat*)scr;
      S_rd = (const xtfloat*)scr;
 
      for (n = 0; n<(blkLen); n++)
      {
        p10 = XT_LSX(TBL_LOG2, 10 * 4);
        p11 = XT_LSX(TBL_LOG2, 11 * 4);
        p12 = XT_LSX(TBL_LOG2, 12 * 4);
        p13 = XT_LSX(TBL_LOG2, 13 * 4);
      
        XT_LSIP(x0, S_rd, sz_f32);
        XT_LSIP(ef0, S_rd, sz_f32);
        XT_LSIP(t2, S_rd, sz_f32);
      
        /* next coefficients are computed in extended precision */
        t0 = XT_MUL_S(x0, t2); t1 = t0;
        XT_MSUB_S(t1, x0, t2);
        w0 = XT_ADD_S(t0, p10);
        w1 = XT_SUB_S(w0, p10);
        w1 = XT_SUB_S(t0, w1);
        w1 = XT_SUB_S(w1, t1);
        t0 = w0; t1 = w1;
        w0 = XT_MUL_S(x0, t0); w1 = w0;
        XT_MSUB_S(w1, x0, t0); t0 = w0;
        XT_MSUB_S(w1, x0, t1); t1 = w1;
        w0 = XT_ADD_S(t0, p11);
        w1 = XT_SUB_S(w0, p11);
        w1 = XT_SUB_S(t0, w1);
        w1 = XT_SUB_S(w1, t1);
        t0 = w0; t1 = w1;
        x0 = XT_NEG_S(x0);
        w0 = XT_MUL_S(x0, t0); w1 = w0;
        XT_MSUB_S(w1, x0, t0); t0 = w0;
        XT_MSUB_S(w1, x0, t1); t1 = w1;
        /* multiply by log2(e) */
        w0 = XT_MUL_S(t0, p12); w1 = w0;
        XT_MSUB_S(w1, t0, p12);
        XT_MADD_S(w1, t1, p12);
        XT_MSUB_S(w1, t0, p13);
        t0 = w0; t1 = w1;
        /* add exponent */
        w0 = XT_ADD_S(t0, ef0);
        w1 = XT_SUB_S(w0, ef0);
        w1 = XT_SUB_S(t0, w1);
        t1 = XT_SUB_S(w1, t1);//!!!!
        t0 = w0; // !!!!!
        XT_SSIP(t0, S_wr, sz_f32);
        XT_SSIP(t1, S_wr, sz_f32);
      }
    }
    __Pragma("no_reorder");
    /* */
    {
      xtfloat xy, dxy, c0, c1, _m1;;
      xtfloat p0, p1, p2, p3, p4, p5, p6;
      S_wr = (      xtfloat*)scr;
      S_rd = (const xtfloat*)scr;
      TBL_POW2 = (const xtfloat *)xa_nnlib_pow2f_coef;
      pY = (const xtfloat*)y;
      _m1 = -1.0f;
      for (n = 0; n<(blkLen); n++)
      {
        XT_LSIP(t0, S_rd, sz_f32);
        XT_LSIP(t1, S_rd, sz_f32);
        XT_LSIP(y0, pY, sz_f32);
        /* compute y*log2(x) and separate result into integer and fractional parts */
        xy = XT_FLOAT_S(XT_ROUND_S(XT_MUL_S(y0, t0), 0), 0);
        dxy = XT_NEG_S(xy);
        XT_MADD_S(dxy, y0, t0);
        XT_MADD_S(dxy, y0, t1);
        c5i = AE_L32_I(TBL, 5 * 4);/*  0.5 */
        c6i = AE_L32_I(TBL, 6 * 4);/*  -0.5 */
        dxy = XT_MIN_S(dxy, _1);
        dxy = XT_MAX_S(dxy, _m1);
        /* compute 2^fract */
        p0 = XT_LSI(TBL_POW2, 0 * 4);
        p1 = XT_LSI(TBL_POW2, 1 * 4);
        p2 = XT_LSI(TBL_POW2, 2 * 4);
        p3 = XT_LSI(TBL_POW2, 3 * 4);
        p4 = XT_LSI(TBL_POW2, 4 * 4);
        p5 = XT_LSI(TBL_POW2, 5 * 4);
        p6 = XT_LSI(TBL_POW2, 6 * 4);
        /* NOTE: do not change the order of computations and way of polynomial decomposition ! */
        XT_MADD_S(p1, dxy, p0);
        XT_MADD_S(p2, dxy, p1);
        XT_MADD_S(p3, dxy, p2);
        XT_MADD_S(p4, dxy, p3);
        XT_MADD_S(p5, dxy, p4);
        XT_MADD_S(p6, dxy, p5);
        z0 = p6;
        /* apply integer part */
        e0 = XT_TRUNC_S(xy, 0);
        c7i = AE_L32_I(TBL, 7 * 4);/* -252 */
        c8i = AE_L32_X(TBL, 8 * 4);/* 254 */
        e0 = AE_MAX32(e0, c7i);
        e0 = AE_MIN32(e0, c8i);
        e0 = AE_ADD32(e0, c8i);
        ex0 = AE_SRAI32(e0, 1);
        e0 = AE_SUB32(e0, ex0);
        ex0 = AE_SLLI32(ex0, 23);
        e0 = AE_SLLI32(e0, 23);
        
        c0 = XT_WFR(e0);
        c1 = XT_WFR(ex0);
        z0 = XT_MUL_S(z0, c1);
        z0 = XT_MUL_S(z0, c0); //!!!!!!!!!!!!
        XT_SSIP(z0, S_wr, sz_f32);

      }
    }
    __Pragma("no_reorder");
    /* */
    {
      xtbool b_yint, b_e0, b0, b_notspec;
      xtbool b_yeqz, b_yinf, b_xeqz, b_xeq1, b_xinf;
      xtbool b_NaN1, b_NaN2, b_one, b_Inf, b_zero;
      uint32_t b0i, b1i;
      uint32_t yeqz, yinf, xeqz, xeq1, xinf, sx, sy, yint;
      uint32_t one, NaN1, Inf, zero;
      xtfloat xabs, spec;
      ae_int32x2 sgn, zi0;

      S_rd = (const xtfloat*)scr;
      pY = (const xtfloat*)y;
      pX = (const xtfloat*)x;
      pZ = (xtfloat*)z;

      for (n = 0; n<(blkLen); n++)
      {
        XT_LSIP(z0, S_rd, sz_f32);
        XT_LSIP(x0, pX, sz_f32);
        XT_LSIP(y0, pY, sz_f32);

        /* Take sign of x and y */
        xi0 = XT_RFR(x0);
        yi0 = XT_RFR(y0);
        bsx = XT_OLT_S(x0, (xtfloat)0.0f);
        bsy = XT_OLT_S(y0, (xtfloat)0.0f);
      
        xabs = XT_ABS_S(x0);
        /* check if y is integer */
        {   /* validate if y is integral - all numbers bigger than 2^23 are assumed as integral */
          xtfloat t, c;
          t = XT_ABS_S((xtfloat)y0);
          c = 8388608.f;
          XT_MOVT_S(c, t, XT_ULT_S(t, 8388608.f));
          t = c;
          t0 = XT_FLOAT_S(XT_TRUNC_S(t, 0), 0);
          b_yint = XT_OEQ_S(XT_FLOAT_S(XT_TRUNC_S(t, 0), 0), t);
        }
      
        /* check if y is odd */
        e0 = XT_TRUNC_S(y0, 0); //temp0
        b_e0 = xtbool2_extract_0(AE_EQ32(e0, MAX_INT32));//~b_tmp0
        b0i = AE_MOVAB(b_e0);
        b1i = AE_MOVAB(b_yint);
        b0i = b1i&(~b0i);
        b0 = AE_MOVBA(b0i);
        AE_MOVF_32(e0, AE_ZERO32(), b0);
        e0 = AE_SLLI32(e0, 31);
        sgn = AE_AND32(e0, xi0);
        /* process special numbers */
        b_yeqz = XT_OEQ_S((xtfloat)0.0f, y0);            /*  y ==0      */
        b_yinf = XT_OEQ_S(XT_ABS_S(y0), xa_nnlib_pow_plusInff.f);     /* |y|==Inf    */
        b_xeqz = XT_OEQ_S(x0, (xtfloat)0.0f);            /*  x ==0      */
        b_xeq1 = XT_OEQ_S(xabs, (xtfloat)1.0f);          /* |x|==1      */
        b_xinf = XT_OEQ_S(xabs, xa_nnlib_pow_plusInff.f);               /* |x|==INF    */
      
        yint = AE_MOVAB(b_yint);
        yeqz = AE_MOVAB(b_yeqz);
        yinf = AE_MOVAB(b_yinf);
        xeqz = AE_MOVAB(b_xeqz);
        xeq1 = AE_MOVAB(b_xeq1);
        xinf = AE_MOVAB(b_xinf);
        sx = AE_MOVAB(bsx);
        sy = AE_MOVAB(bsy);
        one = xeq1 & (yinf | (~sx));  /* |x|==1 && ( |y|==Inf || x>0 )                       */
        one = one | yeqz;           /* ( |x|==1 && ( |y|==Inf || x>0 ) ) || y==0 --> z=1.0 */
        NaN1 = sx&(~yint);          /* x<0 && y is not an integer --> z=NaN                */
        Inf = xinf&(~sy);          /* x==INF && y>0 --> z=INF */
        Inf = Inf | (xeqz & sy);    /* x==0   && y<0 --> z=INF */
        zero = xeqz &(~sy);         /* x==0   && y>0 --> z=0.0 */
        zero = zero | (xinf & sy);  /* x==INF && y<0 --> z=0.0 */
      
        b_NaN1 = AE_MOVBA(NaN1);
        b_NaN2 = XT_UN_S(x0, y0);         /* isnan(x) || isnan(y) --> z=NaN                      */
        b_one = AE_MOVBA(one);
        b_Inf = AE_MOVBA(Inf);
        b_zero = AE_MOVBA(zero);
      
        /* Save special numbers and mask for special numbers */
        spec = (xtfloat)xa_nnlib_pow_qNaNf.f;
        XT_MOVF_S(spec, half, b_NaN1);
        XT_MOVT_S(spec, _0, b_zero);
        XT_MOVT_S(spec, xa_nnlib_pow_plusInff.f, b_Inf);
        XT_MOVT_S(spec, xa_nnlib_pow_qNaNf.f, b_NaN2);
        XT_MOVT_S(spec, _1, b_one);
      
        b_notspec = XT_OEQ_S(spec, half);
        /* Replace result with special numbers if needed */
        XT_MOVF_S(z0, spec, b_notspec);
        /* Restore sign and store result */
        zi0 = XT_RFR(z0);
        zi0 = AE_XOR32(zi0, sgn);
        z0 = XT_WFR(zi0);
        XT_SSIP(z0, pZ, sz_f32);
      }
    }
  }

} /* vec_powf() */
#endif
