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
#include "xa_type_def.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nn_common.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"


#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_clamp_f32xf32xf32_f32,
             (
                FLOAT32 *p_out,
                const FLOAT32 *p_inp,
                const FLOAT32 *p_min,
                const FLOAT32 *p_max,
                WORD32 num_elm
              )
           )
#else
WORD32 xa_nn_elm_clamp_f32xf32xf32_f32(FLOAT32 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp,
                               const FLOAT32 * __restrict__ p_min,
                               const FLOAT32 * __restrict__ p_max,
                               WORD32 num_elm)
{

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
    XA_NNLIB_ARG_CHK_PTR(p_min, -1);
    XA_NNLIB_ARG_CHK_PTR(p_max, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_min, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_max, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    int i;
    xtfloatx2 *inp = (xtfloatx2 *)p_inp;
    xtfloatx2 *min = (xtfloatx2 *)p_min;
    xtfloatx2 *max = (xtfloatx2 *)p_max;
    xtfloatx2 *out =  (xtfloatx2 *)p_out;

    xtfloatx2 x1, d_min, d_max, y;

    if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp)&7) == 0) && ((((unsigned)p_min)&7) == 0) && ((((unsigned)p_max)&7) == 0))
    {
        for(i=0;i < num_elm>>1;i++)
        {
            XT_LSX2IP(x1, inp, 2*sizeof(FLOAT32));
            XT_LSX2IP(d_min, min, 2*sizeof(FLOAT32));
            XT_LSX2IP(d_max, max, 2*sizeof(FLOAT32));

            y = XT_MAX_SX2(x1, d_min);
            y = XT_MIN_SX2(y, d_max);

            XT_SSX2IP( y, out,  2*sizeof(FLOAT32));
        }
    }
    else
    {
        ae_valign inp_a, min_a, max_a, out_a;

        inp_a = XT_LASX2PP(inp);
        min_a = XT_LASX2PP(min);
        max_a = XT_LASX2PP(max);
        out_a = AE_ZALIGN64();
        /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
        for(i=0;i < num_elm>>1;i++)
        {
            XT_LASX2IP(x1, inp_a, inp);
            XT_LASX2IP(d_min, min_a, min);
            XT_LASX2IP(d_max, max_a, max);

            y = XT_MAX_SX2(x1, d_min);
            y = XT_MIN_SX2(y, d_max);

            XT_SASX2IP(y, out_a, out);
        }
        XT_SASX2POSFP(out_a, out);
    }
    // Remainder Loop
    if (num_elm & 1)
    {
        xtfloat a1, a2, a3, a;
        XT_LSIP(a1, (xtfloat *)inp, 0);
        XT_LSIP(a2, (xtfloat *)min, 0);
        XT_LSIP(a3, (xtfloat *)max, 0);
        a = XT_MAX_S(a1, a2); 
        a = XT_MIN_S(a, a3); 
        XT_SSI(a, (xtfloat *)out, 0);
    }
    return 0;
}

static void internal_elm_clamp_broadcast_f32xf32xf32_f32(FLOAT32 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_min,
                    const    FLOAT32 * __restrict__ p_max,
                    const    FLOAT32 * __restrict__ p_inp,
                             WORD32  num_elm,
                             xtbool  sign_flag)
{
  int i;
  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_min;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_max; 
  xtfloatx2  *__restrict__ p_c =  (xtfloatx2 *)p_out;
  xtfloatx2  *__restrict__ input = (xtfloatx2 *)p_inp;

  const int num_simd2_ops = num_elm >> 1;
  const int num_scalar_ops = num_elm & 1;

  xtfloat a0_7, out, in0;
  xtfloatx2 d_inp, x1, x2, y;
  x2 = XT_LSI((xtfloat *)p_b, 0);

/* Min pointer is pointing to actual max and max to min */
  if(sign_flag){
    if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0) && ((((unsigned)input)&7) == 0))
    {
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
        XT_LSX2IP(d_inp, input, 2 * sizeof(FLOAT32));
        y = XT_MAX_SX2(d_inp, x2);
        y = XT_MIN_SX2(y, x1);
        XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32)); 
      }
    }
    else
    {
      ae_valign inp_a, min_a, out_a;
      min_a = XT_LASX2PP(p_a);
      inp_a = XT_LASX2PP(input);
      out_a = AE_ZALIGN64();      
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LASX2IP(x1, min_a, p_a);
        XT_LASX2IP(d_inp, inp_a, input);
        y = XT_MAX_SX2(d_inp, x2);
        y = XT_MIN_SX2(y, x1);
        XT_SASX2IP(y, out_a, p_c);
      }
      XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);   
    }  
    if(num_scalar_ops !=0)
    {
      XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
      XT_LSIP(in0, (xtfloat *)input, 0); 
      out = XT_MAX_S(in0, x2); 
      out = XT_MIN_S(out, a0_7);  	  
      XT_SSI(out, (xtfloat *)p_c, 0);
    }
  }
/* Min pointer is pointing to actual min and max to max.*/
  else
  {
    if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0) && ((((unsigned)input)&7) == 0))
    {
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
        XT_LSX2IP(d_inp, input, 2 * sizeof(FLOAT32));
        y = XT_MAX_SX2(d_inp, x1);
        y = XT_MIN_SX2(y, x2);
        XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32)); 
      }
    }
    else
    {
      ae_valign min_a, out_a, inp_a;
      inp_a = XT_LASX2PP(input);
      min_a = XT_LASX2PP(p_a);
      out_a = AE_ZALIGN64();       
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LASX2IP(x1, min_a, p_a);
        XT_LASX2IP(d_inp, inp_a, input);
        y = XT_MAX_SX2(d_inp, x1);
        y = XT_MIN_SX2(y, x2);
        XT_SASX2IP(y, out_a, p_c);
      }
      XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);
    }
    if(num_scalar_ops !=0)
    {
      XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
      XT_LSIP(in0, (xtfloat *)input, 0); 
      out = XT_MAX_S(in0, a0_7); 
      out = XT_MIN_S(out, x2);
      XT_SSI(out, (xtfloat *)p_c, 0);
    }
  }
}

static void internal_elm_clamp_broadcast_both_f32xf32xf32_f32(FLOAT32 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_min,
                    const    FLOAT32 * __restrict__ p_max,
                    const    FLOAT32 * __restrict__ p_inp,
                             WORD32  num_elm)
{
  int i;
  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_min;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_max; 
  xtfloatx2  *__restrict__ p_c =  (xtfloatx2 *)p_out;
  xtfloatx2  *__restrict__ input = (xtfloatx2 *)p_inp;

  const int num_simd2_ops = num_elm >> 1;
  const int num_scalar_ops = num_elm & 1;

  xtfloat a0_7, out, in0;
  xtfloatx2 d_inp, x1, x2, y;
  x2 = XT_LSI((xtfloat *)p_b, 0);
  x1 = XT_LSI((xtfloat *)p_a, 0);

    if(((((unsigned)p_c)&7) == 0) && ((((unsigned)input)&7) == 0))
    {
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LSX2IP(d_inp, input, 2 * sizeof(FLOAT32));
        y = XT_MAX_SX2(d_inp, x1);
        y = XT_MIN_SX2(y, x2);
        XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32)); 
      }
    }
    else
    {
      ae_valign min_a, out_a, inp_a;
      inp_a = XT_LASX2PP(input);
      out_a = AE_ZALIGN64();       
      for(i=0; i<num_simd2_ops; i++)
      {
        XT_LASX2IP(d_inp, inp_a, input);
        y = XT_MAX_SX2(d_inp, x1);
        y = XT_MIN_SX2(y, x2);
        XT_SASX2IP(y, out_a, p_c);
      }
      XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);
    }
    if(num_scalar_ops !=0)
    {
      XT_LSIP(in0, (xtfloat *)input, 0); 
      out = XT_MAX_S(in0, x1); 
      out = XT_MIN_S(out, x2);
      XT_SSI(out, (xtfloat *)p_c, 0);
    }
}

static void internal_elm_clamp_broadcast_2D_f32xf32xf32_f32(FLOAT32 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_min,
                    const    FLOAT32 * __restrict__ p_max,
                    const    FLOAT32 * __restrict__ p_inp,
                             WORD32  out_lc,
                             WORD32  in_lc,
                             xtbool  sign_flag)
{
  int i, j;

  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_min;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_max; 
  xtfloatx2  *__restrict__  p_c = (xtfloatx2 *)p_out;
  xtfloatx2  *__restrict__ input = (xtfloatx2 *)p_inp;
  
  int num_simd2_ops;
  int num_scalar_ops;

  if(out_lc)
  {
    num_simd2_ops = in_lc >> 1;
    num_scalar_ops = in_lc & 1;
  }
  else
  {
    num_simd2_ops = (in_lc >> 2) << 1;
    num_scalar_ops = in_lc & 3;
  }

    xtfloatx2 d_inp, x1, x2, y;
    xtfloat in0, a0, b0, c0;
    unsigned char con1, con2;
  if(sign_flag){
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx2 *)&p_min[i * in_lc];
      p_b = (xtfloatx2 *)p_max;
      p_c = (xtfloatx2 *)&p_out[i * in_lc];
      input = (xtfloatx2 *)&p_inp[i * in_lc];
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0) && ((((unsigned)input)&7) == 0))
      {
        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          XT_LSX2IP(x2, p_b, 2 * sizeof(FLOAT32));
          XT_LSX2IP(d_inp, input, 2 * sizeof(FLOAT32));
          y = XT_MAX_SX2(d_inp, x2);
          y = XT_MIN_SX2(y, x1);
          XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32));
        }
      }
      else
      {
        ae_valign vinp, vmin, vmax, out_a = AE_ZALIGN64();
        vmin = XT_LASX2PP(p_a);
        vmax = XT_LASX2PP(p_b);
        vinp = XT_LASX2PP(input);
        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LASX2IP(x1, vmin, p_a);
          XT_LASX2IP(x2, vmax, p_b);
          XT_LASX2IP(d_inp, vinp, input);
          y = XT_MAX_SX2(d_inp, x2);
          y = XT_MIN_SX2(y, x1);
          XT_SASX2IP(y, out_a, p_c); 
        }
        XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0, (xtfloat *)p_a, 0);
        XT_LSIP(b0, (xtfloat *)p_b, 0); 
        XT_LSIP(in0, (xtfloat *)input, 0);
        c0 = XT_MAX_S(in0, b0); 
        c0 = XT_MIN_S(a0, c0); 
        XT_SSI(c0, (xtfloat *)p_c, 0);
      }
    }
  }
  else
  {
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx2 *)&p_min[i * in_lc];
      p_b = (xtfloatx2 *)p_max;
      p_c = (xtfloatx2 *)&p_out[i * in_lc];
      input = (xtfloatx2 *)&p_inp[i * in_lc];
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0) && ((((unsigned)input)&7) == 0))
      {
        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          XT_LSX2IP(x2, p_b, 2 * sizeof(FLOAT32));
          XT_LSX2IP(d_inp, input, 2 * sizeof(FLOAT32));
          y = XT_MAX_SX2(d_inp, x1);
          y = XT_MIN_SX2(y, x2);
          XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32)); 
        }
      }
      else
      {
        ae_valign vinp, vmin, vmax, out_a = AE_ZALIGN64();
        vmin = XT_LASX2PP(p_a);
        vmax = XT_LASX2PP(p_b);
        vinp = XT_LASX2PP(input);
        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LASX2IP(x1, vmin, p_a);
          XT_LASX2IP(x2, vmax, p_b);
          XT_LASX2IP(d_inp, vinp, input);
          y = XT_MAX_SX2(d_inp, x1);
          y = XT_MIN_SX2(y, x2);
          XT_SASX2IP(y, out_a, p_c);
        }
        XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0, (xtfloat *)p_a, 0);
        XT_LSIP(b0, (xtfloat *)p_b, 0);
        XT_LSIP(in0, (xtfloat *)input, 0); 
        c0 = XT_MAX_S(in0, a0); 
        c0 = XT_MIN_S(c0, b0); 
        XT_SSI(c0, (xtfloat *)p_c, 0);
      }
    }  
  }
}

static void internal_elm_clamp_broadcast_both_2D_f32xf32xf32_f32(FLOAT32 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_min,
                    const    FLOAT32 * __restrict__ p_max,
                    const    FLOAT32 * __restrict__ p_inp,
                             WORD32  out_lc,
                             WORD32  in_lc)
{
  int i, j;

  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_min;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_max; 
  xtfloatx2  *__restrict__  p_c = (xtfloatx2 *)p_out;
  xtfloatx2  *__restrict__  input = (xtfloatx2 *)p_inp;
  
  int num_simd2_ops;
  int num_scalar_ops;

  if(out_lc)
  {
    num_simd2_ops = in_lc >> 1;
    num_scalar_ops = in_lc & 1;
  }
  else
  {
    num_simd2_ops = (in_lc >> 2) << 1;
    num_scalar_ops = in_lc & 3;
  }

    xtfloatx2 d_inp, x1, x2, y;
    xtfloat in0, a0, b0, c0;
    unsigned char con1, con2;

    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx2 *)p_min;
      p_b = (xtfloatx2 *)p_max;
      p_c = (xtfloatx2 *)&p_out[i * in_lc];
      input = (xtfloatx2 *)&p_inp[i * in_lc];
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0) && ((((unsigned)input)&7) == 0))
      {
        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          XT_LSX2IP(x2, p_b, 2 * sizeof(FLOAT32));
          XT_LSX2IP(d_inp, input, 2 * sizeof(FLOAT32));
          y = XT_MAX_SX2(d_inp, x1);
          y = XT_MIN_SX2(y, x2);
          XT_SSX2IP(y, p_c, 2 * sizeof(FLOAT32)); 
        }
      }
      else
      {
        ae_valign vinp, vmin, vmax, out_a = AE_ZALIGN64();
        vmin = XT_LASX2PP(p_a);
        vmax = XT_LASX2PP(p_b);
        vinp = XT_LASX2PP(input);
        for(j = 0; j < num_simd2_ops; j++)
        {
          XT_LASX2IP(x1, vmin, p_a);
          XT_LASX2IP(x2, vmax, p_b);
          XT_LASX2IP(d_inp, vinp, input);
          y = XT_MAX_SX2(d_inp, x1);
          y = XT_MIN_SX2(y, x2);
          XT_SASX2IP(y, out_a, p_c);
        }
        XT_SASX2POSFP(out_a, (xtfloatx2 *)p_c);
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0, (xtfloat *)p_a, 0);
        XT_LSIP(b0, (xtfloat *)p_b, 0);
        XT_LSIP(in0, (xtfloat *)input, 0); 
        c0 = XT_MAX_S(in0, a0); 
        c0 = XT_MIN_S(c0, b0); 
        XT_SSI(c0, (xtfloat *)p_c, 0);
      }
    }  
}

WORD32 xa_nn_elm_clamp_broadcast_4D_f32Xf32xf32_f32(FLOAT32 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * __restrict__ p_inp,
                      const WORD32 *const p_inp_shape,
                      const FLOAT32 * __restrict__ p_min,
                      const WORD32 *const p_min_shape,
                      const FLOAT32 * __restrict__ p_max,
                      const WORD32 *const p_max_shape
                      )
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_min, -1);
  XA_NNLIB_ARG_CHK_PTR(p_max, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_min_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_max_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_min, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_max, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_min_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_max_shape, sizeof(WORD32), -1);
  /* Check shapes */
  int i;
  xtbool sign_flag;
  for(i = 0; i < 4; i++)
  {  
    if((p_min_shape[i] != p_max_shape[i]) && ((p_min_shape[i] != 1) && (p_max_shape[i] != 1)))
    {
      return -1;
    }   
  }
  const float *p_min_new = p_min;
  for(i = 0; i < 4; i++)
  {
      for(int j=0; j < p_min_shape[i]; j++)
      {
          p_min_new++;
      }
  }
  const FLOAT32 *p_max_new = p_max;
  for(i = 0; i < 4; i++)
  {
      for(int j=0; j < p_max_shape[i]; j++)
      {
          p_max_new++;
      } 
  }
  const FLOAT32 *p_inp_new = p_inp;
  for(i = 0; i < 4; i++)
  {
      for(int j=0; j < p_inp_shape[i]; j++)
      {
          p_inp_new++;
      } 
  }    
  WORD32 min_strides[4], max_strides[4];
  min_strides[3] = 1;
  max_strides[3] = 1;
  for(i = 2; i >= 0; i--)
  {
    ae_int32x2 d_str, d_shape;
    d_str = AE_MOVDA32X2(min_strides[i + 1], max_strides[i + 1]);
    d_shape = AE_MOVDA32X2(p_min_shape[i + 1], p_max_shape[i + 1]);
    d_str = AE_MULP32X2(d_str, d_shape);
    min_strides[i] = AE_MOVAD32_H(d_str);
    max_strides[i] = AE_MOVAD32_L(d_str);    
  }

  int need_broadcast = 0;
  int min_const = 1, max_const = 1;
  for(i = 0; i < 4; i++)
  {
      if(p_min_shape[i] == 1)
      {
          min_strides[i] = 0;
          need_broadcast = 1;
      }
      else
      {
          min_const &= 0;
      }
      if(p_max_shape[i] == 1)
      {
          max_strides[i] = 0;
          need_broadcast = 1;
      }
      else
      {
          max_const &= 0;
      }
  }

  int itr0, itr1, itr2;
  FLOAT32 *p_out_tmp = p_out;
  const FLOAT32 *__restrict p_inp_temp = p_inp;
  const FLOAT32 *__restrict__ p_min_tmp = p_min;
  const FLOAT32 *__restrict__ p_max_tmp = p_max;

  if(need_broadcast == 0)
  {
    sign_flag = 0;
    internal_elm_clamp_broadcast_2D_f32xf32xf32_f32(
                p_out,
                p_min,
                p_max,
                p_inp,
                1,
                p_out_shape[0] * min_strides[0],
                sign_flag);
  }
  else if((min_strides[3] == 1)&& (max_strides[3] == 1))
  {
    WORD32 in_lc, out_lc;
    sign_flag = 0;
    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if((min_strides[2] == 0) && (max_strides[2] == 0))
    {
        in_lc = p_out_shape[3];
        out_lc = p_out_shape[2];
        for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
        {
          const FLOAT32 *__restrict__ p_min_tmp0 = p_min_tmp;
          const FLOAT32 *__restrict__ p_max_tmp0 = p_max_tmp;
          for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
          {
            internal_elm_clamp_broadcast_both_2D_f32xf32xf32_f32(
                p_out_tmp,
                p_min_tmp0,
                p_max_tmp0,
                p_inp_temp,
                out_lc,
                in_lc);
            p_out_tmp += in_lc * out_lc;
            p_min_tmp0 += min_strides[1];
            p_max_tmp0 += max_strides[1];
            p_inp_temp += in_lc * out_lc;
          }
          p_min_tmp += min_strides[0];
          p_max_tmp += max_strides[0];        
        }
    }
    else
    {
        if(min_strides[2] == 0)
        {
          const FLOAT32 *tmp;
          tmp = p_min_tmp;   p_min_tmp = p_max_tmp;    p_max_tmp = tmp;
          sign_flag = 1;
          int tmp_strides[2];
          tmp_strides[0] = min_strides[0];
          tmp_strides[1] = min_strides[1];

          min_strides[0] = max_strides[0];
          min_strides[1] = max_strides[1];

          max_strides[0] = tmp_strides[0];
          max_strides[1] = tmp_strides[1];
          in_lc = p_out_shape[3];
          out_lc = p_out_shape[2];
        }
        else if(max_strides[2] == 0)
        {
          in_lc = p_out_shape[3];
          out_lc = p_out_shape[2];
        }

        for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
        {
          const FLOAT32 *__restrict__ p_min_tmp0 = p_min_tmp;
          const FLOAT32 *__restrict__ p_max_tmp0 = p_max_tmp;
          for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
          {
            internal_elm_clamp_broadcast_2D_f32xf32xf32_f32(
                p_out_tmp,
                p_min_tmp0,
                p_max_tmp0,
                p_inp_temp,
                out_lc,
                in_lc,
                sign_flag);
            p_out_tmp += in_lc * out_lc;
            p_min_tmp0 += min_strides[1];
            p_max_tmp0 += max_strides[1];
            p_inp_temp += in_lc * out_lc;
          }

          p_min_tmp += min_strides[0];
          p_max_tmp += max_strides[0];
        }
    }
  }
  else if(min_const == 1 || max_const == 1)
  {
    if((min_const == 1)&&(max_const == 1))
    {
        internal_elm_clamp_broadcast_both_f32xf32xf32_f32(
            p_out_tmp,
            p_min_tmp,
            p_max_tmp,
            p_inp_temp,
            p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3]);
    }
    else
    {
        sign_flag = 0;
        if(min_strides[3] == 0)
        {
          sign_flag = 1;
          const FLOAT32 *tmp;
          tmp = p_min_tmp;   p_min_tmp = p_max_tmp;    p_max_tmp = tmp;
        }
        internal_elm_clamp_broadcast_f32xf32xf32_f32(
            p_out_tmp,
            p_min_tmp,
            p_max_tmp,
            p_inp_temp,
            p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3],
            sign_flag);
    }
  }
  else
  {
    sign_flag = 0;
    if((min_strides[3] == 0) && (max_strides[3] == 0))
    {
        for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
        {
          const FLOAT32 *__restrict__ p_min_tmp0 = p_min_tmp;
          const FLOAT32 *__restrict__ p_max_tmp0 = p_max_tmp;
          for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
          {
            const FLOAT32 *__restrict__ p_min_tmp1 = p_min_tmp0;
            const FLOAT32 *__restrict__ p_max_tmp1 = p_max_tmp0;
            for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
            {
              {
                internal_elm_clamp_broadcast_both_f32xf32xf32_f32(
                    p_out_tmp,
                    p_min_tmp1,
                    p_max_tmp1,
                    p_inp_temp,
                    p_out_shape[3]);
              }
              p_out_tmp += p_out_shape[3];
              p_min_tmp1 += min_strides[2];
              p_max_tmp1 += max_strides[2];
              p_inp_temp += p_out_shape[3];
            }
            p_min_tmp0 += min_strides[1];
            p_max_tmp0 += max_strides[1];
          }
          p_min_tmp += min_strides[0];
          p_max_tmp += max_strides[0];
        }
    }
    else
    {
        if(min_strides[3] == 0)
        {
          const FLOAT32 *tmp;
          tmp = p_min_tmp;   p_min_tmp = p_max_tmp;    p_max_tmp = tmp;
          sign_flag = 1;
          int tmp_strides[3];
          tmp_strides[0] = min_strides[0];
          tmp_strides[1] = min_strides[1];
          tmp_strides[2] = min_strides[2];

          min_strides[0] = max_strides[0];
          min_strides[1] = max_strides[1];
          min_strides[2] = max_strides[2];

          max_strides[0] = tmp_strides[0];
          max_strides[1] = tmp_strides[1];
          max_strides[2] = tmp_strides[2];
        }
        for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
        {
          const FLOAT32 *__restrict__ p_min_tmp0 = p_min_tmp;
          const FLOAT32 *__restrict__ p_max_tmp0 = p_max_tmp;
          for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
          {
            const FLOAT32 *__restrict__ p_min_tmp1 = p_min_tmp0;
            const FLOAT32 *__restrict__ p_max_tmp1 = p_max_tmp0;
            for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
            {
              {
                internal_elm_clamp_broadcast_f32xf32xf32_f32(
                    p_out_tmp,
                    p_min_tmp1,
                    p_max_tmp1,
                    p_inp_temp,
                    p_out_shape[3], 
                    sign_flag);
              }
              p_out_tmp += p_out_shape[3];
              p_min_tmp1 += min_strides[2];
              p_max_tmp1 += max_strides[2];
              p_inp_temp += p_out_shape[3];
            }
            p_min_tmp0 += min_strides[1];
            p_max_tmp0 += max_strides[1];
          }
          p_min_tmp += min_strides[0];
          p_max_tmp += max_strides[0];
        }
    }
  }
  return 0;
}
#endif
