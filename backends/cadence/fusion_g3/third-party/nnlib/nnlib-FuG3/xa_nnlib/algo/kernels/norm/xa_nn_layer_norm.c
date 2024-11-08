/*******************************************************************************
* Copyright (c) 2024 Cadence Design Systems, Inc.
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
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_internal.h"

/*
 *
 * Mean = (x0 + x1+ .. +xn-1)/n
 * Variance = ((x0*x0 + x1*x1 + .. +xn-1*xn-1)/n - (Mean*Mean))
 * std = sqrt(Variance + eps)
 * norm = (((x - Mean)/std) * weight) + bias
 *
 * */

WORD32 xa_nn_native_layer_norm_f32_f32(FLOAT32 *p_out,
                                       FLOAT32 *p_mean,
                                       FLOAT32 *p_std,
                                       const FLOAT32 *p_inp,
                                       const WORD32 *const p_inp_shape,
                                       WORD32 num_inp_dims,
                                       WORD32 axis,
                                       const FLOAT32 *p_weight,
                                       const FLOAT32 *p_bias,
                                       FLOAT32 eps)
{

    WORD32 i, j, m;
#ifdef ENABLE_HIGH_PRECISION
    xtfloat *p_a0 = (xtfloat *)p_inp;
#endif

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_weight, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_bias, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_mean, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_std, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_weight, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_mean, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_std, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((eps <= 0), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND((num_inp_dims <= 0), UNSUPPORTED_PARAM);

    XA_NNLIB_ARG_CHK_COND((axis < 0), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND((axis >= num_inp_dims), UNSUPPORTED_PARAM);

    const xb_vecMxf32 *restrict p_in_mxf32 = (const xb_vecMxf32 *)p_inp;
    const xb_vecMxf32 *restrict p_in1_mxf32 = p_in_mxf32;
    xb_vecMxf32 *restrict p_out_mxf32 = (xb_vecMxf32 *)p_out;
    xb_vecMxf32 *restrict p_rstd_mxf32 = (xb_vecMxf32 *)p_std;
    xb_vecMxf32 *restrict p_mean_mxf32 = (xb_vecMxf32 *)p_mean;
    const xb_vecMxf32 *restrict p_weight_mxf32;
    const xb_vecMxf32 *restrict p_bias_mxf32;

    /* Initialize number of elements of leading and normalized shapes */
    WORD32 leading_dim = CONST_ONE;
    WORD32 norm_dim = CONST_ONE;

    /* Calculate number of elements of leading dimensions */
    for (int i = 0; i < axis; i++)
    {
        leading_dim *= p_inp_shape[i];
    }

    /* Calculate number of elements of the shape to be normalized */
    for (int i = axis; i < num_inp_dims; i++)
    {
        norm_dim *= p_inp_shape[i];
    }

    if (!leading_dim)
    {
        return UNSUPPORTED_PARAM;
    }

    if (!norm_dim)
    {
        valign ax, ay;
        xb_vecMxf32 *restrict p_mean_mxf32 = (xb_vecMxf32 *)p_mean;
        /* Zeroing align registers */
        ax = PDX_Z_ALIGN();
        ay = PDX_Z_ALIGN();

        FLOAT32 rstd = NAN;

        /* Initialize the mean output with 0 */
        xb_vecMxf32 x;
        x = PDX_ZERO_MXF32();

        /* Initialize the inverse std output with NAN */
        xb_vecMxf32 y = PDX_MOV_MXF32_FROM_F32(rstd);
        y = PDX_REP_MXF32(y, 0);

        /* loop runs for leading_dim/4 iterations */
        for (i = 0; i < (leading_dim >> LOG2_PDX_M); i++)
        {
            /* mean and inverse std values are being stored */
            PDX_SA_MXF32_IP(x, ax, p_mean_mxf32);
            PDX_SA_MXF32_IP(y, ay, p_rstd_mxf32);
        }

        /* Store the remaining mean and rstd values after processing the loop */
        m = (leading_dim & (PDX_M - CONST_ONE)) << LOG2_SIZE_FLOAT;
        PDX_SAV_MXF32_XP(x, ax, p_mean_mxf32, m);
        PDX_SAPOS_MXF32_FP(ax, p_mean_mxf32);

        PDX_SAV_MXF32_XP(y, ay, p_rstd_mxf32, m);
        PDX_SAPOS_MXF32_FP(ay, p_rstd_mxf32);

        return UNSUPPORTED_PARAM;
    }

    xb_vecMxf32 sum_mxf32;
    xb_vecMxf32 sq_sum_mxf32;
    xb_vecMxf32 mean_vec;
    xb_vecMxf32 std_vec;
    xb_vecMxf32 rstd_vec;

    FLOAT32 sum;
    FLOAT32 sq_sum;
    FLOAT32 mean;
    FLOAT32 variance;
    FLOAT32 inv_normalized;

    /* Align load priming of output and inverse std */
    valign a_out = PDX_Z_ALIGN();
    valign a_rstd = PDX_Z_ALIGN();
    valign a_mean = PDX_Z_ALIGN();

    valign a_out_dim1 = PDX_Z_ALIGN();
    valign a_out_dim2 = PDX_Z_ALIGN();
    valign a_out_dim3 = PDX_Z_ALIGN();

    /* Calculate inverse of number of normalized elements */
    PDX_DIV_F32_T(inv_normalized, CONST_ONE, norm_dim, CONST_ONE);

#ifndef ENABLE_HIGH_PRECISION
    xb_vecMxf32 inv_norm_vec = inv_normalized;
    xb_vecMxf32 eps_vec = eps;
#endif
    xb_vecMxf32 x0;

    valign ax;

    /* Calculate number of remaining inputs */
    m = (norm_dim & (PDX_M - CONST_ONE)) << LOG2_SIZE_FLOAT;

    xb_vecMxf32 sum_mxf32_1, sum_mxf32_2, sum_mxf32_3, sum_mxf32_4;
    xb_vecMxf32 sq_sum_mxf32_1, sq_sum_mxf32_2, sq_sum_mxf32_3, sq_sum_mxf32_4;

    const xb_vecMxf32 *restrict p_in_mxf32_st1;
    const xb_vecMxf32 *restrict p_in_mxf32_st2;
    const xb_vecMxf32 *restrict p_in_mxf32_st3;

    xb_vecMxf32 *restrict p_out_mxf32_dim1;
    xb_vecMxf32 *restrict p_out_mxf32_dim2;
    xb_vecMxf32 *restrict p_out_mxf32_dim3;

    WORD32 offset_dim = 4 * norm_dim;

    valign ax_st1, ax_st2, ax_st3;
    const FLOAT32 *p_inp1, *p_inp2, *p_inp3, *p_inp4;
    FLOAT32 *p_out1, *p_out2, *p_out3, *p_out4;

    p_inp1 = p_inp;
    p_inp2 = p_inp + norm_dim;
    p_inp3 = p_inp + CONST_TWO * norm_dim;
    p_inp4 = p_inp + CONST_THREE * norm_dim;

    p_out1 = p_out;
    p_out2 = p_out + norm_dim;
    p_out3 = p_out + CONST_TWO * norm_dim;
    p_out4 = p_out + CONST_THREE * norm_dim;

    /* Loop runs for leading_dim/4 iterations */
    for (i = 0; i < leading_dim >> LOG2_PDX_M; i++)
    {
        FLOAT32 sum1, sum2, sum3;
        FLOAT32 sq_sum1, sq_sum2, sq_sum3;
        xb_vecMxf32 x1, x2, x3;
        xb_vecMxf32 b1, b2, b3;
        xb_vecMxf32 mean_vec1, mean_vec2, mean_vec3, mean_vec4;
#ifdef ENABLE_HIGH_PRECISION
        xb_vecMxf32 std_vec1, std_vec2, std_vec3, std_vec4;
#else
        xb_vecMxf32 rstd_vec1, rstd_vec2, rstd_vec3, rstd_vec4;
        xb_vecMxf32 w1, w2, w3, w4;
#endif

        /* 4 series of computations are done together */
        p_in_mxf32 = (const xb_vecMxf32 *)p_inp1;
        p_in_mxf32_st1 = (const xb_vecMxf32 *)p_inp2;
        p_in_mxf32_st2 = (const xb_vecMxf32 *)p_inp3;
        p_in_mxf32_st3 = (const xb_vecMxf32 *)p_inp4;

        p_out_mxf32 = (xb_vecMxf32 *)p_out1;
        p_out_mxf32_dim1 = (xb_vecMxf32 *)p_out2;
        p_out_mxf32_dim2 = (xb_vecMxf32 *)p_out3;
        p_out_mxf32_dim3 = (xb_vecMxf32 *)p_out4;

        p_out1 += offset_dim;
        p_out2 += offset_dim;
        p_out3 += offset_dim;
        p_out4 += offset_dim;

        ax = PDX_LA_MXF32_PP(p_in_mxf32);
        ax_st1 = PDX_LA_MXF32_PP(p_in_mxf32_st1);
        ax_st2 = PDX_LA_MXF32_PP(p_in_mxf32_st2);
        ax_st3 = PDX_LA_MXF32_PP(p_in_mxf32_st3);

        /* Reset sum and sq_sum vectors to zero */
        sum_mxf32_1 = PDX_ZERO_MXF32();
        sq_sum_mxf32_1 = PDX_ZERO_MXF32();

        sum_mxf32_2 = PDX_ZERO_MXF32();
        sq_sum_mxf32_2 = PDX_ZERO_MXF32();

        sum_mxf32_3 = PDX_ZERO_MXF32();
        sq_sum_mxf32_3 = PDX_ZERO_MXF32();

        sum_mxf32_4 = PDX_ZERO_MXF32();
        sq_sum_mxf32_4 = PDX_ZERO_MXF32();

#ifdef ENABLE_HIGH_PRECISION
        // Loop runs for norm_dim iterations
        xtfloat a0 = 0;
        xtfloat a1 = 0;
        xtfloat a2 = 0;
        xtfloat a3 = 0;

        xtfloat *p_a0 = (xtfloat *)p_inp1;
        xtfloat *p_a1 = (xtfloat *)p_inp2;
        xtfloat *p_a2 = (xtfloat *)p_inp3;
        xtfloat *p_a3 = (xtfloat *)p_inp4;

        sum = 0, sq_sum = 0;
        sum1 = 0, sq_sum1 = 0;
        sum2 = 0, sq_sum2 = 0;
        sum3 = 0, sq_sum3 = 0;

#pragma no_reorder
        for (j = 0; j < (norm_dim); j++)
        {
            xtfloat_loadip(a0, p_a0, 4);
            xtfloat_loadip(a1, p_a1, 4);
            xtfloat_loadip(a2, p_a2, 4);
            xtfloat_loadip(a3, p_a3, 4);

            sum = sum + a0;
            sq_sum = sq_sum + XT_MUL_S(a0, a0);

            sum1 = sum1 + a1;
            sq_sum1 = sq_sum1 + XT_MUL_S(a1, a1);

            sum2 = sum2 + a2;
            sq_sum2 = sq_sum2 + XT_MUL_S(a2, a2);

            sum3 = sum3 + a3;
            sq_sum3 = sq_sum3 + XT_MUL_S(a3, a3);
        }
#else

        /* Loop runs for norm_dim/4 iterations */
        for (j = 0; j < (norm_dim >> LOG2_PDX_M); j++)
        {
            /* Aligning load input (4-way) */
            PDX_LA_MXF32_IP(x0, ax, p_in_mxf32);
            PDX_LA_MXF32_IP(x1, ax_st1, p_in_mxf32_st1);
            PDX_LA_MXF32_IP(x2, ax_st2, p_in_mxf32_st2);
            PDX_LA_MXF32_IP(x3, ax_st3, p_in_mxf32_st3);

            /* Add all the inputs of the dimension to be normalized */
            sum_mxf32_1 = PDX_ADD_MXF32(sum_mxf32_1, x0);
            sum_mxf32_2 = PDX_ADD_MXF32(sum_mxf32_2, x1);
            sum_mxf32_3 = PDX_ADD_MXF32(sum_mxf32_3, x2);
            sum_mxf32_4 = PDX_ADD_MXF32(sum_mxf32_4, x3);

            /* Calculate the sum of squares of the inputs */
            PDX_MULA_MXF32(sq_sum_mxf32_1, x0, x0);
            PDX_MULA_MXF32(sq_sum_mxf32_2, x1, x1);
            PDX_MULA_MXF32(sq_sum_mxf32_3, x2, x2);
            PDX_MULA_MXF32(sq_sum_mxf32_4, x3, x3);
        }

        x0 = 0;
        x1 = 0;
        x2 = 0;
        x3 = 0;

        /* Load remaining inputs */
        PDX_LAV_MXF32_XP(x0, ax, p_in_mxf32, m);
        PDX_LAV_MXF32_XP(x1, ax_st1, p_in_mxf32_st1, m);
        PDX_LAV_MXF32_XP(x2, ax_st2, p_in_mxf32_st2, m);
        PDX_LAV_MXF32_XP(x3, ax_st3, p_in_mxf32_st3, m);

        /* Add all the remaining inputs of the dimension to be normalized */
        sum_mxf32_1 = PDX_ADD_MXF32(sum_mxf32_1, x0);
        sum_mxf32_2 = PDX_ADD_MXF32(sum_mxf32_2, x1);
        sum_mxf32_3 = PDX_ADD_MXF32(sum_mxf32_3, x2);
        sum_mxf32_4 = PDX_ADD_MXF32(sum_mxf32_4, x3);

        /* Calculate the sum of squares of the remaining inputs */
        PDX_MULA_MXF32(sq_sum_mxf32_1, x0, x0);
        PDX_MULA_MXF32(sq_sum_mxf32_2, x1, x1);
        PDX_MULA_MXF32(sq_sum_mxf32_3, x2, x2);
        PDX_MULA_MXF32(sq_sum_mxf32_4, x3, x3);

        sum = PDX_RADD_MXF32(sum_mxf32_1);
        sq_sum = PDX_RADD_MXF32(sq_sum_mxf32_1);

        sum1 = PDX_RADD_MXF32(sum_mxf32_2);
        sq_sum1 = PDX_RADD_MXF32(sq_sum_mxf32_2);

        sum2 = PDX_RADD_MXF32(sum_mxf32_3);
        sq_sum2 = PDX_RADD_MXF32(sq_sum_mxf32_3);

        sum3 = PDX_RADD_MXF32(sum_mxf32_4);
        sq_sum3 = PDX_RADD_MXF32(sq_sum_mxf32_4);
#endif

#ifdef ENABLE_HIGH_PRECISION
        /* Calculate mean */
        xtfloat mean1;
        xtfloat mean2;
        xtfloat mean3;
        xtfloat mean4;

        PDX_DIV_F32_T(mean1,sum,norm_dim,1);
        PDX_DIV_F32_T(mean2,sum1,norm_dim,1);
        PDX_DIV_F32_T(mean3,sum2,norm_dim,1);
        PDX_DIV_F32_T(mean4,sum3,norm_dim,1);

        sum_mxf32 = mean1;
        sum_mxf32_1 = mean2;
        sum_mxf32_2 = mean3;
        sum_mxf32_3 = mean4;

        sum_mxf32 = PDX_SELI_MXF32(sum_mxf32_1, sum_mxf32, SEL_INDEX );
        sum_mxf32_2 = PDX_SELI_MXF32(sum_mxf32_3, sum_mxf32_2, SEL_INDEX );

        /* Mean of each dimension */
        mean_vec = PDX_SELI_MXF32(sum_mxf32_2, sum_mxf32, SEL_INDEX );

        /* Calculate variance */
        xtfloat variance ;
        PDX_DIV_F32_T(variance,sq_sum,norm_dim,1);
        variance -= mean1 * mean1;
        variance = variance + eps;
        xtfloat std1 = variance;

        PDX_DIV_F32_T(variance,sq_sum1,norm_dim,1);
        variance -= mean2 * mean2;
        variance = variance + eps;
        xtfloat std2 = variance;

        PDX_DIV_F32_T(variance,sq_sum2,norm_dim,1);
        variance -= mean3 * mean3;
        variance = variance + eps;
        xtfloat std3 = variance;

        PDX_DIV_F32_T(variance,sq_sum3,norm_dim,1);
        variance -= mean4 * mean4;
        variance = variance + eps;
        xtfloat std4 = variance;

        sq_sum_mxf32 = std1;
        sq_sum_mxf32_1 = std2;
        sq_sum_mxf32_2 = std3;
        sq_sum_mxf32_3 = std4;

        sq_sum_mxf32 = PDX_SELI_MXF32(sq_sum_mxf32_1, sq_sum_mxf32, SEL_INDEX );
        sq_sum_mxf32_2 = PDX_SELI_MXF32(sq_sum_mxf32_3, sq_sum_mxf32_2, SEL_INDEX );

        /* std of each dimension */
        std_vec = PDX_SELI_MXF32(sq_sum_mxf32_2, sq_sum_mxf32, SEL_INDEX );
#else
        sum_mxf32 = sum;
        sum_mxf32_1 = sum1;
        sum_mxf32_2 = sum2;
        sum_mxf32_3 = sum3;

        sum_mxf32 = PDX_SELI_MXF32(sum_mxf32_1, sum_mxf32, SEL_INDEX );
        sum_mxf32_2 = PDX_SELI_MXF32(sum_mxf32_3, sum_mxf32_2, SEL_INDEX );

        /* Sum values of each dimension */
        sum_mxf32 = PDX_SELI_MXF32(sum_mxf32_2, sum_mxf32, SEL_INDEX );

        sq_sum_mxf32 = sq_sum;
        sq_sum_mxf32_1 = sq_sum1;
        sq_sum_mxf32_2 = sq_sum2;
        sq_sum_mxf32_3 = sq_sum3;

        sq_sum_mxf32 = PDX_SELI_MXF32(sq_sum_mxf32_1, sq_sum_mxf32, SEL_INDEX );
        sq_sum_mxf32_2 = PDX_SELI_MXF32(sq_sum_mxf32_3, sq_sum_mxf32_2, SEL_INDEX );

        /* Sum of squares of each dimension */
        sq_sum_mxf32 = PDX_SELI_MXF32(sq_sum_mxf32_2, sq_sum_mxf32, SEL_INDEX );

        /*  Calculate mean */
        mean_vec = PDX_MUL_MXF32(sum_mxf32, inv_norm_vec);

        /* Calculate variance */
        std_vec = PDX_MUL_MXF32(sq_sum_mxf32, inv_norm_vec);
        PDX_MULS_MXF32(std_vec, mean_vec, mean_vec);
        std_vec = PDX_ADD_MXF32(std_vec, eps_vec);
#endif

        /* Calculate std */
        std_vec = PDX_SQRT_MXF32(std_vec);

        /* Calculate inverse std */
        rstd_vec = PDX_DIV_MXF32(CONST_ONE, std_vec);

        /* Store mean and inverse std output for each normalized shape of four dims */
        PDX_SA_MXF32_IP(rstd_vec, a_rstd, p_rstd_mxf32);
        PDX_SA_MXF32_IP(mean_vec, a_mean, p_mean_mxf32);

        mean_vec1 = PDX_REP_MXF32(mean_vec,0);
        mean_vec2 = PDX_REP_MXF32(mean_vec, CONST_ONE);
        mean_vec3 = PDX_REP_MXF32(mean_vec, CONST_TWO);
        mean_vec4 = PDX_REP_MXF32(mean_vec, CONST_THREE);

#ifdef ENABLE_HIGH_PRECISION
        std_vec1 = PDX_REP_MXF32(std_vec,0);
        std_vec2 = PDX_REP_MXF32(std_vec, CONST_ONE);
        std_vec3 = PDX_REP_MXF32(std_vec, CONST_TWO);
        std_vec4 = PDX_REP_MXF32(std_vec, CONST_THREE);
#else
        rstd_vec1 = PDX_REP_MXF32(rstd_vec,0);
        rstd_vec2 = PDX_REP_MXF32(rstd_vec, CONST_ONE);
        rstd_vec3 = PDX_REP_MXF32(rstd_vec, CONST_TWO);
        rstd_vec4 = PDX_REP_MXF32(rstd_vec, CONST_THREE);
#endif

        xb_vecMxf32 w0, b0;
        p_weight_mxf32 = (const xb_vecMxf32 *)p_weight;
        p_bias_mxf32 = (const xb_vecMxf32 *)p_bias;

        /* Align load priming of weight and bias */
        valign a_weight = PDX_LA_MXF32_PP(p_weight_mxf32);
        valign a_bias = PDX_LA_MXF32_PP(p_bias_mxf32);

        p_in_mxf32 = (const xb_vecMxf32 *)p_inp1;
        p_in_mxf32_st1 = (const xb_vecMxf32 *)p_inp2;
        p_in_mxf32_st2 = (const xb_vecMxf32 *)p_inp3;
        p_in_mxf32_st3 = (const xb_vecMxf32 *)p_inp4;

        p_inp1 += offset_dim;
        p_inp2 += offset_dim;
        p_inp3 += offset_dim;
        p_inp4 += offset_dim;

        ax = PDX_LA_MXF32_PP(p_in_mxf32);
        ax_st1 = PDX_LA_MXF32_PP(p_in_mxf32_st1);
        ax_st2 = PDX_LA_MXF32_PP(p_in_mxf32_st2);
        ax_st3 = PDX_LA_MXF32_PP(p_in_mxf32_st3);

        /*
         * The layer norm computations for 4 series are performed
         * in two separate loops i.e. Each loop processes 2 series
         * because performing all 4 series in a single loop would
         * require 10 align registers. However fusion_g3 has only
         * 8 align registers available.
         * To avoid an additional cycle caused by alignment priming
         * along with loads within a single loop, the 4 series of
         * computations are split into two loops.
         *
        */

        /* Calculate normalized values of first two dimensions */
#ifdef ENABLE_HIGH_PRECISION
#pragma no_reorder
#endif
        for (j = 0; j < (norm_dim >> LOG2_PDX_M); j++)
        {
            // Load weight */
            PDX_LA_MXF32_IP(w0, a_weight, p_weight_mxf32);

            /* Load bias */
            PDX_LA_MXF32_IP(b0, a_bias, p_bias_mxf32);

            /* Load input of each dimension */
            PDX_LA_MXF32_IP(x0, ax, p_in_mxf32);
            PDX_LA_MXF32_IP(x1, ax_st1, p_in_mxf32_st1);

            /* x[j] - mean_value */
            x0 = PDX_SUB_MXF32(x0, mean_vec1);
            x1 = PDX_SUB_MXF32(x1, mean_vec2);

#ifdef ENABLE_HIGH_PRECISION
            /* (x[j] - mean_value) / std */
            x0 = PDX_DIV_MXF32(x0, std_vec1);
            x1 = PDX_DIV_MXF32(x1, std_vec2);

            b1 = b0;

            /* (x[j] - mean_value)/std * w + b */
			x0 = PDX_MUL_MXF32(x0, w0);
			b0 = PDX_ADD_MXF32(x0, b0);
			x1 = PDX_MUL_MXF32(x1, w0);
			b1 = PDX_ADD_MXF32(x1, b1);

#else
            /* (1/std)*w -> rstd * w */
            w1 = PDX_MUL_MXF32(rstd_vec1, w0);
            w2 = PDX_MUL_MXF32(rstd_vec2, w0);

            b1 = b0;

            /* (x[j] - mean_value) * (1/std * w) + b */
            PDX_MULA_MXF32(b0, x0, w1);
            PDX_MULA_MXF32(b1, x1, w2);
#endif
            /* Store the normalized data */
            PDX_SA_MXF32_IP(b0, a_out, p_out_mxf32);
            PDX_SA_MXF32_IP(b1, a_out_dim1, p_out_mxf32_dim1);
        }

        p_weight_mxf32 = (const xb_vecMxf32 *)p_weight;
        p_bias_mxf32 = (const xb_vecMxf32 *)p_bias;
        a_weight = PDX_LA_MXF32_PP(p_weight_mxf32);
        a_bias = PDX_LA_MXF32_PP(p_bias_mxf32);

        /* Calculate normalized values of next two dimensions */

#ifdef ENABLE_HIGH_PRECISION
#pragma no_reorder
#endif
        for (j = 0; j < (norm_dim >> LOG2_PDX_M); j++)
        {
            /* Load weight */
            PDX_LA_MXF32_IP(w0, a_weight, p_weight_mxf32);

            /* Load bias */
            PDX_LA_MXF32_IP(b0, a_bias, p_bias_mxf32);

            /* Load input of each dimension */
            PDX_LA_MXF32_IP(x2, ax_st2, p_in_mxf32_st2);
            PDX_LA_MXF32_IP(x3, ax_st3, p_in_mxf32_st3);

            /* x[j] - mean_value */
            x2 = PDX_SUB_MXF32(x2, mean_vec3);
            x3 = PDX_SUB_MXF32(x3, mean_vec4);
#ifdef ENABLE_HIGH_PRECISION
            /* (x[j] - mean_value) / rstd */
            x2 = PDX_DIV_MXF32(x2, std_vec3);
            x3 = PDX_DIV_MXF32(x3, std_vec4);

            b3 = b0;

            /* (x[j] - mean_value)/std * w + b */
			x2 = PDX_MUL_MXF32(x2, w0);
			b0 = PDX_ADD_MXF32(x2, b0);
			x3 = PDX_MUL_MXF32(x3, w0);
			b3 = PDX_ADD_MXF32(x3, b3);
#else
            /* (1/std)*w -> rstd * w */
            w3 = PDX_MUL_MXF32(rstd_vec3, w0);
            w4 = PDX_MUL_MXF32(rstd_vec4, w0);

            b3 = b0;

            /* (x[j] - mean_value) * (1/std * w) + b */
            PDX_MULA_MXF32(b0, x2, w3);
            PDX_MULA_MXF32(b3, x3, w4);
#endif
            /* Store the normalized data */
            PDX_SA_MXF32_IP(b0, a_out_dim2, p_out_mxf32_dim2);
            PDX_SA_MXF32_IP(b3, a_out_dim3, p_out_mxf32_dim3);
        }

        /* Load remaining input data */
        PDX_LAV_MXF32_XP(x0, ax, p_in_mxf32, m);
        PDX_LAV_MXF32_XP(x1, ax_st1, p_in_mxf32_st1, m);
        PDX_LAV_MXF32_XP(x2, ax_st2, p_in_mxf32_st2, m);
        PDX_LAV_MXF32_XP(x3, ax_st3, p_in_mxf32_st3, m);

        /* Load weight */
        PDX_LAV_MXF32_XP(w0, a_weight, p_weight_mxf32, m);

        /* Load bias */
        PDX_LAV_MXF32_XP(b0, a_bias, p_bias_mxf32, m);

        /* x[j] - mean_value */
        x0 = PDX_SUB_MXF32(x0, mean_vec1);
        x1 = PDX_SUB_MXF32(x1, mean_vec2);
        x2 = PDX_SUB_MXF32(x2, mean_vec3);
        x3 = PDX_SUB_MXF32(x3, mean_vec4);

#ifdef ENABLE_HIGH_PRECISION
        /* (x[j] - mean_value) / std */
        x0 = PDX_DIV_MXF32(x0, std_vec1);
        x1 = PDX_DIV_MXF32(x1, std_vec2);
        x2 = PDX_DIV_MXF32(x2, std_vec3);
        x3 = PDX_DIV_MXF32(x3, std_vec4);

        b1 = b0;
        b2 = b0;
        b3 = b0;

        // (x[j] - mean_value)/std * w + b;
		x0 = PDX_MUL_MXF32(x0, w0);
		b0 = PDX_ADD_MXF32(x0, b0);
		x1 = PDX_MUL_MXF32(x1, w0);
		b1 = PDX_ADD_MXF32(x1, b1);
		x2 = PDX_MUL_MXF32(x2, w0);
		b2 = PDX_ADD_MXF32(x2, b2);
		x3 = PDX_MUL_MXF32(x3, w0);
		b3 = PDX_ADD_MXF32(x3, b3);
#else
        /* (1/std)*w -> rstd * w */
        w1 = PDX_MUL_MXF32(rstd_vec1, w0);
        w2 = PDX_MUL_MXF32(rstd_vec2, w0);
        w3 = PDX_MUL_MXF32(rstd_vec3, w0);
        w4 = PDX_MUL_MXF32(rstd_vec4, w0);

        b1 = b0;
        b2 = b0;
        b3 = b0;

        /* (x[j] - mean_value) * (1/std * w) + b */
        PDX_MULA_MXF32(b0, x0, w1);
        PDX_MULA_MXF32(b1, x1, w2);
        PDX_MULA_MXF32(b2, x2, w3);
        PDX_MULA_MXF32(b3, x3, w4);
#endif

        /* Store the normalized data */
        PDX_SAV_MXF32_XP(b0, a_out, p_out_mxf32, m);
        PDX_SAV_MXF32_XP(b1, a_out_dim1, p_out_mxf32_dim1, m);
        PDX_SAV_MXF32_XP(b2, a_out_dim2, p_out_mxf32_dim2, m);
        PDX_SAV_MXF32_XP(b3, a_out_dim3, p_out_mxf32_dim3, m);

        PDX_SAPOS_MXF32_FP(a_out, p_out_mxf32);
        PDX_SAPOS_MXF32_FP(a_out_dim1, p_out_mxf32_dim1);
        PDX_SAPOS_MXF32_FP(a_out_dim2, p_out_mxf32_dim2);
        PDX_SAPOS_MXF32_FP(a_out_dim3, p_out_mxf32_dim3);
    }

    i = i*4;

    p_in_mxf32 = (const xb_vecMxf32 *)p_inp1;
    p_in1_mxf32 = (const xb_vecMxf32 *)p_inp1;

    p_out_mxf32 = (xb_vecMxf32 *)p_out1;

#ifdef ENABLE_HIGH_PRECISION
        p_a0 = (xtfloat *)p_inp1;
#endif

    /* Align load priming */
#ifndef ENABLE_HIGH_PRECISION
    ax = PDX_LA_MXF32_PP(p_in_mxf32);
#endif
    valign ax_inp = PDX_LA_MXF32_PP(p_in1_mxf32);

    /* Process remaining leading dimensions */
    for (; i < leading_dim; i++)
    {
        /* Reset sum and sq_sum vectors to zero */
        sum_mxf32 = PDX_ZERO_MXF32();
        sq_sum_mxf32 = PDX_ZERO_MXF32();

#ifdef ENABLE_HIGH_PRECISION
        xtfloat a0 = 0;
        sum = 0, sq_sum = 0;
#pragma no_reorder

        /* Loop runs for norm_dim iterations */
        for (j = 0; j < (norm_dim); j++)
        {
            xtfloat_loadip(a0, p_a0, 4);
            sum = sum + a0;
            sq_sum = sq_sum + XT_MUL_S(a0, a0);
        }
#else
        /* Loop runs for norm_dim/4 iterations */
        for (j = 0; j < (norm_dim >> LOG2_PDX_M); j++)
        {
            /* Aligning load input (4-way) */
            PDX_LA_MXF32_IP(x0, ax, p_in_mxf32);

            /* Add all the inputs of the dimension to be normalized */
            sum_mxf32 = PDX_ADD_MXF32(sum_mxf32, x0);

            /* Calculate the sum of squares of the inputs */
            PDX_MULA_MXF32(sq_sum_mxf32, x0, x0);
        }

        x0 = 0;
        /* Load remaining inputs */
        PDX_LAV_MXF32_XP(x0, ax, p_in_mxf32, m);

        /* Add all the remaining inputs of the dimension to be normalized */
        sum_mxf32 = PDX_ADD_MXF32(sum_mxf32, x0);

        /* Calculate the sum of squares of the remaining inputs */
        PDX_MULA_MXF32(sq_sum_mxf32, x0, x0);

        sum = PDX_RADD_MXF32(sum_mxf32);
        sq_sum = PDX_RADD_MXF32(sq_sum_mxf32);
#endif

#ifdef ENABLE_HIGH_PRECISION
        mean = PDX_DIV_MXF32(sum,norm_dim);
        variance = PDX_DIV_MXF32(sq_sum,norm_dim);
#else
        mean = sum * inv_normalized;
        variance = sq_sum * inv_normalized;
#endif
        /* Calculate mean */
        mean_vec = mean;

        /* Calculate variance */
        variance -= mean * mean;
        variance = variance + eps;
        std_vec = variance;

        /* Calculate std */
        std_vec = PDX_SQRT_MXF32(std_vec);

        /* Calculate inverse std */
        rstd_vec = PDX_DIV_MXF32(1, std_vec);

        /* Store inverse std output for each normalized shape */
        PDX_SAV_MXF32_XP(rstd_vec, a_rstd, p_rstd_mxf32, SIZE_OF_FLOAT);

        /* Store mean */
        p_mean[i] = mean;

        xb_vecMxf32 w0, b0;
        p_weight_mxf32 = (const xb_vecMxf32 *)p_weight;
        p_bias_mxf32 = (const xb_vecMxf32 *)p_bias;

        // Align load priming of weight and bias
        valign a_weight = PDX_LA_MXF32_PP(p_weight_mxf32);
        valign a_bias = PDX_LA_MXF32_PP(p_bias_mxf32);

#ifdef ENABLE_HIGH_PRECISION
#pragma no_reorder
#endif
        /* Calculate normalized values */
        for (j = 0; j < (norm_dim >> LOG2_PDX_M); j++)
        {
            /* Load input */
            PDX_LA_MXF32_IP(x0, ax_inp, p_in1_mxf32);

            /* Load weight */
            PDX_LA_MXF32_IP(w0, a_weight, p_weight_mxf32);

            /* Load bias */
            PDX_LA_MXF32_IP(b0, a_bias, p_bias_mxf32);

            /* x[j] - mean_value */
            x0 = PDX_SUB_MXF32(x0, mean_vec);

#ifdef ENABLE_HIGH_PRECISION
            /* (x[j] - mean_value) / std */
            x0 = PDX_DIV_MXF32(x0, std_vec);

            /* (x[j] - mean_value)/std * w + b */
    		x0 = PDX_MUL_MXF32(x0, w0);
    		b0 = PDX_ADD_MXF32(x0, b0);

#else
            /* (1/std)*w -> rstd * w */
            w0 = PDX_MUL_MXF32(rstd_vec, w0);

            /* (x[j] - mean_value) * (1/std * w) + b */
            PDX_MULA_MXF32(b0, x0, w0);
#endif

            /* Store the normalized data */
            PDX_SA_MXF32_IP(b0, a_out, p_out_mxf32);
        }

        /* Load remaining input data */
        PDX_LAV_MXF32_XP(x0, ax_inp, p_in1_mxf32, m);

        /* Load weight */
        PDX_LAV_MXF32_XP(w0, a_weight, p_weight_mxf32, m);

        /* Load bias */
        PDX_LAV_MXF32_XP(b0, a_bias, p_bias_mxf32, m);

        /* x[j] - mean_value */
        x0 = PDX_SUB_MXF32(x0, mean_vec);

#ifdef ENABLE_HIGH_PRECISION
        /* (x[j] - mean_value) / std */
        x0 = PDX_DIV_MXF32(x0, std_vec);

        /* (x[j] - mean_value)/std * w + b */
    	x0 = PDX_MUL_MXF32(x0, w0);
    	b0 = PDX_ADD_MXF32(x0, b0);

#else
        /* (1/std)*w -> rstd * w */
        w0 = PDX_MUL_MXF32(rstd_vec, w0);

        /* (x[j] - mean_value) * (1/std * w) + b */
        PDX_MULA_MXF32(b0, x0, w0);
#endif

        /* Store the normalized data */
        PDX_SAV_MXF32_XP(b0, a_out, p_out_mxf32, m);
        PDX_SAPOS_MXF32_FP(a_out, p_out_mxf32);
    }

    PDX_SAPOS_MXF32_FP(a_rstd, p_rstd_mxf32);

    return 0;
}
