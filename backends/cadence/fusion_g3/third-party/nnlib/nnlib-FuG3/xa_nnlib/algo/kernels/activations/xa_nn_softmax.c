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

#include "expf_tbl.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_internal.h"

#define EXPONENT(x0, out_val)                                                       \
{                                                                                   \
    xb_vecMxf32 approx;                                                             \
    xb_vecMx32 in_int, frac, exp, temp, exp0, exp1;                                 \
    in_int = PDX_TRUNC32_MXF32(x0, Q24_SHIFT_BITS);                                 \
    xb_vecMx80 temp0 = PDX_MULW_MX32(in_int, invln2_q30);                           \
    xb_vecMx80 temp1 = PDX_SRAI_MX80(temp0, FRACTIONAL_COMPONENT_SHIFT);            \
    frac = PDX_SRLI_MX32(PDX_PACKV_MX80(temp1), CONST_ONE);                         \
    temp1 = PDX_SRAI_MX80(temp0, EXPONENT_SHIFT_BITS);                              \
    exp = PDX_PACKV_MX80(temp1);                                                    \
    xb_vecMx32 f2 = PDX_PACKSIV_MX80(PDX_MULW_MX32(frac, frac),                     \
                                    POLYNOMIAL_APPROXIMATION_SHIFT);                \
    xb_vecMx32 y1 = PDX_LSR_32_I(expftbl_q30, 0);                                   \
    xb_vecMx32 y2 = PDX_LSR_32_I(expftbl_q30, SIZE_OF_INT);                         \
    xb_vecMx32 c1, c2;                                                              \
    c1 = PDX_LSR_32_I(expftbl_q30, CONST_TWO * SIZE_OF_INT);                        \
    temp = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y1), POLYNOMIAL_APPROXIMATION_SHIFT); \
    y1 = PDX_ADD_MX32(c1, temp);                                                    \
    c2 = PDX_LSR_32_I(expftbl_q30, CONST_THREE * SIZE_OF_INT);                      \
    temp = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y2), POLYNOMIAL_APPROXIMATION_SHIFT); \
    y2 = PDX_ADD_MX32(c2, temp);                                                    \
    c1 = PDX_LSR_32_I(expftbl_q30, CONST_FOUR * SIZE_OF_INT);                       \
    temp = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y1), POLYNOMIAL_APPROXIMATION_SHIFT); \
    y1 = PDX_ADD_MX32(c1, temp);                                                    \
    c2 = PDX_LSR_32_I(expftbl_q30, CONST_FIVE * SIZE_OF_INT);                       \
    temp = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y2), POLYNOMIAL_APPROXIMATION_SHIFT); \
    y2 = PDX_ADD_MX32(c2, temp);                                                    \
    c1 = PDX_LSR_32_I(expftbl_q30, CONST_SIX * SIZE_OF_INT);                        \
    temp = PDX_PACKSIV_MX80(PDX_MULW_MX32(f2, y1), POLYNOMIAL_APPROXIMATION_SHIFT); \
    y1 = PDX_ADD_MX32(c1, temp);                                                    \
    xb_vecMx32 g = PDX_ADD_MX32(y1, PDX_PACKSIV_MX80(PDX_MULW_MX32(frac, y2),       \
                                                  POLYNOMIAL_APPROXIMATION_SHIFT)); \
    approx = PDX_FLOATF32_MX32(g, Q31_SHIFT_BITS);                                              \
    exp1 = PDX_SRAI_MX32(exp, CONST_ONE);                                                   \
    exp0 = PDX_SUB_MX32(exp, exp1);                                                 \
    exp0 = PDX_ADD_MX32(EXPONENT_BIAS, exp0);                                                 \
    exp1 = PDX_ADD_MX32(EXPONENT_BIAS, exp1);                                                 \
    exp0 = PDX_SLLI_MX32(exp0, Q24_SHIFT_BITS_MINUS_ONE);                                                 \
    exp1 = PDX_SLLI_MX32(exp1, Q24_SHIFT_BITS_MINUS_ONE);                                                 \
    xb_vecMxf32 scale0 = PDX_MOV_MXF32_FROM_4MX8(PDX_MOV_4MX8_FROM_MX32(exp0));     \
    xb_vecMxf32 scale1 = PDX_MOV_MXF32_FROM_4MX8(PDX_MOV_4MX8_FROM_MX32(exp1));     \
    out_val = PDX_MUL_MXF32(approx, scale0);                                        \
    out_val = PDX_MUL_MXF32(out_val, scale1);                                       \
}

WORD32 xa_nn_softmax_f32_f32(FLOAT32 *p_out,
                             const FLOAT32 *p_inp,
                             const WORD32 *p_inp_shape,
                             WORD32 num_inp_dims,
                             WORD32 *p_axis)
{
    WORD32 i, j, dim = 0;

    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_inp_dims <= 0), UNSUPPORTED_PARAM);
    if (p_axis != NULL)
    {
        XA_NNLIB_ARG_CHK_COND((*p_axis < 0), UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_COND((*p_axis >= num_inp_dims), UNSUPPORTED_PARAM);
    }

    for (i = 0; i < num_inp_dims; i++)
    {
        XA_NNLIB_ARG_CHK_COND((p_inp_shape[i] <= 0), UNSUPPORTED_PARAM);
    }

    /* Leading dimensions across which softmax calculation is repeated */
    WORD32 leading_dim = CONST_ONE;
    /* number of elements for which softmax is computed */
    WORD32 inner_count = CONST_ONE;
    /* Stride with which the elements are loaded */
    WORD32 inner_stride = CONST_ONE;

    if (p_axis != NULL)
    {
        dim = *p_axis;
        inner_count = p_inp_shape[dim];

        /* Calculate number of elements of leading dimensions */
        for (int i = 0; i < dim; i++)
        {
            leading_dim *= p_inp_shape[i];
        }

        for (int i = dim + 1; i < num_inp_dims; i++)
        {
            inner_stride *= p_inp_shape[i];
        }
    }
    else /* if p_axis is NULL, then softmax is calculated over entire input dimensions */
    {
        for (i = 0; i < num_inp_dims; i++)
        {
            inner_count *= p_inp_shape[i];
        }
    }

    if (inner_stride == CONST_ONE)
    {
        xb_vecMxf32 x0, x1, max_vec;
        xb_vecMxf32 *restrict p_out_mxf32 = (xb_vecMxf32 *)p_out;
        xb_vecMxf32 *restrict p_out_exp_mxf32 = (xb_vecMxf32 *)p_out;
        xb_vecMxf32 *restrict p_out_softmax_mxf32 = p_out_exp_mxf32;

        /* Calculate number of remaining elements after processing inner loop*/
        WORD32 rem_elem_bytes = (inner_count & (PDX_M - 1)) << LOG2_SIZE_FLOAT;

        /* Set 4-way vboolM vector bit to true based on remaining elements */
        xb_vecMx32 rem_elem_vec = rem_elem_bytes;
        xb_vecMx32 list_rem_bytes = {4, 8, 12, 16};

        /* Set flag only for the remaining elements */
        vboolM bool_vec = PDX_GE_MX32(rem_elem_vec, list_rem_bytes);

        const xb_vecMxf32 *restrict p_in_exp_mxf32 = (const xb_vecMxf32 *)p_inp;
        valign ax_inp = PDX_LA_MXF32_PP(p_in_exp_mxf32);

        /* Loop count of maximum value calculation*/
        WORD32 count = inner_count - PDX_M;

        /* Offset from base address to load inputs for maximum value calculation */
        WORD32 offset2 = ((count - (count & (PDX_2M - 1))) >> 1) + PDX_M;

        const FLOAT32 *p_inp_out_itr = p_inp;
        for (i = 0; i < leading_dim; i++)
        {
            const xb_vecMxf32 *restrict p_in_mxf32 = (const xb_vecMxf32 *)p_inp_out_itr;
            const xb_vecMxf32 *restrict p_in_max_mxf32 =
                    (const xb_vecMxf32 *)(p_inp_out_itr + offset2);

            p_inp_out_itr += inner_count;

            /* Align load priming of input */
            valign ax = PDX_LA_MXF32_PP(p_in_mxf32);
            valign ax_max = PDX_LA_MXF32_PP(p_in_max_mxf32);

            FLOAT32 max_elem = 0;

            /* Aligning load input (4-way) */
            PDX_LA_MXF32_IP(max_vec, ax, p_in_mxf32);

            /* Calculate maximum value among elements for which softmax will be computed */

            /* Loop runs for inner_count/8 iterations */
            for (j = 0; j < (count >> LOG2_PDX_2M); j++)
            {
                /* Load input (4-way) */
                PDX_LA_MXF32_IP(x0, ax, p_in_mxf32);
                PDX_LA_MXF32_IP(x1, ax_max, p_in_max_mxf32);

                /* Calculate max value 4-way */
                x0 = PDX_MAXNUM_MXF32(x0, x1);
                max_vec = PDX_MAXNUM_MXF32(max_vec, x0);
            }

            if ((count & (PDX_2M - 1)) >= PDX_M)
            {
                /* Load input (4-way) */
                PDX_LA_MXF32_IP(x1, ax_max, p_in_max_mxf32);

                /* Calculate max value 4-way */
                max_vec = PDX_MAXNUM_MXF32(max_vec, x1);
            }

            vboolM a;

            PDX_LAV_MXF32_XP(x0, ax_max, p_in_max_mxf32, rem_elem_bytes);
            PDX_MAXNUM_MXF32_T(max_vec, max_vec, x0, bool_vec);

            PDX_RBMAXNUM_MXF32(a, max_elem, max_vec);
            max_vec = max_elem;

            valign align_z = PDX_Z_ALIGN();

            FLOAT32 exp_sum = 0;
            xb_vecMxf32 exp_sum_mxf32 = PDX_ZERO_MXF32();
            xb_vecMxf32 out_val;
            xb_vecMxf32 inv_exp_sum_mxf32;

            /* Calculate exponent of each element */
            for (j = 0; j < inner_count >> LOG2_PDX_M; j++)
            {
                /* Aligning load input (4-way) */
                PDX_LA_MXF32_IP(x0, ax_inp, p_in_exp_mxf32);

                /* Sub max value from each input element */
                x0 = PDX_SUB_MXF32(x0, max_vec);

                /* Calculate exponent */
                EXPONENT(x0, out_val);

                /* Accumulate the exp values */
                exp_sum_mxf32 = PDX_ADD_MXF32(exp_sum_mxf32, out_val);

                /* Store output */
                PDX_SA_MXF32_IP(out_val, align_z, p_out_mxf32);
            }

            if (rem_elem_bytes > 0)
            {
                /* Load remaining input data */
                PDX_LAV_MXF32_XP(x0, ax_inp, p_in_exp_mxf32, rem_elem_bytes);

                /* x[j] - mean_value */
                PDX_SUB_MXF32_T(x0, x0, max_vec, bool_vec);

                /* Calculate exponent */
                EXPONENT(x0, out_val);

                /* Accumulate the exp values */
                PDX_ADD_MXF32_T(exp_sum_mxf32, exp_sum_mxf32, out_val, bool_vec);

                /* Store the normalized data */
                PDX_SAV_MXF32_XP(out_val, align_z, p_out_mxf32, rem_elem_bytes);
            }

            PDX_SAPOS_MXF32_FP(align_z, p_out_mxf32);

            exp_sum = PDX_RADD_MXF32(exp_sum_mxf32);

            FLOAT32 inv_exp_sum;
            PDX_DIV_F32_T(inv_exp_sum, 1, exp_sum, 1);
            inv_exp_sum_mxf32 = inv_exp_sum;

            /* Align load priming of output */
            valign a_out = PDX_LA_MXF32_PP(p_out_exp_mxf32);

            align_z = PDX_Z_ALIGN();

            /* Compute softmax for each element */
            for (j = 0; j < inner_count >> LOG2_PDX_M; j++)
            {
                /* Load exp values of each element (4-way) */
                PDX_LA_MXF32_IP(x0, a_out, p_out_exp_mxf32);

                /* Calculate the softmax */
                x0 = PDX_MUL_MXF32(x0, inv_exp_sum_mxf32);

                /* Store the softmax */
                PDX_SA_MXF32_IP(x0, align_z, p_out_softmax_mxf32);
            }

            /* Load remaining input data */
            PDX_LAV_MXF32_XP(x0, a_out, p_out_exp_mxf32, rem_elem_bytes);

            /* Calculate the softmax */
            x0 = PDX_MUL_MXF32(x0, inv_exp_sum_mxf32);

            /* Store the softmax */
            PDX_SAV_MXF32_XP(x0, align_z, p_out_softmax_mxf32, rem_elem_bytes);
            PDX_SAPOS_MXF32_FP(align_z, p_out_softmax_mxf32);
        }
    }
    else
    {
        WORD32 rem_elem_bytes;
        valign ax, ax_inp;
        xb_vecMxf32 x0, x1, max_vec;
        WORD32 k;
        WORD32 offset = inner_stride * inner_count;
        WORD32 inner_stride_bytes = inner_stride << LOG2_SIZE_FLOAT;

        const FLOAT32 *p_inp1 = p_inp;
        const FLOAT32 *p_out1 = p_out;
        /* number of remaining elements to be processed */
        WORD32 rem_elem = (inner_stride & (PDX_M - 1));
        xb_vecMxf32 *restrict p_out_mxf32;

        for (i = 0; i < leading_dim; i++)
        {
            const FLOAT32 *p_inp2 = p_inp1;
            const FLOAT32 *p_out2 = p_out1;
            for (j = 0; j < inner_stride - rem_elem; j += 4)
            {
                p_inp2 = p_inp1 + j;
                p_out2 = p_out1 + j;

                const FLOAT32 *p_inp3;
                const xb_vecMxf32 *restrict p_in_mxf32 =
                     (const xb_vecMxf32 *)(p_inp2);
                const xb_vecMxf32 *restrict p_in_max_mxf32 =
                    (const xb_vecMxf32 *)(p_inp2 + inner_stride);

                ax = PDX_LA_MXF32_PP(p_in_mxf32);
                PDX_LA_MXF32_XP(max_vec, ax, p_in_mxf32, inner_stride_bytes * 2);

                /* inner_count -> group of elements on which softmax is computed */
                for (k = 0; k < (inner_count - 1) >> 1; k++)
                {
                    /* Align load priming of input */
                    ax_inp = PDX_LA_MXF32_PP(p_in_max_mxf32);
                    ax = PDX_LA_MXF32_PP(p_in_mxf32);

                    /* Load input elements with stride "inner_stride" */
                    PDX_LA_MXF32_XP(x1, ax_inp, p_in_max_mxf32,
                                   inner_stride_bytes * 2);
                    PDX_LA_MXF32_XP(x0, ax, p_in_mxf32, inner_stride_bytes * 2);

                    /* Calculate maximum across each lane of vector */
                    x0 = PDX_MAXNUM_MXF32(x0, x1);
                    max_vec = PDX_MAXNUM_MXF32(x0, max_vec);
                }

                WORD32 rem = ((inner_count - 1) & (1));

                if (rem)
                {
                    /* Align load priming of input */
                    ax = PDX_LA_MXF32_PP(p_in_max_mxf32);

                    /* Load input elements with stride "inner_stride" */
                    PDX_LA_MXF32_XP(x0, ax, p_in_max_mxf32, inner_stride_bytes * 2);

                    /* Calculate maximum across each lane of vector */
                    max_vec = PDX_MAXNUM_MXF32(x0, max_vec);
                }

                /* Calculate exponent of each element */
                xb_vecMxf32 exp_sum_mxf32 = PDX_ZERO_MXF32();
                valign align_z = PDX_Z_ALIGN();
                xb_vecMxf32 out_val;
                p_inp3 = p_inp2;
                const FLOAT32 *p_out3 = p_out2;

                p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);

                for (k = 0; k < inner_count; k++)
                {
                    p_out_mxf32 = (xb_vecMxf32 *)p_out3;
                    /* Align load priming of input */
                    ax = PDX_LA_MXF32_PP(p_in_mxf32);

                    /* Load input elements with stride "inner_stride" */
                    PDX_LA_MXF32_XP(x0, ax, p_in_mxf32, inner_stride_bytes);

                    /* Sub max value from each input element */
                    x0 = PDX_SUB_MXF32(x0, max_vec);

                    /* Calculate exponent */
                    EXPONENT(x0, out_val);

                    /* Accumulate the exp values */
                    exp_sum_mxf32 = PDX_ADD_MXF32(exp_sum_mxf32, out_val);

                    /* Store output */
                    PDX_SA_MXF32_IP(out_val, align_z, p_out_mxf32);
                    PDX_SAPOS_MXF32_FP(align_z, p_out_mxf32);
                    p_out3 += inner_stride;
                }

                xb_vecMxf32 inv_exp_sum_mxf32;
                inv_exp_sum_mxf32 = PDX_DIV_MXF32(1, exp_sum_mxf32);

                /* Compute softmax */
                align_z = PDX_Z_ALIGN();
                const xb_vecMxf32 *restrict p_out_exp_mxf32 = (xb_vecMxf32 *)(p_out2);

                for (k = 0; k < inner_count; k++)
                {
                    xb_vecMxf32 *restrict p_out_softmax_mxf32 = (xb_vecMxf32 *)p_out2;
                    /* Align load priming */
                    ax = PDX_LA_MXF32_PP(p_out_exp_mxf32);

                    /* Aligning load exp values of each element (4-way) */
                    PDX_LA_MXF32_XP(x0, ax, p_out_exp_mxf32, inner_stride_bytes);

                    /* Calculate the softmax */
                    x0 = PDX_MUL_MXF32(x0, inv_exp_sum_mxf32);

                    /* Store the softmax */
                    PDX_SA_MXF32_IP(x0, align_z, p_out_softmax_mxf32);
                    PDX_SAPOS_MXF32_FP(align_z, p_out_softmax_mxf32);
                    p_out2 += inner_stride;
                }
            }

            /* Process remaining elements */
            rem_elem_bytes = rem_elem * SIZE_OF_FLOAT;
            p_inp2 = p_inp1 + j;
            p_out2 = p_out1 + j;

            const FLOAT32 *p_inp3 = p_inp2;
            const xb_vecMxf32 *restrict p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);

            ax = PDX_LA_MXF32_PP(p_in_mxf32);
            PDX_LAV_MXF32_XP(max_vec, ax, p_in_mxf32, rem_elem_bytes);

            /* Calculate maximum among group of elements on which softmax is computed */
            for (k = 0; k < inner_count - 1; k++)
            {
                p_inp3 += inner_stride;
                p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);

                /* Align load priming of input */
                ax = PDX_LA_MXF32_PP(p_in_mxf32);

                /* Load input elements with stride "inner_stride" */
                PDX_LAV_MXF32_XP(x0, ax, p_in_mxf32, rem_elem_bytes);

                /* Calculate maximum across each lane of vector */
                max_vec = PDX_MAXNUM_MXF32(x0, max_vec);
            }

            /* Calculate exponent of each element */
            xb_vecMxf32 exp_sum_mxf32 = PDX_ZERO_MXF32();
            valign align_z = PDX_Z_ALIGN();
            xb_vecMxf32 out_val;
            p_inp3 = p_inp2;
            const FLOAT32 *p_out3 = p_out2;

            /* Calculate exp of group of elements on which softmax is calculated */
            for (k = 0; k < inner_count; k++)
            {
                /* const Float32 *p_inp3 = p_inp2 + k*inner_stride; */
                const xb_vecMxf32 *restrict p_in_mxf32 = (const xb_vecMxf32 *)(p_inp3);
                xb_vecMxf32 *restrict p_out_mxf32 = (xb_vecMxf32 *)p_out3;

                p_inp3 += inner_stride;
                p_out3 += inner_stride;

                /* Align load priming of input */
                ax = PDX_LA_MXF32_PP(p_in_mxf32);

                /* Load input elements with stride "inner_stride" */
                PDX_LAV_MXF32_XP(x0, ax, p_in_mxf32, rem_elem_bytes);

                /* Sub max value from each input element */
                x0 = PDX_SUB_MXF32(x0, max_vec);

                /* Calculate exponent */
                EXPONENT(x0, out_val);

                /* Accumulate the exp values */
                exp_sum_mxf32 = PDX_ADD_MXF32(exp_sum_mxf32, out_val);

                /* Store output */
                PDX_SAV_MXF32_XP(out_val, align_z, p_out_mxf32, rem_elem_bytes);
                PDX_SAPOS_MXF32_FP(align_z, p_out_mxf32);
            }

            xb_vecMxf32 inv_exp_sum_mxf32;
            inv_exp_sum_mxf32 = PDX_DIV_MXF32(1, exp_sum_mxf32);

            /* Compute softmax */
            p_inp3 = p_out2;
            p_out3 = p_out2;
            align_z = PDX_Z_ALIGN();

            /* Calculate softmax of group of elements */
            for (k = 0; k < inner_count; k++)
            {
                const xb_vecMxf32 *restrict p_out_exp_mxf32 = (const xb_vecMxf32 *)(p_inp3);
                xb_vecMxf32 *restrict p_out_softmax_mxf32 = (xb_vecMxf32 *)p_out3;

                p_inp3 += inner_stride;
                p_out3 = p_inp3;
                /* Align load priming */
                ax = PDX_LA_MXF32_PP(p_out_exp_mxf32);

                /* Load exp values of each element (4-way) */
                PDX_LAV_MXF32_XP(x0, ax, p_out_exp_mxf32, rem_elem_bytes);

                /* Calculate the softmax */
                x0 = PDX_MUL_MXF32(x0, inv_exp_sum_mxf32);

                /* Store the softmax */
                PDX_SAV_MXF32_XP(x0, align_z, p_out_softmax_mxf32, rem_elem_bytes);
                PDX_SAPOS_MXF32_FP(align_z, p_out_softmax_mxf32);
            }

            p_inp1 = p_inp1 + offset;
            p_out1 = p_out1 + offset;
        }
    }
    return 0;
}
