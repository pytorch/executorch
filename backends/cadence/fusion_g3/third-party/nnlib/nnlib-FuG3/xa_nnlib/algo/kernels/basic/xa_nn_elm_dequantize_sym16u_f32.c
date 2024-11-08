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

WORD32 xa_nn_elm_dequantize_sym16u_f32(FLOAT32 *__restrict__ p_out,
        const UWORD16 *__restrict__ p_inp,
        const WORD32 *const p_inp_shape,
        WORD32 num_inp_dims,
        WORD32 *p_axis,
        FLOAT32 *p_inp_scale)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp_scale, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(UWORD16), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp_scale, sizeof(FLOAT32), UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);

    /* Number of elements to be processed with a stride of 1 */
    WORD32 num_elm = CONST_ONE;
    /* Number of leading dimensions of axis */
    WORD32 leading_dims = CONST_ONE;
    /* Number of trailing dimensions of axis */
    WORD32 trailing_dims = CONST_ONE;
    WORD32 length_per_step = 0;
    WORD32 axis_count = CONST_ONE;

    if (p_axis == NULL)
    {
        for (WORD32 i = 0; i < num_inp_dims; i++)
        {
            num_elm *= p_inp_shape[i];
        }
    }
    else
    {
        WORD32 axis = *p_axis;
        /* Invalid input checks
         * axis should be in the range [0,num_inp_dims-1]
         */
        XA_NNLIB_ARG_CHK_COND(((axis < 0) || (axis >= num_inp_dims)),
                UNSUPPORTED_PARAM);

        /* Calculating leading dims */
        for (WORD32 i = 0; i < axis; i++)
        {
            leading_dims *= p_inp_shape[i];
        }

        /* Calculating trailing dims */
        for (WORD32 i = axis + CONST_ONE; i < num_inp_dims; i++)
        {
            trailing_dims *= p_inp_shape[i];
        }

        num_elm = trailing_dims;

        /* Number of elements to be skipped after trailing number of
         * elements quantized with a scale and zero_bias values to get
         * the next base addresses.
         */
        length_per_step = p_inp_shape[axis] * trailing_dims;

        /* Length of the dimension along axis */
        axis_count = p_inp_shape[axis];
    }

    /* Base pointers that points to the first element in the channel */
    const UWORD16 *__restrict__ inp_base;
    FLOAT32 *__restrict__ out_base;

    /* Vector pointers for the base pointers */
    const xb_vecMxu16 *__restrict__ inp_base_p;
    xb_vecMxf32 *__restrict__ out_base_p;

    WORD32 m = (num_elm & (PDX_M - CONST_ONE));
    WORD32 m_16 = m * SIZE_OF_INT16;
    WORD32 m_32 = m * SIZE_OF_FLOAT;

    WORD32 leading_dim_idx;
    valign align_a, align_out;
    align_out = PDX_Z_ALIGN();

    xb_vecMxu32 x0;
    xb_vecMxf32 y0;

    /* Outermost loop iterates over the channels */
    for (WORD32 axis_index = 0; axis_index < axis_count; axis_index++)
    {
        xb_vecMxf32 d_inp_scale = p_inp_scale[axis_index];
        inp_base = p_inp + (axis_index * trailing_dims);
        out_base = p_out + (axis_index * trailing_dims);

        /* This loop iterates over the leading dims.
         * All the elements are quantized at a time for
         * single scale and zero_bias once loaded
         */
        for (leading_dim_idx = 0; leading_dim_idx < leading_dims;
                leading_dim_idx++)
        {
            inp_base_p = (const xb_vecMxu16*) inp_base;
            align_a = PDX_LA_MXU16_PP(inp_base_p);
            out_base_p = (xb_vecMxf32*) out_base;

            for (WORD32 i = 0; i < (num_elm >> LOG2_PDX_M); i++)
            {
                PDX_LAU32_MX16_IP(x0, align_a, inp_base_p);
                y0 = PDX_MUL_MXF32(x0, d_inp_scale);
                PDX_SA_MXF32_IP(y0, align_out, out_base_p);
            }
            PDX_LAVU32_MX16_XP(x0, align_a, inp_base_p, m_16);
            y0 = PDX_MUL_MXF32(x0, d_inp_scale);
            PDX_SAV_MXF32_XP(y0, align_out, out_base_p, m_32);
            PDX_SAPOS_MXF32_FP(align_out, out_base_p);

            inp_base = inp_base + length_per_step;
            out_base = out_base + length_per_step;
        }
    }

    return 0;
}
