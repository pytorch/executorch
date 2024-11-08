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
 * *******************************************************************************/

#include "xa_type_def.h"
#include "xa_nnlib_err_chk.h"
#include "xa_nnlib_kernels_api.h"
#include "xa_nnlib_common_internal.h"

WORD32 xa_nn_cat(WORD8 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        const WORD8 **pp_inps,
        const WORD32 *const*pp_inps_shape,
        WORD32 num_inp_dims,
        WORD32 num_inp,
        WORD32 axis,
        WORD32 elm_size)
{
    /* NULL Pointer Checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);

    /* Pointer Alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, elm_size, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), UNSUPPORTED_PARAM);

    for (int i = 0; i < num_inp; i++)
    {
        XA_NNLIB_ARG_CHK_PTR(pp_inps[i], UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_ALIGN(pp_inps[i], elm_size, UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_PTR(pp_inps_shape[i], UNSUPPORTED_PARAM);
        XA_NNLIB_ARG_CHK_ALIGN(pp_inps_shape[i], sizeof(WORD32),
                UNSUPPORTED_PARAM);
    }

    /* Invalid Input checks */
    XA_NNLIB_ARG_CHK_COND((axis < 0) || (axis >= num_inp_dims),
                            UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND((num_inp_dims <= 0), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND((num_inp <= 0), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_COND((elm_size <= 0) || (elm_size == CONST_THREE)
             || (elm_size > CONST_FOUR), UNSUPPORTED_PARAM);

    /* Calculate outer_size and inner_size based on axis */
    WORD32 outer_size = CONST_ONE;
    WORD32 inner_size = CONST_ONE;

    for (WORD32 i = 0; i < axis; i++)
    {
        outer_size *= p_out_shape[i];
    }

    for (WORD32 i = axis + 1; i < num_inp_dims; i++)
    {
        inner_size *= p_out_shape[i];
    }

    WORD8 *ptmp_out = p_out;

    /* Calculate the total size in bytes for the inner dimensions of the tensor */
    WORD32 inner_size_bytes = inner_size * elm_size;

    /* Loop over each input tensor */
    for (int i = 0; i < num_inp; i++)
    {
        /*
         * Calculate the number of elements to copy based
         * on the shape of the current input along the axis
         */
        WORD32 copy_size = pp_inps_shape[i][axis] * inner_size_bytes;

        const WORD8 *__restrict__ p_i = pp_inps[i];
        WORD8 *__restrict__ p_o = ptmp_out;

        valign align_in = PDX_LA_4MX8_PP((xb_vec4Mx8*) p_i);

        /* Number of full chunks in copy_size */
        WORD32 t_full_chunks = copy_size >> LOG2_PDX_4M;

        /* Remaining bytes after full chunks */
        WORD32 t_remainder = copy_size & MASK_LOG2_PDX_4M;

        xb_vec4Mx8 *in_ptr = (xb_vec4Mx8*) p_i;
        xb_vec4Mx8 *out_ptr = (xb_vec4Mx8*) p_o;

        /* Loop over each slice in the outer dimension */
        for (WORD32 k = 0; k < outer_size; k++)
        {
            p_o = ptmp_out + k * p_out_shape[axis] * inner_size_bytes;
            valign align_out = PDX_Z_ALIGN();

            xb_vec4Mx8 x0;

            /* Process full vector chunks */
            for (WORD32 m = 0; m < t_full_chunks; m++)
            {
                PDX_LA_4MX8_IP(x0, align_in, in_ptr);
                PDX_SA_4MX8_IP(x0, align_out, out_ptr);
            }

            /* Handle any remaining elements */
            PDX_LAV_4MX8_XP(x0, align_in, in_ptr, t_remainder);
            PDX_SAV_4MX8_XP(x0, align_out, out_ptr, t_remainder);

            /* Store the remaining data if any */
            PDX_SAPOS_4MX8_FP(align_out, out_ptr);
            out_ptr = (xb_vec4Mx8*) (ptmp_out
                    + (k + CONST_ONE) * p_out_shape[axis] * inner_size * elm_size);
        }

        /*
         * Update the input pointer and the output
         * pointer after processing one tensor
         */
        in_ptr += copy_size * elm_size;
        ptmp_out += pp_inps_shape[i][axis] * inner_size * elm_size;
    }
    return 0;
}
