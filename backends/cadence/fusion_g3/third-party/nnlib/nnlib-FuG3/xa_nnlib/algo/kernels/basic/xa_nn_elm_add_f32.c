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

WORD32 xa_nn_elm_add_f32xf32_f32(FLOAT32 *p_out,
        const FLOAT32 *p_inp1,
        const FLOAT32 *p_inp2,
        FLOAT32 alpha,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    WORD32 n, m;

    xb_vecMxf32 x0, y0;
    valign ax, ay, az;

    const xb_vecMxf32 *restrict p_x = (const xb_vecMxf32*) p_inp1;
    const xb_vecMxf32 *restrict p_y = (const xb_vecMxf32*) p_inp2;
    xb_vecMxf32 *restrict p_z = (xb_vecMxf32*) p_out;

    xb_vecMxf32 vec_alpha = alpha;

    ax = PDX_LA_MXF32_PP(p_x);
    ay = PDX_LA_MXF32_PP(p_y);
    az = PDX_Z_ALIGN();

    for (n = 0; n < (num_elm >> LOG2_PDX_M); n++)
    {
        PDX_LA_MXF32_IP(x0, ax, p_x);
        PDX_LA_MXF32_IP(y0, ay, p_y);
        PDX_MULA_MXF32(x0, y0, vec_alpha);
        PDX_SA_MXF32_IP(x0, az, p_z);
    }
    m = (num_elm & (PDX_M - CONST_ONE)) << LOG2_SIZE_FLOAT;
    PDX_LAV_MXF32_XP(x0, ax, p_x, m);
    PDX_LAV_MXF32_XP(y0, ay, p_y, m);
    PDX_MULA_MXF32(x0, y0, vec_alpha);
    PDX_SAV_MXF32_XP(x0, az, p_z, m);
    PDX_SAPOS_MXF32_FP(az, p_z);

    return 0;
}

WORD32 xa_nn_elm_add_scalar_f32xf32_f32(FLOAT32 *p_out,
        const FLOAT32 *p_inp1,
        const FLOAT32 inp2,
        FLOAT32 alpha,
        WORD32 num_elm)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), UNSUPPORTED_PARAM);

    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), UNSUPPORTED_PARAM);

    WORD32 n, m;

    xb_vecMxf32 x, y, vec_alpha;
    vec_alpha = alpha;
    y = inp2;
    valign ax, az;

    const xb_vecMxf32 *restrict p_x = (const xb_vecMxf32*) p_inp1;
    xb_vecMxf32 *restrict p_z = (xb_vecMxf32*) p_out;

    /* Align load priming */
    ax = PDX_LA_MXF32_PP(p_x);

    /* Zeroing align register */
    az = PDX_Z_ALIGN();

    /* loop iterates for multiple of LOG2_PDX_M */
    for (n = 0; n < (num_elm >> LOG2_PDX_M); n++)
    {
        /* Aligning load input1 (4-way 32 bit) */
        PDX_LA_MXF32_IP(x, ax, p_x);
        /* Add input1 and (input2 * alpha) */
        PDX_MULA_MXF32(x, y, vec_alpha);
        /* Aligning store (4-way 32bit elements)
         * with post increment addressing
         */
        PDX_SA_MXF32_IP(x, az, p_z);
    }

    /* Remaining elements after processing the loop */
    m = (num_elm & (PDX_M - CONST_ONE)) << LOG2_SIZE_FLOAT;

    /* Variable aligining load */
    PDX_LAV_MXF32_XP(x, ax, p_x, m);

    PDX_MULA_MXF32(x, y, vec_alpha);
    /* Variable aligining store and flush */
    PDX_SAV_MXF32_XP(x, az, p_z, m);
    PDX_SAPOS_MXF32_FP(az, p_z);

    return 0;
}

static inline void shapes_convert_5D(WORD32 *const __restrict__ p_5d_out_shape,
        WORD32 *const __restrict__ p_5d_inp1_shape,  /* new input1 shapes */
        WORD32 *const __restrict__ p_5d_inp2_shape,  /* new input2 shapes */
        const WORD32 *const __restrict__ p_out_shape,
        const WORD32 *const __restrict__ p_inp1_shape,  /* original input1 shapes */
        const WORD32 *const __restrict__ p_inp2_shape,  /* original input1 shapes */
        const WORD32 num_inp_dims)
{
    /* Convert number of dimension less than 5D to 5D */
    for (WORD32 i = 0; i < num_inp_dims; i++)
    {
        p_5d_out_shape[i + MAX_DIMS - num_inp_dims] = p_out_shape[i];
        p_5d_inp1_shape[i + MAX_DIMS - num_inp_dims] = p_inp1_shape[i];
        p_5d_inp2_shape[i + MAX_DIMS - num_inp_dims] = p_inp2_shape[i];
    }
}

static inline WORD32 check_shapes(const WORD32 *const p_inp1_shape,
        const WORD32 *const p_inp2_shape,
        const WORD32 *const p_out_shape)
{
    /* Check the shapes of input and output */
    for (WORD32 i = 0; i < MAX_DIMS; i++)
    {
        if (((p_inp1_shape[i] != p_inp2_shape[i])
                && (p_inp1_shape[i] != CONST_ONE)
                && (p_inp2_shape[i] != CONST_ONE))
                || (p_out_shape[i]
                        != (p_inp1_shape[i] > p_inp2_shape[i] ?
                                p_inp1_shape[i] : p_inp2_shape[i])))
        {
            return UNSUPPORTED_PARAM;
        }
    }
    return 0;
}

static inline void strides_calculation(const WORD32 *const inp1_shape,
        const WORD32 *const inp2_shape,
        WORD32 *const inp1_strides,
        WORD32 *const inp2_strides)
{
    inp1_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    inp2_strides[MAX_DIMS - CONST_ONE] = CONST_ONE;
    for (WORD32 i = MAX_DIMS - CONST_TWO; i >= 0; i--)
    {
        inp1_strides[i] = inp1_strides[i + CONST_ONE]
                * inp1_shape[i + CONST_ONE];
        inp2_strides[i] = inp2_strides[i + CONST_ONE]
                * inp2_shape[i + CONST_ONE];
    }
}

static inline void internal_elm_add_broadcast_2D_f32xf32_f32(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp1,
        const FLOAT32 *__restrict__ p_inp2,
        WORD32 out_lc,
        WORD32 in_lc,
        const WORD32 *input1_shapes,
        const WORD32 *input2_shapes,
        FLOAT32 alpha)
{
    WORD32 n, m;

    xb_vecMxf32 x0, x1, y0, y1;

    xb_vecMxf32 vec_alpha = alpha;

    const xb_vecMxf32 *__restrict__ p_x = (const xb_vecMxf32*) &p_inp1[0];
    const xb_vecMxf32 *__restrict__ p_y = (const xb_vecMxf32*) &p_inp2[0];

    valign ax, ax0, ax1, ay, ay0, ay1, az, az0, az1;
    ax = PDX_LA_MXF32_PP(p_x);
    ay = PDX_LA_MXF32_PP(p_y);
    az = PDX_Z_ALIGN();

    const FLOAT32 *__restrict__ p_x_baseptr = &p_inp1[0];
    const xb_vecMxf32 *__restrict__ p_x0 = (const xb_vecMxf32*) &p_x_baseptr[0];
    const xb_vecMxf32 *__restrict__ p_x1 = (const xb_vecMxf32*) (&p_x_baseptr[0]
            + ((out_lc / CONST_TWO) * in_lc));

    ax0 = PDX_LA_MXF32_PP(p_x0);
    ax1 = PDX_LA_MXF32_PP(p_x1);

    const FLOAT32 *__restrict__ p_y_baseptr = &p_inp2[0];
    const xb_vecMxf32 *__restrict__ p_y0 = (const xb_vecMxf32*) &p_y_baseptr[0];
    const xb_vecMxf32 *__restrict__ p_y1 = (const xb_vecMxf32*) (&p_y_baseptr[0]
            + ((out_lc / CONST_TWO) * in_lc));

    ay0 = PDX_LA_MXF32_PP(p_y0);
    ay1 = PDX_LA_MXF32_PP(p_y1);

    FLOAT32 *__restrict__ p_z_baseptr = &p_out[0];
    xb_vecMxf32 *__restrict__ p_z0 = (xb_vecMxf32*) &p_z_baseptr[0];
    xb_vecMxf32 *__restrict__ p_z1 = (xb_vecMxf32*) (&p_z_baseptr[0]
            + ((out_lc / CONST_TWO) * in_lc));

    az0 = PDX_Z_ALIGN();
    az1 = PDX_Z_ALIGN();

    if (input1_shapes[3] == CONST_ONE)
    {
        /* input1_shapes[3] is 1 */
        for (WORD32 i = 0; i < out_lc - CONST_ONE; i += CONST_TWO)
        {
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                PDX_LA_MXF32_IP(x0, ax, p_x);
                x1 = x0;

                PDX_LA_MXF32_IP(y0, ay0, p_y0);
                PDX_LA_MXF32_IP(y1, ay1, p_y1);

                PDX_MULA_MXF32(x0, y0, vec_alpha);
                PDX_SA_MXF32_IP(x0, az0, p_z0);

                PDX_MULA_MXF32(x1, y1, vec_alpha);
                PDX_SA_MXF32_IP(x1, az1, p_z1);
            }
            m = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
            PDX_LAV_MXF32_XP(x0, ax, p_x, m);
            x1 = x0;

            PDX_LAV_MXF32_XP(y0, ay0, p_y0, m);
            PDX_LAV_MXF32_XP(y1, ay1, p_y1, m);

            PDX_MULA_MXF32(x0, y0, vec_alpha);
            PDX_SAV_MXF32_XP(x0, az0, p_z0, m);

            PDX_MULA_MXF32(x1, y1, vec_alpha);
            PDX_SAV_MXF32_XP(x1, az1, p_z1, m);

            PDX_SAPOS_MXF32_FP(az0, p_z0);
            PDX_SAPOS_MXF32_FP(az1, p_z1);

            p_x = (const xb_vecMxf32*) &p_inp1[0];
            ax = PDX_LA_MXF32_PP(p_x);
        }
        if (out_lc % CONST_TWO != 0)
        {
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                PDX_LA_MXF32_IP(y1, ay1, p_y1);
                PDX_LA_MXF32_IP(x0, ax, p_x);
                PDX_MULA_MXF32(x0, y1, vec_alpha);
                PDX_SA_MXF32_IP(x0, az1, p_z1);
            }
            m = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
            PDX_LAV_MXF32_XP(y1, ay1, p_y1, m);
            PDX_LAV_MXF32_XP(x0, ax, p_x, m);
            PDX_MULA_MXF32(x0, y1, vec_alpha);
            PDX_SAV_MXF32_XP(x0, az1, p_z1, m);
            PDX_SAPOS_MXF32_FP(az1, p_z1);
        }
    }
    else
    {
        /* input2_shapes[3] is 1 */
        for (WORD32 i = 0; i < out_lc - CONST_ONE; i += CONST_TWO)
        {
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                PDX_LA_MXF32_IP(y0, ay, p_y);

                PDX_LA_MXF32_IP(x0, ax0, p_x0);
                PDX_LA_MXF32_IP(x1, ax1, p_x1);

                PDX_MULA_MXF32(x0, y0, vec_alpha);
                PDX_MULA_MXF32(x1, y0, vec_alpha);

                PDX_SA_MXF32_IP(x0, az0, p_z0);
                PDX_SA_MXF32_IP(x1, az1, p_z1);
            }
            m = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
            PDX_LAV_MXF32_XP(y0, ay, p_y, m);
            PDX_LAV_MXF32_XP(x0, ax0, p_x0, m);
            PDX_LAV_MXF32_XP(x1, ax1, p_x1, m);
            PDX_MULA_MXF32(x0, y0, vec_alpha);
            PDX_MULA_MXF32(x1, y0, vec_alpha);
            PDX_SAV_MXF32_XP(x0, az0, p_z0, m);
            PDX_SAV_MXF32_XP(x1, az1, p_z1, m);
            PDX_SAPOS_MXF32_FP(az0, p_z0);
            PDX_SAPOS_MXF32_FP(az1, p_z1);

            p_y = (const xb_vecMxf32*) &p_inp2[0];
            ay = PDX_LA_MXF32_PP(p_y);
        }
        if (out_lc % CONST_TWO != 0)
        {
            for (n = 0; n < (in_lc >> LOG2_PDX_M); n++)
            {
                PDX_LA_MXF32_IP(y0, ay, p_y);
                PDX_LA_MXF32_IP(x1, ax1, p_x1);
                PDX_MULA_MXF32(x1, y0, vec_alpha);
                PDX_SA_MXF32_IP(x1, az1, p_z1);
            }
            m = (in_lc & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
            PDX_LAV_MXF32_XP(y0, ay, p_y, m);
            PDX_LAV_MXF32_XP(x0, ax1, p_x1, m);
            PDX_MULA_MXF32(x0, y0, vec_alpha);
            PDX_SAV_MXF32_XP(x0, az1, p_z1, m);
            PDX_SAPOS_MXF32_FP(az1, p_z1);
        }
    }
}

static inline void internal_elm_add_broadcast_1D_scalar_f32xf32_f32(
        FLOAT32 *__restrict__ p_out,
        const FLOAT32 *__restrict__ p_inp1,
        const FLOAT32 *__restrict__ p_inp2,
        WORD32 num_elm,
        const WORD32 *__restrict__ input1_shapes,
        const WORD32 inp1_const,
        const FLOAT32 alpha)
{

    xb_vecMxf32 x0, x1, y, vec_alpha = alpha;
    xb_vecMxf32 *restrict p_i1 = (xb_vecMxf32*) p_inp1;
    xb_vecMxf32 *restrict p_i2 = (xb_vecMxf32*) p_inp2;
    xb_vecMxf32 *restrict p_o = (xb_vecMxf32*) p_out;

    valign ax, az;
    az = PDX_Z_ALIGN();

    WORD32 m;

    if ((input1_shapes[4] == CONST_ONE) || (inp1_const == CONST_ONE))
    {
        x0 = PDX_LSR_F32_I(p_inp1, 0);
        ax = PDX_LA_MXF32_PP(p_i2);
        for(WORD32 i = 0; i < (num_elm >> LOG2_PDX_M); i++)
        {
            PDX_LA_MXF32_IP(y, ax, p_i2);
            x1 = x0;
            PDX_MULA_MXF32(x1, y, vec_alpha);
            PDX_SA_MXF32_IP(x1, az, p_o);
        }
        m = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
        PDX_LAV_MXF32_XP(y, ax, p_i2, m);
        x1 = x0;
        PDX_MULA_MXF32(x1, y, vec_alpha);
        PDX_SAV_MXF32_XP(x1, az, p_o, m);
    }
    else
    {
        y = PDX_LSR_F32_I(p_inp2, 0);
        ax = PDX_LA_MXF32_PP(p_i1);
        for (WORD32 i = 0; i < (num_elm >> LOG2_PDX_M); i++)
        {
            PDX_LA_MXF32_IP(x0, ax, p_i1);
            PDX_MULA_MXF32(x0, y, vec_alpha);
            PDX_SA_MXF32_IP(x0, az, p_o);
        }
        m = (num_elm & (PDX_M - CONST_ONE)) * SIZE_OF_FLOAT;
        PDX_LAV_MXF32_XP(x0, ax, p_i1, m);
        PDX_MULA_MXF32(x0, y, vec_alpha);
        PDX_SAV_MXF32_XP(x0, az, p_o, m);
    }
    PDX_SAPOS_MXF32_FP(az, p_o);
}

WORD32 xa_nn_elm_add_broadcast_5D_f32xf32_f32(FLOAT32 *__restrict__ p_out,
        const WORD32 *const p_out_shape,
        const FLOAT32 *__restrict__ p_inp1,
        const WORD32 *const p_inp1_shape,
        const FLOAT32 *__restrict__ p_inp2,
        const WORD32 *const p_inp2_shape,
        WORD32 num_inp_dims,
        FLOAT32 alpha)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_out_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, UNSUPPORTED_PARAM);

    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), UNSUPPORTED_PARAM);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), UNSUPPORTED_PARAM);

    /* UNSUPPORTED_PARAM input checks */
    XA_NNLIB_ARG_CHK_COND(((num_inp_dims <= 0) || (num_inp_dims > MAX_DIMS)),
            UNSUPPORTED_PARAM);

    /* 5D shapes initialization */
    WORD32 p_5d_out_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
            CONST_ONE, CONST_ONE};
    WORD32 p_5d_inp1_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
            CONST_ONE, CONST_ONE};
    WORD32 p_5d_inp2_shape[MAX_DIMS] = {CONST_ONE, CONST_ONE, CONST_ONE,
            CONST_ONE, CONST_ONE};

    shapes_convert_5D(p_5d_out_shape, p_5d_inp1_shape, p_5d_inp2_shape,
            p_out_shape, p_inp1_shape, p_inp2_shape, num_inp_dims);

    /* Check shapes for broadcast compatibility */
    WORD32 error = 0;
    error = check_shapes(p_5d_inp1_shape, p_5d_inp2_shape, p_5d_out_shape);
    if (error)
    {
        return UNSUPPORTED_PARAM;
    }

    /* Strides calculation */
    WORD32 inp1_strides[MAX_DIMS], inp2_strides[MAX_DIMS];
    strides_calculation(p_5d_inp1_shape, p_5d_inp2_shape, inp1_strides,
            inp2_strides);

    /* Check for broadcast need */
    WORD32 need_broadcast = 0;
    WORD32 inp1_const = CONST_ONE, inp2_const = CONST_ONE;
    for (int i = 0; i < MAX_DIMS; i++)
    {
        if (p_5d_inp1_shape[i] != p_5d_inp2_shape[i])
        {
            if (p_5d_inp1_shape[i] == CONST_ONE)
            {
                inp1_strides[i] = 0;
            }
            else
            {
                inp2_strides[i] = 0;
            }
            need_broadcast = CONST_ONE;
        }

        if (p_5d_inp1_shape[i] != CONST_ONE) inp1_const &= 0;
        if (p_5d_inp2_shape[i] != CONST_ONE) inp2_const &= 0;
    }

    const FLOAT32 *__restrict__ p_inp1_base = p_inp1;
    const FLOAT32 *__restrict__ p_inp2_base = p_inp2;
    FLOAT32 *p_out_base = p_out;

    /* If broadcast is not needed */
    if (need_broadcast == 0)
    {
        xa_nn_elm_add_f32xf32_f32(
                p_out_base,
                p_inp1_base,
                p_inp2_base,
                alpha,
                p_5d_out_shape[0] * inp1_strides[0]);
    }

    /* If broadcast is needed */
    else if (inp1_const == CONST_ONE || inp2_const == CONST_ONE)
    {
        internal_elm_add_broadcast_1D_scalar_f32xf32_f32(
                p_out_base,
                p_inp1_base,
                p_inp2_base,
                p_5d_out_shape[0] * p_5d_out_shape[1] * p_5d_out_shape[2]
                        * p_5d_out_shape[3] * p_5d_out_shape[4],
                p_5d_inp1_shape,
                inp1_const,
                alpha);
    }
    /* Check if 4th dim in both inputs is the same */
    else if (inp1_strides[4] == inp2_strides[4])
    {
        WORD32 in_lc, out_lc;
        /* Check if 3rd dim needs to be broadcasted */
        if (inp1_strides[3] == 0 || inp2_strides[3] == 0)
        {
            /* Repeat the 4th dimension as the 3rd dimension needs to be broadcasted */
            in_lc = p_5d_out_shape[4];
            out_lc = p_5d_out_shape[3];
            for (WORD32 itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
            {
                const FLOAT32 *__restrict__ p_inp1_itr0 = p_inp1_base;
                const FLOAT32 *__restrict__ p_inp2_itr0 = p_inp2_base;
                for (WORD32 itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const FLOAT32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                    const FLOAT32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                    for (WORD32 itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        internal_elm_add_broadcast_2D_f32xf32_f32(
                                p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                out_lc,
                                in_lc,
                                p_5d_inp1_shape,
                                p_5d_inp2_shape,
                                alpha);

                        p_out_base += in_lc * out_lc;
                        p_inp1_itr1 += inp1_strides[2];
                        p_inp2_itr1 += inp2_strides[2];
                    }
                    p_inp1_itr0 += inp1_strides[1];
                    p_inp2_itr0 += inp2_strides[1];
                }
                p_inp1_base += inp1_strides[0];
                p_inp2_base += inp2_strides[0];
            }
        }
        else
        {
            /* 3rd and 4th dimensions need not be broadcasted. The lower
             * dimension broadcasting (0th, 1st, 2nd) will be taken care
             * while calculating the input addresses */
            in_lc = p_5d_out_shape[3] * p_5d_out_shape[4];
            for (WORD32 itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
            {
                const FLOAT32 *__restrict__ p_inp1_itr0 = p_inp1_base;
                const FLOAT32 *__restrict__ p_inp2_itr0 = p_inp2_base;
                for (WORD32 itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
                {
                    const FLOAT32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                    const FLOAT32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                    for (WORD32 itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                    {
                        xa_nn_elm_add_f32xf32_f32(
                                p_out_base,
                                p_inp1_itr1,
                                p_inp2_itr1,
                                alpha,
                                in_lc);
                        p_out_base += in_lc;
                        p_inp1_itr1 += inp1_strides[2];
                        p_inp2_itr1 += inp2_strides[2];
                    }
                    p_inp1_itr0 += inp1_strides[1];
                    p_inp2_itr0 += inp2_strides[1];
                }
                p_inp1_base += inp1_strides[0];
                p_inp2_base += inp2_strides[0];
            }
        }
    }
    else
    {
        /* If the last dim itself is broadcastable */
        for (WORD32 itr0 = 0; itr0 < p_5d_out_shape[0]; itr0++)
        {
            const FLOAT32 *__restrict__ p_inp1_itr0 = p_inp1_base;
            const FLOAT32 *__restrict__ p_inp2_itr0 = p_inp2_base;
            for (WORD32 itr1 = 0; itr1 < p_5d_out_shape[1]; itr1++)
            {
                const FLOAT32 *__restrict__ p_inp1_itr1 = p_inp1_itr0;
                const FLOAT32 *__restrict__ p_inp2_itr1 = p_inp2_itr0;
                for (WORD32 itr2 = 0; itr2 < p_5d_out_shape[2]; itr2++)
                {
                    const FLOAT32 *__restrict__ p_inp1_itr2 = p_inp1_itr1;
                    const FLOAT32 *__restrict__ p_inp2_itr2 = p_inp2_itr1;
                    for (WORD32 itr3 = 0; itr3 < p_5d_out_shape[3]; itr3++)
                    {
                        internal_elm_add_broadcast_1D_scalar_f32xf32_f32(
                                p_out_base,
                                p_inp1_itr2,
                                p_inp2_itr2,
                                p_5d_out_shape[4],
                                p_5d_inp1_shape,
                                inp1_const,
                                alpha);
                        p_out_base += p_5d_out_shape[4];
                        p_inp1_itr2 += inp1_strides[3];
                        p_inp2_itr2 += inp2_strides[3];
                    }
                    p_inp1_itr1 += inp1_strides[2];
                    p_inp2_itr1 += inp2_strides[2];
                }
                p_inp1_itr0 += inp1_strides[1];
                p_inp2_itr0 += inp2_strides[1];
            }
            p_inp1_base += inp1_strides[0];
            p_inp2_base += inp2_strides[0];
        }
    }
    return 0;
}
