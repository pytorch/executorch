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

#ifndef __XA_NNLIB_KERNELS_API_H__
#define __XA_NNLIB_KERNELS_API_H__

#if defined(__cplusplus)
    extern "C"
{
#endif

#include "xa_type_def.h"

WORD32 xa_nn_cat(WORD8 * __restrict__ p_out,                                 /*!< [out] output pointer */
        const WORD32 *const p_out_shape,                                     /*!< [in] output pointer dimensions */
        const WORD8 **pp_inps,                                               /*!< [in] pointer to input pointers */
        const WORD32 *const *pp_inps_shape,                                  /*!< [in] input pointer shapes */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 num_inp,                                                      /*!< [in] number of inputs */
        WORD32 axis,                                                         /*!< [in] axis along which the concatenation happens */
        WORD32 elm_size);                                                    /*!< [in] size of one element */

WORD32 xa_nn_elm_add_f32xf32_f32(FLOAT32 * p_out,                            /*!< [out] output pointer */
        const  FLOAT32 * p_inp1,                                             /*!< [in] input1 pointer */
        const  FLOAT32 * p_inp2,                                             /*!< [in] input2 pointer */
        FLOAT32 alpha,                                                       /*!< [in] scale for the input2 */
        WORD32 num_elm);                                                     /*!< [in] total number of elements */

WORD32 xa_nn_elm_add_scalar_f32xf32_f32(FLOAT32 * p_out,                     /*!< [out] output pointer */
        const  FLOAT32 * p_inp1,                                             /*!< [in] input1 pointer */
        const  FLOAT32  inp2,                                                /*!< [in] scalar input */
        FLOAT32 alpha,                                                       /*!< [in] scale for the input2 */
        WORD32 num_elm);                                                     /*!< [in] total number of elements */

WORD32 xa_nn_elm_add_broadcast_5D_f32xf32_f32(FLOAT32 * __restrict__ p_out,  /*!< [out] output pointer */
        const WORD32 * const p_out_shape,                                    /*!< [in] output pointer shape */
        const FLOAT32 * __restrict__ p_inp1,                                 /*!< [in] input1 pointer */
        const WORD32 * const p_inp1_shape,                                   /*!< [in] input1 pointer shape */
        const FLOAT32 * __restrict__ p_inp2,                                 /*!< [in] input2 pointer */
        const WORD32 * const p_inp2_shape,                                   /*!< [in] input2 pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        FLOAT32 alpha);                                                      /*!< [in] scale for the input2 */

WORD32 xa_nn_elm_add_32x32_32(WORD32 * p_out,                                /*!< [out] output pointer */
        const  WORD32 * p_inp1,                                              /*!< [in] input1 pointer */
        const  WORD32 * p_inp2,                                              /*!< [in] input2 pointer */
        WORD32 alpha,                                                        /*!< [in] scale for the input2 */
        WORD32 num_elm);                                                     /*!< [in] total number of elements */

WORD32 xa_nn_elm_add_scalar_32x32_32(WORD32 * p_out,                         /*!< [out] output pointer */
        const  WORD32 * p_inp1,                                              /*!< [in] input1 pointer */
        const  WORD32 inp2,                                                  /*!< [in] scalar input */
        WORD32 alpha,                                                        /*!< [in] scale for the input2 */
        WORD32 num_elm);                                                     /*!< [in] total number of elements */

WORD32 xa_nn_elm_add_broadcast_5D_32x32_32(WORD32 * __restrict__ p_out,      /*!< [out] output pointer */
        const WORD32 * const p_out_shape,                                    /*!< [in] output pointer shape */
        const WORD32 * __restrict__ p_inp1,                                  /*!< [in] input1 pointer */
        const WORD32 * const p_inp1_shape,                                   /*!< [in] input1 pointer shape */
        const WORD32 * __restrict__ p_inp2,                                  /*!< [in] input2 pointer */
        const WORD32 * const p_inp2_shape,                                   /*!< [in] input2 pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 alpha);                                                       /*!< [in] scale for the input2 */

WORD32 xa_nn_elm_dequantize_asym16_f32(FLOAT32 *__restrict__ p_out,          /*!< [out] output pointer */
        const WORD16 *__restrict__ p_inp,                                    /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        WORD32 *p_inp_zero_bias,                                             /*!< [in] pointer to zero_bias */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_sym16_f32(FLOAT32 *__restrict__ p_out,           /*!< [out] output pointer */
        const WORD16 *__restrict__ p_inp,                                    /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_asym16u_f32(FLOAT32 *__restrict__ p_out,         /*!< [out] output pointer */
        const UWORD16 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        WORD32 *p_inp_zero_bias,                                             /*!< [in] pointer to zero_bias */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_sym16u_f32(FLOAT32 *__restrict__ p_out,          /*!< [out] output pointer */
        const UWORD16 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shapes */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_asym8_f32(FLOAT32 *__restrict__ p_out,           /*!< [out] output pointer */
        const WORD8 *__restrict__ p_inp,                                     /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        WORD32 *p_inp_zero_bias,                                             /*!< [in] pointer to zero_bias */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_sym8_f32(FLOAT32 *__restrict__ p_out,            /*!< [out] output pointer */
        const WORD8 *__restrict__ p_inp,                                     /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_asym8u_f32(FLOAT32 *__restrict__ p_out,          /*!< [out] output pointer */
        const UWORD8 *__restrict__ p_inp,                                    /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        WORD32 *p_inp_zero_bias,                                             /*!< [in] pointer to zero_bias */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_sym8u_f32(FLOAT32 *__restrict__ p_out,           /*!< [out] output pointer */
        const UWORD8 *__restrict__ p_inp,                                    /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_asym4_f32(FLOAT32 *__restrict__ p_out,           /*!< [out] output pointer */
        const WORD8 *__restrict__ p_inp,                                     /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shapes */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        WORD32 *p_inp_zero_bias,                                             /*!< [in] pointer to zero_bias */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_sym4_f32(FLOAT32 *__restrict__ p_out,            /*!< [out] output pointer */
        const WORD8 *__restrict__ p_inp,                                     /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_asym4u_f32(FLOAT32 *__restrict__ p_out,          /*!< [out] output pointer */
        const UWORD8 *__restrict__ p_inp,                                    /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        WORD32 *p_inp_zero_bias,                                             /*!< [in] pointer to zero_bias */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */

WORD32 xa_nn_elm_dequantize_sym4u_f32(FLOAT32 *__restrict__ p_out,           /*!< [out] output pointer */
        const UWORD8 *__restrict__ p_inp,                                    /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which dequantization happens */
        FLOAT32 *p_inp_scale);                                               /*!< [in] pointer to input scale */


WORD32 xa_nn_elm_mul_scalar_32x32_32(WORD32 * __restrict__ p_out,            /*!< [out] output pointer */
        const WORD32 * __restrict__ p_inp1,                                  /*!< [in] input1 pointer */
        const WORD32 inp2,                                                   /*!< [in] scalar input */
        WORD32 num_elm);                                                     /*!< [in] total number of elements */

WORD32 xa_nn_elm_mul_32x32_32(WORD32 * __restrict__ p_out,                   /*!< [out] output pointer */
        const WORD32 * __restrict__ p_inp1,                                  /*!< [in] input1 pointer */
        const WORD32 * __restrict__ p_inp2,                                  /*!< [in] input2 pointer */
        WORD32 num_elm);                                                     /*!< [in] total number of elements */  

WORD32 xa_nn_elm_mul_broadcast_5D_32x32_32(WORD32 * __restrict__ p_out,      /*!< [out] output pointer */
        const WORD32 * const p_out_shape,                                    /*!< [in] output pointer shape */
        const WORD32 * __restrict__ p_inp1,                                  /*!< [in] input1 pointer */
        const WORD32 * const p_inp1_shape,                                   /*!< [in] input1 pointer shape */
        const WORD32 * __restrict__ p_inp2,                                  /*!< [in] input2 pointer */
        const WORD32 * const p_inp2_shape,                                   /*!< [in] input2 pointer shape */
        WORD32 num_inp_dims);                                                /*!< [in] number of input dimensions */

WORD32 xa_nn_elm_mul_scalar_f32xf32_f32(FLOAT32 * __restrict__ p_out,        /*!< [out] output pointer */
        const FLOAT32 * __restrict__ p_inp1,                                 /*!< [in] input1 pointer */
        const FLOAT32  inp2,                                                 /*!< [in] scalar input */
        WORD32 num_elm);                                                     /*!< [in] total number of elements */

WORD32 xa_nn_elm_mul_f32xf32_f32(FLOAT32 * __restrict__ p_out,               /*!< [out] output pointer */
        const FLOAT32 * __restrict__ p_inp1,                                 /*!< [in] input1 pointer */
        const FLOAT32 * __restrict__ p_inp2,                                 /*!< [in] input2 pointer */
        WORD32 num_elm);                                                     /*!< [in] total number of elements */

WORD32 xa_nn_elm_mul_broadcast_5D_f32xf32_f32(FLOAT32 * __restrict__ p_out,  /*!< [out] output pointer */
        const WORD32 * const p_out_shape,                                    /*!< [in] output pointer shape */
        const FLOAT32 * __restrict__ p_inp1,                                 /*!< [in] input1 pointer */
        const WORD32 * const p_inp1_shape,                                   /*!< [in] input1 pointer shape */
        const FLOAT32 * __restrict__ p_inp2,                                 /*!< [in] input2 pointer */
        const WORD32 * const p_inp2_shape,                                   /*!< [in] input2 pointer shape */
        WORD32 num_inp_dims);                                                /*!< [in] number of input dimensions */

WORD32 xa_nn_elm_quantize_f32_asym16(WORD16 *__restrict__ p_out,             /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 *p_out_zero_bias,                                             /*!< [in] pointer to zero_bias */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_sym16(WORD16 *__restrict__ p_out,              /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_asym16u(UWORD16 *__restrict__ p_out,           /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 *p_out_zero_bias,                                             /*!< [in] pointer to zero_bias */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_sym16u(UWORD16 *__restrict__ p_out,            /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */


WORD32 xa_nn_elm_quantize_f32_asym8(WORD8 *__restrict__ p_out,               /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 *p_out_zero_bias,                                             /*!< [in] pointer to zero_bias */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_sym8(WORD8 *__restrict__ p_out,                /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_asym8u(UWORD8 *__restrict__ p_out,             /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 *p_out_zero_bias,                                             /*!< [in] pointer to zero_bias */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_sym8u(UWORD8 *__restrict__ p_out,              /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_asym4(WORD8 *__restrict__ p_out,               /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 *p_out_zero_bias,                                             /*!< [in] pointer to zero_bias */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_sym4(WORD8 *__restrict__ p_out,                /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_asym4u(UWORD8 *__restrict__ p_out,             /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 *p_out_zero_bias,                                             /*!< [in] pointer to zero_bias */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */

WORD32 xa_nn_elm_quantize_f32_sym4u(UWORD8 *__restrict__ p_out,              /*!< [out] output pointer */
        const FLOAT32 *__restrict__ p_inp,                                   /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis,                                                      /*!< [in] pointer to axis along which quantization happens */
        FLOAT32 *p_out_scale,                                                /*!< [in] pointer to output scale */
        WORD32 quant_min,                                                    /*!< [in] lower boundary of output */
        WORD32 quant_max);                                                   /*!< [in] upper boundary of output */


WORD32 xa_nn_native_layer_norm_f32_f32(FLOAT32 * p_out,                      /*!< [out] layer_norm output pointer */
        FLOAT32 * p_mean ,                                                   /*!< [out] mean output pointer */
        FLOAT32 * p_std,                                                     /*!< [out] reciprocal of standard deviation pointer */
        const FLOAT32 * p_inp,                                               /*!< [in] input pointer */
        const WORD32 *const p_inp_shape,                                     /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 axis,                                                         /*!< [in] axis along which layer_norm will be calculated */
        const FLOAT32 * p_weight,                                            /*!< [in] pointer to weights */
        const FLOAT32 * p_bias,                                              /*!< [in] pointer to bias */
        FLOAT32  eps);                                                       /*!< [in] epsilon value */

WORD32 xa_nn_softmax_f32_f32(FLOAT32 * p_out,                                /*!< [out] output pointer */
        const FLOAT32 * p_inp,                                               /*!< [in] input pointer */
        const WORD32 * p_inp_shape,                                          /*!< [in] input pointer shape */
        WORD32 num_inp_dims,                                                 /*!< [in] number of input dimensions */
        WORD32 *p_axis);                                                     /*!< [in] pointer to axis along which softmax will be calculated */

#if defined(__cplusplus)
}
#endif
#endif /* __XA_NNLIB_KERNELS_API_H__ */
