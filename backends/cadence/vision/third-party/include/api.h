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
 * API
 */

#ifndef __API_H__
#define __API_H__

#include "dtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

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
Input:
x[N]   input data, Q3.12 floating point
N      Length of input/output data vectors
Output:
y[N]   result, Q7.8 or floating point

Restrictions:
x,y    aligned on 2*BBE_SIMD_WIDTH-bytes boundary (vsoftmax)
x,y    Must not overlap
N      multiple of BBE_SIMD_WIDTH (vsoftmax)
-------------------------------------------------------------------------*/
void vsoftmaxf(float32_t *y, const float32_t *x, int N);

void tensor_transposef(float32_t *restrict ptr_out
    ,const int *const ptr_out_shape
    ,const float32_t *restrict ptr_inp
    ,const int *const ptr_inp_shape
    ,const int *restrict ptr_permute_vec
    ,int num_out_dims
    ,int num_inp_dims);

#ifdef __cplusplus
};
#endif

#endif /* __API_H__ */
