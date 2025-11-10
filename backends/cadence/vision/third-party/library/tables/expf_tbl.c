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
    tables for expf(x) approximation
*/
/* Portable data types. */
#include "expf_tbl.h"
#include "dtypes.h"

/*
   polynomial coefficients for 2^x in range 0...1

   derived by MATLAB code:
   order=6;
   x=(0:pow2(1,-16):1);
   y=2.^x;
   p=polyfit(x,y,6);
   p(order+1)=1;
   p(order)=p(order)-(sum(p)-2);
*/
const int32_t ALIGN_2SIMD expftbl_Q30[8] = {
    234841,
    1329551,
    10400465,
    59570027,
    257946177,
    744260763,
    1073741824,
    0 /* Padding to allow for vector loads */
};

const union ufloat32uint32 ALIGN_2SIMD
    expfminmax[2] = /* minimum and maximum arguments of expf() input */
    {
        {0xc2ce8ed0}, /*-1.0327893066e+002f */
        {0x42b17218} /* 8.8722839355e+001f */
};

const int32_t invln2_Q30 = 1549082005L; /* 1/ln(2), Q30 */

const union ufloat32uint32 ALIGN_2SIMD log2_e[2] = {
    {0x3fb8aa3b}, /* 1.4426950216      */
    {0x32a57060} /* 1.9259629891e-008 */
};

/*
order=6;
x=(0:pow2(1,-16):1);
y=2.^x;
p=polyfit(x,y,order);
p(order+1)=1;
p(order)=p(order)-(sum(p)-2);
num2hex(single(p));
*/
const union ufloat32uint32 ALIGN_2SIMD expftblf[] = {
    {0x39655635},
    {0x3aa24c7a},
    {0x3c1eb2d1},
    {0x3d633ddb},
    {0x3e75ff24},
    {0x3f317212},
    {0x3f800000}};
