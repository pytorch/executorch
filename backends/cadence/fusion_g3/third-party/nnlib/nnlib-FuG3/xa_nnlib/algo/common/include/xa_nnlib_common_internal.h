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
#ifndef __XA_NNLIB_COMMON_INTERNAL_H__
#define __XA_NNLIB_COMMON_INTERNAL_H__

#include <xtensa/config/core-isa.h>
#include <xtensa/tie/xt_core.h>
#include <xtensa/tie/xt_misc.h>
#include <xtensa/tie/xt_pdx4.h>
#include <xtensa/tie/xt_pdxn.h>

/* floating point units detection flags on PDXNX/Fusion G3/Fusion G6 cores */
#if defined (PDX_MUL_MXF32)
#define HAVE_SP_VFPU 1 /* single precision FPU is selected */
#else
#define HAVE_SP_VFPU 0 /* single precision FPU is NOT selected */
#endif
#if defined (PDX_MUL_M2XF64)
#define HAVE_DP_VFPU 1 /* double precision FPU is selected */
#else
#define HAVE_DP_VFPU 0 /* double precision FPU is NOT selected */
#endif
/* scalar FPUs flags */
#define HAVE_SP_FPU XCHAL_HAVE_FP
#define HAVE_DP_FPU XCHAL_HAVE_DFP

#if (HAVE_SP_FPU)
#include <xtensa/tie/xt_FP.h>
#endif
#if (HAVE_DP_FPU)
#include <xtensa/tie/xt_DFP.h>
#endif

#ifndef PDX_M
#define PDX_M 4 /* SIMD width in 32-bit elements */
#endif

#if PDX_M==4
#define PDX_2M  (PDX_M*2)
#define PDX_4M  (PDX_M*4)
#define PDX_M2  (PDX_M/2)
#define PDX_M4  (PDX_M/4)
/* log2(PDX_M) */
#define LOG2_PDX_M     2
#define LOG2_PDX_4M    (LOG2_PDX_M+2)
#define LOG2_PDX_2M    (LOG2_PDX_M+1)
#define LOG2_PDX_M2    (LOG2_PDX_M-1)
#define LOG2_PDX_M4    (LOG2_PDX_M-2)
#elif PDX_M==8
#define PDX_2M  (PDX_M*2)
#define PDX_4M  (PDX_M*4)
#define PDX_M2  (PDX_M/2)
#define PDX_M4  (PDX_M/4)
/* log2(PDX_M) */
#define LOG2_PDX_M     3
#define LOG2_PDX_4M    (LOG2_PDX_M+2)
#define LOG2_PDX_2M    (LOG2_PDX_M+1)
#define LOG2_PDX_M2    (LOG2_PDX_M-1)
#define LOG2_PDX_M4    (LOG2_PDX_M-2)
#else
#error unsupported PDX_M
#endif

#define UNSUPPORTED_PARAM           -1
#define MAX_DIMS                     5
#define MASK_LOG2_PDX_4M             15
#define SIZE_OF_INT                  sizeof(WORD32)
#define SIZE_OF_INT16                sizeof(WORD16)
#define SIZE_OF_INT8                 sizeof(WORD8)
#define SIZE_OF_FLOAT                sizeof(FLOAT32)
/* log2(size of int) */
#define LOG2_SIZE_INT                2 
/* log2(size of float) */
#define LOG2_SIZE_FLOAT              2
#define NAN                          0x7fc00000

#define INT16_LOWER_LIMIT           -32768
#define INT16_UPPER_LIMIT            32767
#define UINT16_LOWER_LIMIT           0
#define UINT16_UPPER_LIMIT           65535
#define INT8_LOWER_LIMIT            -128
#define INT8_UPPER_LIMIT             127
#define UINT8_LOWER_LIMIT            0
#define UINT8_UPPER_LIMIT            255
#define INT4_LOWER_LIMIT            -8
#define INT4_UPPER_LIMIT             7
#define UINT4_LOWER_LIMIT            0
#define UINT4_UPPER_LIMIT            15

#define SHIFT_FACTOR_4_BIT           4
#define SCALE_FACTOR_4_BIT           16

/* Macros for constants */
#define CONST_ONE                    1
#define CONST_TWO                    2
#define CONST_THREE                  3
#define CONST_FOUR                   4
#define CONST_FIVE                   5
#define CONST_SIX                    6

#define LOOP_UNROLL_BY_8             8
#define IS_NOT_32_MULTIPLE           31
#define SEL_INDEX                    30

#define Q24_SHIFT_BITS               24       // Bit shift for Q24 representation
#define FRACTIONAL_COMPONENT_SHIFT   22       // Bit shift for fractional component extraction
#define EXPONENT_SHIFT_BITS          54       // Bit shift for extracting exponent
#define POLYNOMIAL_APPROXIMATION_SHIFT 31     // Bit shift for polynomial approximation
#define EXPONENT_BIAS                127      // Bias for exponent in floating-point representation
#define Q31_SHIFT_BITS               30       // Bit shift for Q31 representation
#define Q24_SHIFT_BITS_MINUS_ONE     23

#define IS_ALIGN(p) ((((int)(p))&0x7) == 0)
#define ALIGN(x)    __attribute__((aligned(x))) 
#define ALIGN_PDX_4M    __attribute__((aligned(PDX_4M)))  /* alignment on PDX_M*4 byte boundary */

/*-----------------------------------------------------
 Common constants
-----------------------------------------------------*/

#define M_PI_FLT  3.14159265358979323846f
#define M_PI_DBL  3.14159265358979323846

#ifdef __cplusplus
#define externC extern "C" 
#else
#define externC extern 
#endif

#endif /* __XA_NNLIB_COMMON_INTERNAL_H__ */
