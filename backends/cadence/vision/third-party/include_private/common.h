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

#ifndef __COMMON_H__
#define __COMMON_H__

#if defined COMPILER_XTENSA
#include <xtensa/config/core-isa.h>
#include <xtensa/tie/xt_ivpn.h>
#include <xtensa/tie/xt_ivpn_verification.h>
#include <xtensa/tie/xt_core.h>
#include <xtensa/tie/xt_density.h>
#include <xtensa/tie/xt_misc.h>
#if XCHAL_HAVE_IDMA
#ifndef IDMA_USE_MULTICHANNEL
  #define IDMA_USE_MULTICHANNEL 1
#endif
#include <xtensa/idma.h>
#endif
#define IVP_SIMD_WIDTH XCHAL_IVPN_SIMD_WIDTH

#include "xtensa/config/core-isa.h"
#include "xtensa/tie/xt_ivpn.h"
#if XCHAL_HAVE_IDMA
#include "xtensa/idma.h"
#endif

#ifdef _MSC_VER
#define ALIGN(x) _declspec(align(x))
#else
#define ALIGN(x) __attribute__((aligned(x)))
#endif

#ifdef COMPILER_XTENSA
#define ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define ATTRIBUTE_NEVER_INLINE __attribute__((noinline))
#define ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define ATTRIBUTE_ALWAYS_INLINE
#define ATTRIBUTE_NEVER_INLINE
#define ATTRIBUTE_UNUSED
#endif

/* 'restrict' qualifier, is applied to pointers only under clang compiler */
#ifdef __clang__
#define restrict_clang restrict
#else
#define restrict_clang
#endif

// Performance measurement macros
#define XTPERF_PRINTF(...) printf(__VA_ARGS__)
#define TIME_DECL(test) long start_time_##test, end_time_##test;
#define TIME_START(test) { start_time_##test = 0;   XT_WSR_CCOUNT(0); }
#define TIME_END(test) { end_time_##test = XT_RSR_CCOUNT(); }
#define TIME_DISPLAY(test, opcnt, opname) { long long cycles_##test = end_time_##test - start_time_##test; \
		XTPERF_PRINTF("PERF_LOG : %s : %d : %s : %lld : cycles : %.2f : %s/cycle : %.2f : cycles/%s\n", \
		       #test, opcnt, opname, cycles_##test, cycles_##test == 0 ? 0 : (double)(opcnt)/cycles_##test, \
           opname, cycles_##test == 0 ? 0 : 1/((double)(opcnt)/cycles_##test), opname); }

//-----------------------------------------------------
// log2(BBE_SIMD_WIDTH)
//-----------------------------------------------------
#define LOG2_IVP_SIMD_WIDTH 5
#define ALIGN_SIMD ALIGN(64)
#define ALIGN_2SIMD ALIGN(128)

#define LOG2_SIMD_N_2 (LOG2_IVP_SIMD_WIDTH - 1)
#define LOG2_SIMD_2N (LOG2_IVP_SIMD_WIDTH + 1)
//-----------------------------------------------------
// some C++ support
//-----------------------------------------------------

// special XCC type casting of pointers
#ifdef __cplusplus
#define castxcc(type_, ptr) (ptr)
#else
#define castxcc(type_, ptr) (type_ *)(ptr)
#endif

//-----------------------------------------------------
// C99 pragma wrapper
//-----------------------------------------------------

#ifdef COMPILER_XTENSA
#define __Pragma(a) _Pragma(a)
#else
#define __Pragma(a)
#endif

//-----------------------------------------------------
// Conditionalization support
//-----------------------------------------------------
/* place DISCARD_FUN(retval_type,name) instead of function definition for
   functions to be discarded from the executable THIS WORKS only for external
   library functions declared as extern "C" and not supported for internal
   references without "C" qualifier!
*/
#ifdef COMPILER_MSVC
#pragma section("$DISCARDED_FUNCTIONS", execute, discard)
#pragma section("$$$$$$$$$$", execute, discard)
#define DISCARD_FUN(retval_type, name, arglist)                                \
  __pragma(alloc_text("$DISCARDED_FUNCTIONS", name))                           \
      __pragma(section("$DISCARDED_FUNCTIONS", execute, discard))              \
          __pragma(warning(push)) __pragma(warning(disable : 4026 4716))       \
              retval_type name arglist {}                                      \
  __pragma(warning(pop))
#endif

#if defined(COMPILER_XTENSA) || defined(COMPILER_GNU)
#define DISCARD_FUN(retval_type, name, arglist)                                \
  __asm__(".type " #name ", @object\n\t.global " #name                         \
          "\n\t.align 4\n\t" #name ":\n\t.long 0x49438B96,0x4D73F192\n\t");
#endif

/*------ LIST OF DEFINES DEPENDING ON ISA OPTIONS ------*/

/* Single-precision Extended Vector Floating-point option */
#if ((XCHAL_HAVE_VISION_SP_VFPU))
#define HAVE_SPX_VFPU 1
#else
#define HAVE_SPX_VFPU 0
#endif

/* all vector single precision/Extended vector floating point instructions */
#if ((XCHAL_HAVE_VISION_SP_VFPU))
#define HAVE_SPX_VFPU 1
#define HAVE_VFPU 1
#else
#define HAVE_SPX_VFPU 0
#define HAVE_VFPU 0
#endif

/* all scalar single precision floating point instructions */
#if ((XCHAL_HAVE_VISION_SP_VFPU) || (XCHAL_HAVE_FP))
#define HAVE_FPU 1
#else
#define HAVE_FPU 0
#endif

#else
#define HAVE_VFPU 0
#define HAVE_FPU 0
#endif

/* detect if half precision FPU is present in a core */
#if ((XCHAL_HAVE_VISION_HP_VFPU))
#define HAVE_HPFPU 1
#include <xtensa/tie/xt_ivpn_scalarfp.h>
#else
#define HAVE_HPFPU 0
#endif

/* detect if double precision FPU is present in a core */
#if ((XCHAL_HAVE_VISION_DP_VFPU))
#define HAVE_DPFPU 1
#include <xtensa/tie/xt_ivpn_scalarfp.h>
#else
#define HAVE_DPFPU 0
#endif

/*
  32x32 multiplier
*/
#if defined(BBE_MULN_2X32)
#define HAVE_32X32 1
#else
#define HAVE_32X32 0
#endif

#ifdef __cplusplus
#define externC extern "C"
#else
#define externC extern
#endif

#endif // __COMMON_H__
