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
 * Cross-platform data type definitions and utility macros
 */

#ifndef __DTYPES_H__
#define __DTYPES_H__

#include <stddef.h>

#ifndef COMPILER_ANSI
/* ----------------------------------------------------------
             Compilers autodetection
 ----------------------------------------------------------*/
#define ___UNKNOWN_COMPILER_YET
#ifdef ___UNKNOWN_COMPILER_YET
#ifdef _MSC_VER

#ifdef _ARM_
#define COMPILER_CEARM9E /* Microsoft Visual C++,ARM9E */
#else
#define COMPILER_MSVC /* Microsoft Visual C++ */
#endif

#undef ___UNKNOWN_COMPILER_YET
#endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
#ifdef _TMS320C6X
#if defined(_TMS320C6400)
#define COMPILER_C64
#undef ___UNKNOWN_COMPILER_YET
#endif
#if defined(_TMS320C6400_PLUS)
#define COMPILER_C64PLUS
#undef ___UNKNOWN_COMPILER_YET
#endif
#endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
#ifdef __TMS320C55X__
#define COMPILER_C55
#undef ___UNKNOWN_COMPILER_YET
#endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
#ifdef __ADSPBLACKFIN__
#define COMPILER_ADSP_BLACKFIN
#undef ___UNKNOWN_COMPILER_YET
#endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
#ifdef __XCC__
#define COMPILER_XTENSA
#undef ___UNKNOWN_COMPILER_YET
#endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
#ifdef __GNUC__
#ifdef __arm__
#ifndef COMPILER_GNU_ARM
#endif
#define COMPILER_GNUARM /* GNU C/C++ compiler*/
#else
/* GNU GCC x86 compiler */
#ifndef COMPILER_GNU
#endif
#define COMPILER_GNU /* GNU C/C++ */
#endif
#undef ___UNKNOWN_COMPILER_YET
#endif
#endif

#ifdef ___UNKNOWN_COMPILER_YET
#error Unknown compiler
#endif

#endif /* #ifndef COMPILER_ANSI */

/* ----------------------------------------------------------
        Language-dependent definitions
 ----------------------------------------------------------*/
#ifdef __cplusplus

#undef extern_C
#define extern_C extern "C"

#else

#undef extern_C
#define extern_C

#ifndef false
#define false 0
#endif
#ifndef true
#define true 1
#endif

#endif

/*    Assertion support                   */
#if !defined(_ASSERT)
#include <assert.h>
#if defined(_DEBUG) /*&& defined(COMPILER_MSVC)*/
#define ASSERT(x)                                                              \
  { assert(x); }
#else

/*#undef ASSERT*/
#ifndef ASSERT
#define ASSERT(_ignore) ((void)0)
#endif

#endif /* _DEBUG */
#else  /* ASSERT*/
#define ASSERT(exp)                                                            \
  {                                                                            \
    extern void ExternalAssertHandler(void *, void *, unsigned);               \
    (void)((exp) || (ExternalAssertHandler(#exp, __FILE__, __LINE__), 0));     \
  }
#endif /* ASSERT */

/*** Inline methods definition ***/
#undef inline_
#if (defined COMPILER_MSVC) || (defined COMPILER_CEARM9E)
#define inline_ __inline
#elif defined(COMPILER_ADSP_BLACKFIN)
#define inline_ inline
#elif defined(COMPILER_ANSI)
#define inline_
#elif (defined COMPILER_GNU) || (defined COMPILER_GNUARM) ||                   \
    (defined COMPILER_ARM)
#define inline_ static inline
#else
#define inline_ static inline
#endif

#ifndef MAX_FLT32
#define MAX_FLT32  (3.402823466e+38F)
#endif
#ifndef MIN_FLT32
#define MIN_FLT32  (1.175494351e-38F)
#endif
#ifndef MAX_INT8
#define MAX_INT8  (0x7f)
#endif
#ifndef MIN_INT8
#define MIN_INT8  (- MAX_INT8 - 1)
#endif
#ifndef MAX_INT16
#define MAX_INT16 (0x7FFF)
#endif
#ifndef MIN_INT16
#define MIN_INT16 (0x8000)
#endif
#ifndef MAX_INT32
#define MAX_INT32 (0x7FFFFFFFL)
#endif
#ifndef MIN_INT32
#define MIN_INT32 (0x80000000L)
#endif
#ifndef MIN_INT64
#define MIN_INT64 (0x8000000000000000LL)
#endif
#ifndef MAX_INT64
#define MAX_INT64 (0x7fffffffffffffffLL)
#endif

/* size of variables in bytes */
#ifdef COMPILER_C55
#define SIZEOF_BYTE(x) (sizeof(x) << 1)
#else
#define SIZEOF_BYTE(x) sizeof(x)
#endif

#ifndef FLT32_SIZE
#define FLT32_SIZE  4
#endif
#ifndef INT8_SIZE
#define INT8_SIZE 1
#endif
#ifndef INT16_SIZE
#define INT16_SIZE 2
#endif
#ifndef INT32_SIZE
#define INT32_SIZE 4
#endif
#ifndef INT64_SIZE
#define INT64_SIZE 8
#endif

/*---------------------------------------
 special keywords definition
 restrict  keyword means that the memory
           is addressed exclusively via
           this pointer
 onchip    keyword means that the memory
           is on-chip and can not be
           accessed via external bus
---------------------------------------*/
#if defined(COMPILER_C55)
#define NASSERT _nassert
#elif defined(COMPILER_C64)
#define onchip
#define NASSERT _nassert
#elif defined(COMPILER_ADSP_BLACKFIN)
#define onchip
#define NASSERT(x) __builtin_assert(x)
#elif defined(COMPILER_GNUARM)
#define onchip
#define NASSERT(x)                                                             \
  { (void)__builtin_expect((x) != 0, 1); }
#define restrict __restrict
#elif defined(COMPILER_GNU)
#define onchip
#define NASSERT(x)                                                             \
  {                                                                            \
    (void)__builtin_expect((x) != 0, 1);                                       \
    ASSERT(x);                                                                 \
  }
#define restrict __restrict
#elif defined(COMPILER_CEARM9E)
#define onchip
#define NASSERT(x)
#define restrict
#elif defined(COMPILER_XTENSA)
#ifndef restrict
#define restrict __restrict
#endif
#define onchip
#define NASSERT(x)                                                             \
  {                                                                            \
    (void)__builtin_expect((x) != 0, 1);                                       \
    ASSERT(x);                                                                 \
  }
#else
#define restrict
#define onchip
#define NASSERT ASSERT
#endif
#if defined(COMPILER_ADSP_BLACKFIN)
#define NASSERT_ALIGN(addr, align) __builtin_aligned(addr, align)
#else
#define NASSERT_ALIGN(addr, align) NASSERT(((uintptr_t)(addr)) % (align) == 0)
#endif
#define NASSERT_ALIGN2(addr) NASSERT_ALIGN(addr, 2)
#define NASSERT_ALIGN4(addr) NASSERT_ALIGN(addr, 4)
#define NASSERT_ALIGN8(addr) NASSERT_ALIGN(addr, 8)
#define NASSERT_ALIGN16(addr) NASSERT_ALIGN(addr, 16)
#define NASSERT_ALIGN32(addr) NASSERT_ALIGN(addr, 32)
#define NASSERT_ALIGN64(addr) NASSERT_ALIGN(addr, 64)
#define NASSERT_ALIGN128(addr) NASSERT_ALIGN(addr, 128)
/* ----------------------------------------------------------
             Common types
 ----------------------------------------------------------*/
#if defined(COMPILER_GNU) | defined(COMPILER_GNUARM) | defined(COMPILER_XTENSA)
/*
  typedef signed char   int8_t;
  typedef unsigned char uint8_t;
*/
#include <inttypes.h>
#elif defined(COMPILER_C64)
#include <stdint.h>
#elif defined(COMPILER_C55)
#include <stdint.h>
typedef signed char int8_t;
typedef unsigned char uint8_t;
#elif defined(COMPILER_ADSP_BLACKFIN)
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned long uint32_t;
typedef unsigned short uint16_t;
typedef long int32_t;
typedef short int16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef uint32_t uintptr_t;
#else
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned long uint32_t;
typedef unsigned short uint16_t;
typedef long int32_t;
typedef short int16_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#endif

#if defined(COMPILER_CEARM9E)
typedef uint32_t uintptr_t;
#endif

#if defined(COMPILER_ARM)
typedef uint32_t uintptr_t;
#endif

typedef int16_t float16_t;
typedef float float32_t;
typedef double float64_t;
typedef int16_t fract16;
typedef int32_t fract32;

typedef union tag_complex_fract16 {
  struct {
    int16_t re, im;
  } s;
  uint32_t a; /* just for 32-bit alignment */
} complex_fract16;

typedef union tag_complex_fract32 {
  struct {
    int32_t re, im;
  } s;
  uint64_t a; /* just for 64-bit alignment */
} complex_fract32;

#if defined(COMPILER_MSVC)
#if 0
/* Note: Visual Studio does not support C99 compatible complex types yet */
typedef union tag_complex_float {
  struct {
    float32_t re, im;
  } s;
  uint64_t a; /* just for 64-bit alignment */
} complex_float;
typedef union tag_complex_double {
  struct {
    float64_t re, im;
  } s;
  uint64_t a[2]; /* only 64-bit alignment under Visual Studio :(( */
} complex_double;

inline_ float32_t crealf(complex_float x) { return x.s.re; }
inline_ float32_t cimagf(complex_float x) { return x.s.im; }
inline_ float64_t creal(complex_double x) { return x.s.re; }
inline_ float64_t cimag(complex_double x) { return x.s.im; }
#else
#include <complex.h>
#define complex_float _Fcomplex
#define complex_double _Dcomplex
#endif

#else
/* C99 compatible type */
#include <complex.h>
#define complex_float __complex__ float
#define complex_double __complex__ double
#endif

/* complex half-precision datatype */
typedef union tag_complex_float16 {
  struct {
    float16_t re, im;
  } s;
  uint32_t a; /* just for 32-bit alignment */
} complex_float16;

inline_ float16_t crealh(complex_float16 x) { return x.s.re; }
inline_ float16_t cimagh(complex_float16 x) { return x.s.im; }
/*    union data type for writing float32_t/float64_t constants in a bitexact
 * form */
union ufloat32uint32 {
  uint32_t u;
  float32_t f;
};
union ufloat64uint64 {
  uint64_t u;
  float64_t f;
};
union ufloat16uint16 {
  uint16_t u;
  float16_t f;
};

#if defined(__RENAMING__)
#include "__renaming__.h"
#endif

#endif /* __DTYPE_H__ */
