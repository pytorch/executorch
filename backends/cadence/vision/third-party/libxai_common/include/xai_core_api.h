/*
 * Copyright (c) 2013-2018 Tensilica Inc. ALL RIGHTS RESERVED.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __XAI_CORE_API_H__
#define __XAI_CORE_API_H__

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

#include "xai_config_api.h"
#include "xai_tile_manager.h"

/* library information */
// _XAI_API_ is defined in glow/externalbackends/Xtensa/Backends/libxai/libxai.h and xtensa-mlir-dialect/include/xtensa/Conversion/xaicnn.h
// They dont use _XAI_API_ from xaicnn/libxai/include/xai_config_api.h and hence they dont get _XAI_API_VAR_
// defining _XAI_API_VAR_ for those cases.

#ifndef _XAI_API_VAR_
#define _XAI_API_VAR_  _XAI_API_
#endif

_XAI_API_VAR_ char XAI_BUILD_CONFIGURATION[];
_XAI_API_VAR_ char XAI_BUILD_TOOLS_VERSION[];
_XAI_API_VAR_ char XAI_BUILD_CORE_ID[];
_XAI_API_VAR_ char XAI_BUILD_ERROR_LEVEL[];
_XAI_API_VAR_ char XAI_BUILD_FEATURES_STR[];

/* Math constants */

#define XAI_PI    3.14159265358979323846
#define XAI_PI_F  3.14159265358979323846f

/* IVP library data types */

typedef int32_t    XAI_ERR_TYPE;
typedef uint8_t    xai_bool;

typedef int16_t    XAI_Q0_15;
typedef int16_t    XAI_Q5_10;
typedef int16_t    XAI_Q6_9;
typedef int16_t    XAI_Q7_8;
typedef int16_t    XAI_Q8_7;
typedef int16_t    XAI_Q12_3;
typedef int16_t    XAI_Q13_2;

typedef int32_t    XAI_Q0_31;
typedef int32_t    XAI_Q1_30;
typedef int32_t    XAI_Q12_19;
typedef int32_t    XAI_Q13_18;
typedef int32_t    XAI_Q15_16;
typedef int32_t    XAI_Q16_15;
typedef int32_t    XAI_Q22_9;
typedef int32_t    XAI_Q28_3;

typedef XAI_Q0_15  XAI_Q15;
typedef uint16_t   XAI_Q0_16;


typedef struct
{
  int16_t x;
  int16_t y;
} xai_point;

typedef struct
{
  int32_t x;
  int32_t y;
} xai_point32;

typedef struct
{
  XAI_Q16_15 x;
  XAI_Q16_15 y;
} xai_point_fpt;

typedef struct
{
  float x;
  float y;
} xai_point_f;

typedef struct
{
  int32_t width;
  int32_t height;
} xai_size;

typedef struct
{
  float a11;
  float a12;
  float a21;
  float a22;
  float xt;
  float yt;
} xai_affine;

typedef struct
{
  XAI_Q13_18 a11;
  XAI_Q13_18 a12;
  XAI_Q13_18 a21;
  XAI_Q13_18 a22;
  XAI_Q13_18 xt;
  XAI_Q13_18 yt;
} xai_affine_fpt;

typedef struct
{
  float a11;
  float a12;
  float a13;
  float a21;
  float a22;
  float a23;
  float a31;
  float a32;
  float a33;
} xai_perspective;

typedef struct
{
  XAI_Q13_18 a11;
  XAI_Q13_18 a12;
  XAI_Q13_18 a13;
  XAI_Q13_18 a21;
  XAI_Q13_18 a22;
  XAI_Q13_18 a23;
  XAI_Q13_18 a31;
  XAI_Q13_18 a32;
  XAI_Q13_18 a33;
} xai_perspective_fpt;

typedef struct
{
  int16_t  x;
  int16_t  y;
  uint16_t width;
  uint16_t height;
} xai_rect;

typedef struct
{
  int16_t   x;
  int16_t   y;
  uint16_t  width;
  uint16_t  height;
  XAI_Q5_10 angle;
} xai_rotated_rect;

typedef struct
{
  float x;
  float y;
  float width;
  float height;
  float angle;
} xai_rotated_rect_f;

typedef struct
{
  int32_t M00;
  int64_t M10;
  int64_t M01;
  int64_t M11;
  int64_t M20;
  int64_t M02;
} xai_moments;

typedef struct
{
  XAI_Q13_18 rho;
  XAI_Q13_18 theta;
} xai_line_polar_fpt;

typedef struct
{
  uint32_t size;      // number of pyramid levels
  float    scale;
  xai_tile2D **levels;   // array of pyramid levels
} xai_pyramid, *xai_pPyramid;
#define XAI_HAS_PYRAMID  1


/* Error codes */

#define XAI_ERR_OK                 0  // no error
#define XAI_ERR_IALIGNMENT         1  // input alignment requirements are not satisfied
#define XAI_ERR_OALIGNMENT         2  // output alignment requirements are not satisfied
#define XAI_ERR_MALIGNMENT         3  // same modulo alignment requirement is not satisfied
#define XAI_ERR_BADARG             4  // arguments are somehow invalid
#define XAI_ERR_MEMLOCAL           5  // tile is not placed in local memory
#define XAI_ERR_INPLACE            6  // inplace operation is not supported
#define XAI_ERR_EDGE               7  // edge extension size is too small
#define XAI_ERR_DATASIZE           8  // input/output tile size is too small or too big or otherwise inconsistent
#define XAI_ERR_TMPSIZE            9  // temporary tile size is too small or otherwise inconsistent
#define XAI_ERR_KSIZE              10 // filer kernel size is not supported
#define XAI_ERR_NORM               11 // invalid normalization divisor or shift value
#define XAI_ERR_COORD              12 // invalid coordinates
#define XAI_ERR_BADTRANSFORM       13 // the transform is singular or otherwise invalid
#define XAI_ERR_NULLARG            14 // one of required arguments is null
#define XAI_ERR_THRESH_INVALID     15 // threshold value is somehow invalid
#define XAI_ERR_SCALE              16 // provided scale factor is not supported
#define XAI_ERR_OVERFLOW           17 // tile size can lead to sum overflow
#define XAI_ERR_NOTIMPLEMENTED     18 // the requested functionality is absent in current version
#define XAI_ERR_CHANNEL_INVALID    19 // invalid channel number
#define XAI_ERR_DATATYPE           20 // argument has invalid data type
#define XAI_ERR_NO_VARIANT         21 // No suitable variant found for the function
#define XAI_ERR_PTR_NULL           22 // Pointer is NULL
#define XAI_ERR_CUSTOMACC_PREPARE  23 // fails to prepare the custom acc hardware
#define XAI_ERR_CUSTOMACC_EXECUTE  24 // fails to execute ops on the custom acc hardware
#define XAI_ERR_CUSTOMACC_REMOVE   25 // fails to remove a network for the custom acc hardware
#define XAI_ERR_LAST               25

/* non-fatal errors */

#define XAI_ERR_POOR_DECOMPOSITION  1024 // computed transform decomposition can produce visual artifacts
#define XAI_ERR_OUTOFTILE           1025 // arguments or results are out of tile
#define XAI_ERR_OBJECTLOST          1026 // tracked object is lost
#define XAI_ERR_RANSAC_NOTFOUND     1027 // there is no found appropriate model for RANSAC
#define XAI_ERR_REPLAY              1028 // function has to be called again for completion


/* helper macro */

#ifdef XCHAL_IVPN_SIMD_WIDTH
#  define XAI_SIMD_WIDTH  XCHAL_IVPN_SIMD_WIDTH
#else
#  define XAI_SIMD_WIDTH  32
#endif

#define XAI_SIZE_AREA(sz)              ((size_t) sz.width * sz.height)
#define XAI_ALIGN_VAL(val, pow2)       (((val) + ((pow2) - 1)) & ~((pow2) - 1))
#define XAI_ALIGN_VALN(val)            XAI_ALIGN_VAL(val, XAI_SIMD_WIDTH)

#define XAI_PTR_TO_ADDR(ptr)           ((uintptr_t) (ptr))
#define XAI_ALIGN_PTR(ptr, alignment)  ((void *) XAI_ALIGN_VAL(XAI_PTR_TO_ADDR((ptr)), (alignment)))

/* temporary space requirement for xaiSort */
#if XCHAL_HAVE_GRIVPEP_HISTOGRAM || XCHAL_HAVE_VISION_HISTOGRAM
#   define XAI_SORT_TMP_SIZE  0                                       // use vector registers only
#elif XCHAL_HAVE_VISION
#   define XAI_SORT_TMP_SIZE  (XAI_SIMD_WIDTH * 256 + XAI_SIMD_WIDTH)   // SIMD_WIDTH histograms by 256 bins + 32 for pointer alignment inside optimized function
#else
#   define XAI_SORT_TMP_SIZE  (2 * 256 + XAI_SIMD_WIDTH)               // 3 histograms by 256 bins + 32 for pointer alignment inside optimized function
#endif


/* error code to text conversion */
_XAI_API_ const char* xaiErrStr(XAI_ERR_TYPE code);
#endif
