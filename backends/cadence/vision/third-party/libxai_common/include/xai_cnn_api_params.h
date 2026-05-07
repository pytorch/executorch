/*
 * Copyright (c) 2018 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
 * These coded instructions, statements, and computer programs are the
 * copyrighted works and confidential proprietary information of
 * Cadence Design Systems Inc.  They may be adapted and modified by bona fide
 * purchasers for internal use, but neither the original nor any adapted
 * or modified version may be disclosed or distributed to third parties
 * in any manner, medium, or form, in whole or in part, without the prior
 * written consent of Cadence Design Systems Inc.  This software and its
 * derivatives are to be executed solely on products incorporating a Cadence
 * Design Systems processor.
 */

#ifndef __XAI_CNN_API_PARAMS_H__
#define __XAI_CNN_API_PARAMS_H__

#include "xai_config_api.h"
#include "xai_core_api.h"
#include "xai_tile_manager.h"
#include <math.h>
#include <stdbool.h>

#define TFL_QUANTIZATION_MODE_BIT_EXACT    1
#define TFL_QUANTIZATION_MODE_APPROXIMATE  2
#define XNNC_QUANTIZATION_MODE             3
#define TFL_USE_ACT_TIE                    4

#ifndef FLT_MIN
#define FLT_MIN  (1.175494351e-38F)
#endif

#ifndef FLT_MAX
#define FLT_MAX  (3.402823466e+38F)
#endif


#if defined(__clang__) && (defined(GLOW_BUILD) || defined(GLOW_WITH_XTENSA))

#ifdef XCHAL_HAVE_VISION_HP_VFPU
#undef XCHAL_HAVE_VISION_HP_VFPU
#define XCHAL_HAVE_VISION_HP_VFPU  1
#endif

#ifdef XCHAL_IVPN_SIMD_WIDTH
#if (XCHAL_IVPN_SIMD_WIDTH == 64)
#define XCHAL_HAVE_CONNX_B_HP_VFPU  1
#define XCHAL_HAVE_VISION_SP_VFPU   1
#define XCHAL_HAVE_BBENEP_SP_VFPU   1
#endif
#endif

#include <math.h>

#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
# undef ENABLE_F16_PRECISION
# define ENABLE_F16_PRECISION  1
#endif

#ifdef BIT_EXACT_FP16_REF
#  undef BIT_EXACT_FP16_REF
#endif

#ifdef BIT_EXACT_FP32_REF
#  undef BIT_EXACT_FP32_REF
#endif

#include "fp16.h"
#include <cstdint>
#include "shared/Common/Float16.h"
#undef xb_f16
typedef shared::float16  xb_f16;

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))

#include <math.h>
#undef ENABLE_F32_PRECISION
#define ENABLE_F32_PRECISION  1

#ifdef BIT_EXACT_FP32_REF
#  undef BIT_EXACT_FP32_REF
#endif
#endif // #if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1))

// MLIR builds cannot use the contents of these include files, but they
// currently do not need the symbols defined in them.
#elif !defined(MLIR_BUILD)
#ifndef XAI_REF_ONLY_COMPILATION
#include <xtensa/tie/xt_misc.h>
#if (XCHAL_HAVE_HIFI1 || XCHAL_HAVE_HIFI3Z || XCHAL_HAVE_HIFI4 || XCHAL_HAVE_HIFI5)
#include <xtensa/tie/xt_hifi2.h>
#else
#include <xtensa/tie/xt_ivpn.h>
#endif
#endif
#if (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION) && !defined(GENERIC_XTENSA_BUILD))
typedef _Float16  xb_f16;
#elif defined(GENERIC_BUILD)
typedef float xb_f16;
#endif
#elif defined(MLIR_BUILD) && defined(XAI_REF_ONLY_COMPILATION)
typedef float xb_f16;
#endif // #if defined(__clang__) && (defined(GLOW_BUILD) || defined(GLOW_WITH_XTENSA))

#if defined (BIT_EXACT_FP16_REF)
#undef XAI_F16_half
#define XAI_F16_half  IVP_CVTF16F32(0.5f)
#else
#undef XAI_F16_half
#define XAI_F16_half  (xb_f16) (0.5f)
#endif

#define XAI_F16_MIN_FLT  (float) (-65504.0f)
#define XAI_F16_MAX_FLT  (float) (65504.0f)
#define XAI_F32_MIN_FLT  (float) (-FLT_MAX)
#define XAI_F32_MAX_FLT  (float) (FLT_MAX)

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_F16_MIN         (xb_f16) (-65504.0f)
#define XAI_F16_MAX         (xb_f16) (65504.0f)
#define XAI_F16_MIN_VECN    (xb_vecNxf16) (-65504.0f)
#define XAI_F16_MAX_VECN    (xb_vecNxf16) (65504.0f)
#define XAI_F16_MIN_VECN32  (xb_vecN_2xf32) (-65504.0f)
#define XAI_F16_MAX_VECN32  (xb_vecN_2xf32) (65504.0f)
#define XAI_F16_POS_MIN     (xb_f16) (6.10352e-5F)
#endif

/***************************************************************************************/
/* log2 function is not defined in Visual Studio 2012 but available in higher versions */
/* _MSC_VER version number check to be performed for visual studio version             */
/* If _MSC_VER <= (Visual Studio 2012) version log2 function  is enabled               */
/* Visual Studio Version Information :                                                 */
/* MSVC++ 14.0 _MSC_VER == 1900 (Visual Studio 2015)                                   */
/* MSVC++ 12.0 _MSC_VER == 1800 (Visual Studio 2013)                                   */
/* MSVC++ 11.0 _MSC_VER == 1700 (Visual Studio 2012)                                   */
/* MSVC++ 10.0 _MSC_VER == 1600 (Visual Studio 2010)                                   */
/* MSVC++ 9.0  _MSC_VER == 1500 (Visual Studio 2008)                                   */
/* MSVC++ 8.0  _MSC_VER == 1400 (Visual Studio 2005)                                   */
/***************************************************************************************/

#if defined(_MSC_VER)
#if _MSC_VER <= 1700
#include "math.h"
static _XAI_INLINE_ double log2(double number)
{
  /* Calculates log2 of number.  */
  return(log(number) / log(2.0));
}
#endif
#endif

#define CNN_CONV_FLAG_RELU                              (1 << 0)
#define CNN_CONV_FLAG_LEFTEDGE                          (1 << 1)
#define CNN_CONV_FLAG_TOPEDGE                           (1 << 2)
#define CNN_CONV_FLAG_INPUT                             (1 << 3)
#define CNN_CONV_FLAG_OUTPUT                            (1 << 4)

#define CNN_POOLING_TOPEDGE_FLAG                        (1 << 1)
#define CNN_POOLING_LEFTEDGE_FLAG                       (1 << 0)

#define CNN_NORMALIZE_ALONG_WIDTH                       (1 << 0)
#define CNN_NORMALIZE_ALONG_HEIGHT                      (1 << 1)
#define CNN_NORMALIZE_ALONG_DEPTH                       (1 << 2)
#define CNN_NORMALIZE_ALONG_BATCH                       (1 << 3)
#define CNN_NORMALIZE_ALONG_WIDTH_AND_HEIGHT            (CNN_NORMALIZE_ALONG_WIDTH | CNN_NORMALIZE_ALONG_HEIGHT)
#define CNN_NORMALIZE_ALONG_WIDTH_AND_HEIGHT_AND_DEPTH  (CNN_NORMALIZE_ALONG_WIDTH | CNN_NORMALIZE_ALONG_HEIGHT | CNN_NORMALIZE_ALONG_DEPTH)
#define CNN_NORMALIZE_CHANNEL_SHARE_FLAG                (1 << 0)

#define CNN_GLOBAL_POOL_INTERMEDIATE_TILE               0
#define CNN_GLOBAL_POOL_FIRST_TILE                      1
#define CNN_GLOBAL_POOL_LAST_TILE                       2
#define CNN_GLOBAL_POOL_FIRST_AND_LAST_TILE             3

#define CNN_NORMALIZE_INTERMEDIATE_TILE                 0
#define CNN_NORMALIZE_FIRST_TILE                        1
#define CNN_NORMALIZE_LAST_TILE                         2
#define CNN_NORMALIZE_FIRST_AND_LAST_TILE               3
#define CNN_EXP_LUT_PARTITION                           3

typedef struct
{
  float   widthScale;
  float   heightScale;
  float   xshift;
  float   yshift;
  int8_t  alignCorners;
  int8_t  halfPixelCenters;
  int32_t zeroPtInput;
  int32_t zeroPtOutput;
  int32_t outMultiplier;
  int32_t outShift;
  int32_t widthFrame;
  int32_t heightFrame;
  int8_t  quantization_mode;
} xai_cnn_resizeA3D_params;

#define XAI_CNN_RESIZE3D_GET_WIDTHSCALE(x)                  ((x)->widthScale)
#define XAI_CNN_RESIZE3D_GET_HEIGHTSCALE(x)                 ((x)->heightScale)
#define XAI_CNN_RESIZE3D_GET_XSHIFT(x)                      ((x)->xshift)
#define XAI_CNN_RESIZE3D_GET_YSHIFT(x)                      ((x)->yshift)
#define XAI_CNN_RESIZE3D_GET_FLAG_ALIGN_CORNERS(x)          ((x)->alignCorners)
#define XAI_CNN_RESIZE3D_GET_FLAG_HALF_PIXEL_CENTERS(x)     ((x)->halfPixelCenters)
#define XAI_CNN_RESIZE3D_GET_ZERO_POINT_INPUT(x)            ((x)->zeroPtInput)
#define XAI_CNN_RESIZE3D_GET_ZERO_POINT_OUTPUT(x)           ((x)->zeroPtOutput)
#define XAI_CNN_RESIZE3D_GET_OUT_MULTIPLIER(x)              ((x)->outMultiplier)
#define XAI_CNN_RESIZE3D_GET_OUT_SHIFT(x)                   ((x)->outShift)
#define XAI_CNN_RESIZE3D_GET_WIDTHFRAME(x)                  ((x)->widthFrame)
#define XAI_CNN_RESIZE3D_GET_HEIGHTFRAME(x)                 ((x)->heightFrame)

#define XAI_CNN_RESIZE3D_SET_WIDTHSCALE(x, v)               ((x)->widthScale = (v))
#define XAI_CNN_RESIZE3D_SET_HEIGHTSCALE(x, v)              ((x)->heightScale = (v))
#define XAI_CNN_RESIZE3D_SET_XSHIFT(x, v)                   ((x)->xshift = (v))
#define XAI_CNN_RESIZE3D_SET_YSHIFT(x, v)                   ((x)->yshift = (v))
#define XAI_CNN_RESIZE3D_SET_FLAG_ALIGN_CORNERS(x, v)       ((x)->alignCorners = v)
#define XAI_CNN_RESIZE3D_SET_FLAG_HALF_PIXEL_CENTERS(x, v)  ((x)->halfPixelCenters = v)
#define XAI_CNN_RESIZE3D_SET_ZERO_POINT_INPUT(x, v)         ((x)->zeroPtInput = (v))
#define XAI_CNN_RESIZE3D_SET_ZERO_POINT_OUTPUT(x, v)        ((x)->zeroPtOutput = (v))
#define XAI_CNN_RESIZE3D_SET_OUT_MULTIPLIER(x, v)           ((x)->outMultiplier = (v))
#define XAI_CNN_RESIZE3D_SET_OUT_SHIFT(x, v)                ((x)->outShift = (v))
#define XAI_CNN_RESIZE3D_SET_WIDTHFRAME(x, v)               ((x)->widthFrame = (v))
#define XAI_CNN_RESIZE3D_SET_HEIGHTFRAME(x, v)              ((x)->heightFrame = (v))

#define XAI_CNN_RESIZE3D_GET_QUANTIZATION_MODE(x)           ((x)->quantization_mode)
#define XAI_CNN_RESIZE3D_SET_QUANTIZATION_MODE(x, v)        ((x)->quantization_mode = (v))

typedef struct
{
  uint8_t  strideX;                // Convolution StrideX
  uint8_t  strideY;                // Convolution StrideY
  uint8_t  accumShift;             // Accumulator Shift - Shift to convert accumulator data to 16 bit
  uint16_t outputScale;            // Amount by which shifted data is scaled
  uint8_t  outputShift;            // Shift amount to convert the scaled data to 16 bit
  uint8_t  flags;
  /*
   *  --------------------------------------------------------------------------
   *  |bit 7 - 5|    bit 4     | bit 3       | bit2      | bit1       | bit0     |
   *  | unused  |FC output flag|FC input flag|topEdgeFlag|leftEdgeFlag|Relu Flag |
   *  --------------------------------------------------------------------------
   */
  uint8_t dilationX;   // dilation along kernel width
  uint8_t dilationY;   // dilation along kernel height
  int32_t reluMin;     // Minimum clamping limit when bit 0 of flags is set
  int32_t reluMax;     // Maximum clamping limit when bit 0 of flags is set
  int8_t  quantization_mode;
  int32_t input_offset;
  int32_t output_offset;
  int32_t coeff_offset;
  int32_t outputScaleTFL;
  int32_t outputShiftTFL;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  xb_f16  reluMinFlt;
  xb_f16  reluMaxFlt;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
  float reluMinFlt32;
  float reluMaxFlt32;
#endif
} xai_cnn_conv_params;

#define XAI_CNN_CONV_GET_STRIDE(x)               ((x)->strideX)
#define XAI_CNN_CONV_SET_STRIDE(x, v)            (x)->strideX = (v); (x)->strideY = (v);
#define XAI_CNN_CONV_GET_STRIDEX(x)              ((x)->strideX)
#define XAI_CNN_CONV_GET_STRIDEY(x)              ((x)->strideY)
#define XAI_CNN_CONV_SET_STRIDE_XY(x, v1, v2)    (x)->strideX = (v1); (x)->strideY = (v2);
#define XAI_CNN_CONV_SET_STRIDEX(x, v)           (x)->strideX = (v);
#define XAI_CNN_CONV_SET_STRIDEY(x, v)           (x)->strideY = (v);
#define XAI_CNN_CONV_GET_ACCUM_SHIFT(x)          ((x)->accumShift)
#define XAI_CNN_CONV_SET_ACCUM_SHIFT(x, v)       ((x)->accumShift = (v))
#define XAI_CNN_CONV_GET_OUTPUT_SCALE(x)         ((x)->outputScale)
#define XAI_CNN_CONV_SET_OUTPUT_SCALE(x, v)      ((x)->outputScale = (v))
#define XAI_CNN_CONV_GET_OUTPUT_SHIFT(x)         ((x)->outputShift)
#define XAI_CNN_CONV_SET_OUTPUT_SHIFT(x, v)      ((x)->outputShift = (v))
#define XAI_CNN_CONV_GET_FLAGS(x)                ((x)->flags)
#define XAI_CNN_CONV_SET_FLAGS(x, v)             ((x)->flags = (v))
#define XAI_CNN_CONV_GET_FLAG_RELU(x)            ((x)->flags & CNN_CONV_FLAG_RELU)
#define XAI_CNN_CONV_SET_FLAG_RELU(x)            ((x)->flags = ((x)->flags | CNN_CONV_FLAG_RELU))
#define XAI_CNN_CONV_RESET_FLAG_RELU(x)          ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_RELU))
#define XAI_CNN_CONV_GET_FLAG_LEFTEDGE(x)        ((x)->flags & CNN_CONV_FLAG_LEFTEDGE)
#define XAI_CNN_CONV_SET_FLAG_LEFTEDGE(x)        ((x)->flags = ((x)->flags | CNN_CONV_FLAG_LEFTEDGE))
#define XAI_CNN_CONV_RESET_FLAG_LEFTEDGE(x)      ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_LEFTEDGE))
#define XAI_CNN_CONV_GET_FLAG_TOPEDGE(x)         ((x)->flags & CNN_CONV_FLAG_TOPEDGE)
#define XAI_CNN_CONV_SET_FLAG_TOPEDGE(x)         ((x)->flags = ((x)->flags | CNN_CONV_FLAG_TOPEDGE))
#define XAI_CNN_CONV_RESET_FLAG_TOPEDGE(x)       ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_TOPEDGE))
#define XAI_CNN_CONV_GET_FLAG_INPUT(x)           ((x)->flags & CNN_CONV_FLAG_INPUT)
#define XAI_CNN_CONV_SET_FLAG_INPUT(x)           ((x)->flags = ((x)->flags | CNN_CONV_FLAG_INPUT))
#define XAI_CNN_CONV_RESET_FLAG_INPUT(x)         ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_INPUT))
#define XAI_CNN_CONV_GET_FLAG_OUTPUT(x)          ((x)->flags & CNN_CONV_FLAG_OUTPUT)
#define XAI_CNN_CONV_SET_FLAG_OUTPUT(x)          ((x)->flags = ((x)->flags | CNN_CONV_FLAG_OUTPUT))
#define XAI_CNN_CONV_RESET_FLAG_OUTPUT(x)        ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_OUTPUT))
#define XAI_CNN_CONV_GET_DILATION(x)             ((x)->dilationX)
#define XAI_CNN_CONV_SET_DILATION(x, v)          (x)->dilationX = (v); (x)->dilationY = (v);
#define XAI_CNN_CONV_GET_DILATIONX(x)            ((x)->dilationX)
#define XAI_CNN_CONV_SET_DILATIONX(x, v)         ((x)->dilationX = (v))
#define XAI_CNN_CONV_GET_DILATIONY(x)            ((x)->dilationY)
#define XAI_CNN_CONV_SET_DILATIONY(x, v)         ((x)->dilationY = (v))
#define XAI_CNN_CONV_SET_DILATION_XY(x, v1, v2)  (x)->dilationX = (v1); (x)->dilationY = (v2);
#define XAI_CNN_CONV_GET_RELU_MIN(x)             ((x)->reluMin)
#define XAI_CNN_CONV_SET_RELU_MIN(x, v)          ((x)->reluMin = (v))
#define XAI_CNN_CONV_GET_RELU_MAX(x)             ((x)->reluMax)
#define XAI_CNN_CONV_SET_RELU_MAX(x, v)          ((x)->reluMax = (v))
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_CONV_GET_RELU_MIN_FLT(x)         ((x)->reluMinFlt)
#define XAI_CNN_CONV_SET_RELU_MIN_FLT(x, v)      ((x)->reluMinFlt = (v))
#define XAI_CNN_CONV_GET_RELU_MAX_FLT(x)         ((x)->reluMaxFlt)
#define XAI_CNN_CONV_SET_RELU_MAX_FLT(x, v)      ((x)->reluMaxFlt = (v))
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
#define XAI_CNN_CONV_GET_RELU_MIN_FLT32(x)        ((x)->reluMinFlt32)
#define XAI_CNN_CONV_SET_RELU_MIN_FLT32(x, v)     ((x)->reluMinFlt32 = (v))
#define XAI_CNN_CONV_GET_RELU_MAX_FLT32(x)        ((x)->reluMaxFlt32)
#define XAI_CNN_CONV_SET_RELU_MAX_FLT32(x, v)     ((x)->reluMaxFlt32 = (v))
#endif
#define XAI_CNN_CONV_GET_QUANTIZATION_MODE(x)     ((x)->quantization_mode)
#define XAI_CNN_CONV_SET_QUANTIZATION_MODE(x, v)  ((x)->quantization_mode = (v))
#define XAI_CNN_CONV_GET_INPUT_OFFSET(x)          ((x)->input_offset)
#define XAI_CNN_CONV_SET_INPUT_OFFSET(x, v)       ((x)->input_offset = (v))
#define XAI_CNN_CONV_GET_OUTPUT_OFFSET(x)         ((x)->output_offset)
#define XAI_CNN_CONV_SET_OUTPUT_OFFSET(x, v)      ((x)->output_offset = (v))
#define XAI_CNN_CONV_GET_COEFF_OFFSET(x)          ((x)->coeff_offset)
#define XAI_CNN_CONV_SET_COEFF_OFFSET(x, v)       ((x)->coeff_offset = (v))
#define XAI_CNN_CONV_GET_OUTPUT_SCALE_TFL(x)      ((x)->outputScaleTFL)
#define XAI_CNN_CONV_SET_OUTPUT_SCALE_TFL(x, v)   ((x)->outputScaleTFL = (v))
#define XAI_CNN_CONV_GET_OUTPUT_SHIFT_TFL(x)      ((x)->outputShiftTFL)
#define XAI_CNN_CONV_SET_OUTPUT_SHIFT_TFL(x, v)   ((x)->outputShiftTFL = (v))

typedef struct
{
  uint8_t  strideX;                // Convolution StrideX
  uint8_t  strideY;                // Convolution StrideY
  uint8_t  accumShift;             // Accumulator Shift - Shift to convert accumulator data to 16 bit
  uint16_t outputScale;            // Amount by which shifted data is scaled
  uint8_t  outputShift;            // Shift amount to convert the scaled data to 16 bit
  uint8_t  flags;
  /*
   *  --------------------------------------------------------------------------
   *  |bit 7 - 5|    bit 4     | bit 3       | bit2      | bit1       | bit0     |
   *  | unused  |FC output flag|FC input flag|topEdgeFlag|leftEdgeFlag|Relu Flag |
   *  --------------------------------------------------------------------------
   */
  uint8_t dilationX;              // dilation along kernel width
  uint8_t dilationY;              // dilation along kernel height
  uint8_t depthMultiplier;        // factor by which output depth size varies from input depth size
  int32_t reluMin;                // Minimum clamping limit when bit 0 of flags is set
  int32_t reluMax;                // Maximum clamping limit when bit 0 of flags is set
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  xb_f16  reluMinFlt;
  xb_f16  reluMaxFlt;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
  float   reluMinFlt32;
  float   reluMaxFlt32;
#endif
  int8_t  quantization_mode;
  int32_t input_offset;
  int32_t output_offset;
} xai_cnn_depthwiseDilatedConv_params;

#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(x)               ((x)->strideX)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_STRIDE(x, v)            (x)->strideX = (v); (x)->strideY = (v)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(x)              ((x)->strideX)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(x)              ((x)->strideY)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_STRIDE_XY(x, v1, v2)    (x)->strideX = (v1); (x)->strideY = (v2)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_STRIDEX(x, v)           (x)->strideX = (v);
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_STRIDEY(x, v)           (x)->strideY = (v);
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_ACCUM_SHIFT(x)          ((x)->accumShift)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_ACCUM_SHIFT(x, v)       ((x)->accumShift = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_OUTPUT_SCALE(x)         ((x)->outputScale)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_OUTPUT_SCALE(x, v)      ((x)->outputScale = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_OUTPUT_SHIFT(x)         ((x)->outputShift)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_OUTPUT_SHIFT(x, v)      ((x)->outputShift = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAGS(x)                ((x)->flags)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_FLAGS(x, v)             ((x)->flags = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_RELU(x)            ((x)->flags & CNN_CONV_FLAG_RELU)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_FLAG_RELU(x)            ((x)->flags = ((x)->flags | CNN_CONV_FLAG_RELU))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_RESET_FLAG_RELU(x)          ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_RELU))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_LEFTEDGE(x)        ((x)->flags & CNN_CONV_FLAG_LEFTEDGE)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_FLAG_LEFTEDGE(x)        ((x)->flags = ((x)->flags | CNN_CONV_FLAG_LEFTEDGE))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_RESET_FLAG_LEFTEDGE(x)      ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_LEFTEDGE))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_TOPEDGE(x)         ((x)->flags & CNN_CONV_FLAG_TOPEDGE)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_FLAG_TOPEDGE(x)         ((x)->flags = ((x)->flags | CNN_CONV_FLAG_TOPEDGE))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_RESET_FLAG_TOPEDGE(x)       ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_TOPEDGE))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_INPUT(x)           ((x)->flags & CNN_CONV_FLAG_INPUT)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_FLAG_INPUT(x)           ((x)->flags = ((x)->flags | CNN_CONV_FLAG_INPUT))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_RESET_FLAG_INPUT(x)         ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_INPUT))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_OUTPUT(x)          ((x)->flags & CNN_CONV_FLAG_OUTPUT)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_FLAG_OUTPUT(x)          ((x)->flags = ((x)->flags | CNN_CONV_FLAG_OUTPUT))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_RESET_FLAG_OUTPUT(x)        ((x)->flags = ((x)->flags & ~CNN_CONV_FLAG_OUTPUT))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(x)             ((x)->dilationX)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_DILATION(x, v)          (x)->dilationX = (v); (x)->dilationY = (v)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONX(x)            ((x)->dilationX)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_DILATIONX(x, v)         ((x)->dilationX = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONY(x)            ((x)->dilationY)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_DILATIONY(x, v)         ((x)->dilationY = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_DILATION_XY(x, v1, v2)  (x)->dilationX = (v1); (x)->dilationY = (v2)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DEPTH_MULTIPLIER(x)     ((x)->depthMultiplier)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_DEPTH_MULTIPLIER(x, v)  ((x)->depthMultiplier = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(x)             ((x)->reluMin)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_RELU_MIN(x, v)          ((x)->reluMin = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(x)             ((x)->reluMax)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_RELU_MAX(x, v)          ((x)->reluMax = (v))
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN_FLT(x)         ((x)->reluMinFlt)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_RELU_MIN_FLT(x, v)      ((x)->reluMinFlt = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX_FLT(x)         ((x)->reluMaxFlt)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_RELU_MAX_FLT(x, v)      ((x)->reluMaxFlt = (v))
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN_FLT32(x)        ((x)->reluMinFlt32)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_RELU_MIN_FLT32(x, v)     ((x)->reluMinFlt32 = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX_FLT32(x)        ((x)->reluMaxFlt32)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_RELU_MAX_FLT32(x, v)     ((x)->reluMaxFlt32 = (v))
#endif
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_QUANTIZATION_MODE(x)     ((x)->quantization_mode)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_QUANTIZATION_MODE(x, v)  ((x)->quantization_mode = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_INPUT_OFFSET(x)          ((x)->input_offset)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_INPUT_OFFSET(x, v)       ((x)->input_offset = (v))
#define XAI_CNN_DEPTHWISE_DILATED_CONV_GET_OUTPUT_OFFSET(x)         ((x)->output_offset)
#define XAI_CNN_DEPTHWISE_DILATED_CONV_SET_OUTPUT_OFFSET(x, v)      ((x)->output_offset = (v))

typedef struct
{
  uint8_t kernelWidth;             // Normalization window width
  uint8_t kernelHeight;            // Normalization window height
  int16_t sigmaScale;              // Factor used to scale the sum of squares of data under the normalization window
  uint8_t sigmaScaleShift;         // Shift to map the scaled sum of squares to LUT index
  uint8_t outputShift;             // Output shift
} xai_cnn_lrn_spatial_params;

typedef struct
{
  uint8_t kernelDepth;             // Normalization window depth
  int16_t sigmaScale;              // Factor used to scale the sum of squares of data under the normalization window
  uint8_t sigmaScaleShift;         // Shift to map the scaled sum of squares to LUT index
  uint8_t outputShift;             // Output shift
} xai_cnn_lrn_depth_params;

#define XAI_CNN_LRN_GET_KERNELWIDTH(x)         ((x)->kernelWidth)
#define XAI_CNN_LRN_SET_KERNELWIDTH(x, v)      ((x)->kernelWidth = (v))
#define XAI_CNN_LRN_GET_KERNELHEIGHT(x)        ((x)->kernelHeight)
#define XAI_CNN_LRN_SET_KERNELHEIGHT(x, v)     ((x)->kernelHeight = (v))
#define XAI_CNN_LRN_GET_KERNELDEPTH(x)         ((x)->kernelDepth)
#define XAI_CNN_LRN_SET_KERNELDEPTH(x, v)      ((x)->kernelDepth = (v))
#define XAI_CNN_LRN_GET_SIGMASCALE(x)          ((x)->sigmaScale)
#define XAI_CNN_LRN_SET_SIGMASCALE(x, v)       ((x)->sigmaScale = (v))
#define XAI_CNN_LRN_GET_SIGMASCALESHIFT(x)     ((x)->sigmaScaleShift)
#define XAI_CNN_LRN_SET_SIGMASCALESHIFT(x, v)  ((x)->sigmaScaleShift = (v))
#define XAI_CNN_LRN_GET_OUTPUTSHIFT(x)         ((x)->outputShift)
#define XAI_CNN_LRN_SET_OUTPUTSHIFT(x, v)      ((x)->outputShift = (v))

typedef struct
{
  int16_t kernelWidth;
  int16_t kernelHeight;
  uint8_t strideX;     // The number of points by which the pooling window
                       // is shifted along X direction.
  uint8_t strideY;     // The number of points by which the pooling window
                       // is shifted along Y direction.
  uint8_t edgeFlag;    // edgeFlag is applicable only for pooling with even kernel sizes. Least significant bit(LSB)
                       // of the flag represents whether minimum left edge size required for pooling should be
                       // greater than the minimum right edge size required. The bit adjacent to LSB decides whether
                       // minimum top edge size required should be greater than minimum bottom edge size.
  int16_t outputScale; // Normalizer value to be multiplied with sum of elements under the pooling window
  uint8_t outputShift; // Shift to be applied on the normalized sum to obtain the average
  int32_t fixUpInit;   // the fixUp term that is used to incorporte Zero Points
  uint8_t enableRelu;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  xb_f16  reluMinFlt;
  xb_f16  reluMaxFlt;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
  float   reluMinFlt32;
  float   reluMaxFlt32;
#endif
  int8_t  quantization_mode;
  int32_t reluMin;
  int32_t reluMax;
} xai_cnn_pooling_params;

#define XAI_CNN_POOLING_GET_KERNELWIDTH(x)        ((x)->kernelWidth)
#define XAI_CNN_POOLING_SET_KERNELWIDTH(x, v)     ((x)->kernelWidth = (v))
#define XAI_CNN_POOLING_GET_KERNELHEIGHT(x)       ((x)->kernelHeight)
#define XAI_CNN_POOLING_SET_KERNELHEIGHT(x, v)    ((x)->kernelHeight = (v))
#define XAI_CNN_POOLING_GET_STRIDE(x)             ((x)->strideX)
#define XAI_CNN_POOLING_SET_STRIDE(x, v)          (x)->strideX = (v); (x)->strideY = (v);
#define XAI_CNN_POOLING_GET_STRIDEX(x)            ((x)->strideX)
#define XAI_CNN_POOLING_GET_STRIDEY(x)            ((x)->strideY)
#define XAI_CNN_POOLING_SET_STRIDE_XY(x, v1, v2)  (x)->strideX = (v1); (x)->strideY = (v2);
#define XAI_CNN_POOLING_SET_STRIDEX(x, v)         (x)->strideX = (v);
#define XAI_CNN_POOLING_SET_STRIDEY(x, v)         (x)->strideY = (v);
#define XAI_CNN_POOLING_GET_TOPEDGE_FLAG(x)       ((x)->edgeFlag & CNN_POOLING_TOPEDGE_FLAG)
#define XAI_CNN_POOLING_SET_TOPEDGE_FLAG(x)       ((x)->edgeFlag = ((x)->edgeFlag | CNN_POOLING_TOPEDGE_FLAG))
#define XAI_CNN_POOLING_RESET_TOPEDGE_FLAG(x)     ((x)->edgeFlag = ((x)->edgeFlag & ~CNN_POOLING_TOPEDGE_FLAG))
#define XAI_CNN_POOLING_GET_LEFTEDGE_FLAG(x)      ((x)->edgeFlag & CNN_POOLING_LEFTEDGE_FLAG)
#define XAI_CNN_POOLING_SET_LEFTEDGE_FLAG(x)      ((x)->edgeFlag = ((x)->edgeFlag | CNN_POOLING_LEFTEDGE_FLAG))
#define XAI_CNN_POOLING_RESET_LEFTEDGE_FLAG(x)    ((x)->edgeFlag = ((x)->edgeFlag & ~CNN_POOLING_LEFTEDGE_FLAG))
#define XAI_CNN_POOLING_GET_OUTPUTSCALE(x)        ((x)->outputScale)
#define XAI_CNN_POOLING_SET_OUTPUTSCALE(x, v)     ((x)->outputScale = (v))
#define XAI_CNN_POOLING_GET_OUTPUTSHIFT(x)        ((x)->outputShift)
#define XAI_CNN_POOLING_SET_OUTPUTSHIFT(x, v)     ((x)->outputShift = (v))
#define XAI_CNN_POOLING_GET_FIXUPINIT(x)          ((x)->fixUpInit)
#define XAI_CNN_POOLING_SET_FIXUPINIT(x, v)       ((x)->fixUpInit = (v))
#define XAI_CNN_POOLING_GET_RELUFLAG(x)           ((x)->enableRelu)
#define XAI_CNN_POOLING_SET_RELUFLAG(x, v)        ((x)->enableRelu = (v))

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_POOLING_GET_RELUMINFLT(x)     ((x)->reluMinFlt)
#define XAI_CNN_POOLING_SET_RELUMINFLT(x, v)  ((x)->reluMinFlt = (v))
#define XAI_CNN_POOLING_GET_RELUMAXFLT(x)     ((x)->reluMaxFlt)
#define XAI_CNN_POOLING_SET_RELUMAXFLT(x, v)  ((x)->reluMaxFlt = (v))
#endif

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
#define XAI_CNN_POOLING_GET_RELU_MIN_FLT32(x)        ((x)->reluMinFlt32)
#define XAI_CNN_POOLING_SET_RELU_MIN_FLT32(x, v)     ((x)->reluMinFlt32 = (v))
#define XAI_CNN_POOLING_GET_RELU_MAX_FLT32(x)        ((x)->reluMaxFlt32)
#define XAI_CNN_POOLING_SET_RELU_MAX_FLT32(x, v)     ((x)->reluMaxFlt32 = (v))
#endif
#define XAI_CNN_POOLING_GET_QUANTIZATION_MODE(x)     ((x)->quantization_mode)
#define XAI_CNN_POOLING_SET_QUANTIZATION_MODE(x, v)  ((x)->quantization_mode = (v))
#define XAI_CNN_POOLING_GET_RELUMIN(x)               ((x)->reluMin)
#define XAI_CNN_POOLING_SET_RELUMIN(x, v)            ((x)->reluMin = (v))
#define XAI_CNN_POOLING_GET_RELUMAX(x)               ((x)->reluMax)
#define XAI_CNN_POOLING_SET_RELUMAX(x, v)            ((x)->reluMax = (v))

typedef struct
{
  int16_t outputScale; //Normalizer value to be multiplied with sum of elements under the pooling window
  uint8_t tileFlag;    // indicates whether the given tile is a first tile, last tile or neither of those
  uint8_t outputShift; //Shift to be applied on the normalized sum to obtain the average
  uint8_t accShift;    //accumulator shift that is applied to bring the data to S32 range
  int32_t fixUpInit;   //the fixUp term that is used to incorporte Zero Points
} xai_cnn_global_pooling_params;

#define XAI_CNN_GLOBAL_POOLING_GET_OUTPUTSCALE(x)     ((x)->outputScale)
#define XAI_CNN_GLOBAL_POOLING_SET_OUTPUTSCALE(x, v)  ((x)->outputScale = (v))
#define XAI_CNN_GLOBAL_POOLING_GET_OUTPUTSHIFT(x)     ((x)->outputShift)
#define XAI_CNN_GLOBAL_POOLING_SET_OUTPUTSHIFT(x, v)  ((x)->outputShift = (v))
#define XAI_CNN_GLOBAL_POOLING_GET_ACCSHIFT(x)        ((x)->accShift)
#define XAI_CNN_GLOBAL_POOLING_SET_ACCSHIFT(x, v)     ((x)->accShift = (v))
#define XAI_CNN_GLOBAL_POOLING_GET_TILE_FLAG(x)       ((x)->tileFlag)
#define XAI_CNN_GLOBAL_POOLING_SET_TILE_FLAG(x, v)    ((x)->tileFlag = (v))
#define XAI_CNN_GLOBAL_POOLING_GET_FIXUPINIT(x)       ((x)->fixUpInit)
#define XAI_CNN_GLOBAL_POOLING_SET_FIXUPINIT(x, v)    ((x)->fixUpInit = (v))

typedef struct
{
  uint16_t spatialScaleX;           // Multiplicative spatial scale factor to translate ROI coords from their
                                    // input scale to the scale used when pooling
                                    //Spatial scale in the X direction
  uint16_t spatialScaleY;           //Spatial scale in the Y direction
  uint16_t spatialScaleShiftX;      //Shift value to apply for spatial scale operations in the X direction
  uint16_t spatialScaleShiftY;      //Shift value to apply for spatial scale operations in the Y direction
  int32_t  pooledHeight;            //Total number of fixed output points along height dimension from ROI
  int32_t  pooledWidth;             //Total number of fixed output points along width dimension from ROI
  uint16_t oneByPooledHeightScale;  //Reciprocal of pooledHeight represented in U15 range
  uint16_t oneByPooledWidthScale;   //Reciprocal of pooledWidth represented in U15 range
  uint16_t oneByPooledHeightShift;  //Shift value to normalize after operating with oneByPooledHeightScale variable
  uint16_t oneByPooledWidthShift;   //Shift value to normalize after operating with oneByPooledWidthScale variable
} xai_cnn_roi_pooling_params;

#define XAI_CNN_ROI_POOLING_GET_SPATIAL_SCALEX(x)                 ((x)->spatialScaleX)
#define XAI_CNN_ROI_POOLING_SET_SPATIAL_SCALEX(x, v)              ((x)->spatialScaleX = (v))
#define XAI_CNN_ROI_POOLING_GET_SPATIAL_SCALEY(x)                 ((x)->spatialScaleY)
#define XAI_CNN_ROI_POOLING_SET_SPATIAL_SCALEY(x, v)              ((x)->spatialScaleY = (v))
#define XAI_CNN_ROI_POOLING_GET_SPATIAL_SCALE_SHIFTX(x)           ((x)->spatialScaleShiftX)
#define XAI_CNN_ROI_POOLING_SET_SPATIAL_SCALE_SHIFTX(x, v)        ((x)->spatialScaleShiftX = (v))
#define XAI_CNN_ROI_POOLING_GET_SPATIAL_SCALE_SHIFTY(x)           ((x)->spatialScaleShiftY)
#define XAI_CNN_ROI_POOLING_SET_SPATIAL_SCALE_SHIFTY(x, v)        ((x)->spatialScaleShiftY = (v))
#define XAI_CNN_ROI_POOLING_GET_POOLED_WIDTH(x)                   ((x)->pooledWidth)
#define XAI_CNN_ROI_POOLING_SET_POOLED_WIDTH(x, v)                ((x)->pooledWidth = (v))
#define XAI_CNN_ROI_POOLING_GET_POOLED_HEIGHT(x)                  ((x)->pooledHeight)
#define XAI_CNN_ROI_POOLING_SET_POOLED_HEIGHT(x, v)               ((x)->pooledHeight = (v))
#define XAI_CNN_ROI_POOLING_GET_ONE_BY_POOLED_WIDTH_SCALE(x)      ((x)->oneByPooledWidthScale)
#define XAI_CNN_ROI_POOLING_SET_ONE_BY_POOLED_WIDTH_SCALE(x, v)   ((x)->oneByPooledWidthScale = (v))
#define XAI_CNN_ROI_POOLING_GET_ONE_BY_POOLED_HEIGHT_SCALE(x)     ((x)->oneByPooledHeightScale)
#define XAI_CNN_ROI_POOLING_SET_ONE_BY_POOLED_HEIGHT_SCALE(x, v)  ((x)->oneByPooledHeightScale = (v))
#define XAI_CNN_ROI_POOLING_GET_ONE_BY_POOLED_WIDTH_SHIFT(x)      ((x)->oneByPooledWidthShift)
#define XAI_CNN_ROI_POOLING_SET_ONE_BY_POOLED_WIDTH_SHIFT(x, v)   ((x)->oneByPooledWidthShift = (v))
#define XAI_CNN_ROI_POOLING_GET_ONE_BY_POOLED_HEIGHT_SHIFT(x)     ((x)->oneByPooledHeightShift)
#define XAI_CNN_ROI_POOLING_SET_ONE_BY_POOLED_HEIGHT_SHIFT(x, v)  ((x)->oneByPooledHeightShift = (v))

typedef struct
{
  uint8_t outputShift;      /* No. of output bits to be right shifted. */
  uint8_t qFactorOutput;    /* No. of bits scaling applied to the reciprocal of the sum of exp(x)*/
  int16_t maxVal;           /* global max value in the 3D tile */
  int8_t  axis;             /* dimension along which softmax is applied*/
  int8_t  quantization_mode;
  int32_t diff_min;          //defines minimum difference with respect to the maximum value
  int32_t inputScale;        //significand of BetaScaleQ5.26
  int32_t inputShift;        //exponent of BetaScaleQ5.26
} xai_cnn_softmax_params;

#define XAI_CNN_SOFTMAX_GET_OUTPUTSHIFT(x)           ((x)->outputShift)
#define XAI_CNN_SOFTMAX_SET_OUTPUTSHIFT(x, v)        ((x)->outputShift = (v))
#define XAI_CNN_SOFTMAX_GET_QFACTOROUTPUT(x)         ((x)->qFactorOutput)
#define XAI_CNN_SOFTMAX_SET_QFACTOROUTPUT(x, v)      ((x)->qFactorOutput = (v))
#define XAI_CNN_SOFTMAX_GET_MAXVAL(x)                ((x)->maxVal)
#define XAI_CNN_SOFTMAX_SET_MAXVAL(x, v)             ((x)->maxVal = (v))
#define XAI_CNN_SOFTMAX_GET_AXIS(x)                  ((x)->axis)
#define XAI_CNN_SOFTMAX_SET_AXIS(x, v)               ((x)->axis = (v))
#define XAI_CNN_SOFTMAX_GET_QUANTIZATION_MODE(x)     ((x)->quantization_mode)
#define XAI_CNN_SOFTMAX_SET_QUANTIZATION_MODE(x, v)  ((x)->quantization_mode = (v))
#define XAI_CNN_SOFTMAX_PARAMS_GET_DIFF_MIN(x)       ((x)->diff_min)
#define XAI_CNN_SOFTMAX_PARAMS_SET_DIFF_MIN(x, v)    ((x)->diff_min = (v))
#define XAI_CNN_SOFTMAX_GET_INPUT_SCALE(x)           ((x)->inputScale)
#define XAI_CNN_SOFTMAX_SET_INPUT_SCALE(x, v)        ((x)->inputScale = (v))
#define XAI_CNN_SOFTMAX_GET_INPUT_SHIFT(x)           ((x)->inputShift)
#define XAI_CNN_SOFTMAX_SET_INPUT_SHIFT(x, v)        ((x)->inputShift = (v))

typedef struct
{
  int8_t  quantization_mode;
  // tfl related parameters
  int32_t inputZeroPoint;
  int32_t outputZeroPoint;
  int16_t reluishMultiplierFixedpointS16;
  int32_t reluishMultiplierExponent;
  int16_t outputMultiplierFixedpointS16;
  int32_t outputMultiplierExponent;
} xai_cnn_tfl_hardSwish_params;

#define XAI_CNN_HARDSWISH_GET_QUANTIZATION_MODE(x)                      ((x)->quantization_mode)
#define XAI_CNN_HARDSWISH_SET_QUANTIZATION_MODE(x, v)                   ((x)->quantization_mode = (v))
#define XAI_CNN_HARDSWISH_GET_INPUT_ZERO_POINT(x)                       ((x)->inputZeroPoint)
#define XAI_CNN_HARDSWISH_SET_INPUT_ZERO_POINT(x, v)                    ((x)->inputZeroPoint = (v))
#define XAI_CNN_HARDSWISH_GET_OUTPUT_ZERO_POINT(x)                      ((x)->outputZeroPoint)
#define XAI_CNN_HARDSWISH_SET_OUTPUT_ZERO_POINT(x, v)                   ((x)->outputZeroPoint = (v))
#define XAI_CNN_HARDSWISH_GET_RELUISH_MULTIPLIER_FIXED_POINT_S16(x)     ((x)->reluishMultiplierFixedpointS16)
#define XAI_CNN_HARDSWISH_SET_RELUISH_MULTIPLIER_FIXED_POINT_S16(x, v)  ((x)->reluishMultiplierFixedpointS16 = (v))
#define XAI_CNN_HARDSWISH_GET_RELUISH_MULTIPLIER_EXPONENT(x)            ((x)->reluishMultiplierExponent)
#define XAI_CNN_HARDSWISH_SET_RELUISH_MULTIPLIER_EXPONENT(x, v)         ((x)->reluishMultiplierExponent = (v))
#define XAI_CNN_HARDSWISH_GET_OUTPUT_MULTIPLIER_FIXED_POINT_S16(x)      ((x)->outputMultiplierFixedpointS16)
#define XAI_CNN_HARDSWISH_SET_OUTOUT_MULTIPLIER_FIXED_POINT_S16(x, v)   ((x)->outputMultiplierFixedpointS16 = (v))
#define XAI_CNN_HARDSWISH_GET_OUTPUT_MULTIPLIER_EXPONENT(x)             ((x)->outputMultiplierExponent)
#define XAI_CNN_HARDSWISH_SET_OUTPUT_MULTIPLIER_EXPONENT(x, v)          ((x)->outputMultiplierExponent = (v))

typedef struct
{
  int8_t  quantization_mode;
  // tfl related parameters
  int32_t inputRangeRadius;
  int32_t inputScale;
  int32_t inputShift;
  int32_t inputZeroPoint;
} xai_cnn_sigmoid_params;

#define XAI_CNN_SIGMOID_GET_QUANTIZATION_MODE(x)      ((x)->quantization_mode)
#define XAI_CNN_SIGMOID_SET_QUANTIZATION_MODE(x, v)   ((x)->quantization_mode = (v))
#define XAI_CNN_SIGMOID_GET_INPUT_RANGE_RADIUS(x)     ((x)->inputRangeRadius)
#define XAI_CNN_SIGMOID_SET_INPUT_RANGE_RADIUS(x, v)  ((x)->inputRangeRadius = (v))
#define XAI_CNN_SIGMOID_GET_INPUT_SCALE(x)            ((x)->inputScale)
#define XAI_CNN_SIGMOID_SET_INPUT_SCALE(x, v)         ((x)->inputScale = (v))
#define XAI_CNN_SIGMOID_GET_INPUT_SHIFT(x)            ((x)->inputShift)
#define XAI_CNN_SIGMOID_SET_INPUT_SHIFT(x, v)         ((x)->inputShift = (v))
#define XAI_CNN_SIGMOID_GET_INPUT_ZERO_POINT(x)       ((x)->inputZeroPoint)
#define XAI_CNN_SIGMOID_SET_INPUT_ZERO_POINT(x, v)    ((x)->inputZeroPoint = (v))

typedef struct
{
  int8_t  quantization_mode;
  // tfl related parameters
  int32_t inputRangeRadius;
  int32_t inputScale;
  int32_t inputShift;
  int32_t inputZeroPoint;
  int32_t outputZeroPoint; //Hack in Glow to keep tanh and sigmoid params different
} xai_cnn_tanh_params;

#define XAI_CNN_TANH_GET_QUANTIZATION_MODE(x)      ((x)->quantization_mode)
#define XAI_CNN_TANH_SET_QUANTIZATION_MODE(x, v)   ((x)->quantization_mode = (v))
#define XAI_CNN_TANH_GET_INPUT_RANGE_RADIUS(x)     ((x)->inputRangeRadius)
#define XAI_CNN_TANH_SET_INPUT_RANGE_RADIUS(x, v)  ((x)->inputRangeRadius = (v))
#define XAI_CNN_TANH_GET_INPUT_SCALE(x)            ((x)->inputScale)
#define XAI_CNN_TANH_SET_INPUT_SCALE(x, v)         ((x)->inputScale = (v))
#define XAI_CNN_TANH_GET_INPUT_SHIFT(x)            ((x)->inputShift)
#define XAI_CNN_TANH_SET_INPUT_SHIFT(x, v)         ((x)->inputShift = (v))
#define XAI_CNN_TANH_GET_INPUT_ZERO_POINT(x)       ((x)->inputZeroPoint)
#define XAI_CNN_TANH_SET_INPUT_ZERO_POINT(x, v)    ((x)->inputZeroPoint = (v))


typedef struct
{
  int32_t outputScaleIdentity;
  int32_t outputShiftIdentity;
  int32_t outputScaleAlpha;
  int32_t outputShiftAlpha;
  int32_t inputOffset;
  int32_t outputOffset;
  int8_t  quantization_mode;
} xai_cnn_tfl_leakyrelu_params;

#define XAI_CNN_LEAKYRELU_GET_OUTPUT_SCALE_IDENTITY(x)     ((x)->outputScaleIdentity)
#define XAI_CNN_LEAKYRELU_SET_OUTPUT_SCALE_IDENTITY(x, v)  ((x)->outputScaleIdentity = (v))
#define XAI_CNN_LEAKYRELU_GET_OUTPUT_SHIFT_IDENTITY(x)     ((x)->outputShiftIdentity)
#define XAI_CNN_LEAKYRELU_SET_OUTPUT_SHIFT_IDENTITY(x, v)  ((x)->outputShiftIdentity = (v))
#define XAI_CNN_LEAKYRELU_GET_OUTPUT_SCALE_ALPHA(x)        ((x)->outputScaleAlpha)
#define XAI_CNN_LEAKYRELU_SET_OUTPUT_SCALE_ALPHA(x, v)     ((x)->outputScaleAlpha = (v))
#define XAI_CNN_LEAKYRELU_GET_OUTPUT_SHIFT_ALPHA(x)        ((x)->outputShiftAlpha)
#define XAI_CNN_LEAKYRELU_SET_OUTPUT_SHIFT_ALPHA(x, v)     ((x)->outputShiftAlpha = (v))
#define XAI_CNN_LEAKYRELU_GET_INPUT_OFFSET(x)              ((x)->inputOffset)
#define XAI_CNN_LEAKYRELU_SET_INPUT_OFFSET(x, v)           ((x)->inputOffset = (v))
#define XAI_CNN_LEAKYRELU_GET_OUTPUT_OFFSET(x)             ((x)->outputOffset)
#define XAI_CNN_LEAKYRELU_SET_OUTPUT_OFFSET(x, v)          ((x)->outputOffset = (v))
#define XAI_CNN_LEAKYRELU_GET_QUANTIZATION_MODE(x)         ((x)->quantization_mode)
#define XAI_CNN_LEAKYRELU_SET_QUANTIZATION_MODE(x, v)      ((x)->quantization_mode = (v))

typedef struct
{
  int32_t outputScalePositive;
  int32_t outputScaleNegative;
  int32_t outputShiftPositive;
  int32_t outputShiftNegative;
  int32_t inputOffset;
  int32_t outputOffset;
  int32_t alphaOffset;
  int8_t  quantization_mode;
} xai_cnn_tfl_prelu_params;

#define XAI_CNN_PRELU_GET_OUTPUT_SCALE_POSITIVE(x)     ((x)->outputScalePositive)
#define XAI_CNN_PRELU_SET_OUTPUT_SCALE_POSITIVE(x, v)  ((x)->outputScalePositive = (v))
#define XAI_CNN_PRELU_GET_OUTPUT_SHIFT_POSITIVE(x)     ((x)->outputShiftPositive)
#define XAI_CNN_PRELU_SET_OUTPUT_SHIFT_POSITIVE(x, v)  ((x)->outputShiftPositive = (v))
#define XAI_CNN_PRELU_GET_OUTPUT_SCALE_NEGATIVE(x)     ((x)->outputScaleNegative)
#define XAI_CNN_PRELU_SET_OUTPUT_SCALE_NEGATIVE(x, v)  ((x)->outputScaleNegative = (v))
#define XAI_CNN_PRELU_GET_OUTPUT_SHIFT_NEGATIVE(x)     ((x)->outputShiftNegative)
#define XAI_CNN_PRELU_SET_OUTPUT_SHIFT_NEGATIVE(x, v)  ((x)->outputShiftNegative = (v))
#define XAI_CNN_PRELU_GET_INPUT_OFFSET(x)              ((x)->inputOffset)
#define XAI_CNN_PRELU_SET_INPUT_OFFSET(x, v)           ((x)->inputOffset = (v))
#define XAI_CNN_PRELU_GET_OUTPUT_OFFSET(x)             ((x)->outputOffset)
#define XAI_CNN_PRELU_SET_OUTPUT_OFFSET(x, v)          ((x)->outputOffset = (v))
#define XAI_CNN_PRELU_GET_ALPHA_OFFSET(x)              ((x)->alphaOffset)
#define XAI_CNN_PRELU_SET_ALPHA_OFFSET(x, v)           ((x)->alphaOffset = (v))
#define XAI_CNN_PRELU_GET_QUANTIZATION_MODE(x)         ((x)->quantization_mode)
#define XAI_CNN_PRELU_SET_QUANTIZATION_MODE(x, v)      ((x)->quantization_mode = (v))

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
typedef struct
{
  int32_t axis;                  // axis along which softmax is to be computed
  xb_f16  beta;                  // multiplication factor
} xai_cnn_softmaxA3D_F16_params;

#define XAI_CNN_SOFTMAXAF16_PARAMS_GET_AXIS(x)     ((x)->axis)
#define XAI_CNN_SOFTMAXAF16_PARAMS_GET_BETA(x)     ((x)->beta)
#define XAI_CNN_SOFTMAXAF16_PARAMS_SET_AXIS(x, v)  ((x)->axis = (v))
#define XAI_CNN_SOFTMAXAF16_PARAMS_SET_BETA(x, v)  ((x)->beta = (v))
#endif // #if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))

#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
typedef struct
{
  int32_t axis;                 // axis along which softmax is to be computed
  float   beta;                 // multiplication factor
} xai_cnn_softmaxA3D_F32_params;

#define XAI_CNN_SOFTMAXAF32_PARAMS_GET_AXIS(x)     ((x)->axis)
#define XAI_CNN_SOFTMAXAF32_PARAMS_GET_BETA(x)     ((x)->beta)
#define XAI_CNN_SOFTMAXAF32_PARAMS_SET_AXIS(x, v)  ((x)->axis = (v))
#define XAI_CNN_SOFTMAXAF32_PARAMS_SET_BETA(x, v)  ((x)->beta = (v))
#endif // #if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))

typedef struct
{
  int16_t maxVal;   /* global max value of a 3D tile */
  uint8_t tileFlag; /* tileFlag can take values 0-3.
                       0 : neither first not last tile
                       1 : first tile
                       2 : last tile
                       3 : first and last tile. */
} xai_cnn_maxval_params;

#define XAI_CNN_MAXVAL_GET_MAXVAL(x)       ((x)->maxVal)
#define XAI_CNN_MAXVAL_SET_MAXVAL(x, v)    ((x)->maxVal = (v))
#define XAI_CNN_MAXVAL_GET_TILEFLAG(x)     ((x)->tileFlag)
#define XAI_CNN_MAXVAL_SET_TILEFLAG(x, v)  ((x)->tileFlag = (v))

typedef struct
{
  uint16_t input1Scale;  /* Scaling factor for 1st input */
  uint16_t input2Scale;  /* Scaling factor for 2nd input */
  uint8_t  accumShift;   /* Accumulator Shift to bring data to 16b after scaling and addition */
  uint16_t outputScale;  /* Scaling factor for Output */
  uint8_t  outputShift;  /* Shift value to bring the final sum to 8b */
  uint8_t  reluFlag;     /* Enable/Disable Relu at the output */
  int32_t  minVal;       /* minimum Value for clamping if reluFlag is set to 1 */
  int32_t  maxVal;       /* maximum Value for clamping if reluFlag is set to 1 */
  uint8_t  stride;       /* Stride factor */
  int32_t  fixUpInit;    /* The fixUp term that is used to incorporte Zero Points*/
  uint8_t  sat11;        /* Dummy. Not used for xai_cnn_eltwise_params. Used only in xaicnne. Added it for consistency */
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  xb_f16   reluMinFlt;
  xb_f16   reluMaxFlt;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
  float reluMinFlt32;
  float reluMaxFlt32;
#endif
} xai_cnn_eltwise_params;

typedef struct
{
  int32_t input1Scale;   /* Scaling factor for 1st input */
  int32_t input2Scale;   /* Scaling factor for 2nd input */
  int32_t input1Shift;   /* Shift for 1st input */
  int32_t input2Shift;   /* Shift for 2nd input */
  int32_t leftShift;     /* Left Shift for both input */
  int32_t outputScale;   /* Scaling factor for Output */
  int32_t outputShift;   /* Shift value to bring the final sum to 8b */
  int32_t input1Offset;
  int32_t input2Offset;
  int32_t outputOffset;
  uint8_t reluFlag;      /* Enable/Disable Relu at the output */
  int32_t minVal;        /* minimum Value for clamping if reluFlag is set to 1 */
  int32_t maxVal;        /* maximum Value for clamping if reluFlag is set to 1 */
  uint8_t stride;        /* Stride factor */
  int8_t  quantization_mode;
}xai_cnn_tfl_eltwise_params;

typedef struct
{
  int16_t  input1Scale;       /* Scaling factor for 1st input */
  int16_t  input2Scale;       /* Scaling factor for 2nd input */
  uint8_t  accumShift;        /* Accumulator Shift to bring data to 16b after scaling and addition */
  uint16_t outputScale;       /* Scaling factor for Output */
  uint8_t  outputShift;       /* Shift value to bring the final sum to 8b */
  uint8_t  reluFlag;          /* Enable/Disable Relu at the output */
  int32_t  minVal;            /* minimum Value for clamping if reluFlag is set to 1 */
  int32_t  maxVal;            /* maximum Value for clamping if reluFlag is set to 1 */
  uint8_t  stride;            /* Stride factor */
  int32_t  fixUpInit;         /* The fixUp term that is used to incorporte Zero Points*/
  uint8_t  sat11;             /* Quantization saturation: 0 - 10 bit; 1 - 11 bit; */
} xnne_eltwise_params;

#define XAI_CNN_ELTWISE_GET_INPUT1SCALE(x)           ((x)->input1Scale)
#define XAI_CNN_ELTWISE_SET_INPUT1SCALE(x, v)        ((x)->input1Scale = (v))
#define XAI_CNN_ELTWISE_GET_INPUT2SCALE(x)           ((x)->input2Scale)
#define XAI_CNN_ELTWISE_SET_INPUT2SCALE(x, v)        ((x)->input2Scale = (v))
#define XAI_CNN_ELTWISE_GET_INPUT1SHIFT(x)           ((x)->input1Shift)
#define XAI_CNN_ELTWISE_SET_INPUT1SHIFT(x, v)        ((x)->input1Shift = (v))
#define XAI_CNN_ELTWISE_GET_INPUT2SHIFT(x)           ((x)->input2Shift)
#define XAI_CNN_ELTWISE_SET_INPUT2SHIFT(x, v)        ((x)->input2Shift = (v))
#define XAI_CNN_ELTWISE_GET_LEFTSHIFT(x)             ((x)->leftShift)
#define XAI_CNN_ELTWISE_SET_LEFTSHIFT(x, v)          ((x)->leftShift = (v))
#define XAI_CNN_ELTWISE_GET_ACCUMSHIFT(x)            ((x)->accumShift)
#define XAI_CNN_ELTWISE_SET_ACCUMSHIFT(x, v)         ((x)->accumShift = (v))
#define XAI_CNN_ELTWISE_GET_OUTPUTSCALE(x)           ((x)->outputScale)
#define XAI_CNN_ELTWISE_SET_OUTPUTSCALE(x, v)        ((x)->outputScale = (v))
#define XAI_CNN_ELTWISE_GET_OUTPUTSHIFT(x)           ((x)->outputShift)
#define XAI_CNN_ELTWISE_SET_OUTPUTSHIFT(x, v)        ((x)->outputShift = (v))
#define XAI_CNN_ELTWISE_GET_INPUT1_OFFSET(x)         ((x)->input1Offset)
#define XAI_CNN_ELTWISE_SET_INPUT1_OFFSET(x, v)      ((x)->input1Offset = (v))
#define XAI_CNN_ELTWISE_GET_INPUT2_OFFSET(x)         ((x)->input2Offset)
#define XAI_CNN_ELTWISE_SET_INPUT2_OFFSET(x, v)      ((x)->input2Offset = (v))
#define XAI_CNN_ELTWISE_GET_OUTPUT_OFFSET(x)         ((x)->outputOffset)
#define XAI_CNN_ELTWISE_SET_OUTPUT_OFFSET(x, v)      ((x)->outputOffset = (v))
#define XAI_CNN_ELTWISE_GET_QUANTIZATION_MODE(x)     ((x)->quantization_mode)
#define XAI_CNN_ELTWISE_SET_QUANTIZATION_MODE(x, v)  ((x)->quantization_mode = (v))
#define XAI_CNN_ELTWISE_GET_RELUFLAG(x)              ((x)->reluFlag)
#define XAI_CNN_ELTWISE_SET_RELUFLAG(x, v)           ((x)->reluFlag = (v))
#define XAI_CNN_ELTWISE_GET_MIN_VAL(x)               ((x)->minVal)
#define XAI_CNN_ELTWISE_SET_MIN_VAL(x, v)            ((x)->minVal = (v))
#define XAI_CNN_ELTWISE_GET_MAX_VAL(x)               ((x)->maxVal)
#define XAI_CNN_ELTWISE_SET_MAX_VAL(x, v)            ((x)->maxVal = (v))
#define XAI_CNN_ELTWISE_GET_STRIDE(x)                ((x)->stride)
#define XAI_CNN_ELTWISE_SET_STRIDE(x, v)             ((x)->stride = (v))
#define XAI_CNN_ELTWISE_GET_FIXUPINIT(x)             ((x)->fixUpInit)
#define XAI_CNN_ELTWISE_SET_FIXUPINIT(x, v)          ((x)->fixUpInit = (v))
#define XAI_CNN_ELTWISE_GET_SAT11(x)                 ((x)->sat11)
#define XAI_CNN_ELTWISE_SET_SAT11(x, v)              ((x)->sat11 = (v))
#define XAI_CNN_ELTWISE_ADD_STRIDE_J1    (1)
#define XAI_CNN_ELTWISE_ADD_STRIDE_J2    (2)
#define XAI_CNN_ELTWISE_ADD_STRIDE_J1J2  (3)
#define XAI_CNN_ELTWISE_SUB_STRIDE_J1    (1)
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_ELTWISE_GET_RELU_MIN_FLT(x)     ((x)->reluMinFlt)
#define XAI_CNN_ELTWISE_SET_RELU_MIN_FLT(x, v)  ((x)->reluMinFlt = (v))
#define XAI_CNN_ELTWISE_GET_RELU_MAX_FLT(x)     ((x)->reluMaxFlt)
#define XAI_CNN_ELTWISE_SET_RELU_MAX_FLT(x, v)  ((x)->reluMaxFlt = (v))
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
#define XAI_CNN_ELTWISE_GET_RELU_MIN_FLT32(x)     ((x)->reluMinFlt32)
#define XAI_CNN_ELTWISE_SET_RELU_MIN_FLT32(x, v)  ((x)->reluMinFlt32 = (v))
#define XAI_CNN_ELTWISE_GET_RELU_MAX_FLT32(x)     ((x)->reluMaxFlt32)
#define XAI_CNN_ELTWISE_SET_RELU_MAX_FLT32(x, v)  ((x)->reluMaxFlt32 = (v))
#endif

typedef struct
{
  uint16_t inputScale;     /* Scaling factor for Input */
  uint8_t  inputShift;     /* Input Shift to bring data to 16b after scaling */
  int32_t  minIdx;         /* Minimum value of input. Corresponds to first element of LUT array. */
  int32_t  maxIdx;         /* Maximum value of input. Corresponds to last element of LUT array. */
  uint8_t  tableType;      /* Value to describe the type of Table: 0/1/2 - Normal/Symmetric/Asymmetric */
  int32_t  lut1Offset;     /* Offset of the 0th entry of lut1Array in Full range LUT table(minIdx <= lut1Offset <= maxIdx). */
  int32_t  lut2Offset;     /* Offset of the 0th entry of lut2Array in Full range LUT table(minIdx <= lut2Offset <= maxIdx). */
} xai_cnn_lut_params;

#define XAI_LUT_TYPE_NORMAL         0
#define XAI_LUT_TYPE_EVENSYMMETRIC  1
#define XAI_LUT_TYPE_ODDSYMMETRIC   2

#define XAI_CNN_LUT_GET_INPUTSCALE(x)      ((x)->inputScale)
#define XAI_CNN_LUT_SET_INPUTSCALE(x, v)   ((x)->inputScale = (v))
#define XAI_CNN_LUT_GET_INPUTSHIFT(x)      ((x)->inputShift)
#define XAI_CNN_LUT_SET_INPUTSHIFT(x, v)   ((x)->inputShift = (v))
#define XAI_CNN_LUT_GET_MIN_IDX(x)         ((x)->minIdx)
#define XAI_CNN_LUT_SET_MIN_IDX(x, v)      ((x)->minIdx = (v))
#define XAI_CNN_LUT_GET_MAX_IDX(x)         ((x)->maxIdx)
#define XAI_CNN_LUT_SET_MAX_IDX(x, v)      ((x)->maxIdx = (v))
#define XAI_CNN_LUT_GET_TABLE_TYPE(x)      ((x)->tableType)
#define XAI_CNN_LUT_SET_TABLE_TYPE(x, v)   ((x)->tableType = (v))
#define XAI_CNN_LUT_GET_LUT1_OFFSET(x)     ((x)->lut1Offset)
#define XAI_CNN_LUT_SET_LUT1_OFFSET(x, v)  ((x)->lut1Offset = v)
#define XAI_CNN_LUT_GET_LUT2_OFFSET(x)     ((x)->lut2Offset)
#define XAI_CNN_LUT_SET_LUT2_OFFSET(x, v)  ((x)->lut2Offset = v)

typedef struct
{
  int16_t outputScale;   /* Scaling factor for Output */
  uint8_t outputShift;   /* Shift value to bring the final product to output datatype */
  uint8_t reluFlag;      /* Enable/Disable Relu at the output */
  int32_t minVal;        /* minimum Value for clamping */
  int32_t maxVal;        /* maximum Value for clamping */
  int32_t inZero1;
  int32_t inZero2;
  int32_t fixUpInit;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  xb_f16  reluMinFlt;
  xb_f16  reluMaxFlt;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
  float reluMinFlt32;
  float reluMaxFlt32;
#endif
} xai_cnn_eltwiseMul_params;

#define XAI_CNN_ELTWISE_MUL_GET_OUTPUTSCALE(x)      ((x)->outputScale)
#define XAI_CNN_ELTWISE_MUL_SET_OUTPUTSCALE(x, v)   ((x)->outputScale = (v))
#define XAI_CNN_ELTWISE_MUL_GET_OUTPUTSHIFT(x)      ((x)->outputShift)
#define XAI_CNN_ELTWISE_MUL_SET_OUTPUTSHIFT(x, v)   ((x)->outputShift = (v))
#define XAI_CNN_ELTWISE_MUL_GET_RELUFLAG(x)         ((x)->reluFlag)
#define XAI_CNN_ELTWISE_MUL_SET_RELUFLAG(x, v)      ((x)->reluFlag = (v))
#define XAI_CNN_ELTWISE_MUL_GET_MIN_VAL(x)          ((x)->minVal)
#define XAI_CNN_ELTWISE_MUL_SET_MIN_VAL(x, v)       ((x)->minVal = (v))
#define XAI_CNN_ELTWISE_MUL_GET_MAX_VAL(x)          ((x)->maxVal)
#define XAI_CNN_ELTWISE_MUL_SET_MAX_VAL(x, v)       ((x)->maxVal = (v))
#define XAI_CNN_ELTWISE_MUL_GET_INZERO_1(x)         ((x)->inZero1)
#define XAI_CNN_ELTWISE_MUL_SET_INZERO_1(x, v)      ((x)->inZero1 = (v))
#define XAI_CNN_ELTWISE_MUL_GET_INZERO_2(x)         ((x)->inZero2)
#define XAI_CNN_ELTWISE_MUL_SET_INZERO_2(x, v)      ((x)->inZero2 = (v))
#define XAI_CNN_ELTWISE_MUL_GET_FIXUPINIT(x)        ((x)->fixUpInit)
#define XAI_CNN_ELTWISE_MUL_SET_FIXUPINIT(x, v)     ((x)->fixUpInit = (v))
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_ELTWISE_MUL_GET_RELU_MIN_FLT(x)     ((x)->reluMinFlt)
#define XAI_CNN_ELTWISE_MUL_SET_RELU_MIN_FLT(x, v)  ((x)->reluMinFlt = (v))
#define XAI_CNN_ELTWISE_MUL_GET_RELU_MAX_FLT(x)     ((x)->reluMaxFlt)
#define XAI_CNN_ELTWISE_MUL_SET_RELU_MAX_FLT(x, v)  ((x)->reluMaxFlt = (v))
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
#define XAI_CNN_ELTWISE_MUL_GET_RELU_MIN_FLT32(x)     ((x)->reluMinFlt32)
#define XAI_CNN_ELTWISE_MUL_SET_RELU_MIN_FLT32(x, v)  ((x)->reluMinFlt32 = (v))
#define XAI_CNN_ELTWISE_MUL_GET_RELU_MAX_FLT32(x)     ((x)->reluMaxFlt32)
#define XAI_CNN_ELTWISE_MUL_SET_RELU_MAX_FLT32(x, v)  ((x)->reluMaxFlt32 = (v))
#endif

/*SVDF structure */
typedef struct
{
  int32_t nInput;
  int32_t nFilter;
  int32_t nMemory;
  int32_t nBatch;
  int32_t nRank;
  int32_t biasFlag;
  uint8_t shift1;
  uint8_t shift2;
  uint8_t accShift1;
  uint8_t accShift2;
  int32_t preset;
  int32_t minVal;
  int32_t maxVal;
  uint8_t reluFlag;
} xai_cnn_svdf_params;

#define S24_MIN                  (-(((int32_t) 1) << 23))
#define S24_MAX                  ((((int32_t) 1) << 23) - 1)
#define XCHAL_IVPN_SIMD_WIDTH_2  (XCHAL_IVPN_SIMD_WIDTH >> 1)
#define USE_24_BIT_ACCUMULATOR
#define MULQISA                  1

#define XAI_CNN_SVDF_GET_NUMINPUT(x)         ((x)->nInput)
#define XAI_CNN_SVDF_SET_NUMINPUT(x, v)      ((x)->nInput = (v))
#define XAI_CNN_SVDF_GET_MIN_VAL(x)          ((x)->minVal)
#define XAI_CNN_SVDF_SET_MIN_VAL(x, v)       ((x)->minVal = (v))
#define XAI_CNN_SVDF_GET_MAX_VAL(x)          ((x)->maxVal)
#define XAI_CNN_SVDF_SET_MAX_VAL(x, v)       ((x)->maxVal = (v))
#define XAI_CNN_SVDF_GET_RELUFLAG(x)         ((x)->reluFlag)
#define XAI_CNN_SVDF_SET_RELUFLAG(x, v)      ((x)->reluFlag = (v))
#define XAI_CNN_SVDF_GET_NUMFILTER(x)        ((x)->nFilter)
#define XAI_CNN_SVDF_SET_NUMFILTER(x, v)     ((x)->nFilter = (v))
#define XAI_CNN_SVDF_GET_NUMMEMORY(x)        ((x)->nMemory)
#define XAI_CNN_SVDF_SET_NUMMEMORY(x, v)     ((x)->nMemory = (v))
#define XAI_CNN_SVDF_GET_NUMBATCH(x)         ((x)->nBatch)
#define XAI_CNN_SVDF_SET_NUMBATCH(x, v)      ((x)->nBatch = (v))
#define XAI_CNN_SVDF_GET_BIASFLAG(x)         ((x)->biasFlag)
#define XAI_CNN_SVDF_SET_BIASFLAG(x, v)      ((x)->biasFlag = (v))
#define XAI_CNN_SVDF_GET_RANK(x)             ((x)->nRank)
#define XAI_CNN_SVDF_SET_RANK(x, v)          ((x)->nRank = (v))
#define XAI_CNN_SVDF_GET_NUNIT(x)            ((x)->nUnit
#define XAI_CNN_SVDF_SET_NUNIT(x, v)         ((x)->nUnit = (v))
#define XAI_CNN_SVDF_GET_OUTPUTSHIFT1(x)     ((x)->shift1)
#define XAI_CNN_SVDF_SET_OUTPUTSHIFT1(x, v)  ((x)->shift1 = (v))
#define XAI_CNN_SVDF_GET_OUTPUTSHIFT2(x)     ((x)->shift2)
#define XAI_CNN_SVDF_SET_OUTPUTSHIFT2(x, v)  ((x)->shift2 = (v))
#define XAI_CNN_SVDF_GET_ACCSHIFT1(x)        ((x)->accShift1)
#define XAI_CNN_SVDF_SET_ACCSHIFT1(x, v)     ((x)->accShift1 = (v))
#define XAI_CNN_SVDF_GET_ACCSHIFT2(x)        ((x)->accShift2)
#define XAI_CNN_SVDF_SET_ACCSHIFT2(x, v)     ((x)->accShift2 = (v))
#define XAI_CNN_SVDF_GET_PRESET(x)           ((x)->preset)
#define XAI_CNN_SVDF_SET_PRESET(x, v)        ((x)->preset = (v))

typedef struct
{
  uint16_t tableLength0;    /* Minor table (Table 0) length */
  uint16_t tableLength1;    /* Major table (Table 1) length */
  uint16_t inMask0;         /* Mask applied on input while accessing minor table entry */
  uint16_t inMask1;         /* Mask applied on input while accessing major table entry */
  uint8_t  inShift0;        /* Shift applied on input while accessing minor table entry */
  uint8_t  inShift1;        /* Shift applied on input while accessing major table entry */
  uint8_t  outputShift;     /* No. of output bits to be right shifted. */
} xai_cnn_exponent_params;

#define XAI_CNN_EXPONENT_GET_OUTPUTSHIFT(x)       ((x)->outputShift)
#define XAI_CNN_EXPONENT_SET_OUTPUTSHIFT(x, v)    ((x)->outputShift = (v))
#define XAI_CNN_EXPONENT_GET_TABLELENGTH_0(x)     ((x)->tableLength0)
#define XAI_CNN_EXPONENT_SET_TABLELENGTH_0(x, v)  ((x)->tableLength0 = (v))
#define XAI_CNN_EXPONENT_GET_TABLELENGTH_1(x)     ((x)->tableLength1)
#define XAI_CNN_EXPONENT_SET_TABLELENGTH_1(x, v)  ((x)->tableLength1 = (v))
#define XAI_CNN_EXPONENT_GET_MASK_0(x)            ((x)->inMask0)
#define XAI_CNN_EXPONENT_SET_MASK_0(x, v)         ((x)->inMask0 = (v))
#define XAI_CNN_EXPONENT_GET_MASK_1(x)            ((x)->inMask1)
#define XAI_CNN_EXPONENT_SET_MASK_1(x, v)         ((x)->inMask1 = (v))
#define XAI_CNN_EXPONENT_GET_SHIFT_0(x)           ((x)->inShift0)
#define XAI_CNN_EXPONENT_SET_SHIFT_0(x, v)        ((x)->inShift0 = (v))
#define XAI_CNN_EXPONENT_GET_SHIFT_1(x)           ((x)->inShift1)
#define XAI_CNN_EXPONENT_SET_SHIFT_1(x, v)        ((x)->inShift1 = (v))

typedef struct
{
  uint8_t stride;       /* Stride factor */
  uint8_t reverse;      /* Flag to indicate direction of reorg */
} xai_cnn_reorg_params;

#define XAI_CNN_REORG_GET_STRIDE(x)      ((x)->stride)
#define XAI_CNN_REORG_SET_STRIDE(x, v)   ((x)->stride = (v))
#define XAI_CNN_REORG_GET_REVERSE(x)     ((x)->reverse)
#define XAI_CNN_REORG_SET_REVERSE(x, v)  ((x)->reverse = (v))

typedef struct
{
  uint8_t strideX;      /* StrideX factor */
  uint8_t strideY;      /* StrideY factor */
  uint8_t reverse;      /* Flag to indicate direction of reorg */
} xai_cnn_reorg4D_params;

#define XAI_CNN_REORG4D_GET_STRIDEX(x)     ((x)->strideX)
#define XAI_CNN_REORG4D_SET_STRIDEX(x, v)  ((x)->strideX = (v))
#define XAI_CNN_REORG4D_GET_STRIDEY(x)     ((x)->strideY)
#define XAI_CNN_REORG4D_SET_STRIDEY(x, v)  ((x)->strideY = (v))
#define XAI_CNN_REORG4D_GET_REVERSE(x)     ((x)->reverse)
#define XAI_CNN_REORG4D_SET_REVERSE(x, v)  ((x)->reverse = (v))

typedef struct
{
  uint8_t order1;   /* inTile dimension which will be transposed into dimension 1 of outTile */
  uint8_t order2;   /* inTile dimension which will be transposed into dimension 2 of outTile */
  uint8_t order3;   /* inTile dimension which will be transposed into dimension 3 of outTile */
  uint8_t order4;   /* inTile dimension which will be transposed into dimension 4 of outTile */
}xai_cnn_permute4D_params;

#define XAI_CNN_PERMUTE4D_GET_ORDER1(x)     ((x)->order1)
#define XAI_CNN_PERMUTE4D_SET_ORDER1(x, v)  ((x)->order1 = (v))
#define XAI_CNN_PERMUTE4D_GET_ORDER2(x)     ((x)->order2)
#define XAI_CNN_PERMUTE4D_SET_ORDER2(x, v)  ((x)->order2 = (v))
#define XAI_CNN_PERMUTE4D_GET_ORDER3(x)     ((x)->order3)
#define XAI_CNN_PERMUTE4D_SET_ORDER3(x, v)  ((x)->order3 = (v))
#define XAI_CNN_PERMUTE4D_GET_ORDER4(x)     ((x)->order4)
#define XAI_CNN_PERMUTE4D_SET_ORDER4(x, v)  ((x)->order4 = (v))

typedef struct
{
  uint32_t groups;       /* Input Groups */
} xai_cnn_shuffle3D_params;

#define XAI_CNN_SHUFFLE_GET_INTERLEAVEGROUPS(x)     ((x)->groups)
#define XAI_CNN_SHUFFLE_SET_INTERLEAVEGROUPS(x, v)  ((x)->groups = (v))

typedef struct
{
  int32_t xscale;    //Q13.18 format in xaicnn and Q21.10 format in TFL
  int32_t yscale;    //Q13.18 format in xaicnn and Q21.10 format in TFL
  int32_t xshift;    //Q13.18 format in xaicnn and Q21.10 format in TFL
  int32_t yshift;    //Q13.18 format in xaicnn and Q21.10 format in TFL
  uint8_t extrapolationFlag;
  int32_t extrapolationValue;
  int32_t inputFrameWidth;
  int32_t inputFrameHeight;
  int8_t  alignCorners;
  int8_t  halfPixelCenters;
  float   xscaleFlt;
  float   yscaleFlt;
  float   xshiftFlt;
  float   yshiftFlt;
  int8_t  quantization_mode;
} xai_cnn_interp3D_params;

#define XAI_CNN_INTERP3D_GET_XSCALE(x)                      ((x)->xscale)
#define XAI_CNN_INTERP3D_GET_YSCALE(x)                      ((x)->yscale)
#define XAI_CNN_INTERP3D_GET_XSHIFT(x)                      ((x)->xshift)
#define XAI_CNN_INTERP3D_GET_YSHIFT(x)                      ((x)->yshift)
#define XAI_CNN_INTERP3D_GET_EXTRAPOLATION_FLAG(x)          ((x)->extrapolationFlag)
#define XAI_CNN_INTERP3D_GET_EXTRAPOLATION_VALUE(x)         ((x)->extrapolationValue)
#define XAI_CNN_INTERP3D_GET_FRAME_WIDTH(x)                 ((x)->inputFrameWidth)
#define XAI_CNN_INTERP3D_GET_FRAME_HEIGHT(x)                ((x)->inputFrameHeight)
#define XAI_CNN_INTERP3D_GET_FLAG_ALIGN_CORNERS(x)          ((x)->alignCorners)
#define XAI_CNN_INTERP3D_GET_FLAG_HALF_PIXEL_CENTERS(x)     ((x)->halfPixelCenters)
#define XAI_CNN_INTERP3D_GET_XSCALE_FLT(x)                  ((x)->xscaleFlt)
#define XAI_CNN_INTERP3D_GET_YSCALE_FLT(x)                  ((x)->yscaleFlt)
#define XAI_CNN_INTERP3D_GET_XSHIFT_FLT(x)                  ((x)->xshiftFlt)
#define XAI_CNN_INTERP3D_GET_YSHIFT_FLT(x)                  ((x)->yshiftFlt)

#define XAI_CNN_INTERP3D_SET_XSCALE(x, v)                   ((x)->xscale = (v))
#define XAI_CNN_INTERP3D_SET_YSCALE(x, v)                   ((x)->yscale = (v))
#define XAI_CNN_INTERP3D_SET_XSHIFT(x, v)                   ((x)->xshift = (v))
#define XAI_CNN_INTERP3D_SET_YSHIFT(x, v)                   ((x)->yshift = (v))
#define XAI_CNN_INTERP3D_SET_EXTRAPOLATION_FLAG(x, v)       ((x)->extrapolationFlag = (v))
#define XAI_CNN_INTERP3D_SET_EXTRAPOLATION_VALUE(x, v)      ((x)->extrapolationValue = (v))
#define XAI_CNN_INTERP3D_SET_FRAME_WIDTH(x, v)              ((x)->inputFrameWidth = (v))
#define XAI_CNN_INTERP3D_SET_FRAME_HEIGHT(x, v)             ((x)->inputFrameHeight = (v))
#define XAI_CNN_INTERP3D_SET_FLAG_ALIGN_CORNERS(x, v)       ((x)->alignCorners = v)
#define XAI_CNN_INTERP3D_SET_FLAG_HALF_PIXEL_CENTERS(x, v)  ((x)->halfPixelCenters = v)
#define XAI_CNN_INTERP3D_SET_XSCALE_FLT(x, v)               ((x)->xscaleFlt = (v))
#define XAI_CNN_INTERP3D_SET_YSCALE_FLT(x, v)               ((x)->yscaleFlt = (v))
#define XAI_CNN_INTERP3D_SET_XSHIFT_FLT(x, v)               ((x)->xshiftFlt = (v))
#define XAI_CNN_INTERP3D_SET_YSHIFT_FLT(x, v)               ((x)->yshiftFlt = (v))
#define XAI_CNN_INTERP3D_GET_QUANTIZATION_MODE(x)           ((x)->quantization_mode)
#define XAI_CNN_INTERP3D_SET_QUANTIZATION_MODE(x, v)        ((x)->quantization_mode = (v))

typedef struct
{
  int32_t xscale;  //Q13.18 format
  int32_t yscale;  //Q13.18 format
  int32_t xshift;  //Q13.18 format
  int32_t yshift;  //Q13.18 format
  int32_t inputFrameWidth;
  int32_t inputFrameHeight;
  int8_t  alignCorners;
  int8_t  halfPixelCenters;
  float   xscaleFlt;
  float   yscaleFlt;
  float   xshiftFlt;
  float   yshiftFlt;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  xb_f16  xscaleFlt16;
  xb_f16  yscaleFlt16;
#endif
  int8_t  quantization_mode;
} xai_cnn_resize_nearest3D_params;

#define XAI_CNN_RESIZENEAREST3D_GET_XSCALE(x)                   ((x)->xscale)
#define XAI_CNN_RESIZENEAREST3D_GET_YSCALE(x)                   ((x)->yscale)
#define XAI_CNN_RESIZENEAREST3D_GET_XSHIFT(x)                   ((x)->xshift)
#define XAI_CNN_RESIZENEAREST3D_GET_YSHIFT(x)                   ((x)->yshift)
#define XAI_CNN_RESIZENEAREST3D_GET_FLAG_ALIGN_CORNERS(x)       ((x)->alignCorners)
#define XAI_CNN_RESIZENEAREST3D_GET_FLAG_HALF_PIXEL_CENTERS(x)  ((x)->halfPixelCenters)
#define XAI_CNN_RESIZENEAREST3D_GET_FRAME_WIDTH(x)              ((x)->inputFrameWidth)
#define XAI_CNN_RESIZENEAREST3D_GET_FRAME_HEIGHT(x)             ((x)->inputFrameHeight)
#define XAI_CNN_RESIZENEAREST3D_GET_XSCALE_FLT(x)               ((x)->xscaleFlt)
#define XAI_CNN_RESIZENEAREST3D_GET_YSCALE_FLT(x)               ((x)->yscaleFlt)
#define XAI_CNN_RESIZENEAREST3D_GET_XSHIFT_FLT(x)               ((x)->xshiftFlt)
#define XAI_CNN_RESIZENEAREST3D_GET_YSHIFT_FLT(x)               ((x)->yshiftFlt)
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_RESIZENEAREST3D_GET_XSCALE_FLT16(x)             ((x)->xscaleFlt16)
#define XAI_CNN_RESIZENEAREST3D_GET_YSCALE_FLT16(x)             ((x)->yscaleFlt16)
#endif

#define XAI_CNN_RESIZENEAREST3D_SET_XSCALE(x, v)                   ((x)->xscale = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_YSCALE(x, v)                   ((x)->yscale = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_XSHIFT(x, v)                   ((x)->xshift = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_YSHIFT(x, v)                   ((x)->yshift = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_FLAG_ALIGN_CORNERS(x, v)       ((x)->alignCorners = v)
#define XAI_CNN_RESIZENEAREST3D_SET_FLAG_HALF_PIXEL_CENTERS(x, v)  ((x)->halfPixelCenters = v)
#define XAI_CNN_RESIZENEAREST3D_SET_FRAME_WIDTH(x, v)              ((x)->inputFrameWidth = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_FRAME_HEIGHT(x, v)             ((x)->inputFrameHeight = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_XSCALE_FLT(x, v)               ((x)->xscaleFlt = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_YSCALE_FLT(x, v)               ((x)->yscaleFlt = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_XSHIFT_FLT(x, v)               ((x)->xshiftFlt = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_YSHIFT_FLT(x, v)               ((x)->yshiftFlt = (v))
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_RESIZENEAREST3D_SET_XSCALE_FLT16(x, v)             ((x)->xscaleFlt16 = (v))
#define XAI_CNN_RESIZENEAREST3D_SET_YSCALE_FLT16(x, v)             ((x)->yscaleFlt16 = (v))
#endif
#define XAI_CNN_RESIZENEAREST3D_GET_QUANTIZATION_MODE(x)           ((x)->quantization_mode)
#define XAI_CNN_RESIZENEAREST3D_SET_QUANTIZATION_MODE(x, v)        ((x)->quantization_mode = (v))


typedef struct
{
  int16_t epsilon;                    // Always added or max val is considered based on tileFlag.
  uint8_t normType;                   // (1= L1 Norm, 2 = L2 Norm)
  uint8_t normAxis;                   // indicates the combination of axes along which to normalize
  uint8_t channelShareFlag;           // indicates whether we have a single scale value or an array equal to number of channels
  uint8_t tileFlag;                   // indicates whether the given tile is a first tile, last tile or neither of those
  uint8_t tensorFlowFlag;             // describes the usage of epsilon
  int8_t  quantScaleTableShift;       // shift value for scalar table
  int8_t  rSqrtTableShift;            // shift value for recip square root table
  int8_t  recipTableShift;            // shift value for recip table
  int8_t  rSqrtIndexShift;            // shift value recip-square-root table index
  int8_t  sumSquareShift;             // shift value for sum of squares
  float  epsilonFlt;                 // floating point epsilon to be added to avoid divide by zero
  float  sumSqScaleFlt;              // floating point scale value to be multiplied to sum of squares, to account for divide by N factor
  int8_t  quantization_mode;
} xai_cnn_normalize3D_params;

#define XAI_CNN_NORMALIZE3D_GET_EPSILON(x)                               ((x)->epsilon)
#define XAI_CNN_NORMALIZE3D_SET_EPSILON(x, v)                            ((x)->epsilon = (v))
#define XAI_CNN_NORMALIZE3D_GET_NORM_TYPE(x)                             ((x)->normType)
#define XAI_CNN_NORMALIZE3D_SET_NORM_TYPE(x, v)                          ((x)->normType = (v))
#define XAI_CNN_NORMALIZE3D_GET_NORMALIZE_ALONG_WIDTH(x)                 ((x)->normAxis & CNN_NORMALIZE_ALONG_WIDTH)
#define XAI_CNN_NORMALIZE3D_SET_NORMALIZE_ALONG_WIDTH(x)                 ((x)->normAxis = ((x)->normAxis | CNN_NORMALIZE_ALONG_WIDTH))
#define XAI_CNN_NORMALIZE3D_RESET_NORMALIZE_ALONG_WIDTH(x)               ((x)->normAxis = ((x)->normAxis & ~CNN_NORMALIZE_ALONG_WIDTH))
#define XAI_CNN_NORMALIZE3D_GET_NORMALIZE_ALONG_HEIGHT(x)                ((x)->normAxis & CNN_NORMALIZE_ALONG_HEIGHT)
#define XAI_CNN_NORMALIZE3D_SET_NORMALIZE_ALONG_HEIGHT(x)                ((x)->normAxis = ((x)->normAxis | CNN_NORMALIZE_ALONG_HEIGHT))
#define XAI_CNN_NORMALIZE3D_RESET_NORMALIZE_ALONG_HEIGHT(x)              ((x)->normAxis = ((x)->normAxis & ~CNN_NORMALIZE_ALONG_HEIGHT))
#define XAI_CNN_NORMALIZE3D_GET_NORMALIZE_ALONG_WIDTH_AND_HEIGHT(x)      ((x)->normAxis & CNN_NORMALIZE_ALONG_WIDTH_AND_HEIGHT)
#define XAI_CNN_NORMALIZE3D_SET_NORMALIZE_ALONG_WIDTH_AND_HEIGHT(x)      ((x)->normAxis = ((x)->normAxis | CNN_NORMALIZE_ALONG_WIDTH_AND_HEIGHT))
#define XAI_CNN_NORMALIZE3D_RESET_NORMALIZE_ALONG_WIDTH_AND_HEIGHT(x)    ((x)->normAxis = ((x)->normAxis & ~CNN_NORMALIZE_ALONG_WIDTH_AND_HEIGHT))
#define XAI_CNN_NORMALIZE3D_GET_NORMALIZE_ALONG_DEPTH(x)                 ((x)->normAxis & CNN_NORMALIZE_ALONG_DEPTH)
#define XAI_CNN_NORMALIZE3D_SET_NORMALIZE_ALONG_DEPTH(x)                 ((x)->normAxis = ((x)->normAxis | CNN_NORMALIZE_ALONG_DEPTH))
#define XAI_CNN_NORMALIZE3D_RESET_NORMALIZE_ALONG_DEPTH(x)               ((x)->normAxis = ((x)->normAxis & ~CNN_NORMALIZE_ALONG_DEPTH))
#define XAI_CNN_NORMALIZE3D_GET_NORMALIZE_ALONG_BATCH(x)                 ((x)->normAxis & CNN_NORMALIZE_ALONG_BATCH)
#define XAI_CNN_NORMALIZE3D_SET_NORMALIZE_ALONG_BATCH(x)                 ((x)->normAxis = ((x)->normAxis | CNN_NORMALIZE_ALONG_BATCH))
#define XAI_CNN_NORMALIZE3D_RESET_NORMALIZE_ALONG_BATCH(x)               ((x)->normAxis = ((x)->normAxis & ~CNN_NORMALIZE_ALONG_BATCH))
#define XAI_CNN_NORMALIZE3D_GET_NORMALIZE_ALONG_WIDTH_HEIGHT_DEPTH(x)    ((x)->normAxis & CNN_NORMALIZE_ALONG_WIDTH_AND_HEIGHT_AND_DEPTH)
#define XAI_CNN_NORMALIZE3D_SET_NORMALIZE_ALONG_WIDTH_HEIGHT_DEPTH(x)    ((x)->normAxis = ((x)->normAxis | CNN_NORMALIZE_ALONG_WIDTH_AND_HEIGHT_AND_DEPTH))
#define XAI_CNN_NORMALIZE3D_RESET_NORMALIZE_ALONG_WIDTH_HEIGHT_DEPTH(x)  ((x)->normAxis = ((x)->normAxis & ~CNN_NORMALIZE_ALONG_WIDTH_AND_HEIGHT_AND_DEPTH))
#define XAI_CNN_NORMALIZE3D_GET_CHANNEL_SHARE_FLAG(x)                    ((x)->channelShareFlag & CNN_NORMALIZE_CHANNEL_SHARE_FLAG)
#define XAI_CNN_NORMALIZE3D_SET_CHANNEL_SHARE_FLAG(x)                    ((x)->channelShareFlag = ((x)->channelShareFlag | CNN_NORMALIZE_CHANNEL_SHARE_FLAG))
#define XAI_CNN_NORMALIZE3D_RESET_CHANNEL_SHARE_FLAG(x)                  ((x)->channelShareFlag = ((x)->channelShareFlag & ~CNN_NORMALIZE_CHANNEL_SHARE_FLAG))
#define XAI_CNN_NORMALIZE3D_GET_TILE_FLAG(x)                             ((x)->tileFlag)
#define XAI_CNN_NORMALIZE3D_SET_TILE_FLAG(x, v)                          ((x)->tileFlag = (v))
#define XAI_CNN_NORMALIZE3D_GET_TENSORFLOW_FLAG(x)                       ((x)->tensorFlowFlag)
#define XAI_CNN_NORMALIZE3D_SET_TENSORFLOW_FLAG(x, v)                    ((x)->tensorFlowFlag = (v))
#define XAI_CNN_NORMALIZE3D_GET_RSQRT_TABLE_SHIFT(x)                     ((x)->rSqrtTableShift)
#define XAI_CNN_NORMALIZE3D_SET_RSQRT_TABLE_SHIFT(x, v)                  ((x)->rSqrtTableShift = (v))
#define XAI_CNN_NORMALIZE3D_GET_RECIP_TABLE_SHIFT(x)                     ((x)->recipTableShift)
#define XAI_CNN_NORMALIZE3D_SET_RECIP_TABLE_SHIFT(x, v)                  ((x)->recipTableShift = (v))
#define XAI_CNN_NORMALIZE3D_GET_RSQRT_INDEX_SHIFT(x)                     ((x)->rSqrtIndexShift)
#define XAI_CNN_NORMALIZE3D_SET_RSQRT_INDEX_SHIFT(x, v)                  ((x)->rSqrtIndexShift = (v))
#define XAI_CNN_NORMALIZE3D_GET_SUM_SQUARE_SHIFT(x)                      ((x)->sumSquareShift)
#define XAI_CNN_NORMALIZE3D_SET_SUM_SQUARE_SHIFT(x, v)                   ((x)->sumSquareShift = (v))
#define XAI_CNN_NORMALIZE3D_GET_QUANT_SCALE_TABLE_SHIFT(x)               ((x)->quantScaleTableShift)
#define XAI_CNN_NORMALIZE3D_SET_QUANT_SCALE_TABLE_SHIFT(x, v)            ((x)->quantScaleTableShift = (v))
#define XAI_CNN_NORMALIZE3D_GET_EPSILON_FLT(x)                           ((x)->epsilonFlt)
#define XAI_CNN_NORMALIZE3D_SET_EPSILON_FLT(x, v)                        ((x)->epsilonFlt = (v))
#define XAI_CNN_NORMALIZE3D_GET_SUM_SQ_SCALE_FLT(x)                      ((x)->sumSqScaleFlt)
#define XAI_CNN_NORMALIZE3D_SET_SUM_SQ_SCALE_FLT(x, v)                   ((x)->sumSqScaleFlt = (v))
#define XAI_CNN_NORMALIZE3D_GET_QUANTIZATION_MODE(x)                     ((x)->quantization_mode)
#define XAI_CNN_NORMALIZE3D_SET_QUANTIZATION_MODE(x, v)                  ((x)->quantization_mode = (v))

typedef struct
{
  uint8_t outputShift;   /* Shift value to bring the final value to 8b */
  uint8_t tileFlag;
  uint8_t meanShift;  /* set to a S to do the division */
  uint8_t sqAccShift; /*  set to a shift value of accumulation of squares to 32 bits*/
  int32_t meanScale;  /*Scale = (1<<S) /H*W */
  uint8_t reluFlag;   /* Enable/Disable Relu at the output */
  int32_t minVal;     /* minimum Value for clamping if reluFlag is set to 1 */
  int32_t maxVal;     /* maximum Value for clamping if reluFlag is set to 1 */
  int32_t axis;
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  xb_f16  reluMinFlt;
  xb_f16  reluMaxFlt;
  xb_f16  epsilon;
  xb_f16  meanScaleFlt;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
  float reluMinFlt32;
  float reluMaxFlt32;
  float epsilonFlt32;
  float meanScaleFlt32;
#endif
} xai_cnn_instance_norm_param;

#define XAI_CNN_INSTANCE_NORM_GET_OUTPUTSHIFT(x)       ((x)->outputShift)
#define XAI_CNN_INSTANCE_NORM_SET_OUTPUTSHIFT(x, v)    ((x)->outputShift = (v))
#define XAI_CNN_INSTANCE_NORM_GET_TILEFLAG(x)          ((x)->tileFlag)
#define XAI_CNN_INSTANCE_NORM_SET_TILEFLAG(x, v)       ((x)->tileFlag = (v))
#define XAI_CNN_INSTANCE_NORM_GET_MEANSCALE(x)         ((x)->meanScale)
#define XAI_CNN_INSTANCE_NORM_SET_MEANSCALE(x, v)      ((x)->meanScale = (v))
#define XAI_CNN_INSTANCE_NORM_GET_MEANSHIFT(x)         ((x)->meanShift)
#define XAI_CNN_INSTANCE_NORM_SET_MEANSHIFT(x, v)      ((x)->meanShift = (v))
#define XAI_CNN_INSTANCE_NORM_GET_RELUFLAG(x)          ((x)->reluFlag)
#define XAI_CNN_INSTANCE_NORM_SET_RELUFLAG(x, v)       ((x)->reluFlag = (v))
#define XAI_CNN_INSTANCE_NORM_GET_MIN_VAL(x)           ((x)->minVal)
#define XAI_CNN_INSTANCE_NORM_SET_MIN_VAL(x, v)        ((x)->minVal = (v))
#define XAI_CNN_INSTANCE_NORM_GET_MAX_VAL(x)           ((x)->maxVal)
#define XAI_CNN_INSTANCE_NORM_SET_MAX_VAL(x, v)        ((x)->maxVal = (v))
#define XAI_CNN_INSTANCE_NORM_GET_SQACCSHIFT(x)        ((x)->sqAccShift)
#define XAI_CNN_INSTANCE_NORM_SET_SQACCSHIFT(x, v)     ((x)->sqAccShift = (v))
#define XAI_CNN_INSTANCE_NORM_GET_AXIS(x)              ((x)->axis)
#define XAI_CNN_INSTANCE_NORM_SET_AXIS(x, v)           ((x)->axis = (v))
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_INSTANCE_NORM_GET_RELU_MIN_FLT(x)      ((x)->reluMinFlt)
#define XAI_CNN_INSTANCE_NORM_SET_RELU_MIN_FLT(x, v)   ((x)->reluMinFlt = (v))
#define XAI_CNN_INSTANCE_NORM_GET_RELU_MAX_FLT(x)      ((x)->reluMaxFlt)
#define XAI_CNN_INSTANCE_NORM_SET_RELU_MAX_FLT(x, v)   ((x)->reluMaxFlt = (v))
#define XAI_CNN_INSTANCE_NORM_GET_EPSILON_FLT(x)       ((x)->epsilon)
#define XAI_CNN_INSTANCE_NORM_SET_EPSILON_FLT(x, v)    ((x)->epsilon = (v))
#define XAI_CNN_INSTANCE_NORM_GET_MEANSCALE_FLT(x)     ((x)->meanScaleFlt)
#define XAI_CNN_INSTANCE_NORM_SET_MEANSCALE_FLT(x, v)  ((x)->meanScaleFlt = (v))
#define XAI_CNN_INSTANCE_NORM_GET_MEANSCALE_FLT(x)     ((x)->meanScaleFlt)
#define XAI_CNN_INSTANCE_NORM_SET_MEANSCALE_FLT(x, v)  ((x)->meanScaleFlt = (v))
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
#define XAI_CNN_INSTANCE_NORM_GET_RELU_MIN_FLT32(x)      ((x)->reluMinFlt32)
#define XAI_CNN_INSTANCE_NORM_SET_RELU_MIN_FLT32(x, v)   ((x)->reluMinFlt32 = (v))
#define XAI_CNN_INSTANCE_NORM_GET_RELU_MAX_FLT32(x)      ((x)->reluMaxFlt32)
#define XAI_CNN_INSTANCE_NORM_SET_RELU_MAX_FLT32(x, v)   ((x)->reluMaxFlt32 = (v))
#define XAI_CNN_INSTANCE_NORM_GET_EPSILON_FLT32(x)       ((x)->epsilonFlt32)
#define XAI_CNN_INSTANCE_NORM_SET_EPSILON_FLT32(x, v)    ((x)->epsilonFlt32 = (v))
#define XAI_CNN_INSTANCE_NORM_GET_MEANSCALE_FLT32(x)     ((x)->meanScaleFlt32)
#define XAI_CNN_INSTANCE_NORM_SET_MEANSCALE_FLT32(x, v)  ((x)->meanScaleFlt32 = (v))
#define XAI_CNN_INSTANCE_NORM_GET_MEANSCALE_FLT32(x)     ((x)->meanScaleFlt32)
#define XAI_CNN_INSTANCE_NORM_SET_MEANSCALE_FLT32(x, v)  ((x)->meanScaleFlt32 = (v))
#endif

typedef struct
{
  uint32_t valueR;   /* constant value which needs to be divided with divisor
                        for each channel , can take a maximum range of (2^15) - 1
                        for I8 input and 2^31-1 for S16 input */
  uint8_t outShift;  /* Shift value applied to scaled output */
} xai_cnn_divide3D_params;

#define XAI_CNN_CHANNELWISE_DIVIDE_GET_VALUE_R(x)       ((x)->valueR)
#define XAI_CNN_CHANNELWISE_DIVIDE_SET_VALUE_R(x, v)    ((x)->valueR = (v))
#define XAI_CNN_CHANNELWISE_DIVIDE_GET_OUT_SHIFT(x)     ((x)->outShift)
#define XAI_CNN_CHANNELWISE_DIVIDE_SET_OUT_SHIFT(x, v)  ((x)->outShift = (v))

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))

#define CNNA_CONV_F16_FLAG_RELU      1
#define CNNA_CONV_F16_FLAG_LEFTEDGE  (1 << 1)
#define CNNA_CONV_F16_FLAG_TOPEDGE   (1 << 2)
#endif // #if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
typedef struct
{
  uint16_t spatialScaleShiftX;      /* Shift value to apply for spatial scale operations in the X direction  */
  uint16_t spatialScaleShiftY;      /* Shift value to apply for spatial scale operations in the Y direction  */
  uint16_t outShift;                /* Is either 7, 8, 15, 16, or 23 depending on the datatype of input      */
  int32_t  extrapolationValue;      /* Extrapolate value to be used during extrapolation                     */
  int32_t  roiStride;               /* ROI coordinates' stride                                               */
  uint8_t  tensorFlowFlag;          /* Flag to change box coordinates ordering from Caffe2 to TensorFlow         */
} xai_cnn_cropResize3D_params;

#define XAI_CNN_CROP_RESIZE3D_GET_SPATIAL_SCALE_SHIFTX(x)     ((x)->spatialScaleShiftX)
#define XAI_CNN_CROP_RESIZE3D_SET_SPATIAL_SCALE_SHIFTX(x, v)  ((x)->spatialScaleShiftX = (v))
#define XAI_CNN_CROP_RESIZE3D_GET_SPATIAL_SCALE_SHIFTY(x)     ((x)->spatialScaleShiftY)
#define XAI_CNN_CROP_RESIZE3D_SET_SPATIAL_SCALE_SHIFTY(x, v)  ((x)->spatialScaleShiftY = (v))
#define XAI_CNN_CROP_RESIZE3D_GET_OUT_SHIFT(x)                ((x)->outShift)
#define XAI_CNN_CROP_RESIZE3D_SET_OUT_SHIFT(x, v)             ((x)->outShift = (v))
#define XAI_CNN_CROP_RESIZE3D_GET_EXTRAPOLATION_VALUE(x)      ((x)->extrapolationValue)
#define XAI_CNN_CROP_RESIZE3D_SET_EXTRAPOLATION_VALUE(x, v)   ((x)->extrapolationValue = (v))
#define XAI_CNN_CROP_RESIZE3D_GET_ROI_STRIDE(x)               ((x)->roiStride)
#define XAI_CNN_CROP_RESIZE3D_SET_ROI_STRIDE(x, v)            ((x)->roiStride = (v))
#define XAI_CNN_CROP_RESIZE3D_GET_TENSORFLOW_FLAG(x)          ((x)->tensorFlowFlag)
#define XAI_CNN_CROP_RESIZE3D_SET_TENSORFLOW_FLAG(x, v)       ((x)->tensorFlowFlag = (v))

typedef struct
{
  uint8_t outputShift;   /* Shift value to bring the final value to 8b */
  uint8_t reluFlag;      /* Enable/Disable Relu at the output */
  int32_t minVal;        /* minimum Value for clamping if reluFlag is set to 1 */
  int32_t maxVal;        /* maximum Value for clamping if reluFlag is set to 1 */
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  xb_f16  reluMinFlt;
  xb_f16  reluMaxFlt;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
  float reluMinFlt32;
  float reluMaxFlt32;
#endif
} xai_cnn_batchnorm_params;

#define XAI_CNN_BATCHNORM_GET_OUTPUTSHIFT(x)      ((x)->outputShift)
#define XAI_CNN_BATCHNORM_SET_OUTPUTSHIFT(x, v)   ((x)->outputShift = (v))
#define XAI_CNN_BATCHNORM_GET_RELUFLAG(x)         ((x)->reluFlag)
#define XAI_CNN_BATCHNORM_SET_RELUFLAG(x, v)      ((x)->reluFlag = (v))
#define XAI_CNN_BATCHNORM_GET_MIN_VAL(x)          ((x)->minVal)
#define XAI_CNN_BATCHNORM_SET_MIN_VAL(x, v)       ((x)->minVal = (v))
#define XAI_CNN_BATCHNORM_GET_MAX_VAL(x)          ((x)->maxVal)
#define XAI_CNN_BATCHNORM_SET_MAX_VAL(x, v)       ((x)->maxVal = (v))
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_BATCHNORM_GET_RELU_MIN_FLT(x)     ((x)->reluMinFlt)
#define XAI_CNN_BATCHNORM_SET_RELU_MIN_FLT(x, v)  ((x)->reluMinFlt = (v))
#define XAI_CNN_BATCHNORM_GET_RELU_MAX_FLT(x)     ((x)->reluMaxFlt)
#define XAI_CNN_BATCHNORM_SET_RELU_MAX_FLT(x, v)  ((x)->reluMaxFlt = (v))
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
#define XAI_CNN_BATCHNORM_GET_RELU_MIN_FLT32(x)     ((x)->reluMinFlt32)
#define XAI_CNN_BATCHNORM_SET_RELU_MIN_FLT32(x, v)  ((x)->reluMinFlt32 = (v))
#define XAI_CNN_BATCHNORM_GET_RELU_MAX_FLT32(x)     ((x)->reluMaxFlt32)
#define XAI_CNN_BATCHNORM_SET_RELU_MAX_FLT32(x, v)  ((x)->reluMaxFlt32 = (v))
#endif

typedef struct
{
#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  xb_f16 lambdaF16;
  xb_f16 alphaF16;
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
  float lambdaF32;
  float alphaF32;
#endif
} xai_cnn_selu_params;

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CNN_SELU_GET_LAMBDA16(x)     ((x)->lambdaF16)
#define XAI_CNN_SELU_SET_LAMBDA16(x, v)  ((x)->lambdaF16 = (v))
#define XAI_CNN_SELU_GET_ALPHA16(x)      ((x)->alphaF16)
#define XAI_CNN_SELU_SET_ALPHA16(x, v)   ((x)->alphaF16 = (v))
#endif
#if ((XCHAL_HAVE_VISION_SP_VFPU == 1) || (XCHAL_HAVE_BBENEP_SP_VFPU == 1) || defined(XAI_REF_ONLY_COMPILATION))
#define XAI_CNN_SELU_GET_LAMBDA32(x)     ((x)->lambdaF32)
#define XAI_CNN_SELU_SET_LAMBDA32(x, v)  ((x)->lambdaF32 = (v))
#define XAI_CNN_SELU_GET_ALPHA32(x)      ((x)->alphaF32)
#define XAI_CNN_SELU_SET_ALPHA32(x, v)   ((x)->alphaF32 = (v))
#endif

typedef struct
{
  int16_t  ZeroIn;      /* Zero Point value for Input Tile*/
  int16_t  ZeroOut;     /* Zero Point value for output*/
  uint16_t renormScale; /* Scale applied on (input - ZeroIn) */
  uint8_t  renormShift; /* Shift applied to obtain S8 output */
} xai_cnn_renorm_params;

#define XAI_CNN_RENORM_GET_ZEROIN(x)          ((x)->ZeroIn)
#define XAI_CNN_RENORM_SET_ZEROIN(x, v)       ((x)->ZeroIn = (v))
#define XAI_CNN_RENORM_GET_ZEROOUT(x)         ((x)->ZeroOut)
#define XAI_CNN_RENORM_SET_ZEROOUT(x, v)      ((x)->ZeroOut = (v))
#define XAI_CNN_RENORM_GET_RENORMSCALE(x)     ((x)->renormScale)
#define XAI_CNN_RENORM_SET_RENORMSCALE(x, v)  ((x)->renormScale = (v))
#define XAI_CNN_RENORM_GET_RENORMSHIFT(x)     ((x)->renormShift)
#define XAI_CNN_RENORM_SET_RENORMSHIFT(x, v)  ((x)->renormShift = (v))

#if (XCHAL_HAVE_HIFI1 || XCHAL_HAVE_HIFI3Z || XCHAL_HAVE_HIFI4 || XCHAL_HAVE_HIFI5)
typedef struct
{
  int32_t ZeroIn;
  int32_t ZeroOut;
  int32_t requantScale;
  int32_t requantShift;
  int8_t  quantization_mode;
} xai_cnn_tfl_requantize_params;

#define XAI_CNN_REQUANT_GET_ZEROIN(x)                ((x)->ZeroIn)
#define XAI_CNN_REQUANT_SET_ZEROIN(x, v)             ((x)->ZeroIn = (v))
#define XAI_CNN_REQUANT_GET_ZEROOUT(x)               ((x)->ZeroOut)
#define XAI_CNN_REQUANT_SET_ZEROOUT(x, v)            ((x)->ZeroOut = (v))
#define XAI_CNN_REQUANT_GET_REQUANTSCALE(x)          ((x)->requantScale)
#define XAI_CNN_REQUANT_SET_REQUANTSCALE(x, v)       ((x)->requantScale = (v))
#define XAI_CNN_REQUANT_GET_REQUANTSHIFT(x)          ((x)->requantShift)
#define XAI_CNN_REQUANT_SET_REQUANTSHIFT(x, v)       ((x)->requantShift = (v))
#define XAI_CNN_REQUANT_GET_QUANTIZATION_MODE(x)     ((x)->quantization_mode)
#define XAI_CNN_REQUANT_SET_QUANTIZATION_MODE(x, v)  ((x)->quantization_mode = (v))

#else
typedef struct
{
  int16_t ZeroIn;
  int16_t ZeroOut;
  int32_t requantScale;
  int32_t requantShift;
  int8_t  quantization_mode;
} xai_cnn_tfl_requantize_params;

#define XAI_CNN_REQUANT_GET_ZEROIN(x)                ((x)->ZeroIn)
#define XAI_CNN_REQUANT_SET_ZEROIN(x, v)             ((x)->ZeroIn = (v))
#define XAI_CNN_REQUANT_GET_ZEROOUT(x)               ((x)->ZeroOut)
#define XAI_CNN_REQUANT_SET_ZEROOUT(x, v)            ((x)->ZeroOut = (v))
#define XAI_CNN_REQUANT_GET_REQUANTSCALE(x)          ((x)->requantScale)
#define XAI_CNN_REQUANT_SET_REQUANTSCALE(x, v)       ((x)->requantScale = (v))
#define XAI_CNN_REQUANT_GET_REQUANTSHIFT(x)          ((x)->requantShift)
#define XAI_CNN_REQUANT_SET_REQUANTSHIFT(x, v)       ((x)->requantShift = (v))
#define XAI_CNN_REQUANT_GET_QUANTIZATION_MODE(x)     ((x)->quantization_mode)
#define XAI_CNN_REQUANT_SET_QUANTIZATION_MODE(x, v)  ((x)->quantization_mode = (v))
#endif

typedef struct
{
  uint16_t outputScale; /* Scaling factor for Output */
  uint8_t  outputShift; /* Shift value to bring the output to 16b */
} xai_cnn_relu_params;

#define XAI_CNN_RELU_GET_OUTPUTSCALE(x)     ((x)->outputScale)
#define XAI_CNN_RELU_SET_OUTPUTSCALE(x, v)  ((x)->outputScale = (v))
#define XAI_CNN_RELU_GET_OUTPUTSHIFT(x)     ((x)->outputShift)
#define XAI_CNN_RELU_SET_OUTPUTSHIFT(x, v)  ((x)->outputShift = (v))

typedef struct
{
  uint8_t  config;       // Determines reduction across particular dimensions
  uint8_t  tileFlag;     // Determines which tile is currently being processed
                         // 0-> intermediate tile, 1-> first tile, 2 --> last tile, 3 --> first and last tile
  int32_t  fixUpInit;    // The fixUp term that is used to incorporte Zero Points
  uint8_t  accShiftU;    // The value by which the accumulated value is right shifted
  uint8_t  outShiftU;    // The value by which the intermediate output value is right shifted
  uint16_t outScale;     // The value by which acc-shifted value is multiplied to give intermediate output value
  uint8_t  enableReLu;   // Indicates if relu functionality needs to be enabled (1) or not (0)
  int64_t  reluMin;      // The lower limit value which will be used for clamping the outputs
  int64_t  reluMax;      // The upper limit value which will be used for clamping the outputs
  bool     take_abs;     // Indicates if absolute value needs to be taken (true) or not (false)
  int32_t  redEleCount;  // Total number of elements reduced in the output
} xai_cnn_reduce_params;

#define XAI_CNN_REDUCE_GET_CONFIG(x)                     ((x)->config)
#define XAI_CNN_REDUCE_GET_TILEFLAG(x)                   ((x)->tileFlag)
#define XAI_CNN_REDUCE_GET_FIXUPINIT(x)                  ((x)->fixUpInit)
#define XAI_CNN_REDUCE_GET_ACCSHIFT(x)                   ((x)->accShiftU)
#define XAI_CNN_REDUCE_GET_OUTPUTSHIFT(x)                ((x)->outShiftU)
#define XAI_CNN_REDUCE_GET_OUTPUTSCALE(x)                ((x)->outScale)
#define XAI_CNN_REDUCE_GET_FLAG_RELU(x)                  ((x)->enableReLu)
#define XAI_CNN_REDUCE_GET_RELU_MIN(x)                   ((x)->reluMin)
#define XAI_CNN_REDUCE_GET_RELU_MAX(x)                   ((x)->reluMax)
#define XAI_CNN_REDUCE_GET_TAKEABS(x)                    ((x)->take_abs)
#define XAI_CNN_REDUCE_GET_REDUCED_ELEMENTS_COUNT(x)     ((x)->redEleCount)

#define XAI_CNN_REDUCE_SET_CONFIG(x, v)                  ((x)->config = v)
#define XAI_CNN_REDUCE_SET_TILEFLAG(x, v)                ((x)->tileFlag = v)
#define XAI_CNN_REDUCE_SET_FIXUPINIT(x, v)               ((x)->fixUpInit = v)
#define XAI_CNN_REDUCE_SET_ACCSHIFT(x, v)                ((x)->accShiftU = v)
#define XAI_CNN_REDUCE_SET_OUTPUTSHIFT(x, v)             ((x)->outShiftU = v)
#define XAI_CNN_REDUCE_SET_OUTPUTSCALE(x, v)             ((x)->outScale = v)
#define XAI_CNN_REDUCE_SET_FLAG_RELU(x, v)               ((x)->enableReLu = v)
#define XAI_CNN_REDUCE_SET_RELU_MIN(x, v)                ((x)->reluMin = v)
#define XAI_CNN_REDUCE_SET_RELU_MAX(x, v)                ((x)->reluMax = v)
#define XAI_CNN_REDUCE_SET_TAKEABS(x, v)                 ((x)->take_abs = v)
#define XAI_CNN_REDUCE_SET_REDUCED_ELEMENTS_COUNT(x, v)  ((x)->redEleCount = v)

#define XAI_CNN_REDUCE_DIM1               (0x1)
#define XAI_CNN_REDUCE_DIM2               (0x2)
#define XAI_CNN_REDUCE_DIM3               (0x4)
#define XAI_CNN_REDUCE_DIM4               (0x8)

#define XAI_CNN_REDUCE_DIM12              (XAI_CNN_REDUCE_DIM1 | XAI_CNN_REDUCE_DIM2)
#define XAI_CNN_REDUCE_DIM13              (XAI_CNN_REDUCE_DIM1 | XAI_CNN_REDUCE_DIM3)
#define XAI_CNN_REDUCE_DIM14              (XAI_CNN_REDUCE_DIM1 | XAI_CNN_REDUCE_DIM4)
#define XAI_CNN_REDUCE_DIM23              (XAI_CNN_REDUCE_DIM2 | XAI_CNN_REDUCE_DIM3)
#define XAI_CNN_REDUCE_DIM24              (XAI_CNN_REDUCE_DIM2 | XAI_CNN_REDUCE_DIM4)

#define XAI_CNN_REDUCE_DIM34              (XAI_CNN_REDUCE_DIM3 | XAI_CNN_REDUCE_DIM4)
#define XAI_CNN_REDUCE_DIM123             (XAI_CNN_REDUCE_DIM1 | XAI_CNN_REDUCE_DIM2 | XAI_CNN_REDUCE_DIM3)
#define XAI_CNN_REDUCE_DIM124             (XAI_CNN_REDUCE_DIM1 | XAI_CNN_REDUCE_DIM2 | XAI_CNN_REDUCE_DIM4)
#define XAI_CNN_REDUCE_DIM134             (XAI_CNN_REDUCE_DIM1 | XAI_CNN_REDUCE_DIM3 | XAI_CNN_REDUCE_DIM4)

#define XAI_CNN_REDUCE_DIM234             (XAI_CNN_REDUCE_DIM2 | XAI_CNN_REDUCE_DIM3 | XAI_CNN_REDUCE_DIM4)
#define XAI_CNN_REDUCE_DIM1234            (XAI_CNN_REDUCE_DIM1 | XAI_CNN_REDUCE_DIM2 | XAI_CNN_REDUCE_DIM3 | XAI_CNN_REDUCE_DIM4)

#define XAI_CNN_REDUCE_INTERMEDIATE_TILE  0
#define XAI_CNN_REDUCE_FIRST_TILE         1
#define XAI_CNN_REDUCE_LAST_TILE          2
#define XAI_CNN_REDUCE_FIRST_LAST_TILE    3

/* Matrix Multiplication Params */
typedef struct
{
  uint8_t  accumShift;                   // Accumulator Shift - Shift to convert accumulator data to 16 bit
  uint16_t outputScale;                  // Amount by which shifted data is scaled
  uint8_t  outputShift;                  // Shift amount to convert the scaled data to 16 bit
  int8_t   zeroPointIn1;                 // zero point for assymetric input1 data
  int8_t   zeroPointIn2;                 // zero point for assymetric input2 data
} xai_cnn_matmul_params;

#define XAI_CNN_MATMUL_GET_ACCUM_SHIFT(x)      ((x)->accumShift)
#define XAI_CNN_MATMUL_SET_ACCUM_SHIFT(x, v)   ((x)->accumShift = (v))
#define XAI_CNN_MATMUL_GET_OUTPUT_SCALE(x)     ((x)->outputScale)
#define XAI_CNN_MATMUL_SET_OUTPUT_SCALE(x, v)  ((x)->outputScale = (v))
#define XAI_CNN_MATMUL_GET_OUTPUT_SHIFT(x)     ((x)->outputShift)
#define XAI_CNN_MATMUL_SET_OUTPUT_SHIFT(x, v)  ((x)->outputShift = (v))
#define XAI_CNN_MATMUL_GET_ZERO_POINT1(x)      ((x)->zeroPointIn1)
#define XAI_CNN_MATMUL_SET_ZERO_POINT1(x, v)   ((x)->zeroPointIn1 = (v))
#define XAI_CNN_MATMUL_GET_ZERO_POINT2(x)      ((x)->zeroPointIn2)
#define XAI_CNN_MATMUL_SET_ZERO_POINT2(x, v)   ((x)->zeroPointIn2 = (v))

/* Matrix Multiplication TFL Params */
typedef struct
{
  int32_t outputScale;
  int32_t outputShift;
  int32_t lhsTranspose;  // Can be 0 or 1
  int32_t rhsTranspose;  // Can be 0 or 1
  int32_t lhsOffset;
  int32_t rhsOffset;
  int32_t outOffset;
  int32_t lhsBatch0;
  int32_t lhsBatch1;
  int32_t lhsBatch2;
  int32_t rhsBatch0;
  int32_t rhsBatch1;
  int32_t rhsBatch2;
  int32_t outBatch0;
  int32_t outBatch1;
  int32_t outBatch2;
  int8_t  quantization_mode;
} xai_cnn_tfl_matmul_params;

#define XAI_CNN_MATMUL_GET_OUTPUT_SCALE_TFL(x)      ((x)->outputScale)
#define XAI_CNN_MATMUL_SET_OUTPUT_SCALE_TFL(x, v)   ((x)->outputScale = (v))
#define XAI_CNN_MATMUL_GET_OUTPUT_SHIFT_TFL(x)      ((x)->outputShift)
#define XAI_CNN_MATMUL_SET_OUTPUT_SHIFT_TFL(x, v)   ((x)->outputShift = (v))
#define XAI_CNN_MATMUL_GET_LHS_TRANSPOSE_TFL(x)     ((x)->lhsTranspose)
#define XAI_CNN_MATMUL_SET_LHS_TRANSPOSE_TFL(x, v)  ((x)->lhsTranspose = (v))
#define XAI_CNN_MATMUL_GET_RHS_TRANSPOSE_TFL(x)     ((x)->rhsTranspose)
#define XAI_CNN_MATMUL_SET_RHS_TRANSPOSE_TFL(x, v)  ((x)->rhsTranspose = (v))
#define XAI_CNN_MATMUL_GET_LHS_OFFSET_TFL(x)        ((x)->lhsOffset)
#define XAI_CNN_MATMUL_SET_LHS_OFFSET_TFL(x, v)     ((x)->lhsOffset = (v))
#define XAI_CNN_MATMUL_GET_RHS_OFFSET_TFL(x)        ((x)->rhsOffset)
#define XAI_CNN_MATMUL_SET_RHS_OFFSET_TFL(x, v)     ((x)->rhsOffset = (v))
#define XAI_CNN_MATMUL_GET_OUT_OFFSET_TFL(x)        ((x)->outOffset)
#define XAI_CNN_MATMUL_SET_OUT_OFFSET_TFL(x, v)     ((x)->outOffset = (v))
#define XAI_CNN_MATMUL_GET_LHS_BATCH0_TFL(x)        ((x)->lhsBatch0)
#define XAI_CNN_MATMUL_SET_LHS_BATCH0_TFL(x, v)     ((x)->lhsBatch0 = (v))
#define XAI_CNN_MATMUL_GET_LHS_BATCH1_TFL(x)        ((x)->lhsBatch1)
#define XAI_CNN_MATMUL_SET_LHS_BATCH1_TFL(x, v)     ((x)->lhsBatch1 = (v))
#define XAI_CNN_MATMUL_GET_LHS_BATCH2_TFL(x)        ((x)->lhsBatch2)
#define XAI_CNN_MATMUL_SET_LHS_BATCH2_TFL(x, v)     ((x)->lhsBatch2 = (v))
#define XAI_CNN_MATMUL_GET_RHS_BATCH0_TFL(x)        ((x)->rhsBatch0)
#define XAI_CNN_MATMUL_SET_RHS_BATCH0_TFL(x, v)     ((x)->rhsBatch0 = (v))
#define XAI_CNN_MATMUL_GET_RHS_BATCH1_TFL(x)        ((x)->rhsBatch1)
#define XAI_CNN_MATMUL_SET_RHS_BATCH1_TFL(x, v)     ((x)->rhsBatch1 = (v))
#define XAI_CNN_MATMUL_GET_RHS_BATCH2_TFL(x)        ((x)->rhsBatch2)
#define XAI_CNN_MATMUL_SET_RHS_BATCH2_TFL(x, v)     ((x)->rhsBatch2 = (v))
#define XAI_CNN_MATMUL_GET_OUT_BATCH0_TFL(x)        ((x)->outBatch0)
#define XAI_CNN_MATMUL_SET_OUT_BATCH0_TFL(x, v)     ((x)->outBatch0 = (v))
#define XAI_CNN_MATMUL_GET_OUT_BATCH1_TFL(x)        ((x)->outBatch1)
#define XAI_CNN_MATMUL_SET_OUT_BATCH1_TFL(x, v)     ((x)->outBatch1 = (v))
#define XAI_CNN_MATMUL_GET_OUT_BATCH2_TFL(x)        ((x)->outBatch2)
#define XAI_CNN_MATMUL_SET_OUT_BATCH2_TFL(x, v)     ((x)->outBatch2 = (v))
#define XAI_CNN_MATMUL_GET_QUANTIZATION_MODE(x)     ((x)->quantization_mode)
#define XAI_CNN_MATMUL_SET_QUANTIZATION_MODE(x, v)  ((x)->quantization_mode = (v))

/*Crop3DWithStride Params*/
typedef struct
{
  int32_t offsH;
  int32_t offsW;
  int32_t offsD;
  int32_t strideH;
  int32_t strideW;
  int32_t strideD;
} xai_cnn_crop3DWithStride_params;

#define XAI_CNN_CROP3DWITHSTRIDE_GET_OFFSD(x)       ((x)->offsD);
#define XAI_CNN_CROP3DWITHSTRIDE_GET_OFFSW(x)       ((x)->offsW);
#define XAI_CNN_CROP3DWITHSTRIDE_GET_OFFSH(x)       ((x)->offsH);
#define XAI_CNN_CROP3DWITHSTRIDE_GET_STRIDED(x)     ((x)->strideD);
#define XAI_CNN_CROP3DWITHSTRIDE_GET_STRIDEW(x)     ((x)->strideW);
#define XAI_CNN_CROP3DWITHSTRIDE_GET_STRIDEH(x)     ((x)->strideH);
#define XAI_CNN_CROP3DWITHSTRIDE_SET_OFFSD(x, v)    ((x)->offsD = (v))
#define XAI_CNN_CROP3DWITHSTRIDE_SET_OFFSW(x, v)    ((x)->offsW = (v))
#define XAI_CNN_CROP3DWITHSTRIDE_SET_OFFSH(x, v)    ((x)->offsH = (v))
#define XAI_CNN_CROP3DWITHSTRIDE_SET_STRIDED(x, v)  ((x)->strideD = (v))
#define XAI_CNN_CROP3DWITHSTRIDE_SET_STRIDEW(x, v)  ((x)->strideW = (v))
#define XAI_CNN_CROP3DWITHSTRIDE_SET_STRIDEH(x, v)  ((x)->strideH = (v))

typedef struct
{
  float   scale;
  int32_t offset;
  int32_t axis;
} xai_cnn_quantDequantA_params;
#define XAI_CNN_QUANT_DEQUANT_GET_SCALE(x)      ((x)->scale)
#define XAI_CNN_QUANT_DEQUANT_SET_SCALE(x, v)   ((x)->scale = (v))
#define XAI_CNN_QUANT_DEQUANT_GET_OFFSET(x)     ((x)->offset)
#define XAI_CNN_QUANT_DEQUANT_SET_OFFSET(x, v)  ((x)->offset = (v))
#define XAI_CNN_QUANT_DEQUANT_GET_AXIS(x)       ((x)->axis)
#define XAI_CNN_QUANT_DEQUANT_SET_AXIS(x, v)    ((x)->axis = (v))

typedef struct
{
  xai_cnn_conv_params        fcInputParamIG;
  xai_cnn_conv_params        fcInputParamFG;
  xai_cnn_conv_params        fcInputParamOG;
  xai_cnn_conv_params        fcInputParamMI;
  xai_cnn_conv_params        fcHiddenParamIG;
  xai_cnn_conv_params        fcHiddenParamFG;
  xai_cnn_conv_params        fcHiddenParamOG;
  xai_cnn_conv_params        fcHiddenParamMI;

  xai_cnn_sigmoid_params     sigmoidParamIG;
  xai_cnn_sigmoid_params     sigmoidParamFG;
  xai_cnn_sigmoid_params     sigmoidParamOG;
  xai_cnn_tanh_params        tanhParamMI;

  xai_cnn_tfl_eltwise_params eltMulParamHS1;
  xai_cnn_tfl_eltwise_params eltMulParamHS2;

  xai_cnn_tanh_params        tanhParamCS;
  xai_cnn_tfl_eltwise_params eltMulParamCS;

  int16_t                    clipMin;
  int16_t                    clipMax;

  int32_t                    timeMajorAxis;
  int32_t                    direction;
} xai_lstm_tfl_params;

#define XAI_CNN_LSTM_GET_FC_INPUT_IG_PARAM(x)         ((x)->fcInputParamIG)
#define XAI_CNN_LSTM_SET_FC_INPUT_IG_PARAM(x, v)      ((x)->fcInputParamIG = (v))
#define XAI_CNN_LSTM_GET_FC_INPUT_FG_PARAM(x)         ((x)->fcInputParamFG)
#define XAI_CNN_LSTM_SET_FC_INPUT_FG_PARAM(x, v)      ((x)->fcInputParamFG = (v))
#define XAI_CNN_LSTM_GET_FC_INPUT_OG_PARAM(x)         ((x)->fcInputParamOG)
#define XAI_CNN_LSTM_SET_FC_INPUT_OG_PARAM(x, v)      ((x)->fcInputParamOG = (v))
#define XAI_CNN_LSTM_GET_FC_INPUT_MI_PARAM(x)         ((x)->fcInputParamMI)
#define XAI_CNN_LSTM_SET_FC_INPUT_MI_PARAM(x, v)      ((x)->fcInputParamMI = (v))

#define XAI_CNN_LSTM_GET_FC_HIDDEN_IG_PARAM(x)        ((x)->fcHiddenParamIG)
#define XAI_CNN_LSTM_SET_FC_HIDDEN_IG_PARAM(x, v)     ((x)->fcHiddenParamIG = (v))
#define XAI_CNN_LSTM_GET_FC_HIDDEN_FG_PARAM(x)        ((x)->fcHiddenParamFG)
#define XAI_CNN_LSTM_SET_FC_HIDDEN_FG_PARAM(x, v)     ((x)->fcHiddenParamFG = (v))
#define XAI_CNN_LSTM_GET_FC_HIDDEN_OG_PARAM(x)        ((x)->fcHiddenParamOG)
#define XAI_CNN_LSTM_SET_FC_HIDDEN_OG_PARAM(x, v)     ((x)->fcHiddenParamOG = (v))
#define XAI_CNN_LSTM_GET_FC_HIDDEN_MI_PARAM(x)        ((x)->fcHiddenParamMI)
#define XAI_CNN_LSTM_SET_FC_HIDDEN_MI_PARAM(x, v)     ((x)->fcHiddenParamMI = (v))

#define XAI_CNN_LSTM_GET_SIGMOID_IG_PARAM(x)          ((x)->sigmoidParamIG)
#define XAI_CNN_LSTM_SET_SIGMOID_IG_PARAM(x, v)       ((x)->sigmoidParamIG = (v))
#define XAI_CNN_LSTM_GET_SIGMOID_FG_PARAM(x)          ((x)->sigmoidParamFG)
#define XAI_CNN_LSTM_SET_SIGMOID_FG_PARAM(x, v)       ((x)->sigmoidParamFG = (v))
#define XAI_CNN_LSTM_GET_SIGMOID_OG_PARAM(x)          ((x)->sigmoidParamOG)
#define XAI_CNN_LSTM_SET_SIGMOID_OG_PARAM(x, v)       ((x)->sigmoidParamOG = (v))
#define XAI_CNN_LSTM_GET_TANH_MI_PARAM(x)             ((x)->tanhParamMI)
#define XAI_CNN_LSTM_SET_TANH_MI_PARAM(x, v)          ((x)->tanhParamMI = (v))

#define XAI_CNN_LSTM_GET_ELTWISE_MUL_HS1_PARAM(x)     ((x)->eltMulParamHS1)
#define XAI_CNN_LSTM_SET_ELTWISE_MUL_HS1_PARAM(x, v)  ((x)->eltMulParamHS1 = (v))
#define XAI_CNN_LSTM_GET_ELTWISE_MUL_HS2_PARAM(x)     ((x)->eltMulParamHS2)
#define XAI_CNN_LSTM_SET_ELTWISE_MUL_HS2_PARAM(x, v)  ((x)->eltMulParamHS2 = (v))

#define XAI_CNN_LSTM_GET_TANH_CS_PARAM(x)             ((x)->tanhParamCS)
#define XAI_CNN_LSTM_SET_TANH_CS_PARAM(x, v)          ((x)->tanhParamCS = (v))
#define XAI_CNN_LSTM_GET_ELTWISE_MUL_CS_PARAM(x)      ((x)->eltMulParamCS)
#define XAI_CNN_LSTM_SET_ELTWISE_MUL_CS_PARAM(x, v)   ((x)->eltMulParamCS = (v))

#define XAI_CNN_LSTM_GET_CLIP_MIN(x)                  ((x)->clipMin)
#define XAI_CNN_LSTM_SET_CLIP_MIN(x, v)               ((x)->clipMin = (v))
#define XAI_CNN_LSTM_GET_CLIP_MAX(x)                  ((x)->clipMax)
#define XAI_CNN_LSTM_SET_CLIP_MAX(x, v)               ((x)->clipMax = (v))

#define XAI_CNN_LSTM_GET_TIME_MAJOR_AXIS(x)           ((x)->timeMajorAxis)
#define XAI_CNN_LSTM_SET_TIME_MAJOR_AXIS(x, v)        ((x)->timeMajorAxis = (v))
#define XAI_CNN_LSTM_GET_DIRECTION(x)                 ((x)->direction)
#define XAI_CNN_LSTM_SET_DIRECTION(x, v)              ((x)->direction = (v))

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
typedef struct
{
  xai_cnn_conv_params       fcInputParamIG;
  xai_cnn_conv_params       fcInputParamFG;
  xai_cnn_conv_params       fcInputParamOG;
  xai_cnn_conv_params       fcInputParamMI;
  xai_cnn_conv_params       fcHiddenParamIG;
  xai_cnn_conv_params       fcHiddenParamFG;
  xai_cnn_conv_params       fcHiddenParamOG;
  xai_cnn_conv_params       fcHiddenParamMI;

  xai_cnn_eltwiseMul_params eltMulParamHS1;
  xai_cnn_eltwiseMul_params eltMulParamHS2;

  xai_cnn_eltwiseMul_params eltMulParamCS;

  xb_f16                    clipMinFP16;
  xb_f16                    clipMaxFP16;

  int32_t                   timeMajorAxis;
  int32_t                   direction;
} xai_lstm_F16_params;

#define XAI_CNN_LSTM_F16_GET_FC_INPUT_IG_PARAM(x)         ((x)->fcInputParamIG)
#define XAI_CNN_LSTM_F16_SET_FC_INPUT_IG_PARAM(x, v)      ((x)->fcInputParamIG = (v))
#define XAI_CNN_LSTM_F16_GET_FC_INPUT_FG_PARAM(x)         ((x)->fcInputParamFG)
#define XAI_CNN_LSTM_F16_SET_FC_INPUT_FG_PARAM(x, v)      ((x)->fcInputParamFG = (v))
#define XAI_CNN_LSTM_F16_GET_FC_INPUT_OG_PARAM(x)         ((x)->fcInputParamOG)
#define XAI_CNN_LSTM_F16_SET_FC_INPUT_OG_PARAM(x, v)      ((x)->fcInputParamOG = (v))
#define XAI_CNN_LSTM_F16_GET_FC_INPUT_MI_PARAM(x)         ((x)->fcInputParamMI)
#define XAI_CNN_LSTM_F16_SET_FC_INPUT_MI_PARAM(x, v)      ((x)->fcInputParamMI = (v))

#define XAI_CNN_LSTM_F16_GET_FC_HIDDEN_IG_PARAM(x)        ((x)->fcHiddenParamIG)
#define XAI_CNN_LSTM_F16_SET_FC_HIDDEN_IG_PARAM(x, v)     ((x)->fcHiddenParamIG = (v))
#define XAI_CNN_LSTM_F16_GET_FC_HIDDEN_FG_PARAM(x)        ((x)->fcHiddenParamFG)
#define XAI_CNN_LSTM_F16_SET_FC_HIDDEN_FG_PARAM(x, v)     ((x)->fcHiddenParamFG = (v))
#define XAI_CNN_LSTM_F16_GET_FC_HIDDEN_OG_PARAM(x)        ((x)->fcHiddenParamOG)
#define XAI_CNN_LSTM_F16_SET_FC_HIDDEN_OG_PARAM(x, v)     ((x)->fcHiddenParamOG = (v))
#define XAI_CNN_LSTM_F16_GET_FC_HIDDEN_MI_PARAM(x)        ((x)->fcHiddenParamMI)
#define XAI_CNN_LSTM_F16_SET_FC_HIDDEN_MI_PARAM(x, v)     ((x)->fcHiddenParamMI = (v))

#define XAI_CNN_LSTM_F16_GET_ELTWISE_MUL_HS1_PARAM(x)     ((x)->eltMulParamHS1)
#define XAI_CNN_LSTM_F16_SET_ELTWISE_MUL_HS1_PARAM(x, v)  ((x)->eltMulParamHS1 = (v))
#define XAI_CNN_LSTM_F16_GET_ELTWISE_MUL_HS2_PARAM(x)     ((x)->eltMulParamHS2)
#define XAI_CNN_LSTM_F16_SET_ELTWISE_MUL_HS2_PARAM(x, v)  ((x)->eltMulParamHS2 = (v))

#define XAI_CNN_LSTM_F16_GET_ELTWISE_MUL_CS_PARAM(x)      ((x)->eltMulParamCS)
#define XAI_CNN_LSTM_F16_SET_ELTWISE_MUL_CS_PARAM(x, v)   ((x)->eltMulParamCS = (v))

#define XAI_CNN_LSTM_F16_GET_CLIP_MIN(x)                  ((x)->clipMinFP16)
#define XAI_CNN_LSTM_F16_SET_CLIP_MIN(x, v)               ((x)->clipMinFP16 = (v))
#define XAI_CNN_LSTM_F16_GET_CLIP_MAX(x)                  ((x)->clipMaxFP16)
#define XAI_CNN_LSTM_F16_SET_CLIP_MAX(x, v)               ((x)->clipMaxFP16 = (v))

#define XAI_CNN_LSTM_F16_GET_TIME_MAJOR_AXIS(x)           ((x)->timeMajorAxis)
#define XAI_CNN_LSTM_F16_SET_TIME_MAJOR_AXIS(x, v)        ((x)->timeMajorAxis = (v))
#define XAI_CNN_LSTM_F16_GET_DIRECTION(x)                 ((x)->direction)
#define XAI_CNN_LSTM_F16_SET_DIRECTION(x, v)              ((x)->direction = (v))
#endif

typedef struct
{
  xai_cnn_conv_params        fcInputParamRG;
  xai_cnn_conv_params        fcInputParamUG;
  xai_cnn_conv_params        fcInputParamMS;
  xai_cnn_conv_params        fcHiddenParamRG;
  xai_cnn_conv_params        fcHiddenParamUG;
  xai_cnn_conv_params        fcHiddenParamMS;

  xai_cnn_sigmoid_params     sigmoidParamRG;
  xai_cnn_sigmoid_params     sigmoidParamUG;
  xai_cnn_tfl_eltwise_params eltMulParamMS;
  xai_cnn_tanh_params        tanhParamMS;

  xai_cnn_tfl_eltwise_params eltMulParamHS1;
  xai_cnn_tfl_eltwise_params eltMulParamHS2;

  int32_t                    eltAddOutOffsetHS; // NOTE: eltAddOutOffsetHS is not used in S16 variant. For S16 variant, set it to 0.
  int32_t                    timeMajorAxis;
  int32_t                    direction;
} xai_gru_tfl_params;

#define XAI_CNN_GRU_GET_FC_INPUT_RG_PARAM(x)             ((x)->fcInputParamRG)
#define XAI_CNN_GRU_SET_FC_INPUT_RG_PARAM(x, v)          ((x)->fcInputParamRG = (v))
#define XAI_CNN_GRU_GET_FC_INPUT_UG_PARAM(x)             ((x)->fcInputParamUG)
#define XAI_CNN_GRU_SET_FC_INPUT_UG_PARAM(x, v)          ((x)->fcInputParamUG = (v))
#define XAI_CNN_GRU_GET_FC_INPUT_MS_PARAM(x)             ((x)->fcInputParamMS)
#define XAI_CNN_GRU_SET_FC_INPUT_MS_PARAM(x, v)          ((x)->fcInputParamMS = (v))

#define XAI_CNN_GRU_GET_FC_HIDDEN_RG_PARAM(x)            ((x)->fcHiddenParamRG)
#define XAI_CNN_GRU_SET_FC_HIDDEN_RG_PARAM(x, v)         ((x)->fcHiddenParamRG = (v))
#define XAI_CNN_GRU_GET_FC_HIDDEN_UG_PARAM(x)            ((x)->fcHiddenParamUG)
#define XAI_CNN_GRU_SET_FC_HIDDEN_UG_PARAM(x, v)         ((x)->fcHiddenParamUG = (v))
#define XAI_CNN_GRU_GET_FC_HIDDEN_MS_PARAM(x)            ((x)->fcHiddenParamMS)
#define XAI_CNN_GRU_SET_FC_HIDDEN_MS_PARAM(x, v)         ((x)->fcHiddenParamMS = (v))

#define XAI_CNN_GRU_GET_SIGMOID_RG_PARAM(x)              ((x)->sigmoidParamRG)
#define XAI_CNN_GRU_SET_SIGMOID_RG_PARAM(x, v)           ((x)->sigmoidParamRG = (v))
#define XAI_CNN_GRU_GET_SIGMOID_UG_PARAM(x)              ((x)->sigmoidParamUG)
#define XAI_CNN_GRU_SET_SIGMOID_UG_PARAM(x, v)           ((x)->sigmoidParamUG = (v))
#define XAI_CNN_GRU_GET_ELTWISE_MUL_MS_PARAM(x)          ((x)->eltMulParamMS)
#define XAI_CNN_GRU_SET_ELTWISE_MUL_MS_PARAM(x, v)       ((x)->eltMulParamMS = (v))
#define XAI_CNN_GRU_GET_TANH_MS_PARAM(x)                 ((x)->tanhParamMS)
#define XAI_CNN_GRU_SET_TANH_MS_PARAM(x, v)              ((x)->tanhParamMS = (v))

#define XAI_CNN_GRU_GET_ELTWISE_MUL_HS1_PARAM(x)         ((x)->eltMulParamHS1)
#define XAI_CNN_GRU_SET_ELTWISE_MUL_HS1_PARAM(x, v)      ((x)->eltMulParamHS1 = (v))
#define XAI_CNN_GRU_GET_ELTWISE_MUL_HS2_PARAM(x)         ((x)->eltMulParamHS2)
#define XAI_CNN_GRU_SET_ELTWISE_MUL_HS2_PARAM(x, v)      ((x)->eltMulParamHS2 = (v))

#define XAI_CNN_GRU_GET_ELTWISE_ADD_HS_OUT_OFFSET(x)     ((x)->eltAddOutOffsetHS)
#define XAI_CNN_GRU_SET_ELTWISE_ADD_HS_OUT_OFFSET(x, v)  ((x)->eltAddOutOffsetHS = (v))
#define XAI_CNN_GRU_GET_TIME_MAJOR_AXIS(x)               ((x)->timeMajorAxis)
#define XAI_CNN_GRU_SET_TIME_MAJOR_AXIS(x, v)            ((x)->timeMajorAxis = (v))
#define XAI_CNN_GRU_GET_DIRECTION(x)                     ((x)->direction)
#define XAI_CNN_GRU_SET_DIRECTION(x, v)                  ((x)->direction = (v))
#endif // #ifndef __XAI_CNN_API_PARAMS_H__
