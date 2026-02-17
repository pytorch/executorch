/*
 * Copyright (c) 2021 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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

#ifndef __XAI_CNN_COMMON_H__
#define __XAI_CNN_COMMON_H__

#include "xai_tile_manager.h"
#include "xai_core.h"
#include "xai_cnn_api_common.h"
#include "limits.h"

// frequently used macros for rounding and clamping
#ifndef MAX2
#define MAX2(a, b)  (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN2
#define MIN2(a, b)                        (((a) > (b)) ? (b) : (a))
#endif
#define CLAMP(v, min, max)                ((v) < (min) ? (min) : (v) > (max) ? (max) : (v))
#define ROUND(x, s)                       (((s) == 0) ? (x) : (((x) + (1 << ((s) - 1))) >> (s)))
#define ROUND_N_CLAMP(x, s, min, max)     (((s) == 0) ? (CLAMP(x, min, max)) : (CLAMP(ROUND(x, s), min, max)))
#define ROUND64B(x, s)                    (((s) == 0) ? (x) : \
                                           (((x) + ((int64_t) 1 << ((s) - 1))) >> (s)))
#define ROUND_N_CLAMP64B(x, s, min, max)  (((s) == 0) ? (CLAMP(x, min, max)) : \
                                           (CLAMP(ROUND64B(x, s), min, max)))
#define ROI_CEIL(x, s)                    (((s) == 0) ? (x) : (((x) + (1 << ((s)))) >> (s)))

#ifndef XCHAL_IVPN_SIMD_WIDTH
#define XCHAL_IVPN_SIMD_WIDTH  32
#endif

/* Macros used for morphing various APIs */
#define SIGNED8BIT                  1
#define UNSIGNED8BIT                2
#define SIGNED16BIT                 3
#define UNSIGNED16BIT               4
#define SIGNED32BIT                 5
#define INTEGER8BIT                 6
#define INTEGER16BIT                7
#define FLOAT16BIT                  8
#define FLOAT32BIT                  9
#define SIGNED8BITUNSIGNED8BIT      10
#define UNSIGNED8BITSIGNED8BIT      11
#define SIGNED8BITSIGNED16BIT       12
#define UNSIGNED8BITSIGNED16BIT     13
#define SIGNED16BITSIGNED16BIT      14
#define UNSIGNED32BIT               16
#define INPUT16BITFLOAT             17
#define INPUT8BIT                   18
#define INPUT16BIT                  19
#define INPUT32BIT                  20
#define SIGNED64BIT                 21
#define UNSIGNED64BIT               22

#define QP_DEPTH_U8                 ((uint8_t) UCHAR_MAX)
#define QP_DEPTH_U16                ((uint16_t) USHRT_MAX)
#define QP_DEPTH_S16                ((int16_t) SHRT_MAX)
#define QP_DEPTH_S8                 ((uint8_t) SCHAR_MAX)

#define ADAPTIVE_AVG_POOL_Q_FORMAT  15

#define CALC_NSA_32(input, count)                                           \
    {                                                                       \
      count = 0;                                                            \
      int32_t mask  = 0x80000000;                                           \
      int32_t index = 31;                                                   \
      /*Determining the sign of the input*/                                 \
      int32_t sign = (input & mask) >> index & 0x00000001;                  \
      mask = 0x40000000;                                                    \
      index--;                                                              \
      /*Finding the count leading zeros incase of positive number           \
         and count leading ones in case of negative number excluding        \
         the sign bit*/                                                     \
      while ((sign == ((input & mask) >> index)) && (mask != 0))            \
      {                                                                     \
        count += 1;                                                         \
        mask   = mask >> 1;                                                 \
        index--;                                                            \
      }                                                                     \
    }

#define CONVERT_FP16_TO_FP32(F16Data)  (                       \
    {                                                          \
      int signBit, scaleSign, storedExponent;                  \
      int trueExponent;                                        \
      int significand, i;                                      \
      float expVal, bitVal, temp, fractionFloat;               \
      float implicitSignificand_val;                           \
                                                               \
      trueExponent = 0;                                        \
      implicitSignificand_val = 0;                             \
      float floatVal = 0;                                      \
                                                               \
      unsigned short F16Data_U16 = (unsigned short) F16Data;   \
      int hex_val_fp16 = (int) F16Data_U16;                    \
                                                               \
      signBit = (hex_val_fp16 >> 15);                          \
      scaleSign = ((signBit == 0) ? (1) : (-1));               \
      storedExponent = ((hex_val_fp16 & 0x7fff) >> 10);        \
      significand = (hex_val_fp16 & 0x03ff);                   \
                                                               \
      if (storedExponent == 31)                                \
      {                                                        \
        if (scaleSign == 1)                                    \
        {                                                      \
          if (significand == 0)                                \
          {                                                    \
            floatVal = +INFINITY;                              \
            return (floatVal);                                 \
          }                                                    \
          else if (significand != 0)                           \
          {                                                    \
            floatVal = -NAN; /* +nan */                        \
            return (floatVal);                                 \
          }                                                    \
        }                                                      \
        else if (scaleSign == -1)                              \
        {                                                      \
          if (significand == 0)                                \
          {                                                    \
            floatVal = -INFINITY;                              \
            return (floatVal);                                 \
          }                                                    \
          else if (significand != 0)                           \
          {                                                    \
            floatVal = NAN; /* -nan */                         \
            return (floatVal);                                 \
          }                                                    \
        }                                                      \
      }                                                        \
      else if (storedExponent == 0)                            \
      {                                                        \
        trueExponent = -14;                                    \
        implicitSignificand_val = 0.0f;                        \
                                                               \
        if (scaleSign == 1)                                    \
        {                                                      \
          if (significand == 0)                                \
          {                                                    \
            floatVal = 0;                                      \
            return (floatVal);                                 \
          }                                                    \
        }                                                      \
        else if (scaleSign == -1)                              \
        {                                                      \
          if (significand == 0)                                \
          {                                                    \
            floatVal = -0;                                     \
            return (floatVal);                                 \
          }                                                    \
        }                                                      \
      }                                                        \
      else if ((storedExponent > 0) && (storedExponent < 31))  \
      {                                                        \
        trueExponent = storedExponent - 15;                    \
        implicitSignificand_val = 1.0f;                        \
      }                                                        \
                                                               \
      expVal = powf(2, (float) trueExponent);                  \
                                                               \
      fractionFloat = 0.0f;                                    \
      for (i = 10; i > 0; i--)                                 \
      {                                                        \
        bitVal = (float) (significand & 0x1);                  \
        temp = bitVal / (1 << i);                              \
        fractionFloat = fractionFloat + temp;                  \
                                                               \
        significand = significand >> 1;                        \
      }                                                        \
      fractionFloat = fractionFloat + implicitSignificand_val; \
                                                               \
      scaleSign * expVal * fractionFloat;                      \
    })

#define XAI_CHECK_TILE3D_EDGE(tile, edge)                                                                    \
  if (XAI_TILE3D_GET_DATA_ORDER(tile) == XAI_WHD)                                                            \
  {                                                                                                          \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1_EDGE1(tile) >= edge && XAI_TILE3D_GET_DIM1_EDGE2(tile) >= edge &&    \
                    XAI_TILE3D_GET_DIM2_EDGE1(tile) >= edge && XAI_TILE3D_GET_DIM2_EDGE2(tile) >= edge,      \
                    XAI_ERR_EDGE, "The (" #tile ") tile must have at least " #edge "-pixel edge extension"); \
  }                                                                                                          \
  else if (XAI_TILE3D_GET_DATA_ORDER(tile) == XAI_DWH)                                                       \
  {                                                                                                          \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2_EDGE1(tile) >= edge && XAI_TILE3D_GET_DIM2_EDGE2(tile) >= edge &&    \
                    XAI_TILE3D_GET_DIM3_EDGE1(tile) >= edge && XAI_TILE3D_GET_DIM3_EDGE2(tile) >= edge,      \
                    XAI_ERR_EDGE, "The (" #tile ") tile must have at least " #edge "-pixel edge extension"); \
  }                                                                                                          \

#define XAI_CHECK_TILE3D_EDGE2(tile, edge1, edge2)                                                                   \
  if (XAI_TILE3D_GET_DATA_ORDER(tile) == XAI_WHD)                                                                    \
  {                                                                                                                  \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1_EDGE1(tile) >= edge1 && XAI_TILE3D_GET_DIM1_EDGE2(tile) >= edge1 &&          \
                    XAI_TILE3D_GET_DIM2_EDGE1(tile) >= edge2 && XAI_TILE3D_GET_DIM2_EDGE2(tile) >= edge2,            \
                    XAI_ERR_EDGE, "The (" #tile ") tile must have at least " #edge1 #edge2 "-pixel edge extension"); \
  }                                                                                                                  \
  else if (XAI_TILE3D_GET_DATA_ORDER(tile) == XAI_DWH)                                                               \
  {                                                                                                                  \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2_EDGE1(tile) >= edge1 && XAI_TILE3D_GET_DIM2_EDGE2(tile) >= edge1 &&          \
                    XAI_TILE3D_GET_DIM3_EDGE1(tile) >= edge2 && XAI_TILE3D_GET_DIM3_EDGE2(tile) >= edge2,            \
                    XAI_ERR_EDGE, "The (" #tile ") tile must have at least " #edge1 #edge2 "-pixel edge extension"); \
  }

#define XAI_CHECK_TILE3D_DATA_ORDER(tile, type) \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(tile) == type, XAI_ERR_BADARG, "The Data Order of (" #tile ") is not supported by this function")

#define XAI_CHECK_TILE4D_DATA_ORDER(tile, type) \
  XAI_CHECK_ERROR(XAI_TILE4D_GET_DATA_ORDER(tile) == type, XAI_ERR_BADARG, "The Data Order of (" #tile ") is not supported by this function")

#define XAI_CHECK_KERNEL_SIZE(coeffT, size)                                                         \
  if (XAI_TILE4D_GET_DATA_ORDER(coeffT) == XAI_WHDN)                                                \
  {                                                                                                 \
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffT) == size) && (XAI_TILE4D_GET_DIM2(coeffT) == size), \
                    XAI_ERR_KSIZE, "The Coefficient Kernel Size is not supported");                 \
  }                                                                                                 \
  else if (XAI_TILE4D_GET_DATA_ORDER(coeffT) == XAI_NDWH)                                           \
  {                                                                                                 \
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffT) == size) && (XAI_TILE4D_GET_DIM4(coeffT) == size), \
                    XAI_ERR_KSIZE, "The Coefficient Kernel Size is not supported");                 \
  }

#define XAI_CHECK_CONV_OUTPUT_TILE3D(outTile)                                                          \
  XAI_CHECK_TILE3D(outTile);                                                                           \
  XAI_CHECK_ERROR((XAI_TILE3D_CHECK_TYPE(outTile, XAI_U8)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8)) \
                  || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)),                                        \
                  XAI_ERR_DATATYPE, "The argument (" #outTile ") has wrong type");

#define XAI_CHECK_CONV_I16_OUTPUT_TILE3D(outTile)                                                            \
  XAI_CHECK_TILE3D(outTile);                                                                                 \
  XAI_CHECK_ERROR((XAI_TILE3D_CHECK_TYPE(outTile, XAI_U8)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))       \
                  || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16)), \
                  XAI_ERR_DATATYPE, "The argument (" #outTile ") has wrong type");
#define XAI_CHECK_CONV_OUTPUT_IX_TILE3D(outTile)                                                             \
  XAI_CHECK_TILE3D(outTile);                                                                                 \
  XAI_CHECK_ERROR((XAI_TILE3D_CHECK_TYPE(outTile, XAI_U8)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))       \
                  || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16)), \
                  XAI_ERR_DATATYPE, "The argument (" #outTile ") has wrong type");

#define XAI_CHECK_CONV_OUTPUT_TILE4D(outTile)                                                          \
  XAI_CHECK_TILE4D(outTile);                                                                           \
  XAI_CHECK_ERROR((XAI_TILE4D_CHECK_TYPE(outTile, XAI_U8)) || (XAI_TILE4D_CHECK_TYPE(outTile, XAI_S8)) \
                  || (XAI_TILE4D_CHECK_TYPE(outTile, XAI_S16)),                                        \
                  XAI_ERR_DATATYPE, "The argument (" #outTile ") has wrong type");

#define XAI_CHECK_STRIDE(param, stride) \
  XAI_CHECK_ERROR(XAI_CNN_CONV_GET_STRIDE(param) == stride, XAI_ERR_BADARG, "The stride amount provided is not supported.");

#define XAI_CHECK_DILATION(param, dilation) \
  XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATION(param) == dilation, XAI_ERR_BADARG, "The dilation value provided is not supported.");


#define XAI_CHECK_POOLING_STRIDE(param, stride) \
  XAI_CHECK_ERROR(XAI_CNN_POOLING_GET_STRIDE(param) == stride, XAI_ERR_BADARG, "The stride amount provided is not supported.");

#define XAI_CHECK_CONSISTENCY_MOD_DWH(inT, coeffT, biasArr, outT, param)                                                                       \
  uint16_t dilatedKW_MOD = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1);                             \
  uint16_t dilatedKH_MOD = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM4(coeffT) - 1) + 1);                             \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(inT) == XAI_TILE4D_GET_DIM2(coeffT), XAI_ERR_DATASIZE,                                                   \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                              \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outT) == XAI_TILE4D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                                  \
                  "Number of Output Channels not equal to the number of Kernels.");                                                            \
  if (dilatedKW_MOD % 2 != 0)                                                                                                                  \
  {                                                                                                                                            \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKW_MOD >> 1)                                           \
                                                     + (dilatedKW_MOD >> 1) - dilatedKW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                       \
  }                                                                                                                                            \
  else                                                                                                                                         \
  {                                                                                                                                            \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKW_MOD >> 1)                                           \
                                                     + ((dilatedKW_MOD >> 1) - 1) - dilatedKW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                       \
  }                                                                                                                                            \
  if (dilatedKH_MOD % 2 != 0)                                                                                                                  \
  {                                                                                                                                            \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedKH_MOD >> 1)                                           \
                                                     + (dilatedKH_MOD >> 1) - dilatedKH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent..");                                                     \
  }                                                                                                                                            \
  else                                                                                                                                         \
  {                                                                                                                                            \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedKH_MOD >> 1)                                           \
                                                     + ((dilatedKH_MOD >> 1) - 1) - dilatedKH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent..");                                                     \
  }                                                                                                                                            \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE4D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                               \
                  "Width of Bias Array is less than number of Kernels.");                                                                      \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                                       \
                  "Height of Bias Array should be greater than zero.");

#define XAI_CHECK_CONSISTENCY_MOD_DWH_IN16DWH(inT, offsetArr, coeffT, biasArr, outT, param)                                                                  \
  uint16_t dilatedKW_MOD = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM2(coeffT) - 1) + 1);                                           \
  uint16_t dilatedKH_MOD = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1);                                           \
  XAI_CHECK_ERROR((XAI_ALIGN_VAL(XAI_TILE3D_GET_DIM1(inT), 2 * XCHAL_IVPN_SIMD_WIDTH)) == (XAI_TILE4D_GET_DIM1(coeffT) >> 4), XAI_ERR_DATASIZE,              \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                                            \
  XAI_CHECK_ERROR((XAI_ALIGN_VAL(XAI_TILE3D_GET_DIM1(outT), 2 * XCHAL_IVPN_SIMD_WIDTH)) == (XAI_TILE4D_GET_DIM4(coeffT) << 4), XAI_ERR_DATASIZE,             \
                  "Number of Output Channels not equal to the number of Kernels.");                                                                          \
  if (dilatedKW_MOD % 2 != 0)                                                                                                                                \
  {                                                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKW_MOD >> 1)                                                         \
                                                     + (dilatedKW_MOD >> 1) - dilatedKW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),                     \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                     \
  }                                                                                                                                                          \
  else                                                                                                                                                       \
  {                                                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKW_MOD >> 1)                                                         \
                                                     + ((dilatedKW_MOD >> 1) - 1) - dilatedKW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),               \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                     \
  }                                                                                                                                                          \
  if (dilatedKH_MOD % 2 != 0)                                                                                                                                \
  {                                                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedKH_MOD >> 1)                                                         \
                                                     + (dilatedKH_MOD >> 1) - dilatedKH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                     \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent..");                                                                   \
  }                                                                                                                                                          \
  else                                                                                                                                                       \
  {                                                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedKH_MOD >> 1)                                                         \
                                                     + ((dilatedKH_MOD >> 1) - 1) - dilatedKH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),               \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent..");                                                                   \
  }                                                                                                                                                          \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= (XAI_TILE3D_GET_DIM1(outT)), XAI_ERR_DATASIZE,                                                             \
                  "Width of Bias Array is less than number of Kernels.");                                                                                    \
  XAI_CHECK_ERROR((XAI_ARRAY_GET_WIDTH(offsetArr) >=                                                                                                         \
                   (XAI_TILE4D_GET_DIM2(coeffT) * XAI_TILE4D_GET_DIM3(coeffT) * (XAI_ALIGN_VAL(XAI_TILE3D_GET_DIM1(inT), 2 * XCHAL_IVPN_SIMD_WIDTH) >> 4))), \
                  XAI_ERR_DATASIZE, "Input offset Array size should be equal to kernelHeight * kernelWidth * (ALIGN(InputChannels,16)/16).");                \

#define XAI_CHECK_CONSISTENCY_MOD_WHD_DWH(inT, coeffT, biasArr, outT, param)                                                                   \
  uint16_t dilatedKW_MOD = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1);                             \
  uint16_t dilatedKH_MOD = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM4(coeffT) - 1) + 1);                             \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inT) == XAI_TILE4D_GET_DIM2(coeffT), XAI_ERR_DATASIZE,                                                   \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                              \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outT) == XAI_TILE4D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                                  \
                  "Number of Output Channels not equal to the number of Kernels.");                                                            \
  if (dilatedKW_MOD % 2 != 0)                                                                                                                  \
  {                                                                                                                                            \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (dilatedKW_MOD >> 1)                                           \
                                                     + (dilatedKW_MOD >> 1) - dilatedKW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                       \
  }                                                                                                                                            \
  else                                                                                                                                         \
  {                                                                                                                                            \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (dilatedKW_MOD >> 1)                                           \
                                                     + ((dilatedKW_MOD >> 1) - 1) - dilatedKW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                       \
  }                                                                                                                                            \
  if (dilatedKH_MOD % 2 != 0)                                                                                                                  \
  {                                                                                                                                            \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKH_MOD >> 1)                                           \
                                                     + (dilatedKH_MOD >> 1) - dilatedKH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent..");                                                     \
  }                                                                                                                                            \
  else                                                                                                                                         \
  {                                                                                                                                            \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKH_MOD >> 1)                                           \
                                                     + ((dilatedKH_MOD >> 1) - 1) - dilatedKH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent..");                                                     \
  }                                                                                                                                            \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE4D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                               \
                  "Width of Bias Array is less than number of Kernels.");                                                                      \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                                       \
                  "Height of Bias Array should be greater than zero.");

#define XAI_CHECK_CONSISTENCY_MOW_WHD(inT, coeffT, biasArr, outT, param)                                                                      \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inT) == XAI_TILE4D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                                                  \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                             \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outT) == XAI_TILE4D_GET_DIM4(coeffT), XAI_ERR_DATASIZE,                                                 \
                  "Number of Output Channels not equal to the number of Kernels.");                                                           \
  uint16_t dilatedKW_MOW = (uint16_t) (XAI_CNN_CONV_GET_DILATION(param) * (XAI_TILE4D_GET_DIM1(coeffTile) - 1) + 1);                          \
  uint16_t dilatedKH_MOW = (uint16_t) (XAI_CNN_CONV_GET_DILATION(param) * (XAI_TILE4D_GET_DIM2(coeffTile) - 1) + 1);                          \
  if (dilatedKW_MOW % 2 != 0)                                                                                                                 \
  {                                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (dilatedKW_MOW >> 1)                                          \
                                                     + (dilatedKW_MOW >> 1) - dilatedKW_MOW) / (XAI_CNN_CONV_GET_STRIDE(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                      \
  }                                                                                                                                           \
  else                                                                                                                                        \
  {                                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (dilatedKW_MOW >> 1)                                          \
                                                     + ((dilatedKW_MOW >> 1) - 1) - dilatedKW_MOW) / (XAI_CNN_CONV_GET_STRIDE(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                      \
  }                                                                                                                                           \
  if (dilatedKH_MOW % 2 != 0)                                                                                                                 \
  {                                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKH_MOW >> 1)                                          \
                                                     + (dilatedKH_MOW >> 1) - dilatedKH_MOW) / (XAI_CNN_CONV_GET_STRIDE(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent..");                                                    \
  }                                                                                                                                           \
  else                                                                                                                                        \
  {                                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKH_MOW >> 1)                                          \
                                                     + ((dilatedKH_MOW >> 1) - 1) - dilatedKH_MOW) / (XAI_CNN_CONV_GET_STRIDE(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent..");                                                    \
  }                                                                                                                                           \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM3(outT), XAI_ERR_DATASIZE,                                                \
                  "Width of Bias Array is less than number of Kernels.");                                                                     \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                                      \
                  "Height of Bias Array should be greater than zero.");

/* outT is assumed to be ID16WH */
/* inT is assumed to be DWH */
/* coeffT is assumed to be RMOD_DWH_ID16WH */
#if (XCHAL_IVPN_SIMD_WIDTH == 64)
#define XAI_CHECK_CONSISTENCY_MOD_DWH_ID16WH(inT, coeffT, biasArr, outT, param)                                                       \
  {                                                                                                                                   \
    uint16_t dilationX     = XAI_CNN_CONV_GET_DILATIONX(param);                                                                       \
    uint16_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);                                                                       \
    int32_t dilatedkWidth  = dilationX * (XAI_TILE4D_GET_DIM2(coeffT) - 1) + 1;                                                       \
    int32_t dilatedkHeight = dilationY * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1;                                                       \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) << 4) == (XAI_TILE4D_GET_DIM4(coeffT) << 4), XAI_ERR_DATASIZE,                         \
                    "Number of Output Channels not equal to the number of Kernels.");                                                 \
    if (dilatedkWidth % 2 != 0)                                                                                                       \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 4) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedkWidth >> 1) + (dilatedkWidth >> 1)  \
                                                              - dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),            \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 4) <= (((XAI_TILE3D_GET_DIM2(inT) +                                              \
                                                              (dilatedkWidth >> 1) + ((dilatedkWidth >> 1) - 1) -                     \
                                                              dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),              \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    if (dilatedkHeight % 2 != 0)                                                                                                      \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + (dilatedkHeight >> 1)       \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + ((dilatedkHeight >> 1) - 1) \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    if (XAI_TILE4D_GET_DATA_ORDER(coeffT) == XAI_RMOD_DWH_I16_ID16WH)                                                                 \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE4D_GET_DIM1(coeffT) >> 4) == 4), XAI_ERR_DATASIZE,                                                    \
                      "Number of Input Channels in the kernel after zero padding (if any) should be 4.");                             \
    }                                                                                                                                 \
    else if (XAI_TILE4D_GET_DATA_ORDER(coeffT) == XAI_RMOD_DWH_ID16WH)                                                                \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE4D_GET_DIM1(coeffT) >> 5) == 4), XAI_ERR_DATASIZE,                                                    \
                      "Number of Input Channels in the kernel after zero padding (if any) should be 4.");                             \
    }                                                                                                                                 \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(inT) <= 4), XAI_ERR_DATASIZE,                                                                \
                    "Number of Input Channels should be less than equal to 4.");                                                      \
    XAI_CHECK_ERROR((XAI_ALIGN_VAL(XAI_ARRAY_GET_WIDTH(biasArr), 16) == (XAI_TILE3D_GET_DIM2(outT) << 4)), XAI_ERR_DATASIZE,          \
                    "Width of Bias Array is less than or equal to the number of output channels.");                                   \
    XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                            \
                    "Height of Bias Array should be greater than zero.");                                                             \
  }

#else
#define XAI_CHECK_CONSISTENCY_MOD_DWH_ID16WH(inT, coeffT, biasArr, outT, param)                                                       \
  {                                                                                                                                   \
    uint16_t dilationX     = XAI_CNN_CONV_GET_DILATIONX(param);                                                                       \
    uint16_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);                                                                       \
    int32_t dilatedkWidth  = dilationX * (XAI_TILE4D_GET_DIM2(coeffT) - 1) + 1;                                                       \
    int32_t dilatedkHeight = dilationY * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1;                                                       \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) << 4) == (XAI_TILE4D_GET_DIM4(coeffT) << 4), XAI_ERR_DATASIZE,                         \
                    "Number of Output Channels not equal to the number of Kernels.");                                                 \
    if (dilatedkWidth % 2 != 0)                                                                                                       \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 4) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedkWidth >> 1) + (dilatedkWidth >> 1)  \
                                                              - dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),            \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 4) <= (((XAI_TILE3D_GET_DIM2(inT) +                                              \
                                                              (dilatedkWidth >> 1) + ((dilatedkWidth >> 1) - 1) -                     \
                                                              dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),              \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    if (dilatedkHeight % 2 != 0)                                                                                                      \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + (dilatedkHeight >> 1)       \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + ((dilatedkHeight >> 1) - 1) \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    XAI_CHECK_ERROR(((XAI_TILE4D_GET_DIM1(coeffT) >> 4) == 4), XAI_ERR_DATASIZE,                                                      \
                    "Number of Input Channels in the kernel after zero padding (if any) should be 4.");                               \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(inT) <= 4), XAI_ERR_DATASIZE,                                                                \
                    "Number of Input Channels should be less than equal to 4.");                                                      \
    XAI_CHECK_ERROR((XAI_ALIGN_VAL(XAI_ARRAY_GET_WIDTH(biasArr), 16) == (XAI_TILE3D_GET_DIM2(outT) << 4)), XAI_ERR_DATASIZE,          \
                    "Width of Bias Array is less than or equal to the number of output channels.");                                   \
    XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                            \
                    "Height of Bias Array should be greater than zero.");                                                             \
  }
#endif

#define XAI_CHECK_CONSISTENCY_DEPTHWISE_DILATED_MOW_WHD(inT, coeffT, biasArr, outT, param)                                                                \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inT) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DEPTH_MULTIPLIER(param) ==                                                \
                  XAI_TILE3D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                                                                                          \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                                         \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outT) == XAI_TILE3D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                                                             \
                  "Number of Output Channels not equal to the number of Kernels.");                                                                       \
  uint16_t dilatedKW_MOW = (uint16_t) (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(param) *                                                               \
                                       (XAI_TILE3D_GET_DIM1(coeffTile) - 1) + 1);                                                                         \
  uint16_t dilatedKH_MOW = (uint16_t) (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(param) *                                                               \
                                       (XAI_TILE3D_GET_DIM2(coeffTile) - 1) + 1);                                                                         \
  if (dilatedKW_MOW % 2 != 0)                                                                                                                             \
  {                                                                                                                                                       \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (dilatedKW_MOW >> 1)                                                      \
                                                     + (dilatedKW_MOW >> 1) - dilatedKW_MOW) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                  \
  }                                                                                                                                                       \
  else                                                                                                                                                    \
  {                                                                                                                                                       \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (dilatedKW_MOW >> 1)                                                      \
                                                     + ((dilatedKW_MOW >> 1) - 1) - dilatedKW_MOW) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(param)))  \
                                                   + 1)), XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                            \
  }                                                                                                                                                       \
  if (dilatedKH_MOW % 2 != 0)                                                                                                                             \
  {                                                                                                                                                       \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKH_MOW >> 1)                                                      \
                                                     + (dilatedKH_MOW >> 1) - dilatedKH_MOW) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                 \
  }                                                                                                                                                       \
  else                                                                                                                                                    \
  {                                                                                                                                                       \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKH_MOW >> 1)                                                      \
                                                     + ((dilatedKH_MOW >> 1) - 1) - dilatedKH_MOW) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(param)))  \
                                                   + 1)), XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                           \
  }                                                                                                                                                       \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM3(outT), XAI_ERR_DATASIZE,                                                            \
                  "Width of Bias Array is less than number of Kernels.");                                                                                 \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                                                  \
                  "Height of Bias Array should be greater than zero.");

#define XAI_CHECK_CONSISTENCY_SO_DWH(inT, coeffT, biasArr, outT, param)                                                                      \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(inT) == XAI_TILE4D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                                 \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                            \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outT) == XAI_TILE4D_GET_DIM4(coeffT), XAI_ERR_DATASIZE,                                                \
                  "Number of Output Channels not equal to the number of Kernels.");                                                          \
  uint16_t dilatedKW_SO = (uint16_t) (XAI_CNN_CONV_GET_DILATION(param) * (XAI_TILE4D_GET_DIM2(coeffTile) - 1) + 1);                          \
  uint16_t dilatedKH_SO = (uint16_t) (XAI_CNN_CONV_GET_DILATION(param) * (XAI_TILE4D_GET_DIM3(coeffTile) - 1) + 1);                          \
  if (dilatedKW_SO % 2 != 0)                                                                                                                 \
  {                                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKW_SO >> 1)                                          \
                                                     + (dilatedKW_SO >> 1) - dilatedKW_SO) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                     \
  }                                                                                                                                          \
  else                                                                                                                                       \
  {                                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (dilatedKW_SO >> 1)                                          \
                                                     + ((dilatedKW_SO >> 1) - 1) - dilatedKW_SO) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                     \
  }                                                                                                                                          \
  if (dilatedKH_SO % 2 != 0)                                                                                                                 \
  {                                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedKH_SO >> 1)                                          \
                                                     + (dilatedKH_SO >> 1) - dilatedKH_SO) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                    \
  }                                                                                                                                          \
  else                                                                                                                                       \
  {                                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedKH_SO >> 1)                                          \
                                                     + ((dilatedKH_SO >> 1) - 1) - dilatedKH_SO) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                    \
  }                                                                                                                                          \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE4D_GET_DIM4(coeffT), XAI_ERR_DATASIZE,                                             \
                  "Width of Bias Array is less than number of Kernels.");                                                                    \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                                     \
                  "Height of Bias Array should be greater than zero.");

#define XAI_CHECK_COEFFTILE_CONTIGUOUS(coeffT, param)                                   \
  XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1_PITCH(coeffT) == XAI_TILE4D_GET_DIM1(coeffT)) && \
                  (XAI_TILE4D_GET_DIM2_PITCH(coeffT) == XAI_TILE4D_GET_DIM1(coeffT) *   \
                   XAI_TILE4D_GET_DIM2(coeffT)), XAI_ERR_BADARG,                        \
                  "CoeffTile is not contiguous.");

#define XAI_CHECK_CONSISTENCY_POOL_WHD(inT, outT, param)                                                                                                                                        \
  XAI_CHECK_ERROR((XAI_CNN_POOLING_GET_STRIDEX(param) > 0) && (XAI_CNN_POOLING_GET_STRIDEY(param) > 0),                                                                                         \
                  XAI_ERR_BADARG, "\nStrideX = %hhu, StrideY = %hhu\nStride along width and height must be greater than 0",                                                                     \
                  XAI_CNN_POOLING_GET_STRIDEX(param), XAI_CNN_POOLING_GET_STRIDEY(param));                                                                                                      \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inT) == XAI_TILE3D_GET_DIM3(outT),                                                                                                                        \
                  XAI_ERR_CHANNEL_INVALID, "Number of input and output channels don't match");                                                                                                  \
  if (XAI_CNN_POOLING_GET_KERNELWIDTH(param) % 2 != 0)                                                                                                                                          \
  {                                                                                                                                                                                             \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&                                                                                          \
                     XAI_TILE3D_GET_DIM1_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),                                                                                           \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1)                                                                   \
                                                     + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - XAI_CNN_POOLING_GET_KERNELWIDTH(param)) / (XAI_CNN_POOLING_GET_STRIDEX(param))) + 1)),   \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                                        \
  }                                                                                                                                                                                             \
  else                                                                                                                                                                                          \
  {                                                                                                                                                                                             \
    if (XAI_CNN_POOLING_GET_LEFTEDGE_FLAG(param))                                                                                                                                               \
    {                                                                                                                                                                                           \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&                                                                                        \
                       XAI_TILE3D_GET_DIM1_EDGE2(inT) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1)),                                                                                   \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag set");                                                                                               \
    }                                                                                                                                                                                           \
    else                                                                                                                                                                                        \
    {                                                                                                                                                                                           \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inT) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1) &&                                                                                  \
                       XAI_TILE3D_GET_DIM1_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),                                                                                         \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag reset");                                                                                             \
    }                                                                                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - 1)                                                             \
                                                     + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - XAI_CNN_POOLING_GET_KERNELWIDTH(param)) / (XAI_CNN_POOLING_GET_STRIDEX(param))) + 1)),   \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                                        \
  }                                                                                                                                                                                             \
  if (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) % 2 != 0)                                                                                                                                         \
  {                                                                                                                                                                                             \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&                                                                                         \
                     XAI_TILE3D_GET_DIM2_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),                                                                                          \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1)                                                                  \
                                                     + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - XAI_CNN_POOLING_GET_KERNELHEIGHT(param)) / (XAI_CNN_POOLING_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                                                       \
  }                                                                                                                                                                                             \
  else                                                                                                                                                                                          \
  {                                                                                                                                                                                             \
    if (XAI_CNN_POOLING_GET_TOPEDGE_FLAG(param))                                                                                                                                                \
    {                                                                                                                                                                                           \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&                                                                                       \
                       XAI_TILE3D_GET_DIM2_EDGE2(inT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1)),                                                                                  \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag set");                                                                                                \
    }                                                                                                                                                                                           \
    else                                                                                                                                                                                        \
    {                                                                                                                                                                                           \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1) &&                                                                                 \
                       XAI_TILE3D_GET_DIM2_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),                                                                                        \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag reset");                                                                                              \
    }                                                                                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - 1)                                                            \
                                                     + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - XAI_CNN_POOLING_GET_KERNELHEIGHT(param)) / (XAI_CNN_POOLING_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                                                       \
  }

#define XAI_CHECK_CONSISTENCY_POOL_DWH(inT, outT, param)                                                                                                                                        \
  XAI_CHECK_ERROR((XAI_CNN_POOLING_GET_STRIDEX(param) > 0) && (XAI_CNN_POOLING_GET_STRIDEY(param) > 0),                                                                                         \
                  XAI_ERR_BADARG, "\nStrideX = %hhu, StrideY = %hhu\nStride along width and height must be greater than 0",                                                                     \
                  XAI_CNN_POOLING_GET_STRIDEX(param), XAI_CNN_POOLING_GET_STRIDEY(param));                                                                                                      \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(inT) == XAI_TILE3D_GET_DIM1(outT),                                                                                                                        \
                  XAI_ERR_CHANNEL_INVALID, "Number of input and output channels don't match");                                                                                                  \
  if ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) % 2 != 0))                                                                                                                                        \
  {                                                                                                                                                                                             \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&                                                                                          \
                     XAI_TILE3D_GET_DIM2_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),                                                                                           \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1)                                                                   \
                                                     + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - XAI_CNN_POOLING_GET_KERNELWIDTH(param)) / (XAI_CNN_POOLING_GET_STRIDEX(param))) + 1)),   \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                                        \
  }                                                                                                                                                                                             \
  else                                                                                                                                                                                          \
  {                                                                                                                                                                                             \
    if (XAI_CNN_POOLING_GET_LEFTEDGE_FLAG(param))                                                                                                                                               \
    {                                                                                                                                                                                           \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&                                                                                        \
                       XAI_TILE3D_GET_DIM2_EDGE2(inT) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1)),                                                                                   \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag set");                                                                                               \
    }                                                                                                                                                                                           \
    else                                                                                                                                                                                        \
    {                                                                                                                                                                                           \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inT) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1) &&                                                                                  \
                       XAI_TILE3D_GET_DIM2_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),                                                                                         \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag reset");                                                                                             \
    }                                                                                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - 1)                                                             \
                                                     + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - XAI_CNN_POOLING_GET_KERNELWIDTH(param)) / (XAI_CNN_POOLING_GET_STRIDEX(param))) + 1)),   \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                                        \
  }                                                                                                                                                                                             \
  if ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) % 2 != 0))                                                                                                                                       \
  {                                                                                                                                                                                             \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&                                                                                         \
                     XAI_TILE3D_GET_DIM3_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),                                                                                          \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                                                                                                          \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1)                                                                  \
                                                     + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - XAI_CNN_POOLING_GET_KERNELHEIGHT(param)) / (XAI_CNN_POOLING_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                                                       \
  }                                                                                                                                                                                             \
  else                                                                                                                                                                                          \
  {                                                                                                                                                                                             \
    if (XAI_CNN_POOLING_GET_TOPEDGE_FLAG(param))                                                                                                                                                \
    {                                                                                                                                                                                           \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&                                                                                       \
                       XAI_TILE3D_GET_DIM3_EDGE2(inT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1)),                                                                                  \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag set");                                                                                                \
    }                                                                                                                                                                                           \
    else                                                                                                                                                                                        \
    {                                                                                                                                                                                           \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1) &&                                                                                 \
                       XAI_TILE3D_GET_DIM3_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),                                                                                        \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag reset");                                                                                              \
    }                                                                                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - 1)                                                            \
                                                     + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - XAI_CNN_POOLING_GET_KERNELHEIGHT(param)) / (XAI_CNN_POOLING_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                                                       \
  }

#define XAI_CHECK_CONSISTENCY_POOL_ID32WH(inT, outT, param)                                                                                                                                          \
  XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(inT) << 5) == (XAI_TILE3D_GET_DIM2(outT) << 5),                                                                                                               \
                  XAI_ERR_CHANNEL_INVALID, "Number of input and output channels don't match");                                                                                                       \
  if ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) % 2 != 0))                                                                                                                                             \
  {                                                                                                                                                                                                  \
    XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inT) >> 5) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&                                                                                        \
                     (XAI_TILE3D_GET_DIM1_EDGE2(inT) >> 5) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),                                                                                         \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                                                                                                               \
    XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 5) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 5) + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1)                                                          \
                                                            + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - XAI_CNN_POOLING_GET_KERNELWIDTH(param)) / (XAI_CNN_POOLING_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                                             \
  }                                                                                                                                                                                                  \
  else                                                                                                                                                                                               \
  {                                                                                                                                                                                                  \
    if (XAI_CNN_POOLING_GET_LEFTEDGE_FLAG(param))                                                                                                                                                    \
    {                                                                                                                                                                                                \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inT) >> 5) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&                                                                                      \
                       (XAI_TILE3D_GET_DIM1_EDGE2(inT) >> 5) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1)),                                                                                 \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag set");                                                                                                    \
    }                                                                                                                                                                                                \
    else                                                                                                                                                                                             \
    {                                                                                                                                                                                                \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inT) >> 5) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1) &&                                                                                \
                       (XAI_TILE3D_GET_DIM1_EDGE2(inT) >> 5) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),                                                                                       \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag reset");                                                                                                  \
    }                                                                                                                                                                                                \
    XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 5) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 5) + ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - 1)                                                    \
                                                            + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - XAI_CNN_POOLING_GET_KERNELWIDTH(param)) / (XAI_CNN_POOLING_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                                             \
  }                                                                                                                                                                                                  \
  if ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) % 2 != 0))                                                                                                                                            \
  {                                                                                                                                                                                                  \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&                                                                                              \
                     XAI_TILE3D_GET_DIM3_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),                                                                                               \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1)                                                                       \
                                                     + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - XAI_CNN_POOLING_GET_KERNELHEIGHT(param)) / (XAI_CNN_POOLING_GET_STRIDEY(param))) + 1)),      \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                                                            \
  }                                                                                                                                                                                                  \
  else                                                                                                                                                                                               \
  {                                                                                                                                                                                                  \
    if (XAI_CNN_POOLING_GET_TOPEDGE_FLAG(param))                                                                                                                                                     \
    {                                                                                                                                                                                                \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&                                                                                            \
                       XAI_TILE3D_GET_DIM3_EDGE2(inT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1)),                                                                                       \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag set");                                                                                                     \
    }                                                                                                                                                                                                \
    else                                                                                                                                                                                             \
    {                                                                                                                                                                                                \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1) &&                                                                                      \
                       XAI_TILE3D_GET_DIM3_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),                                                                                             \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag reset");                                                                                                   \
    }                                                                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - 1)                                                                 \
                                                     + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - XAI_CNN_POOLING_GET_KERNELHEIGHT(param)) / (XAI_CNN_POOLING_GET_STRIDEY(param))) + 1)),      \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                                                            \
  }
#define XAI_CHECK_CONSISTENCY_POOL_ID16WH(inT, outT, param)                                                                                                                                          \
  XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(inT) << 4) == (XAI_TILE3D_GET_DIM2(outT) << 4),                                                                                                               \
                  XAI_ERR_CHANNEL_INVALID, "Number of input and output channels don't match");                                                                                                       \
  if ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) % 2 != 0))                                                                                                                                             \
  {                                                                                                                                                                                                  \
    XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inT) >> 4) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&                                                                                        \
                     (XAI_TILE3D_GET_DIM1_EDGE2(inT) >> 4) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),                                                                                         \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                                                                                                               \
    XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 4) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 4) + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1)                                                          \
                                                            + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - XAI_CNN_POOLING_GET_KERNELWIDTH(param)) / (XAI_CNN_POOLING_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                                             \
  }                                                                                                                                                                                                  \
  else                                                                                                                                                                                               \
  {                                                                                                                                                                                                  \
    if (XAI_CNN_POOLING_GET_LEFTEDGE_FLAG(param))                                                                                                                                                    \
    {                                                                                                                                                                                                \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inT) >> 4) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&                                                                                      \
                       (XAI_TILE3D_GET_DIM1_EDGE2(inT) >> 4) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1)),                                                                                 \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag set");                                                                                                    \
    }                                                                                                                                                                                                \
    else                                                                                                                                                                                             \
    {                                                                                                                                                                                                \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inT) >> 4) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1) &&                                                                                \
                       (XAI_TILE3D_GET_DIM1_EDGE2(inT) >> 4) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),                                                                                       \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag reset");                                                                                                  \
    }                                                                                                                                                                                                \
    XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 4) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 4) + ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - 1)                                                    \
                                                            + (XAI_CNN_POOLING_GET_KERNELWIDTH(param) >> 1) - XAI_CNN_POOLING_GET_KERNELWIDTH(param)) / (XAI_CNN_POOLING_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                                             \
  }                                                                                                                                                                                                  \
  if ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) % 2 != 0))                                                                                                                                            \
  {                                                                                                                                                                                                  \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&                                                                                              \
                     XAI_TILE3D_GET_DIM3_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),                                                                                               \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1)                                                                       \
                                                     + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - XAI_CNN_POOLING_GET_KERNELHEIGHT(param)) / (XAI_CNN_POOLING_GET_STRIDEY(param))) + 1)),      \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                                                            \
  }                                                                                                                                                                                                  \
  else                                                                                                                                                                                               \
  {                                                                                                                                                                                                  \
    if (XAI_CNN_POOLING_GET_TOPEDGE_FLAG(param))                                                                                                                                                     \
    {                                                                                                                                                                                                \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&                                                                                            \
                       XAI_TILE3D_GET_DIM3_EDGE2(inT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1)),                                                                                       \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag set");                                                                                                     \
    }                                                                                                                                                                                                \
    else                                                                                                                                                                                             \
    {                                                                                                                                                                                                \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1) &&                                                                                      \
                       XAI_TILE3D_GET_DIM3_EDGE2(inT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),                                                                                             \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag reset");                                                                                                   \
    }                                                                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - 1)                                                                 \
                                                     + (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) >> 1) - XAI_CNN_POOLING_GET_KERNELHEIGHT(param)) / (XAI_CNN_POOLING_GET_STRIDEY(param))) + 1)),      \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                                                            \
  }
#define XAI_CHECK_CONSISTENCY_UNPOOL_WHD(inT, outT, param)                                                       \
  /* Width & Height Divisible by stride */                                                                       \
  XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1(outT) - 1) % XAI_CNN_POOLING_GET_STRIDEX(param)) == 0),                 \
                  XAI_ERR_DATASIZE, "Number of output widths to be generated should be a multiple of strideX");  \
  XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM2(outT) - 1) % XAI_CNN_POOLING_GET_STRIDEY(param)) == 0),                 \
                  XAI_ERR_DATASIZE, "Number of output heights to be generated should be a multiple of strideY"); \
                                                                                                                 \
  /* Depth Should be same for in and out tiles */                                                                \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inT) == XAI_TILE3D_GET_DIM3(outT),                                         \
                  XAI_ERR_CHANNEL_INVALID, "Number of input and output channels don't match");                   \
                                                                                                                 \
  /* Minimum required input width to compute requested output width */                                           \
  XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(inT) >= ((XAI_TILE3D_GET_DIM1(outT) - 1) /                                \
                                                XAI_CNN_POOLING_GET_STRIDEX(param)) + 1), XAI_ERR_DATASIZE,      \
                  "Insufficient input width to generate requested output width");                                \
                                                                                                                 \
  /* Minimum required input height to compute requested output height */                                         \
  XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(inT) >= ((XAI_TILE3D_GET_DIM2(outT) - 1) /                                \
                                                XAI_CNN_POOLING_GET_STRIDEY(param)) + 1), XAI_ERR_DATASIZE,      \
                  "Insufficient input height to generate requested output height");                              \
                                                                                                                 \
  if (XAI_CNN_POOLING_GET_KERNELWIDTH(param) % 2 != 0)                                                           \
  {                                                                                                              \
    /* Odd Width Kernel Edge Consistency */                                                                      \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(outT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&          \
                     XAI_TILE3D_GET_DIM1_EDGE2(outT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),           \
                    XAI_ERR_EDGE, "Invalid left/right edge for odd kernel width.");                              \
  }                                                                                                              \
  else                                                                                                           \
  {                                                                                                              \
    /* Even Width Kernel Edge Consistency */                                                                     \
    if (XAI_CNN_POOLING_GET_LEFTEDGE_FLAG(param))                                                                \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(outT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&        \
                       XAI_TILE3D_GET_DIM1_EDGE2(outT) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1)),   \
                      XAI_ERR_EDGE, "Invalid left/right edge for even kernel width with leftedge flag set");     \
    }                                                                                                            \
    else                                                                                                         \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(outT) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1) &&  \
                       XAI_TILE3D_GET_DIM1_EDGE2(outT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),         \
                      XAI_ERR_EDGE, "Invalid left/right edge for even kernel width with leftedge flag reset");   \
    }                                                                                                            \
  }                                                                                                              \
  if (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) % 2 != 0)                                                          \
  {                                                                                                              \
    /* Odd Height Kernel Edge Consistency */                                                                     \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(outT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&         \
                     XAI_TILE3D_GET_DIM2_EDGE2(outT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),          \
                    XAI_ERR_EDGE, "Invalid Top/Bottom edge for odd kernel height.");                             \
  }                                                                                                              \
  else                                                                                                           \
  {                                                                                                              \
    /* Even Height Kernel Edge Consistency */                                                                    \
    if (XAI_CNN_POOLING_GET_TOPEDGE_FLAG(param))                                                                 \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(outT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&       \
                       XAI_TILE3D_GET_DIM2_EDGE2(outT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1)),  \
                      XAI_ERR_EDGE, "Invalid top/bottom edge for even kernel height with topedge flag set");     \
    }                                                                                                            \
    else                                                                                                         \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(outT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1) && \
                       XAI_TILE3D_GET_DIM2_EDGE2(outT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),        \
                      XAI_ERR_EDGE, "Invalid top/bottom edge for even kernel height with topedge flag reset");   \
    }                                                                                                            \
  }

#define XAI_CHECK_CONSISTENCY_UNPOOL_DWH(inT, outT, param)                                                       \
  /* Width & Height Divisible by stride */                                                                       \
  XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM2(outT) - 1) % XAI_CNN_POOLING_GET_STRIDEX(param)) == 0),                 \
                  XAI_ERR_DATASIZE, "Number of output widths to be generated should be a multiple of strideX");  \
  XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM3(outT) - 1) % XAI_CNN_POOLING_GET_STRIDEY(param)) == 0),                 \
                  XAI_ERR_DATASIZE, "Number of output heights to be generated should be a multiple of strideY"); \
                                                                                                                 \
  /* Depth Should be same for in and out tiles */                                                                \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(inT) == XAI_TILE3D_GET_DIM1(outT),                                         \
                  XAI_ERR_CHANNEL_INVALID, "Number of input and output channels don't match");                   \
                                                                                                                 \
  /* Minimum required input width to compute requested output width */                                           \
  XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(inT) >= ((XAI_TILE3D_GET_DIM2(outT) - 1) /                                \
                                                XAI_CNN_POOLING_GET_STRIDEX(param)) + 1), XAI_ERR_DATASIZE,      \
                  "Insufficient input width to generate requested output width");                                \
                                                                                                                 \
  /* Minimum required input height to compute requested output height */                                         \
  XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(inT) >= ((XAI_TILE3D_GET_DIM3(outT) - 1) /                                \
                                                XAI_CNN_POOLING_GET_STRIDEY(param)) + 1), XAI_ERR_DATASIZE,      \
                  "Insufficient input height to generate requested output height");                              \
                                                                                                                 \
  if (XAI_CNN_POOLING_GET_KERNELWIDTH(param) % 2 != 0)                                                           \
  {                                                                                                              \
    /* Odd Width Kernel Edge Consistency */                                                                      \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(outT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&          \
                     XAI_TILE3D_GET_DIM2_EDGE2(outT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),           \
                    XAI_ERR_EDGE, "Invalid left/right edge for odd kernel width.");                              \
  }                                                                                                              \
  else                                                                                                           \
  {                                                                                                              \
    /* Even Width Kernel Edge Consistency */                                                                     \
    if (XAI_CNN_POOLING_GET_LEFTEDGE_FLAG(param))                                                                \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(outT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) &&        \
                       XAI_TILE3D_GET_DIM2_EDGE2(outT) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1)),   \
                      XAI_ERR_EDGE, "Invalid left/right edge for even kernel width with leftedge flag set");     \
    }                                                                                                            \
    else                                                                                                         \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(outT) >= ((XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2) - 1) &&  \
                       XAI_TILE3D_GET_DIM2_EDGE2(outT) >= (XAI_CNN_POOLING_GET_KERNELWIDTH(param) / 2)),         \
                      XAI_ERR_EDGE, "Invalid left/right edge for even kernel width with leftedge flag reset");   \
    }                                                                                                            \
  }                                                                                                              \
  if (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) % 2 != 0)                                                          \
  {                                                                                                              \
    /* Odd Height Kernel Edge Consistency */                                                                     \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(outT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&         \
                     XAI_TILE3D_GET_DIM3_EDGE2(outT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),          \
                    XAI_ERR_EDGE, "Invalid Top/Bottom edge for odd kernel height.");                             \
  }                                                                                                              \
  else                                                                                                           \
  {                                                                                                              \
    /* Even Height Kernel Edge Consistency */                                                                    \
    if (XAI_CNN_POOLING_GET_TOPEDGE_FLAG(param))                                                                 \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(outT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) &&       \
                       XAI_TILE3D_GET_DIM3_EDGE2(outT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1)),  \
                      XAI_ERR_EDGE, "Invalid top/bottom edge for even kernel height with topedge flag set");     \
    }                                                                                                            \
    else                                                                                                         \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(outT) >= ((XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2) - 1) && \
                       XAI_TILE3D_GET_DIM3_EDGE2(outT) >= (XAI_CNN_POOLING_GET_KERNELHEIGHT(param) / 2)),        \
                      XAI_ERR_EDGE, "Invalid top/bottom edge for even kernel height with topedge flag reset");   \
    }                                                                                                            \
  }

#define XAI_CHECK_EDGES_MOW_WHD(inTile, coeffTile, param)                                                        \
  uint16_t dilatedKW = (uint16_t) (XAI_CNN_CONV_GET_DILATION(param) * (XAI_TILE4D_GET_DIM1(coeffTile) - 1) + 1); \
  uint16_t dilatedKH = (uint16_t) (XAI_CNN_CONV_GET_DILATION(param) * (XAI_TILE4D_GET_DIM2(coeffTile) - 1) + 1); \
  if (dilatedKW % 2 != 0)                                                                                        \
  {                                                                                                              \
    if (dilatedKH % 2 != 0)                                                                                      \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                       \
                      && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                    \
                      && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                    \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                   \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                       \
    }                                                                                                            \
    else                                                                                                         \
    {                                                                                                            \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                  \
      {                                                                                                          \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                     \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                         \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                     \
      }                                                                                                          \
      else                                                                                                       \
      {                                                                                                          \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                     \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1))                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                     \
      }                                                                                                          \
    }                                                                                                            \
  }                                                                                                              \
  else                                                                                                           \
  {                                                                                                              \
    if (dilatedKH % 2 != 0)                                                                                      \
    {                                                                                                            \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                 \
      {                                                                                                          \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                     \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((dilatedKW / 2) - 1))                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                     \
      }                                                                                                          \
      else                                                                                                       \
      {                                                                                                          \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1))                             \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                     \
      }                                                                                                          \
    }                                                                                                            \
    else                                                                                                         \
    {                                                                                                            \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                  \
      {                                                                                                          \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                               \
        {                                                                                                        \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (dilatedKW / 2) &&                               \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKH / 2) &&                               \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                          \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) &&                               \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKH / 2) &&                               \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                          \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
        }                                                                                                        \
      }                                                                                                          \
      else                                                                                                       \
      {                                                                                                          \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                               \
        {                                                                                                        \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (dilatedKW / 2) &&                               \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2 - 1) &&                           \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKH / 2)),                                \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) &&                               \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKH / 2)),                                \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
        }                                                                                                        \
      }                                                                                                          \
    }                                                                                                            \
  }

/* outT is assumed to be ID4WH */
#define XAI_CHECK_CONSISTENCY_MOD_ID4WH(inT, coeffT, biasArr, outT, param)                                                            \
  {                                                                                                                                   \
    uint16_t dilationX     = XAI_CNN_CONV_GET_DILATIONX(param);                                                                       \
    uint16_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);                                                                       \
    int32_t dilatedkWidth  = dilationX * (XAI_TILE4D_GET_DIM2(coeffT) - 1) + 1;                                                       \
    int32_t dilatedkHeight = dilationY * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1;                                                       \
    XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM2(outT) << 2) + 15) & (~15)) == (XAI_TILE4D_GET_DIM1(coeffT) >> 2), XAI_ERR_DATASIZE,        \
                    "Number of Output Channels not equal to the number of Kernels.");                                                 \
    XAI_CHECK_ERROR((XAI_ALIGN_VAL(XAI_ARRAY_GET_WIDTH(biasArr), 16) >= (XAI_TILE3D_GET_DIM2(outT) << 2)), XAI_ERR_DATASIZE,          \
                    "Width of Bias Array is less than number of output channels.");                                                   \
    if (dilatedkWidth % 2 != 0)                                                                                                       \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 2) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 2) +                                       \
                                                              (dilatedkWidth >> 1) + (dilatedkWidth >> 1) -                           \
                                                              dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),              \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 2) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 2) +                                       \
                                                              (dilatedkWidth >> 1) + ((dilatedkWidth >> 1) - 1) -                     \
                                                              dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),              \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(inT) << 2) == (XAI_TILE4D_GET_DIM4(coeffT) << 2), XAI_ERR_DATASIZE,                          \
                    "Number of Input Channels not equal to the number of channels in the Kernel.");                                   \
    if (dilatedkHeight % 2 != 0)                                                                                                      \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + (dilatedkHeight >> 1)       \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + ((dilatedkHeight >> 1) - 1) \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                            \
                    "Height of Bias Array should be greater than zero.");                                                             \
  }

/* outT is assumed to be ID16WH */
#define XAI_CHECK_CONSISTENCY_MOD_ID16WH(inT, coeffT, biasArr, outT, param)                                                           \
  {                                                                                                                                   \
    uint16_t dilationX     = XAI_CNN_CONV_GET_DILATIONX(param);                                                                       \
    uint16_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);                                                                       \
    int32_t dilatedkWidth  = dilationX * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1;                                                       \
    int32_t dilatedkHeight = dilationY * (XAI_TILE4D_GET_DIM2(coeffT) - 1) + 1;                                                       \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) << 4) == (XAI_TILE4D_GET_DIM4(coeffT) << 4), XAI_ERR_DATASIZE,                         \
                    "Number of Output Channels not equal to the number of Kernels.");                                                 \
    XAI_CHECK_ERROR((XAI_ALIGN_VAL(XAI_ARRAY_GET_WIDTH(biasArr), 16) >= (XAI_TILE3D_GET_DIM2(outT) << 4)), XAI_ERR_DATASIZE,          \
                    "Width of Bias Array is less than number of output channels.");                                                   \
    if (dilatedkWidth % 2 != 0)                                                                                                       \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 4) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 4) +                                       \
                                                              (dilatedkWidth >> 1) + (dilatedkWidth >> 1) -                           \
                                                              dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),              \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 4) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 4) +                                       \
                                                              (dilatedkWidth >> 1) + ((dilatedkWidth >> 1) - 1) -                     \
                                                              dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),              \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(inT) << 4) == (XAI_TILE4D_GET_DIM1(coeffT) >> 4), XAI_ERR_DATASIZE,                          \
                    "Number of Input Channels not equal to the number of channels in the Kernel.");                                   \
    if (dilatedkHeight % 2 != 0)                                                                                                      \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + (dilatedkHeight >> 1)       \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + ((dilatedkHeight >> 1) - 1) \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                            \
                    "Height of Bias Array should be greater than zero.");                                                             \
  }

#define XAI_CHECK_CONSISTENCY_MOD_ID32WH(inT, coeffT, biasArr, outT, param)                                                           \
  {                                                                                                                                   \
    uint16_t dilationX     = XAI_CNN_CONV_GET_DILATIONX(param);                                                                       \
    uint16_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);                                                                       \
    int32_t dilatedkWidth  = dilationX * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1;                                                       \
    int32_t dilatedkHeight = dilationY * (XAI_TILE4D_GET_DIM2(coeffT) - 1) + 1;                                                       \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) << 5) == (XAI_TILE4D_GET_DIM4(coeffT) << 5), XAI_ERR_DATASIZE,                         \
                    "Number of Output Channels not equal to the number of Kernels.");                                                 \
    XAI_CHECK_ERROR((XAI_ALIGN_VAL(XAI_ARRAY_GET_WIDTH(biasArr), 32) >= (XAI_TILE3D_GET_DIM2(outT) << 5)), XAI_ERR_DATASIZE,          \
                    "Width of Bias Array is less than number of output channels.");                                                   \
    if (dilatedkWidth % 2 != 0)                                                                                                       \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 5) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 5) +                                       \
                                                              (dilatedkWidth >> 1) + (dilatedkWidth >> 1) -                           \
                                                              dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),              \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outT) >> 5) <= ((((XAI_TILE3D_GET_DIM1(inT) >> 5) +                                       \
                                                              (dilatedkWidth >> 1) + ((dilatedkWidth >> 1) - 1) -                     \
                                                              dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),              \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                            \
    }                                                                                                                                 \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(inT) << 5) == (XAI_TILE4D_GET_DIM1(coeffT) >> 5), XAI_ERR_DATASIZE,                          \
                    "Number of Input Channels not equal to the number of channels in the Kernel.");                                   \
    if (dilatedkHeight % 2 != 0)                                                                                                      \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + (dilatedkHeight >> 1)       \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    else                                                                                                                              \
    {                                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (dilatedkHeight >> 1) + ((dilatedkHeight >> 1) - 1) \
                                                       - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                  \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                           \
    }                                                                                                                                 \
    XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                            \
                    "Height of Bias Array should be greater than zero.");                                                             \
  }
// Assuming that "inTile" is in DWH format
// Assuming that "coeffTile" is in RMOD_DWH_ID16WH format
#define XAI_CHECK_EDGES_MOD_DWH_ID16WH(inTile, coeffTile, param)                               \
  uint16_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);                                      \
  uint16_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);                                  \
  int32_t dilatedkWidth  = dilationX * (XAI_TILE4D_GET_DIM2(coeffTile) - 1) + 1;               \
  int32_t dilatedkHeight = dilationY * (XAI_TILE4D_GET_DIM3(coeffTile) - 1) + 1;               \
  if (dilatedkWidth % 2 != 0)                                                                  \
  {                                                                                            \
    if (dilatedkHeight % 2 != 0)                                                               \
    {                                                                                          \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedkWidth / 2)                 \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedkWidth / 2)              \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedkHeight / 2)             \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedkHeight / 2),            \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");     \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedkWidth / 2)               \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedkWidth / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedkHeight / 2)           \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedkHeight / 2) - 1)),  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedkWidth / 2)               \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedkWidth / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedkHeight / 2) - 1))   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedkHeight / 2),          \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
  }                                                                                            \
  else                                                                                         \
  {                                                                                            \
    if (dilatedkHeight % 2 != 0)                                                               \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                               \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedkWidth / 2)               \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedkWidth / 2) - 1)      \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedkHeight / 2)           \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedkHeight / 2),          \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedkWidth / 2) - 1))       \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedkWidth / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedkHeight / 2)           \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedkHeight / 2),          \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedkWidth / 2) &&         \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedkWidth / 2) - 1) &&   \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedkHeight / 2) &&        \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedkHeight / 2) - 1)),   \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedkWidth / 2) - 1) &&   \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedkWidth / 2) &&         \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedkHeight / 2) &&        \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedkHeight / 2) - 1)),   \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedkWidth / 2) &&         \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedkWidth / 2) - 1) &&   \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedkHeight / 2) - 1) &&  \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedkHeight / 2)),         \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedkWidth / 2) - 1) &&   \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedkWidth / 2) &&         \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedkHeight / 2) - 1) &&  \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedkHeight / 2)),         \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
    }                                                                                          \
  }
#define XAI_CHECK_EDGES_DEPTHWISE_DILATED_MOW_WHD(inTile, coeffTile, param)                    \
  uint16_t dilatedKW = (uint16_t) (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(param) *        \
                                   (XAI_TILE3D_GET_DIM1(coeffTile) - 1) + 1);                  \
  uint16_t dilatedKH = (uint16_t) (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(param) *        \
                                   (XAI_TILE3D_GET_DIM2(coeffTile) - 1) + 1);                  \
  if (dilatedKW % 2 != 0)                                                                      \
  {                                                                                            \
    if (dilatedKH % 2 != 0)                                                                    \
    {                                                                                          \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                     \
                      && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                  \
                      && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                  \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                 \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");     \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_TOPEDGE(param))                              \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                   \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),       \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                   \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1))        \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),               \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
  }                                                                                            \
  else                                                                                         \
  {                                                                                            \
    if (dilatedKH % 2 != 0)                                                                    \
    {                                                                                          \
      if (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_LEFTEDGE(param))                             \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                   \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((dilatedKW / 2) - 1))        \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),               \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1))           \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),               \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_TOPEDGE(param))                              \
      {                                                                                        \
        if (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_LEFTEDGE(param))                           \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (dilatedKW / 2) &&             \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&       \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKH / 2) &&             \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),        \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&       \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) &&             \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKH / 2) &&             \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),        \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        if (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_LEFTEDGE(param))                           \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (dilatedKW / 2) &&             \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2 - 1) &&         \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&       \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKH / 2)),              \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&       \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) &&             \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&       \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKH / 2)),              \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
    }                                                                                          \
  }

#define XAI_CHECK_EDGES_MOD_WHD(inTile, coeffTile, param)                                                        \
  uint16_t dilatedKW = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM3(coeffTile) - 1) + 1); \
  uint16_t dilatedKH = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM4(coeffTile) - 1) + 1); \
  if (dilatedKW % 2 != 0)                                                                                        \
  {                                                                                                              \
    if (dilatedKH % 2 != 0)                                                                                      \
    {                                                                                                            \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                       \
                      && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                    \
                      && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                    \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                   \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                       \
    }                                                                                                            \
    else                                                                                                         \
    {                                                                                                            \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                  \
      {                                                                                                          \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                     \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                         \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                     \
      }                                                                                                          \
      else                                                                                                       \
      {                                                                                                          \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                     \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1))                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                     \
      }                                                                                                          \
    }                                                                                                            \
  }                                                                                                              \
  else                                                                                                           \
  {                                                                                                              \
    if (dilatedKH % 2 != 0)                                                                                      \
    {                                                                                                            \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                 \
      {                                                                                                          \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                     \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) - 1)                            \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                     \
      }                                                                                                          \
      else                                                                                                       \
      {                                                                                                          \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1))                             \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                     \
      }                                                                                                          \
    }                                                                                                            \
    else                                                                                                         \
    {                                                                                                            \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                  \
      {                                                                                                          \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                               \
        {                                                                                                        \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (dilatedKW / 2) &&                               \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKH / 2) &&                               \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                          \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) &&                               \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKH / 2) &&                               \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                          \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
        }                                                                                                        \
      }                                                                                                          \
      else                                                                                                       \
      {                                                                                                          \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                               \
        {                                                                                                        \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (dilatedKW / 2) &&                               \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKH / 2)),                                \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) &&                               \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                         \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKH / 2)),                                \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
        }                                                                                                        \
      }                                                                                                          \
    }                                                                                                            \
  }


#define XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param)                                                         \
  uint16_t dilatedKW = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM3(coeffTile) - 1) + 1); \
  uint16_t dilatedKH = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM4(coeffTile) - 1) + 1); \
  if (dilatedKW % 2 != 0)                                                                                         \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                        \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                     \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                     \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                    \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                        \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                          \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1))                           \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
  }                                                                                                               \
  else                                                                                                            \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                  \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) - 1)                             \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1))                              \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
    }                                                                                                             \
  }

#define XAI_CHECK_EDGES_MOD_WHD_DWH(inTile, coeffTile, param)                                                     \
  uint16_t dilatedKW = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM3(coeffTile) - 1) + 1); \
  uint16_t dilatedKH = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM4(coeffTile) - 1) + 1); \
  if (dilatedKW % 2 != 0)                                                                                         \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                        \
                      && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                     \
                      && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                     \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                    \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                        \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                          \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1))                           \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
  }                                                                                                               \
  else                                                                                                            \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                  \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) - 1)                             \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1))                              \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
    }                                                                                                             \
  }

#define XAI_CHECK_TILES3D_CHECK_EDGES_QUANT(inTile, outTile)                                                                               \
  {                                                                                                                                        \
    if (XAI_TILE3D_GET_DATA_PTR(inTile) == XAI_TILE3D_GET_DATA_PTR(outTile))                                                               \
    {                                                                                                                                      \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(outTile) + XAI_TILE3D_GET_DIM1_EDGE2(outTile)) <=                                        \
                       (2 * (XAI_TILE3D_GET_DIM1_PITCH(inTile) - XAI_TILE3D_GET_DIM1(inTile)))), XAI_ERR_BADARG,                           \
                      "Output and Input tile edges constraints have not been met along dimension 1");                                      \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_PITCH(outTile) * (XAI_TILE3D_GET_DIM2_EDGE1(outTile) + XAI_TILE3D_GET_DIM2_EDGE2(outTile))) <= \
                       (2 * (XAI_TILE3D_GET_DIM2_PITCH(inTile) - (XAI_TILE3D_GET_DIM1_PITCH(inTile) * XAI_TILE3D_GET_DIM2(inTile))))),     \
                      XAI_ERR_BADARG, "Output and Input tile edges constraints have not been met  along dimension 2");                     \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM3_EDGE1(outTile) + XAI_TILE3D_GET_DIM3_EDGE2(outTile)) <=                                        \
                       (2 * (XAI_TILE3D_GET_DIM3_EDGE1(inTile) + XAI_TILE3D_GET_DIM3_EDGE2(inTile)))), XAI_ERR_BADARG,                     \
                      "Output and Input tile edges constraints have not been met  along dimension 3");                                     \
      XAI_CHECK_ERROR(((size_t) (XAI_TILE3D_GET_BUFF_PTR(inTile)) <= ((size_t) (XAI_TILE3D_GET_BUFF_PTR(outTile)))), XAI_ERR_BADARG,       \
                      "Output tile buffer pointer should be greater than or equal to input tile buffer pointer");                          \
    }                                                                                                                                      \
  }
#define XAI_CHECK_TILES4D_CHECK_EDGES_QUANT(inTile, outTile)                                                                               \
  {                                                                                                                                        \
    if (XAI_TILE4D_GET_DATA_PTR(inTile) == XAI_TILE4D_GET_DATA_PTR(outTile))                                                               \
    {                                                                                                                                      \
      XAI_CHECK_ERROR(((XAI_TILE4D_GET_DIM1_EDGE1(outTile) + XAI_TILE4D_GET_DIM1_EDGE2(outTile)) <=                                        \
                       (2 * (XAI_TILE4D_GET_DIM1_PITCH(inTile) - XAI_TILE4D_GET_DIM1(inTile)))), XAI_ERR_BADARG,                           \
                      "Output and Input tile edges constraints have not been met along dimension 1");                                      \
      XAI_CHECK_ERROR(((XAI_TILE4D_GET_DIM1_PITCH(outTile) * (XAI_TILE4D_GET_DIM2_EDGE1(outTile) + XAI_TILE4D_GET_DIM2_EDGE2(outTile))) <= \
                       (2 * (XAI_TILE4D_GET_DIM2_PITCH(inTile) - (XAI_TILE3D_GET_DIM1_PITCH(inTile) * XAI_TILE4D_GET_DIM2(inTile))))),     \
                      XAI_ERR_BADARG, "Output and Input tile edges constraints have not been met  along dimension 2");                     \
      XAI_CHECK_ERROR(((XAI_TILE4D_GET_DIM2_PITCH(outTile) * (XAI_TILE4D_GET_DIM3_EDGE1(outTile) + XAI_TILE4D_GET_DIM3_EDGE2(outTile))) <= \
                       (2 * (XAI_TILE4D_GET_DIM3_PITCH(inTile) - (XAI_TILE3D_GET_DIM2_PITCH(inTile) * XAI_TILE4D_GET_DIM3(inTile))))),     \
                      XAI_ERR_BADARG, "Output and Input tile edges constraints have not been met  along dimension 3");                     \
      XAI_CHECK_ERROR(((size_t) (XAI_TILE4D_GET_BUFF_PTR(inTile)) <= ((size_t) (XAI_TILE4D_GET_BUFF_PTR(outTile)))), XAI_ERR_BADARG,       \
                      "Output tile buffer pointer should be greater than or equal to input tile buffer pointer");                          \
    }                                                                                                                                      \
  }
#define XAI_CHECK_TILES3D_CHECK_EDGES_DEQUANT(inTile, outTile)                                                                            \
  {                                                                                                                                       \
    if (XAI_TILE3D_GET_DATA_PTR(inTile) == XAI_TILE3D_GET_DATA_PTR(outTile))                                                              \
    {                                                                                                                                     \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) + XAI_TILE3D_GET_DIM1_EDGE2(inTile)) <=                                         \
                       (2 * (XAI_TILE3D_GET_DIM1_PITCH(outTile) - XAI_TILE3D_GET_DIM1(outTile)))), XAI_ERR_BADARG,                        \
                      "Output and Input tile edges constraints have not been met along dimension 1");                                     \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_PITCH(inTile) * (XAI_TILE3D_GET_DIM2_EDGE1(inTile) + XAI_TILE3D_GET_DIM2_EDGE2(inTile))) <=   \
                       (2 * (XAI_TILE3D_GET_DIM2_PITCH(outTile) - (XAI_TILE3D_GET_DIM1_PITCH(outTile) * XAI_TILE3D_GET_DIM2(outTile))))), \
                      XAI_ERR_BADARG, "Output and Input tile edges constraints have not been met  along dimension 2");                    \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM3_EDGE1(inTile) + XAI_TILE3D_GET_DIM3_EDGE2(inTile)) <=                                         \
                       (2 * (XAI_TILE3D_GET_DIM3_EDGE1(outTile) + XAI_TILE3D_GET_DIM3_EDGE2(outTile)))), XAI_ERR_BADARG,                  \
                      "Output and Input tile edges constraints have not been met  along dimension 3");                                    \
      XAI_CHECK_ERROR(((size_t) (XAI_TILE3D_GET_BUFF_PTR(outTile)) <= ((size_t) (XAI_TILE3D_GET_BUFF_PTR(inTile)))), XAI_ERR_BADARG,      \
                      "Input tile buffer pointer should be greater than or equal to output tile buffer pointer");                         \
    }                                                                                                                                     \
  }
#define XAI_CHECK_TILES4D_CHECK_EDGES_DEQUANT(inTile, outTile)                                                                            \
  {                                                                                                                                       \
    if (XAI_TILE4D_GET_DATA_PTR(inTile) == XAI_TILE4D_GET_DATA_PTR(outTile))                                                              \
    {                                                                                                                                     \
      XAI_CHECK_ERROR(((XAI_TILE4D_GET_DIM1_EDGE1(inTile) + XAI_TILE4D_GET_DIM1_EDGE2(inTile)) <=                                         \
                       (2 * (XAI_TILE4D_GET_DIM1_PITCH(outTile) - XAI_TILE4D_GET_DIM1(outTile)))), XAI_ERR_BADARG,                        \
                      "Output and Input tile edges constraints have not been met along dimension 1");                                     \
      XAI_CHECK_ERROR(((XAI_TILE4D_GET_DIM1_PITCH(inTile) * (XAI_TILE4D_GET_DIM2_EDGE1(inTile) + XAI_TILE4D_GET_DIM2_EDGE2(inTile))) <=   \
                       (2 * (XAI_TILE4D_GET_DIM2_PITCH(outTile) - (XAI_TILE3D_GET_DIM1_PITCH(outTile) * XAI_TILE4D_GET_DIM2(outTile))))), \
                      XAI_ERR_BADARG, "Output and Input tile edges constraints have not been met  along dimension 2");                    \
      XAI_CHECK_ERROR(((XAI_TILE4D_GET_DIM2_PITCH(inTile) * (XAI_TILE4D_GET_DIM3_EDGE1(inTile) + XAI_TILE4D_GET_DIM3_EDGE2(inTile))) <=   \
                       (2 * (XAI_TILE4D_GET_DIM3_PITCH(outTile) - (XAI_TILE3D_GET_DIM2_PITCH(outTile) * XAI_TILE4D_GET_DIM3(outTile))))), \
                      XAI_ERR_BADARG, "Output and Input tile edges constraints have not been met  along dimension 3");                    \
      XAI_CHECK_ERROR(((size_t) (XAI_TILE4D_GET_BUFF_PTR(outTile)) <= ((size_t) (XAI_TILE4D_GET_BUFF_PTR(inTile)))), XAI_ERR_BADARG,      \
                      "Input tile buffer pointer should be greater than or equal to output tile buffer pointer");                         \
    }                                                                                                                                     \
  }

#define XAI_CHECK_EDGES_MOD_DWH_IN16DWH(inTile, coeffTile, param)                                                 \
  uint16_t dilatedKW = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM2(coeffTile) - 1) + 1); \
  uint16_t dilatedKH = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM3(coeffTile) - 1) + 1); \
  if (dilatedKW % 2 != 0)                                                                                         \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                        \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                     \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                     \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                    \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                        \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                          \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1))                           \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
  }                                                                                                               \
  else                                                                                                            \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                  \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) - 1)                             \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1))                              \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
    }                                                                                                             \
  }

#if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
#define XAI_CHECK_EDGES_F16_MOD_DWH(inTile, coeffTile, param)                                                     \
  uint16_t dilatedKW = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM3(coeffTile) - 1) + 1); \
  uint16_t dilatedKH = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM4(coeffTile) - 1) + 1); \
  if (dilatedKW % 2 != 0)                                                                                         \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                        \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                     \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                     \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                    \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                        \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                          \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1))                           \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
  }                                                                                                               \
  else                                                                                                            \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                  \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) - 1)                             \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1))                              \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
    }                                                                                                             \
  }

#define XAI_CHECK_EDGES_DEPTHWISE_F16_MOD_DWH(inTile, coeffTile, param)                        \
  int32_t kW = XAI_TILE3D_GET_DIM2(coeffTile);                                                 \
  int32_t kH = XAI_TILE3D_GET_DIM3(coeffTile);                                                 \
  if (kW % 2 != 0)                                                                             \
  {                                                                                            \
    if (kH % 2 != 0)                                                                           \
    {                                                                                          \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                            \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                         \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                         \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                        \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");     \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),              \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1))               \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
  }                                                                                            \
  else                                                                                         \
  {                                                                                            \
    if (kH % 2 != 0)                                                                           \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                               \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) - 1)                 \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1))                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kH / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kH / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kH / 2)),                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kH / 2)),                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
    }                                                                                          \
  }
#define XAI_CHECK_CONSISTENCY_DEPTHWISE_F16_MOD_DWH(inT, coeffT, biasArr, outT, param)                                           \
  int32_t KW_MOD = XAI_TILE3D_GET_DIM2(coeffT);                                                                                  \
  int32_t KH_MOD = XAI_TILE3D_GET_DIM3(coeffT);                                                                                  \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(inT) == XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                     \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outT) == XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                    \
                  "Number of Output Channels not equal to the number of channels in the Kernel.");                               \
  if (KW_MOD % 2 != 0)                                                                                                           \
  {                                                                                                                              \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                    \
                                                     + (KW_MOD >> 1) - KW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                         \
  }                                                                                                                              \
  else                                                                                                                           \
  {                                                                                                                              \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                    \
                                                     + ((KW_MOD >> 1) - 1) - KW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                         \
  }                                                                                                                              \
  if (KH_MOD % 2 != 0)                                                                                                           \
  {                                                                                                                              \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                    \
                                                     + (KH_MOD >> 1) - KH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                        \
  }                                                                                                                              \
  else                                                                                                                           \
  {                                                                                                                              \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                    \
                                                     + ((KH_MOD >> 1) - 1) - KH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                        \
  }                                                                                                                              \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                 \
                  "Width of Bias Array is less than number of channels in the Kernel.");                                         \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArr) > 0, XAI_ERR_DATASIZE,                                                           \
                  "Height of Bias Array should be greater than zero.");
#define XAI_CHECK_CONV_RELU_LIMITS_F16(param, outTile)  {                                                                                                                                   \
    if (XAI_CNN_CONV_GET_FLAG_RELU(param))                                                                                                                                                  \
    {                                                                                                                                                                                       \
      XAI_CHECK_ERROR((XAI_CNN_CONV_GET_RELU_MIN_FLT(param) <= XAI_CNN_CONV_GET_RELU_MAX_FLT(param)), XAI_ERR_BADARG,                                                                       \
                      "\nMinimum Value of RELU = %f,\nMaximum Value of RELU = %f , Min Limit should not be greater than Max Limit",                                                         \
                      CONVERT_FP16_TO_FP32(XAI_CNN_CONV_GET_RELU_MIN_FLT(param)), CONVERT_FP16_TO_FP32(XAI_CNN_CONV_GET_RELU_MAX_FLT(param)));                                              \
      XAI_CHECK_ERROR((XAI_CNN_CONV_GET_RELU_MIN_FLT(param) >= XAI_F16_MIN &&                                                                                                               \
                       XAI_CNN_CONV_GET_RELU_MAX_FLT(param) <= XAI_F16_MAX), XAI_ERR_BADARG,                                                                                                \
                      "\nMinimum Value of RELU = %f, value should be greater than or equal to XAI_F16_MIN \nMaximum Value of RELU = %f, value should be less than or equal to XAI_F16_MAX", \
                      CONVERT_FP16_TO_FP32(XAI_CNN_CONV_GET_RELU_MIN_FLT(param)), CONVERT_FP16_TO_FP32(XAI_CNN_CONV_GET_RELU_MAX_FLT(param)));                                              \
    }                                                                                                                                                                                       \
}
#endif //if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))

#if (XCHAL_HAVE_VISION_SP_VFPU == 1 || XCHAL_HAVE_BBENEP_SP_VFPU == 1 || defined(XAI_REF_ONLY_COMPILATION))
#define XAI_CHECK_EDGES_F32_MOD_DWH(inTile, coeffTile, param)                                                     \
  uint16_t dilatedKW = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM3(coeffTile) - 1) + 1); \
  uint16_t dilatedKH = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM4(coeffTile) - 1) + 1); \
  if (dilatedKW % 2 != 0)                                                                                         \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                        \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                     \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                     \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                    \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                        \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                          \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1))                           \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
  }                                                                                                               \
  else                                                                                                            \
  {                                                                                                               \
    if (dilatedKH % 2 != 0)                                                                                       \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                  \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= dilatedKW / 2)                                      \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) - 1)                             \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1))                              \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= dilatedKW / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2)                                   \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                  \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                      \
      }                                                                                                           \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                   \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((dilatedKH / 2) - 1)),                           \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
      else                                                                                                        \
      {                                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
        else                                                                                                      \
        {                                                                                                         \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((dilatedKW / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2) &&                                \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((dilatedKH / 2) - 1) &&                          \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2)),                                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                    \
        }                                                                                                         \
      }                                                                                                           \
    }                                                                                                             \
  }
#define XAI_CHECK_EDGES_DEPTHWISE_F32_MOD_DWH(inTile, coeffTile, param)                        \
  int32_t kW = XAI_TILE3D_GET_DIM2(coeffTile);                                                 \
  int32_t kH = XAI_TILE3D_GET_DIM3(coeffTile);                                                 \
  if (kW % 2 != 0)                                                                             \
  {                                                                                            \
    if (kH % 2 != 0)                                                                           \
    {                                                                                          \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                            \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                         \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                         \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                        \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");     \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),              \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1))               \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
  }                                                                                            \
  else                                                                                         \
  {                                                                                            \
    if (kH % 2 != 0)                                                                           \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                               \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) - 1)                 \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1))                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kH / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kH / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kH / 2)),                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kH / 2)),                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
    }                                                                                          \
  }
#define XAI_CHECK_CONSISTENCY_DEPTHWISE_F32_MOD_DWH(inT, coeffT, biasArr, outT, param)                                           \
  int32_t KW_MOD = XAI_TILE3D_GET_DIM2(coeffT);                                                                                  \
  int32_t KH_MOD = XAI_TILE3D_GET_DIM3(coeffT);                                                                                  \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(inT) == XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                     \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outT) == XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                    \
                  "Number of Output Channels not equal to the number of channels in the Kernel.");                               \
  if (KW_MOD % 2 != 0)                                                                                                           \
  {                                                                                                                              \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                    \
                                                     + (KW_MOD >> 1) - KW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                         \
  }                                                                                                                              \
  else                                                                                                                           \
  {                                                                                                                              \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                    \
                                                     + ((KW_MOD >> 1) - 1) - KW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                         \
  }                                                                                                                              \
  if (KH_MOD % 2 != 0)                                                                                                           \
  {                                                                                                                              \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                    \
                                                     + (KH_MOD >> 1) - KH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                        \
  }                                                                                                                              \
  else                                                                                                                           \
  {                                                                                                                              \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                    \
                                                     + ((KH_MOD >> 1) - 1) - KH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                        \
  }                                                                                                                              \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                 \
                  "Width of Bias Array is less than number of channels in the Kernel.");                                         \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArr) > 0, XAI_ERR_DATASIZE,                                                           \
                  "Height of Bias Array should be greater than zero.");
#define XAI_CHECK_CONV_RELU_LIMITS_F32(param, outTile)  {                                                                                                                                           \
    if (XAI_CNN_CONV_GET_FLAG_RELU(param))                                                                                                                                                          \
    {                                                                                                                                                                                               \
      XAI_CHECK_ERROR((XAI_CNN_CONV_GET_RELU_MIN_FLT32(param) <= XAI_CNN_CONV_GET_RELU_MAX_FLT32(param)), XAI_ERR_BADARG,                                                                           \
                      "\nMinimum Value of RELU = %f,\nMaximum Value of RELU = %f , Min Limit should not be greater than Max Limit",                                                                 \
                      XAI_CNN_CONV_GET_RELU_MIN_FLT32(param), XAI_CNN_CONV_GET_RELU_MAX_FLT32(param));                                                                                              \
      XAI_CHECK_ERROR((XAI_CNN_CONV_GET_RELU_MIN_FLT32(param) >= XAI_F32_MIN_FLT &&                                                                                                                 \
                       XAI_CNN_CONV_GET_RELU_MAX_FLT32(param) <= XAI_F32_MAX_FLT), XAI_ERR_BADARG,                                                                                                  \
                      "\nMinimum Value of RELU = %f, value should be greater than or equal to XAI_F32_MIN_FLT \nMaximum Value of RELU = %f, value should be less than or equal to XAI_F32_MAX_FLT", \
                      XAI_CNN_CONV_GET_RELU_MIN_FLT32(param), XAI_CNN_CONV_GET_RELU_MAX_FLT32(param));                                                                                              \
    }                                                                                                                                                                                               \
}
#endif //if (XCHAL_HAVE_VISION_SP_VFPU == 1 || XCHAL_HAVE_BBENEP_SP_VFPU == 1 || defined(XAI_REF_ONLY_COMPILATION))

#define XAI_CHECK_EDGES_SO(inTile, coeffTile, param)                                                              \
  uint16_t dilatedKW = (uint16_t) (XAI_CNN_CONV_GET_DILATIONX(param) * (XAI_TILE4D_GET_DIM2(coeffTile) - 1) + 1); \
  uint16_t dilatedKH = (uint16_t) (XAI_CNN_CONV_GET_DILATIONY(param) * (XAI_TILE4D_GET_DIM3(coeffTile) - 1) + 1); \
  if (dilatedKW % 2 != 0)                                                                                         \
  {                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2)) &&                                     \
                    (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2)),                                       \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                            \
  }                                                                                                               \
  else                                                                                                            \
  {                                                                                                               \
    if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                                    \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2)) &&                                   \
                      (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2 - 1)),                                 \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag set");                 \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (dilatedKW / 2 - 1)) &&                               \
                      (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (dilatedKW / 2)),                                     \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with left edge flag reset");               \
    }                                                                                                             \
  }                                                                                                               \
  if (dilatedKH % 2 != 0)                                                                                         \
  {                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= dilatedKH / 2) &&                                       \
                    (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= dilatedKH / 2),                                         \
                    XAI_ERR_EDGE, "Invalid edge for odd kernel size");                                            \
  }                                                                                                               \
  else                                                                                                            \
  {                                                                                                               \
    if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                     \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2)) &&                                   \
                      (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2 - 1)),                                 \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag set");                  \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (dilatedKH / 2 - 1)) &&                               \
                      (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (dilatedKH / 2)),                                     \
                      XAI_ERR_EDGE, "Invalid edge for even kernel size with top edge flag reset");                \
    }                                                                                                             \
  }                                                                                                               \

#define XAI_CHECK_EDGES_MOD_ID16WH(inTile, coeffT, param)                                         \
  int32_t kWidthMOD, kHeightMOD;                                                                  \
  uint16_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);                                         \
  uint16_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);                                         \
  kWidthMOD  = dilationX * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1;                                 \
  kHeightMOD = dilationY * (XAI_TILE4D_GET_DIM2(coeffT) - 1) + 1;                                 \
  if (kWidthMOD % 2 != 0)                                                                         \
  {                                                                                               \
    if (kHeightMOD % 2 != 0)                                                                      \
    {                                                                                             \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= kWidthMOD / 2)                 \
                      && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= kWidthMOD / 2)              \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                    \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                   \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");        \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                   \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= kWidthMOD / 2)               \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= kWidthMOD / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1)),         \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
      else                                                                                        \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= kWidthMOD / 2)               \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= kWidthMOD / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1))          \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
  else                                                                                            \
  {                                                                                               \
    if (kHeightMOD % 2 != 0)                                                                      \
    {                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                  \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= kWidthMOD / 2)               \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= (kWidthMOD / 2) - 1)      \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
      else                                                                                        \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= ((kWidthMOD / 2) - 1))       \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= kWidthMOD / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                   \
      {                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= (kWidthMOD / 2)) &&       \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= ((kWidthMOD / 2) - 1)) && \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kHeightMOD / 2)) &&             \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1))),        \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= ((kWidthMOD / 2) - 1)) && \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= (kWidthMOD / 2)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kHeightMOD / 2)) &&             \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1))),        \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
      }                                                                                           \
      else                                                                                        \
      {                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= (kWidthMOD / 2)) &&       \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= ((kWidthMOD / 2) - 1)) && \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kHeightMOD / 2))),              \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= ((kWidthMOD / 2) - 1)) && \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= (kWidthMOD / 2)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kHeightMOD / 2))),              \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
      }                                                                                           \
    }                                                                                             \
  }

#define XAI_CHECK_EDGES_MOD_ID32WH(inTile, coeffT, param)                                         \
  int32_t kWidthMOD, kHeightMOD;                                                                  \
  uint16_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);                                         \
  uint16_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);                                         \
  kWidthMOD  = dilationX * (XAI_TILE4D_GET_DIM3(coeffT) - 1) + 1;                                 \
  kHeightMOD = dilationY * (XAI_TILE4D_GET_DIM2(coeffT) - 1) + 1;                                 \
  if (kWidthMOD % 2 != 0)                                                                         \
  {                                                                                               \
    if (kHeightMOD % 2 != 0)                                                                      \
    {                                                                                             \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= kWidthMOD / 2)                 \
                      && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= kWidthMOD / 2)              \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                    \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                   \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");        \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                   \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= kWidthMOD / 2)               \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= kWidthMOD / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1)),         \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
      else                                                                                        \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= kWidthMOD / 2)               \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= kWidthMOD / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1))          \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
  else                                                                                            \
  {                                                                                               \
    if (kHeightMOD % 2 != 0)                                                                      \
    {                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                  \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= kWidthMOD / 2)               \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= (kWidthMOD / 2) - 1)      \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
      else                                                                                        \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= ((kWidthMOD / 2) - 1))       \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= kWidthMOD / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                   \
      {                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= (kWidthMOD / 2)) &&       \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= ((kWidthMOD / 2) - 1)) && \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kHeightMOD / 2)) &&             \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1))),        \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= ((kWidthMOD / 2) - 1)) && \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= (kWidthMOD / 2)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kHeightMOD / 2)) &&             \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1))),        \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
      }                                                                                           \
      else                                                                                        \
      {                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= (kWidthMOD / 2)) &&       \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= ((kWidthMOD / 2) - 1)) && \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kHeightMOD / 2))),              \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= ((kWidthMOD / 2) - 1)) && \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= (kWidthMOD / 2)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kHeightMOD / 2))),              \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
      }                                                                                           \
    }                                                                                             \
  }
#define XAI_CHECK_EDGES_DEPTHWISE_MOD_ID16WH(inTile, coeffT, param)                               \
  int32_t kWidthMOD, kHeightMOD;                                                                  \
  uint16_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);                                         \
  uint16_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);                                         \
  kWidthMOD  = dilationX * (XAI_TILE3D_GET_DIM2(coeffT) - 1) + 1;                                 \
  kHeightMOD = dilationY * (XAI_TILE3D_GET_DIM3(coeffT) - 1) + 1;                                 \
  if (kWidthMOD % 2 != 0)                                                                         \
  {                                                                                               \
    if (kHeightMOD % 2 != 0)                                                                      \
    {                                                                                             \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= kWidthMOD / 2)                 \
                      && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= kWidthMOD / 2)              \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                    \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                   \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");        \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                   \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= kWidthMOD / 2)               \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= kWidthMOD / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1)),         \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
      else                                                                                        \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= kWidthMOD / 2)               \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= kWidthMOD / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1))          \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
  else                                                                                            \
  {                                                                                               \
    if (kHeightMOD % 2 != 0)                                                                      \
    {                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                  \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= kWidthMOD / 2)               \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= (kWidthMOD / 2) - 1)      \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
      else                                                                                        \
      {                                                                                           \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= ((kWidthMOD / 2) - 1))       \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= kWidthMOD / 2)            \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
      }                                                                                           \
    }                                                                                             \
    else                                                                                          \
    {                                                                                             \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                   \
      {                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= (kWidthMOD / 2)) &&       \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= ((kWidthMOD / 2) - 1)) && \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kHeightMOD / 2)) &&             \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1))),        \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= ((kWidthMOD / 2) - 1)) && \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= (kWidthMOD / 2)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kHeightMOD / 2)) &&             \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1))),        \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
      }                                                                                           \
      else                                                                                        \
      {                                                                                           \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= (kWidthMOD / 2)) &&       \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= ((kWidthMOD / 2) - 1)) && \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kHeightMOD / 2))),              \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
          XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 4) >= ((kWidthMOD / 2) - 1)) && \
                           ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 4) >= (kWidthMOD / 2)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1)) &&       \
                           (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kHeightMOD / 2))),              \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
        }                                                                                         \
      }                                                                                           \
    }                                                                                             \
  }

#define XAI_CHECK_CONSISTENCY_DEPTHWISE_MOD_ID16WH(inTile, coeffT, outTile, param)                                                                                        \
  {                                                                                                                                                                       \
    uint16_t dilationX     = XAI_CNN_CONV_GET_DILATIONX(param);                                                                                                           \
    uint16_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);                                                                                                           \
    int32_t dilatedkWidth  = dilationX * (XAI_TILE3D_GET_DIM2(coeffT) - 1) + 1;                                                                                           \
    int32_t dilatedkHeight = dilationY * (XAI_TILE3D_GET_DIM3(coeffT) - 1) + 1;                                                                                           \
    XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM2(inTile) << 4) == (XAI_TILE3D_GET_DIM2(outTile) << 4)),                                                                          \
                    XAI_ERR_DATASIZE, "Number of input and output channel should be equal.");                                                                             \
    if (dilatedkWidth % 2 != 0)                                                                                                                                           \
    {                                                                                                                                                                     \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outTile) >> 4) <= ((((XAI_TILE3D_GET_DIM1(inTile) >> 4) +                                                                     \
                                                                 (dilatedkWidth >> 1) + (dilatedkWidth >> 1) - dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)), \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                \
    }                                                                                                                                                                     \
    else                                                                                                                                                                  \
    {                                                                                                                                                                     \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outTile) >> 4) <= ((((XAI_TILE3D_GET_DIM1(inTile) >> 4) +                                                                     \
                                                                 (dilatedkWidth >> 1) + ((dilatedkWidth >> 1) - 1) -                                                      \
                                                                 dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),                                               \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                \
    }                                                                                                                                                                     \
    if (dilatedkHeight % 2 != 0)                                                                                                                                          \
    {                                                                                                                                                                     \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outTile) <= (((XAI_TILE3D_GET_DIM3(inTile) + (dilatedkHeight >> 1) +                                                           \
                                                          (dilatedkHeight >> 1) - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                             \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                               \
    }                                                                                                                                                                     \
    else                                                                                                                                                                  \
    {                                                                                                                                                                     \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outTile) <= (((XAI_TILE3D_GET_DIM3(inTile) + (dilatedkHeight >> 1) +                                                           \
                                                          ((dilatedkHeight >> 1) - 1) - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                       \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                               \
    }                                                                                                                                                                     \
  }

#define XAI_CHECK_EDGES_DEPTHWISE_MOD_ID32WH(inTile, coeffT, param)  {                              \
    int32_t kWidthMOD, kHeightMOD;                                                                  \
    uint16_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);                                         \
    uint16_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);                                         \
    kWidthMOD  = dilationX * (XAI_TILE3D_GET_DIM2(coeffT) - 1) + 1;                                 \
    kHeightMOD = dilationY * (XAI_TILE3D_GET_DIM3(coeffT) - 1) + 1;                                 \
    if (kWidthMOD % 2 != 0)                                                                         \
    {                                                                                               \
      if (kHeightMOD % 2 != 0)                                                                      \
      {                                                                                             \
        XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= kWidthMOD / 2)                 \
                        && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= kWidthMOD / 2)              \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                    \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                   \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");        \
      }                                                                                             \
      else                                                                                          \
      {                                                                                             \
        if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                   \
        {                                                                                           \
          XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= kWidthMOD / 2)               \
                          && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= kWidthMOD / 2)            \
                          && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                          && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1)),         \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
          XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= kWidthMOD / 2)               \
                          && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= kWidthMOD / 2)            \
                          && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1))          \
                          && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
        }                                                                                           \
      }                                                                                             \
    }                                                                                               \
    else                                                                                            \
    {                                                                                               \
      if (kHeightMOD % 2 != 0)                                                                      \
      {                                                                                             \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                  \
        {                                                                                           \
          XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= kWidthMOD / 2)               \
                          && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= (kWidthMOD / 2) - 1)      \
                          && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                          && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
          XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= ((kWidthMOD / 2) - 1))       \
                          && ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= kWidthMOD / 2)            \
                          && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kHeightMOD / 2)                  \
                          && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kHeightMOD / 2),                 \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");      \
        }                                                                                           \
      }                                                                                             \
      else                                                                                          \
      {                                                                                             \
        if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                   \
        {                                                                                           \
          if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                \
          {                                                                                         \
            XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= (kWidthMOD / 2)) &&       \
                             ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= ((kWidthMOD / 2) - 1)) && \
                             (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kHeightMOD / 2)) &&             \
                             (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1))),        \
                            XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
          }                                                                                         \
          else                                                                                      \
          {                                                                                         \
            XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= ((kWidthMOD / 2) - 1)) && \
                             ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= (kWidthMOD / 2)) &&       \
                             (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kHeightMOD / 2)) &&             \
                             (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kHeightMOD / 2) - 1))),        \
                            XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
          }                                                                                         \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
          if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                \
          {                                                                                         \
            XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= (kWidthMOD / 2)) &&       \
                             ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= ((kWidthMOD / 2) - 1)) && \
                             (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1)) &&       \
                             (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kHeightMOD / 2))),              \
                            XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
          }                                                                                         \
          else                                                                                      \
          {                                                                                         \
            XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >> 5) >= ((kWidthMOD / 2) - 1)) && \
                             ((XAI_TILE3D_GET_DIM1_EDGE2(inTile) >> 5) >= (kWidthMOD / 2)) &&       \
                             (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kHeightMOD / 2) - 1)) &&       \
                             (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kHeightMOD / 2))),              \
                            XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");    \
          }                                                                                         \
        }                                                                                           \
      }                                                                                             \
    }                                                                                               \
}
#define XAI_CHECK_CONSISTENCY_DEPTHWISE_MOD_ID32WH(inTile, coeffT, outTile, param)                                                                                        \
  {                                                                                                                                                                       \
    uint16_t dilationX     = XAI_CNN_CONV_GET_DILATIONX(param);                                                                                                           \
    uint16_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);                                                                                                           \
    int32_t dilatedkWidth  = dilationX * (XAI_TILE3D_GET_DIM2(coeffT) - 1) + 1;                                                                                           \
    int32_t dilatedkHeight = dilationY * (XAI_TILE3D_GET_DIM3(coeffT) - 1) + 1;                                                                                           \
    XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM2(inTile) << 5) == (XAI_TILE3D_GET_DIM2(outTile) << 5)),                                                                          \
                    XAI_ERR_DATASIZE, "Number of input and output channel should be equal.");                                                                             \
    if (dilatedkWidth % 2 != 0)                                                                                                                                           \
    {                                                                                                                                                                     \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outTile) >> 5) <= ((((XAI_TILE3D_GET_DIM1(inTile) >> 5) +                                                                     \
                                                                 (dilatedkWidth >> 1) + (dilatedkWidth >> 1) - dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)), \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                \
    }                                                                                                                                                                     \
    else                                                                                                                                                                  \
    {                                                                                                                                                                     \
      XAI_CHECK_ERROR(((XAI_TILE3D_GET_DIM1(outTile) >> 5) <= ((((XAI_TILE3D_GET_DIM1(inTile) >> 5) +                                                                     \
                                                                 (dilatedkWidth >> 1) + ((dilatedkWidth >> 1) - 1) -                                                      \
                                                                 dilatedkWidth) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),                                               \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                                                \
    }                                                                                                                                                                     \
    if (dilatedkHeight % 2 != 0)                                                                                                                                          \
    {                                                                                                                                                                     \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outTile) <= (((XAI_TILE3D_GET_DIM3(inTile) + (dilatedkHeight >> 1) +                                                           \
                                                          (dilatedkHeight >> 1) - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                             \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                               \
    }                                                                                                                                                                     \
    else                                                                                                                                                                  \
    {                                                                                                                                                                     \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outTile) <= (((XAI_TILE3D_GET_DIM3(inTile) + (dilatedkHeight >> 1) +                                                           \
                                                          ((dilatedkHeight >> 1) - 1) - dilatedkHeight) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),                       \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                                               \
    }                                                                                                                                                                     \
  }

#define XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile)                               {                                                                                              \
    if (XAI_CNN_CONV_GET_FLAG_RELU(param))                                                                                                                                         \
    {                                                                                                                                                                              \
      XAI_CHECK_ERROR((XAI_CNN_CONV_GET_RELU_MIN(param) <= XAI_CNN_CONV_GET_RELU_MAX(param)), XAI_ERR_BADARG,                                                                      \
                      "\nMinimum Value of RELU = %d,\nMaximum Value of RELU = %d , Min Limit should not be greater than Max Limit",                                                \
                      XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param));                                                                                         \
      if (XAI_TYPE_ELEMENT_TYPE(outTile->type) == XAI_U8)                                                                                                                          \
      {                                                                                                                                                                            \
        XAI_CHECK_ERROR((XAI_CNN_CONV_GET_RELU_MIN(param) >= 0 &&                                                                                                                  \
                         XAI_CNN_CONV_GET_RELU_MAX(param) <= UCHAR_MAX), XAI_ERR_BADARG,                                                                                           \
                        "\nMinimum Value of RELU = %d, value should be greater than or equal to 0 \nMaximum Value of RELU = %d, value should be less than or equal to 255",        \
                        XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param));                                                                                       \
      }                                                                                                                                                                            \
      else if (XAI_TYPE_ELEMENT_TYPE(outTile->type) == XAI_S8)                                                                                                                     \
      {                                                                                                                                                                            \
        XAI_CHECK_ERROR((XAI_CNN_CONV_GET_RELU_MIN(param) >= SCHAR_MIN &&                                                                                                          \
                         XAI_CNN_CONV_GET_RELU_MAX(param) <= SCHAR_MAX), XAI_ERR_BADARG,                                                                                           \
                        "\nMinimum Value of RELU = %d, value should be greater than or equal to -128 \nMaximum Value of RELU = %d, value should be less than or equal to 127",     \
                        XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param));                                                                                       \
      }                                                                                                                                                                            \
      else if (XAI_TYPE_ELEMENT_TYPE(outTile->type) == XAI_S16)                                                                                                                    \
      {                                                                                                                                                                            \
        XAI_CHECK_ERROR((XAI_CNN_CONV_GET_RELU_MIN(param) >= SHRT_MIN &&                                                                                                           \
                         XAI_CNN_CONV_GET_RELU_MAX(param) <= SHRT_MAX), XAI_ERR_BADARG,                                                                                            \
                        "\nMinimum Value of RELU = %d, value should be greater than or equal to -32768 \nMaximum Value of RELU = %d, value should be less than or equal to 32767", \
                        XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param));                                                                                       \
      }                                                                                                                                                                            \
      else if (XAI_TYPE_ELEMENT_TYPE(outTile->type) == XAI_U16)                                                                                                                    \
      {                                                                                                                                                                            \
        XAI_CHECK_ERROR((XAI_CNN_CONV_GET_RELU_MIN(param) >= 0 &&                                                                                                                  \
                         XAI_CNN_CONV_GET_RELU_MAX(param) <= USHRT_MAX), XAI_ERR_BADARG,                                                                                           \
                        "\nMinimum Value of RELU = %d, value should be greater than or equal to 0 \nMaximum Value of RELU = %d, value should be less than or equal to 65535",      \
                        XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param));                                                                                       \
      }                                                                                                                                                                            \
      else                                                                                                                                                                         \
      {                                                                                                                                                                            \
        XAI_CHECK_ERROR(0, XAI_ERR_NO_VARIANT, "Output tile datatype is not supported by XAI_CHECK_CONV_RELU_LIMITS_IX");                                                          \
      }                                                                                                                                                                            \
    }                                                                                                                                                                              \
}

#define XAI_CHECK_DEPTHWISE_DILATED_CONV_RELU_LIMITS_IX(param, outTile)             {                                                               \
    if (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_FLAG_RELU(param))                                                                                        \
    {                                                                                                                                               \
      XAI_CHECK_ERROR((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param) <= XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param)),                   \
                      XAI_ERR_BADARG, "\nMinimum Value of RELU = %d,\nMaximum Value of RELU = %d , Min Limit should not be greater than Max Limit", \
                      XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param), XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param));                      \
      if (XAI_TYPE_ELEMENT_TYPE(outTile->type) == XAI_U8)                                                                                           \
      {                                                                                                                                             \
        XAI_CHECK_ERROR((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param) >= 0 &&                                                                 \
                         XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param) <= UCHAR_MAX), XAI_ERR_BADARG,                                          \
                        "\nMinimum Value of RELU = %d, value should be greater than or equal to 0 \nMaximum Value of RELU = %d,"                    \
                        "value should be less than or equal to 255",                                                                                \
                        XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param), XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param));                    \
      }                                                                                                                                             \
      else if (XAI_TYPE_ELEMENT_TYPE(outTile->type) == XAI_S8)                                                                                      \
      {                                                                                                                                             \
        XAI_CHECK_ERROR((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param) >= SCHAR_MIN &&                                                         \
                         XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param) <= SCHAR_MAX), XAI_ERR_BADARG,                                          \
                        "\nMinimum Value of RELU = %d, value should be greater than or equal to -128 \nMaximum Value of RELU = %d,"                 \
                        "value should be less than or equal to 127",                                                                                \
                        XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param), XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param));                    \
      }                                                                                                                                             \
      else if (XAI_TYPE_ELEMENT_TYPE(outTile->type) == XAI_S16)                                                                                     \
      {                                                                                                                                             \
        XAI_CHECK_ERROR((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param) >= SHRT_MIN &&                                                          \
                         XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param) <= SHRT_MAX), XAI_ERR_BADARG,                                           \
                        "\nMinimum Value of RELU = %d, value should be greater than or equal to -32768 \nMaximum Value of RELU = %d,"               \
                        "value should be less than or equal to 32767",                                                                              \
                        XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param), XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param));                    \
      }                                                                                                                                             \
      else if (XAI_TYPE_ELEMENT_TYPE(outTile->type) == XAI_U16)                                                                                     \
      {                                                                                                                                             \
        XAI_CHECK_ERROR((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param) >= 0 &&                                                                 \
                         XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param) <= USHRT_MAX), XAI_ERR_BADARG,                                          \
                        "\nMinimum Value of RELU = %d, value should be greater than or equal to 0 \nMaximum Value of RELU = %d,"                    \
                        "value should be less than or equal to 65535",                                                                              \
                        XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MIN(param), XAI_CNN_DEPTHWISE_DILATED_CONV_GET_RELU_MAX(param));                    \
      }                                                                                                                                             \
      else                                                                                                                                          \
      {                                                                                                                                             \
        XAI_CHECK_ERROR(0, XAI_ERR_NO_VARIANT, "Output tile datatype is not supported by XAI_CHECK_DEPTHWISE_DILATED_CONV_RELU_LIMITS_IX");         \
      }                                                                                                                                             \
    }                                                                                                                                               \
}

#define XAI_CHECK_CONSISTENCY_DEPTHWISE_MOD_DWH(inT, coeffT, biasArr, outT, param)  {                                              \
    int32_t KW_MOD = XAI_TILE3D_GET_DIM2(coeffT);                                                                                  \
    int32_t KH_MOD = XAI_TILE3D_GET_DIM3(coeffT);                                                                                  \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(inT) == XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                     \
                    "Number of Input Channels not equal to the number of channels in the Kernel.");                                \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outT) == XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                    \
                    "Number of Output Channels not equal to the number of channels in the Kernel.");                               \
    if (KW_MOD % 2 != 0)                                                                                                           \
    {                                                                                                                              \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                    \
                                                       + (KW_MOD >> 1) - KW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)),       \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                         \
    }                                                                                                                              \
    else                                                                                                                           \
    {                                                                                                                              \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                    \
                                                       + ((KW_MOD >> 1) - 1) - KW_MOD) / (XAI_CNN_CONV_GET_STRIDEX(param))) + 1)), \
                      XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                         \
    }                                                                                                                              \
    if (KH_MOD % 2 != 0)                                                                                                           \
    {                                                                                                                              \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                    \
                                                       + (KH_MOD >> 1) - KH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)),       \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                        \
    }                                                                                                                              \
    else                                                                                                                           \
    {                                                                                                                              \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                    \
                                                       + ((KH_MOD >> 1) - 1) - KH_MOD) / (XAI_CNN_CONV_GET_STRIDEY(param))) + 1)), \
                      XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                        \
    }                                                                                                                              \
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                 \
                    "Width of Bias Array is less than number of channels in the Kernel.");                                         \
    XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArr) > 0, XAI_ERR_DATASIZE,                                                           \
                    "Height of Bias Array should be greater than zero.");                                                          \
}

#if (((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1)) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))
  #define XAI_CHECK_CONSISTENCY_F16_MOD_DWH(inT, coeffT, biasArr, outT, param)  {                                                        \
    int32_t KW_MOD = (XAI_TILE3D_GET_DIM2(coeffT) - 1) * XAI_CNN_CONV_GET_DILATIONX(param) + 1;                                          \
    int32_t KH_MOD = (XAI_TILE3D_GET_DIM3(coeffT) - 1) * XAI_CNN_CONV_GET_DILATIONY(param) + 1;                                          \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(inT) == XAI_TILE3D_GET_DIM2(coeffT), XAI_ERR_DATASIZE,                                           \
                    "Number of Input Channels not equal to the number of channels in the Kernel.");                                      \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outT) == XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                          \
                    "Number of Output Channels not equal to the number of channels in the Kernel.");                                     \
    if (KW_MOD % 2 != 0)                                                                                                                 \
    {                                                                                                                                    \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                          \
                                                       + (KW_MOD >> 1) - KW_MOD) >> (XAI_CNN_CONV_GET_STRIDEX(param) >> 1)) + 1)),       \
                      XAI_ERR_DATASIZE, "Output Width is invalid.");                                                                     \
    }                                                                                                                                    \
    else                                                                                                                                 \
    {                                                                                                                                    \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                          \
                                                       + ((KW_MOD >> 1) - 1) - KW_MOD) >> (XAI_CNN_CONV_GET_STRIDEX(param) >> 1)) + 1)), \
                      XAI_ERR_DATASIZE, "Output Width is invalid.");                                                                     \
    }                                                                                                                                    \
    if (KH_MOD % 2 != 0)                                                                                                                 \
    {                                                                                                                                    \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                          \
                                                       + (KH_MOD >> 1) - KH_MOD) >> (XAI_CNN_CONV_GET_STRIDEY(param) >> 1)) + 1)),       \
                      XAI_ERR_DATASIZE, "Output Height is invalid.");                                                                    \
    }                                                                                                                                    \
    else                                                                                                                                 \
    {                                                                                                                                    \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                          \
                                                       + ((KH_MOD >> 1) - 1) - KH_MOD) >> (XAI_CNN_CONV_GET_STRIDEY(param) >> 1)) + 1)), \
                      XAI_ERR_DATASIZE, "Output Height is invalid.");                                                                    \
    }                                                                                                                                    \
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                       \
                    "Width of Bias Array is less than number of channels in the Kernel.");                                               \
    XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArray) > 0, XAI_ERR_DATASIZE,                                                               \
                    "Height of Bias Array should be greater than zero.");                                                                \
}
#endif //if ((XCHAL_HAVE_VISION_HP_VFPU == 1) || (XCHAL_HAVE_CONNX_B_HP_VFPU == 1) || (defined(__clang__) && defined(XAI_REF_ONLY_COMPILATION)))

#define XAI_CHECK_CONSISTENCY_DEPTHWISE_DILATED_MOD_DWH(inT, coeffT, biasArr, outT, param)                                                         \
  int32_t KW_MOD = (XAI_TILE3D_GET_DIM2(coeffT) - 1) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONX(param) + 1;                                    \
  int32_t KH_MOD = (XAI_TILE3D_GET_DIM3(coeffT) - 1) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONY(param) + 1;                                    \
  XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(inT) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DEPTH_MULTIPLIER(param))                                          \
                  == XAI_TILE3D_GET_DIM1(coeffT),                                                                                                  \
                  XAI_ERR_DATASIZE,                                                                                                                \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                                  \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outT) == XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                                      \
                  "Number of Output Channels not equal to the number of channels in the Kernel.");                                                 \
  if (KW_MOD % 2 != 0)                                                                                                                             \
  {                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                                      \
                                                     + (KW_MOD >> 1) - KW_MOD) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                           \
  }                                                                                                                                                \
  else                                                                                                                                             \
  {                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                                      \
                                                     + ((KW_MOD >> 1) - 1) - KW_MOD) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                           \
  }                                                                                                                                                \
  if (KH_MOD % 2 != 0)                                                                                                                             \
  {                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                                      \
                                                     + (KH_MOD >> 1) - KH_MOD) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                          \
  }                                                                                                                                                \
  else                                                                                                                                             \
  {                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                                      \
                                                     + ((KH_MOD >> 1) - 1) - KH_MOD) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                          \
  }                                                                                                                                                \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                                   \
                  "Width of Bias Array is less than number of channels in the Kernel.");                                                           \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArr) > 0, XAI_ERR_DATASIZE,                                                                             \
                  "Height of Bias Array should be greater than zero.");

#define XAI_CHECK_CONSISTENCY_DEPTHWISE_DILATED_VQ_MOD_DWH(inT, coeffT, biasArr, outputScaleArray, outT, param)                                    \
  int32_t KW_MOD = (XAI_TILE3D_GET_DIM2(coeffT) - 1) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONX(param) + 1;                                    \
  int32_t KH_MOD = (XAI_TILE3D_GET_DIM3(coeffT) - 1) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONY(param) + 1;                                    \
  XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(inT) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DEPTH_MULTIPLIER(param))                                          \
                  == XAI_TILE3D_GET_DIM1(coeffT),                                                                                                  \
                  XAI_ERR_DATASIZE,                                                                                                                \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                                  \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outT) == XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                                      \
                  "Number of Output Channels not equal to the number of channels in the Kernel.");                                                 \
  if (KW_MOD % 2 != 0)                                                                                                                             \
  {                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                                      \
                                                     + (KW_MOD >> 1) - KW_MOD) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                           \
  }                                                                                                                                                \
  else                                                                                                                                             \
  {                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KW_MOD >> 1)                                                      \
                                                     + ((KW_MOD >> 1) - 1) - KW_MOD) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                           \
  }                                                                                                                                                \
  if (KH_MOD % 2 != 0)                                                                                                                             \
  {                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                                      \
                                                     + (KH_MOD >> 1) - KH_MOD) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                          \
  }                                                                                                                                                \
  else                                                                                                                                             \
  {                                                                                                                                                \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(outT) <= (((XAI_TILE3D_GET_DIM3(inT) + (KH_MOD >> 1)                                                      \
                                                     + ((KH_MOD >> 1) - 1) - KH_MOD) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                          \
  }                                                                                                                                                \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                                   \
                  "Width of Bias Array is less than number of channels in the Kernel.");                                                           \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArr) > 0, XAI_ERR_DATASIZE,                                                                             \
                  "Height of Bias Array should be greater than zero.");                                                                            \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE3D_GET_DIM1(coeffT), XAI_ERR_DATASIZE,                                          \
                  "Width of Bias Array is less than number of channels in the Kernel.");                                                           \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(outputScaleArray) > 0, XAI_ERR_DATASIZE,                                                                    \
                  "Height of Bias Array should be greater than zero.");

#define XAI_CHECK_CONSISTENCY_DEPTHWISE_DILATED_VQ_MOW_WHD(inT, coeffT, biasArr, outputScaleArray, outT, param)                                   \
  int32_t KW_MOW = (XAI_TILE3D_GET_DIM1(coeffT) - 1) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONX(param) + 1;                                   \
  int32_t KH_MOW = (XAI_TILE3D_GET_DIM2(coeffT) - 1) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONY(param) + 1;                                   \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inT) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DEPTH_MULTIPLIER(param)                                           \
                  == XAI_TILE3D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                                                                               \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                                                 \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outT) == XAI_TILE3D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                                                     \
                  "Number of Output Channels not equal to the number of channels in the Kernel.");                                                \
  if (KW_MOW % 2 != 0)                                                                                                                            \
  {                                                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (KW_MOW >> 1)                                                     \
                                                     + (KW_MOW >> 1) - KW_MOW) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                          \
  }                                                                                                                                               \
  else                                                                                                                                            \
  {                                                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (KW_MOW >> 1)                                                     \
                                                     + ((KW_MOW >> 1) - 1) - KW_MOW) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                                          \
  }                                                                                                                                               \
  if (KH_MOW % 2 != 0)                                                                                                                            \
  {                                                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KH_MOW >> 1)                                                     \
                                                     + (KH_MOW >> 1) - KH_MOW) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                         \
  }                                                                                                                                               \
  else                                                                                                                                            \
  {                                                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (KH_MOW >> 1)                                                     \
                                                     + ((KH_MOW >> 1) - 1) - KH_MOW) / (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                                         \
  }                                                                                                                                               \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                                                  \
                  "Width of Bias Array is less than number of channels in the Kernel.");                                                          \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArr) > 0, XAI_ERR_DATASIZE,                                                                            \
                  "Height of Bias Array should be greater than zero.");                                                                           \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE3D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                                         \
                  "Width of Bias Array is less than number of channels in the Kernel.");                                                          \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(outputScaleArray) > 0, XAI_ERR_DATASIZE,                                                                   \
                  "Height of Bias Array should be greater than zero.");

#define XAI_CHECK_CONSISTENCY_DEPTHWISE_MOW_WHD(inT, coeffT, biasArr, outT, param)                                            \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inT) == XAI_TILE3D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                                  \
                  "Number of Input Channels not equal to the number of channels in the Kernel.");                             \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outT) == XAI_TILE3D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                                 \
                  "Number of Output Channels not equal to the number of channels in the Kernel.");                            \
  int32_t kW_MOW = XAI_TILE3D_GET_DIM1(coeffT);                                                                               \
  int32_t kH_MOW = XAI_TILE3D_GET_DIM2(coeffT);                                                                               \
  if (kW_MOW % 2 != 0)                                                                                                        \
  {                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (kW_MOW >> 1) +                               \
                                                     (kW_MOW >> 1) - kW_MOW) / (XAI_CNN_CONV_GET_STRIDE(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                      \
  }                                                                                                                           \
  else                                                                                                                        \
  {                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outT) <= (((XAI_TILE3D_GET_DIM1(inT) + (kW_MOW >> 1) +                               \
                                                     ((kW_MOW >> 1) - 1) - kW_MOW) / (XAI_CNN_CONV_GET_STRIDE(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile widths are inconsistent.");                                      \
  }                                                                                                                           \
  if (kH_MOW % 2 != 0)                                                                                                        \
  {                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (kH_MOW >> 1) +                               \
                                                     (kH_MOW >> 1) - kH_MOW) / (XAI_CNN_CONV_GET_STRIDE(param))) + 1)),       \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                     \
  }                                                                                                                           \
  else                                                                                                                        \
  {                                                                                                                           \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(outT) <= (((XAI_TILE3D_GET_DIM2(inT) + (kH_MOW >> 1) +                               \
                                                     ((kH_MOW >> 1) - 1) - kH_MOW) / (XAI_CNN_CONV_GET_STRIDE(param))) + 1)), \
                    XAI_ERR_DATASIZE, "Input and Output tile heights are inconsistent.");                                     \
  }                                                                                                                           \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(biasArr) >= XAI_TILE3D_GET_DIM3(coeffT), XAI_ERR_DATASIZE,                              \
                  "Width of Bias Array is less than number of channels in the Kernel.");                                      \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(biasArr) > 0, XAI_ERR_DATASIZE,                                                        \
                  "Height of Bias Array should be greater than zero.");

#define XAI_CHECK_KERNEL_SIZE_DEPTHWISE(coeffT, size)                                               \
  if (XAI_TILE3D_GET_DATA_ORDER(coeffT) == XAI_WHD)                                                 \
  {                                                                                                 \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(coeffT) == size) && (XAI_TILE3D_GET_DIM2(coeffT) == size), \
                    XAI_ERR_KSIZE, "The Coefficient Kernel Size is not supported");                 \
  }                                                                                                 \
  else if (XAI_TILE3D_GET_DATA_ORDER(coeffT) == XAI_DWH)                                            \
  {                                                                                                 \
    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(coeffT) == size) && (XAI_TILE3D_GET_DIM3(coeffT) == size), \
                    XAI_ERR_KSIZE, "The Coefficient Kernel Size is not supported");                 \
  }

#define XAI_CHECK_EDGES_DEPTHWISE_MOW_WHD(inTile, coeffTile, param)                            \
  int32_t kW = XAI_TILE3D_GET_DIM1(coeffTile);                                                 \
  int32_t kH = XAI_TILE3D_GET_DIM2(coeffTile);                                                 \
  if (kW % 2 != 0)                                                                             \
  {                                                                                            \
    if (kH % 2 != 0)                                                                           \
    {                                                                                          \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= kW / 2)                            \
                      && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= kW / 2)                         \
                      && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kH / 2)                         \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kH / 2),                        \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");     \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kH / 2) - 1)),              \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kH / 2) - 1))               \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
  }                                                                                            \
  else                                                                                         \
  {                                                                                            \
    if (kH % 2 != 0)                                                                           \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                               \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((kW / 2) - 1))               \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((kW / 2) - 1))                  \
                        && (XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kH / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kH / 2) - 1)),               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kH / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kH / 2) - 1)),               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (kW / 2 - 1) &&                \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kH / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kH / 2)),                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1_EDGE1(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM1_EDGE2(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kH / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kH / 2)),                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
    }                                                                                          \
  }

#define XAI_CHECK_EDGES_DEPTHWISE_MOD_DWH(inTile, coeffTile, param)                            \
  int32_t kW = XAI_TILE3D_GET_DIM2(coeffTile);                                                 \
  int32_t kH = XAI_TILE3D_GET_DIM3(coeffTile);                                                 \
  if (kW % 2 != 0)                                                                             \
  {                                                                                            \
    if (kH % 2 != 0)                                                                           \
    {                                                                                          \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                            \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                         \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                         \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                        \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");     \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),              \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1))               \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
  }                                                                                            \
  else                                                                                         \
  {                                                                                            \
    if (kH % 2 != 0)                                                                           \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                               \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) - 1)                 \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1))                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");   \
      }                                                                                        \
    }                                                                                          \
    else                                                                                       \
    {                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kH / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kH / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
      else                                                                                     \
      {                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                             \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kH / 2)),                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) &&                    \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1) &&              \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kH / 2)),                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data"); \
        }                                                                                      \
      }                                                                                        \
    }                                                                                          \
  }

#define XAI_CHECK_EDGES_DEPTHWISE_DILATED_MOD_DWH(inTile, coeffTile, param)                                    \
  int32_t kW = (XAI_TILE3D_GET_DIM2(coeffTile) - 1) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONX(param) + 1; \
  int32_t kH = (XAI_TILE3D_GET_DIM3(coeffTile) - 1) * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONY(param) + 1; \
  if (kW % 2 != 0)                                                                                             \
  {                                                                                                            \
    if (kH % 2 != 0)                                                                                           \
    {                                                                                                          \
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                                            \
                      && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                                         \
                      && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                                         \
                      && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                                        \
                      XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                     \
    }                                                                                                          \
    else                                                                                                       \
    {                                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                \
      {                                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),                              \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
      }                                                                                                        \
      else                                                                                                     \
      {                                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1))                               \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
      }                                                                                                        \
    }                                                                                                          \
  }                                                                                                            \
  else                                                                                                         \
  {                                                                                                            \
    if (kH % 2 != 0)                                                                                           \
    {                                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                               \
      {                                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= kW / 2)                                          \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) - 1)                                 \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
      }                                                                                                        \
      else                                                                                                     \
      {                                                                                                        \
        XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1))                                  \
                        && (XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= kW / 2)                                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= kH / 2)                                       \
                        && (XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= kH / 2),                                      \
                        XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                   \
      }                                                                                                        \
    }                                                                                                          \
    else                                                                                                       \
    {                                                                                                          \
      if (XAI_CNN_CONV_GET_FLAG_TOPEDGE(param))                                                                \
      {                                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                             \
        {                                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kW / 2) &&                                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kW / 2) - 1) &&                              \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kH / 2) &&                                    \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),                               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                 \
        }                                                                                                      \
        else                                                                                                   \
        {                                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1) &&                              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) &&                                    \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= (kH / 2) &&                                    \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= ((kH / 2) - 1)),                               \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                 \
        }                                                                                                      \
      }                                                                                                        \
      else                                                                                                     \
      {                                                                                                        \
        if (XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param))                                                             \
        {                                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= (kW / 2) &&                                    \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= ((kW / 2) - 1) &&                              \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1) &&                              \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kH / 2)),                                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                 \
        }                                                                                                      \
        else                                                                                                   \
        {                                                                                                      \
          XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2_EDGE1(inTile) >= ((kW / 2) - 1) &&                              \
                           XAI_TILE3D_GET_DIM2_EDGE2(inTile) >= (kW / 2) &&                                    \
                           XAI_TILE3D_GET_DIM3_EDGE1(inTile) >= ((kH / 2) - 1) &&                              \
                           XAI_TILE3D_GET_DIM3_EDGE2(inTile) >= (kH / 2)),                                     \
                          XAI_ERR_EDGE, "The input Tile doesn't have the required Edge Data");                 \
        }                                                                                                      \
      }                                                                                                        \
    }                                                                                                          \
  }

#define XAI_CHECK_ROI_POOLING_PARAMS(param)                                                                                                                      \
  XAI_CHECK_ERROR(((XAI_CNN_ROI_POOLING_GET_SPATIAL_SCALEX(param) <= 32767) && (XAI_CNN_ROI_POOLING_GET_SPATIAL_SCALEY(param) <= 32767)),                        \
                  XAI_ERR_NORM, "spatialScaleX & spatialScaleY should be less than U15_MAX");                                                                    \
  XAI_CHECK_ERROR(((XAI_CNN_ROI_POOLING_GET_ONE_BY_POOLED_WIDTH_SCALE(param) <= 32767) && (XAI_CNN_ROI_POOLING_GET_ONE_BY_POOLED_HEIGHT_SCALE(param) <= 32767)), \
                  XAI_ERR_NORM, "oneByPooledWidth & oneByPooledHeight should be less than U15_MAX");                                                             \
  XAI_CHECK_ERROR(((XAI_CNN_ROI_POOLING_GET_SPATIAL_SCALE_SHIFTX(param) < 32) && (XAI_CNN_ROI_POOLING_GET_SPATIAL_SCALE_SHIFTY(param) < 32)),                    \
                  XAI_ERR_NORM, "spatialScaleShiftX & spatialScaleShiftY should be less than 32 (scalar shift value)");                                          \
  XAI_CHECK_ERROR(((XAI_CNN_ROI_POOLING_GET_ONE_BY_POOLED_WIDTH_SHIFT(param) < 32) && (XAI_CNN_ROI_POOLING_GET_ONE_BY_POOLED_HEIGHT_SHIFT(param) < 32)),         \
                  XAI_ERR_NORM, "shiftPool should be less than 32 (scalar shift value)");                                                                        \

#define XAI_CHECK_REORG_PARAMS_DWH(inTile, outTile, params)                                         \
  if (XAI_CNN_REORG_GET_REVERSE(params))                                                            \
  {                                                                                                 \
    XAI_CHECK_ERROR(XAI_CNN_REORG_GET_STRIDE(params) * XAI_CNN_REORG_GET_STRIDE(params) *           \
                    XAI_TILE3D_GET_DIM1(outTile) == XAI_TILE3D_GET_DIM1(inTile),                    \
                    XAI_ERR_DATASIZE, "The depth dimension of inTile and outTile is inconsistent"); \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTile) == XAI_TILE3D_GET_DIM2(inTile) *                   \
                    XAI_CNN_REORG_GET_STRIDE(params), XAI_ERR_DATASIZE,                             \
                    "The width dimension of inTile and outTile is inconsistent");                   \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTile) == XAI_TILE3D_GET_DIM3(inTile) *                   \
                    XAI_CNN_REORG_GET_STRIDE(params), XAI_ERR_DATASIZE,                             \
                    "The height dimension of inTile and outTile is inconsistent");                  \
  }                                                                                                 \
  else                                                                                              \
  {                                                                                                 \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTile) == XAI_CNN_REORG_GET_STRIDE(params) *              \
                    XAI_CNN_REORG_GET_STRIDE(params) * XAI_TILE3D_GET_DIM1(inTile),                 \
                    XAI_ERR_DATASIZE, "The depth dimension of inTile and outTile is inconsistent"); \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTile) * XAI_CNN_REORG_GET_STRIDE(params) ==              \
                    XAI_TILE3D_GET_DIM2(inTile), XAI_ERR_DATASIZE,                                  \
                    "The width dimension of inTile and outTile is inconsistent");                   \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTile) * XAI_CNN_REORG_GET_STRIDE(params) ==              \
                    XAI_TILE3D_GET_DIM3(inTile), XAI_ERR_DATASIZE,                                  \
                    "The height dimension of inTile and outTile is inconsistent");                  \
  }

#define XAI_CHECK_REORG4D_PARAMS_WHDN(inTile, outTile, params)                                      \
  if (XAI_CNN_REORG_GET_REVERSE(params))                                                            \
  {                                                                                                 \
    XAI_CHECK_ERROR(XAI_CNN_REORG4D_GET_STRIDEX(params) * XAI_CNN_REORG4D_GET_STRIDEY(params) *     \
                    XAI_TILE4D_GET_DIM4(outTile) == XAI_TILE4D_GET_DIM4(inTile),                    \
                    XAI_ERR_DATASIZE, "The batch dimension of inTile and outTile is inconsistent"); \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM1(outTile) == XAI_TILE4D_GET_DIM1(inTile) *                   \
                    XAI_CNN_REORG4D_GET_STRIDEX(params), XAI_ERR_DATASIZE,                          \
                    "The width dimension of inTile and outTile is inconsistent");                   \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM2(outTile) == XAI_TILE4D_GET_DIM2(inTile) *                   \
                    XAI_CNN_REORG4D_GET_STRIDEY(params), XAI_ERR_DATASIZE,                          \
                    "The height dimension of inTile and outTile is inconsistent");                  \
  }                                                                                                 \
  else                                                                                              \
  {                                                                                                 \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM4(outTile) == XAI_CNN_REORG4D_GET_STRIDEX(params) *           \
                    XAI_CNN_REORG4D_GET_STRIDEY(params) * XAI_TILE4D_GET_DIM4(inTile),              \
                    XAI_ERR_DATASIZE, "The batch dimension of inTile and outTile is inconsistent"); \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM1(outTile) * XAI_CNN_REORG4D_GET_STRIDEX(params) ==           \
                    XAI_TILE4D_GET_DIM1(inTile), XAI_ERR_DATASIZE,                                  \
                    "The width dimension of inTile and outTile is inconsistent");                   \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM2(outTile) * XAI_CNN_REORG4D_GET_STRIDEY(params) ==           \
                    XAI_TILE4D_GET_DIM2(inTile), XAI_ERR_DATASIZE,                                  \
                    "The height dimension of inTile and outTile is inconsistent");                  \
  }
#define XAI_CHECK_REORG4D_PARAMS_DWHN(inTile, outTile, params)                                      \
  if (XAI_CNN_REORG_GET_REVERSE(params))                                                            \
  {                                                                                                 \
    XAI_CHECK_ERROR(XAI_CNN_REORG4D_GET_STRIDEX(params) * XAI_CNN_REORG4D_GET_STRIDEY(params) *     \
                    XAI_TILE4D_GET_DIM4(outTile) == XAI_TILE4D_GET_DIM4(inTile),                    \
                    XAI_ERR_DATASIZE, "The batch dimension of inTile and outTile is inconsistent"); \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM2(outTile) == XAI_TILE4D_GET_DIM2(inTile) *                   \
                    XAI_CNN_REORG4D_GET_STRIDEX(params), XAI_ERR_DATASIZE,                          \
                    "The width dimension of inTile and outTile is inconsistent");                   \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM3(outTile) == XAI_TILE4D_GET_DIM3(inTile) *                   \
                    XAI_CNN_REORG4D_GET_STRIDEY(params), XAI_ERR_DATASIZE,                          \
                    "The height dimension of inTile and outTile is inconsistent");                  \
  }                                                                                                 \
  else                                                                                              \
  {                                                                                                 \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM4(outTile) == XAI_CNN_REORG4D_GET_STRIDEX(params) *           \
                    XAI_CNN_REORG4D_GET_STRIDEY(params) * XAI_TILE4D_GET_DIM4(inTile),              \
                    XAI_ERR_DATASIZE, "The batch dimension of inTile and outTile is inconsistent"); \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM2(outTile) * XAI_CNN_REORG4D_GET_STRIDEX(params) ==           \
                    XAI_TILE4D_GET_DIM2(inTile), XAI_ERR_DATASIZE,                                  \
                    "The width dimension of inTile and outTile is inconsistent");                   \
                                                                                                    \
    XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM3(outTile) * XAI_CNN_REORG4D_GET_STRIDEY(params) ==           \
                    XAI_TILE4D_GET_DIM3(inTile), XAI_ERR_DATASIZE,                                  \
                    "The height dimension of inTile and outTile is inconsistent");                  \
  }

#define XAI_CHECK_REORG_PARAMS_WHD(inT, outT, param)                                                  \
  if (XAI_CNN_REORG_GET_REVERSE(param))                                                               \
  {                                                                                                   \
    XAI_CHECK_ERROR((XAI_CNN_REORG_GET_STRIDE(param) * XAI_CNN_REORG_GET_STRIDE(param) *              \
                     XAI_TILE3D_GET_DIM3(outT)) == XAI_TILE3D_GET_DIM3(inT), XAI_ERR_DATASIZE,        \
                    "Number of output channels is strideX * strideY times number of input channels"); \
                                                                                                      \
    XAI_CHECK_ERROR(XAI_CNN_REORG_GET_STRIDE(param) * XAI_TILE3D_GET_DIM1(inT) ==                     \
                    XAI_TILE3D_GET_DIM1(outT), XAI_ERR_DATASIZE,                                      \
                    "Input width is strideX times output width");                                     \
                                                                                                      \
    XAI_CHECK_ERROR(XAI_CNN_REORG_GET_STRIDE(param) * XAI_TILE3D_GET_DIM2(inT) ==                     \
                    XAI_TILE3D_GET_DIM2(outT), XAI_ERR_DATASIZE,                                      \
                    "Input height is strideY times output height");                                   \
                                                                                                      \
  }                                                                                                   \
  else                                                                                                \
  {                                                                                                   \
    XAI_CHECK_ERROR((XAI_CNN_REORG_GET_STRIDE(param) * XAI_CNN_REORG_GET_STRIDE(param) *              \
                     XAI_TILE3D_GET_DIM3(inT)) == XAI_TILE3D_GET_DIM3(outT), XAI_ERR_DATASIZE,        \
                    "Number of output channels is strideX * strideY times number of input channels"); \
                                                                                                      \
    XAI_CHECK_ERROR(XAI_CNN_REORG_GET_STRIDE(param) * XAI_TILE3D_GET_DIM1(outT) ==                    \
                    XAI_TILE3D_GET_DIM1(inT), XAI_ERR_DATASIZE,                                       \
                    "Input width is strideX times output width");                                     \
                                                                                                      \
    XAI_CHECK_ERROR(XAI_CNN_REORG_GET_STRIDE(param) * XAI_TILE3D_GET_DIM2(outT) ==                    \
                    XAI_TILE3D_GET_DIM2(inT), XAI_ERR_DATASIZE,                                       \
                    "Input height is strideY times output height");                                   \
  }

#if XAI_ERROR_LEVEL != XAI_ERROR_LEVEL_NO_ERROR
#define XAI_CHECK_INTERP_BOUNDARY(xDstCoordinate, yDstCoordinate, zDstCoordinate, xSrcCoordinate, ySrcCoordinate, zSrcCoordinate,                        \
                                  xScale, yScale, xShift, yShift, inDataWidth, inDataHeight, inDataDepth, outDataWidth, outDataHeight, outDataDepth,     \
                                  edge1AcrossWidth, edge2AcrossWidth, edge1AcrossHeight, edge2AcrossHeight,                                              \
                                  inFrameWidth, inFrameHeight)                                                                                           \
  {                                                                                                                                                      \
    int32_t insideFrameX;                                                                                                                                \
    int32_t insideFrameY;                                                                                                                                \
                                                                                                                                                         \
    int32_t xmax = (((xDstCoordinate + outDataWidth - 1) * xScale + xShift) >> 18) + 1;                                                                  \
    int32_t ymax = (((yDstCoordinate + outDataHeight - 1) * yScale + yShift) >> 18) + 1;                                                                 \
    int32_t zmax = (zDstCoordinate + outDataDepth);                                                                                                      \
                                                                                                                                                         \
    insideFrameX = (xmax < inFrameWidth);                                                                                                                \
    insideFrameY = (ymax < inFrameHeight);                                                                                                               \
                                                                                                                                                         \
    XAI_CHECK_ERROR(((((xDstCoordinate * xScale + xShift < 0) || ((xDstCoordinate * xScale + xShift) >> 18) >= (xSrcCoordinate - edge1AcrossWidth))) &&  \
                     (((yDstCoordinate * yScale + yShift < 0) || ((yDstCoordinate * yScale + yShift) >> 18) >= (ySrcCoordinate - edge1AcrossHeight))) && \
                     (((zDstCoordinate) >= (zSrcCoordinate))) &&                                                                                         \
                     (((xmax + insideFrameX) <= (xSrcCoordinate + inDataWidth + edge2AcrossWidth))) &&                                                   \
                     (((ymax + insideFrameY) <= (ySrcCoordinate + inDataHeight + edge2AcrossHeight))) &&                                                 \
                     ((zmax <= (zSrcCoordinate + inDataDepth)))),                                                                                        \
                    XAI_ERR_DATASIZE, "The input tile size requirements is in sufficient");                                                              \
  }
#else
#define XAI_CHECK_INTERP_BOUNDARY(xDstCoordinate, yDstCoordinate, zDstCoordinate, xSrcCoordinate, ySrcCoordinate, zSrcCoordinate,                    \
                                  xScale, yScale, xShift, yShift, inDataWidth, inDataHeight, inDataDepth, outDataWidth, outDataHeight, outDataDepth, \
                                  edge1AcrossWidth, edge2AcrossWidth, edge1AcrossHeight, edge2AcrossHeight,                                          \
                                  inFrameWidth, inFrameHeight)
#endif

#if XAI_ERROR_LEVEL != XAI_ERROR_LEVEL_NO_ERROR
#define XAI_CHECK_RESIZENEAREST_BOUNDARY(xDstCoordinate, yDstCoordinate, zDstCoordinate,               \
                                         xSrcCoordinate, ySrcCoordinate, zSrcCoordinate,               \
                                         xScale, yScale, xShift, yShift,                               \
                                         inDataWidth, inDataHeight, inDataDepth,                       \
                                         outDataWidth, outDataHeight, outDataDepth,                    \
                                         edge1AcrossWidth, edge2AcrossWidth, edge1AcrossHeight,        \
                                         edge2AcrossHeight, inFrameWidth, inFrameHeight)               \
  {                                                                                                    \
    int32_t xmin = ((xDstCoordinate * xScale) + xShift);                                               \
    int32_t ymin = ((yDstCoordinate * yScale) + yShift);                                               \
    int32_t zmin = (zDstCoordinate);                                                                   \
    int32_t xmax = (((xDstCoordinate + outDataWidth - 1) * xScale + xShift) >> 18) + 1;                \
    int32_t ymax = (((yDstCoordinate + outDataHeight - 1) * yScale + yShift) >> 18) + 1;               \
    int32_t zmax = (zDstCoordinate + outDataDepth);                                                    \
                                                                                                       \
    int32_t insideFrameX = (xmax < inFrameWidth);                                                      \
    int32_t insideFrameY = (ymax < inFrameHeight);                                                     \
                                                                                                       \
    XAI_CHECK_ERROR((((xmin < 0 || (xmin >> 18) >= (xSrcCoordinate - edge1AcrossWidth))) &&            \
                     ((ymin < 0 || (ymin >> 18) >= (ySrcCoordinate - edge1AcrossHeight))) &&           \
                     (zmin >= (zSrcCoordinate)) &&                                                     \
                     ((xmax + insideFrameX) <= (xSrcCoordinate + inDataWidth + edge2AcrossWidth)) &&   \
                     ((ymax + insideFrameY) <= (ySrcCoordinate + inDataHeight + edge2AcrossHeight)) && \
                     (zmax <= (zSrcCoordinate + inDataDepth))),                                        \
                    XAI_ERR_DATASIZE, "The input tile size requirements is in sufficient");            \
  }
#else
#define XAI_CHECK_RESIZENEAREST_BOUNDARY(xDstCoordinate, yDstCoordinate, zDstCoordinate,        \
                                         xSrcCoordinate, ySrcCoordinate, zSrcCoordinate,        \
                                         xScale, yScale, xShift, yShift,                        \
                                         inDataWidth, inDataHeight, inDataDepth,                \
                                         outDataWidth, outDataHeight, outDataDepth,             \
                                         edge1AcrossWidth, edge2AcrossWidth, edge1AcrossHeight, \
                                         edge2AcrossHeight, inFrameWidth, inFrameHeight)
#endif

#define XAI_CHECK_CONSISTENCY_MAXVALARR8(maxValArr, params, tileFlag)                                       \
  {                                                                                                         \
    if (XAI_CNN_MAXVAL_GET_TILEFLAG(params) != tileFlag)                                                    \
    {                                                                                                       \
      XAI_CHECK_ARRAY_S8(maxValArr);                                                                        \
      XAI_CHECK_ERROR((XAI_ARRAY_GET_WIDTH(maxValArr) >= XCHAL_IVPN_SIMD_WIDTH),                            \
                      XAI_ERR_BADARG, "Length of maxValArr should not be less than XCHAL_IVPN_SIMD_WIDTH"); \
      XAI_CHECK_ERROR((XAI_ARRAY_GET_HEIGHT(maxValArr) > 0), XAI_ERR_BADARG,                                \
                      "maxValArr height parameter is not set as required");                                 \
    }                                                                                                       \
  }
#define XAI_CHECK_CONSISTENCY_MAXVALARR(maxValArr, params, tileFlag)                                        \
  {                                                                                                         \
    if (XAI_CNN_MAXVAL_GET_TILEFLAG(params) != tileFlag)                                                    \
    {                                                                                                       \
      XAI_CHECK_ARRAY_S16(maxValArr);                                                                       \
      XAI_CHECK_ERROR((XAI_ARRAY_GET_WIDTH(maxValArr) >= XCHAL_IVPN_SIMD_WIDTH),                            \
                      XAI_ERR_BADARG, "Length of maxValArr should not be less than XCHAL_IVPN_SIMD_WIDTH"); \
      XAI_CHECK_ERROR((XAI_ARRAY_GET_HEIGHT(maxValArr) > 0), XAI_ERR_BADARG,                                \
                      "maxValArr height parameter is not set as required");                                 \
    }                                                                                                       \
  }

#define XAI_CHECK_PERMUTE_PARAMS(params)                                                                   \
  XAI_CHECK_ERROR((XAI_CNN_PERMUTE4D_GET_ORDER1(params) > 0 && XAI_CNN_PERMUTE4D_GET_ORDER2(params) > 0 && \
                   XAI_CNN_PERMUTE4D_GET_ORDER3(params) > 0 && XAI_CNN_PERMUTE4D_GET_ORDER4(params) > 0),  \
                  XAI_ERR_BADARG, "The order should be greater than 0");                                   \
  XAI_CHECK_ERROR((XAI_CNN_PERMUTE4D_GET_ORDER1(params) < 5 && XAI_CNN_PERMUTE4D_GET_ORDER2(params) < 5 && \
                   XAI_CNN_PERMUTE4D_GET_ORDER3(params) < 5 && XAI_CNN_PERMUTE4D_GET_ORDER4(params) < 5),  \
                  XAI_ERR_BADARG, "The order should be greater than 0");                                   \
  XAI_CHECK_ERROR(((XAI_CNN_PERMUTE4D_GET_ORDER1(params) != XAI_CNN_PERMUTE4D_GET_ORDER2(params)) &&       \
                   (XAI_CNN_PERMUTE4D_GET_ORDER1(params) != XAI_CNN_PERMUTE4D_GET_ORDER3(params)) &&       \
                   (XAI_CNN_PERMUTE4D_GET_ORDER1(params) != XAI_CNN_PERMUTE4D_GET_ORDER4(params)) &&       \
                   (XAI_CNN_PERMUTE4D_GET_ORDER2(params) != XAI_CNN_PERMUTE4D_GET_ORDER3(params)) &&       \
                   (XAI_CNN_PERMUTE4D_GET_ORDER2(params) != XAI_CNN_PERMUTE4D_GET_ORDER4(params)) &&       \
                   (XAI_CNN_PERMUTE4D_GET_ORDER3(params) != XAI_CNN_PERMUTE4D_GET_ORDER4(params))),        \
                  XAI_ERR_BADARG, "The order values should not be equal to one another");

#if XAI_ERROR_LEVEL != XAI_ERROR_LEVEL_NO_ERROR
#define XAI_CHECK_CONSISTENCY_PERMUTE(inT, outT, params)                                                                  \
  {                                                                                                                       \
    uint8_t order[4] = { XAI_CNN_PERMUTE4D_GET_ORDER1(params),                                                            \
                         XAI_CNN_PERMUTE4D_GET_ORDER2(params),                                                            \
                         XAI_CNN_PERMUTE4D_GET_ORDER3(params),                                                            \
                         XAI_CNN_PERMUTE4D_GET_ORDER4(params) };                                                          \
    int32_t inDim[4] = { XAI_TILE4D_GET_DIM1(inT),                                                                        \
                         XAI_TILE4D_GET_DIM2(inT),                                                                        \
                         XAI_TILE4D_GET_DIM3(inT),                                                                        \
                         XAI_TILE4D_GET_DIM4(inT) };                                                                      \
                                                                                                                          \
    const int32_t transposedDim1 = inDim[order[0] - 1];                                                                   \
    const int32_t transposedDim2 = inDim[order[1] - 1];                                                                   \
    const int32_t transposedDim3 = inDim[order[2] - 1];                                                                   \
    const int32_t transposedDim4 = inDim[order[3] - 1];                                                                   \
    XAI_CHECK_ERROR((transposedDim1 == XAI_TILE4D_GET_DIM1(outT) && transposedDim2 == XAI_TILE4D_GET_DIM2(outT)           \
                     && transposedDim3 == XAI_TILE4D_GET_DIM3(outT) && transposedDim4 == XAI_TILE4D_GET_DIM4(outT)),      \
                    XAI_ERR_DATASIZE, "The dimensions of the output tile should be equal to the transposed dimensions of the \
                        input tile whose order is specified by the parameter in the xai_cnn_permute4D_params structure"); \
  }
#else
#define XAI_CHECK_CONSISTENCY_PERMUTE(inT, outT, params)
#endif
#endif

#define XAI_CHECK_CONSISTENCY_ARGMAX_3D_DIM1(inTile, outTileIdx, outTileVal, numLargestVal)             \
  {                                                                                                     \
    if (outTileIdx != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D_U16(outTileIdx);                                                                 \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileIdx);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileIdx);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileIdx) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output index tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileIdx) == XAI_TILE3D_GET_DIM2(inTile), XAI_ERR_DATASIZE, \
                      "Output index tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileIdx) == XAI_TILE3D_GET_DIM3(inTile), XAI_ERR_DATASIZE, \
                      "Output index tile size is incorrect");                                           \
    }                                                                                                   \
    if (outTileVal != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D(outTileVal);                                                                     \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_TYPE(inTile) == XAI_TILE3D_GET_TYPE(outTileVal), XAI_ERR_DATATYPE, \
                      "Data type of output tile must be same as input tile");                           \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileVal);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileVal);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileVal) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileVal) == XAI_TILE3D_GET_DIM2(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileVal) == XAI_TILE3D_GET_DIM3(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
    }                                                                                                   \
    if ((outTileVal != NULL) && (outTileIdx != NULL))                                                   \
    {                                                                                                   \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(outTileIdx, outTileVal);                                        \
    }                                                                                                   \
  }

#define XAI_CHECK_CONSISTENCY_MERGE_TOPK_ARGMAX_ARGMIN_3D_DIM1(inTileIdx, inTileVal, outTileIdx, outTileVal, numVal) \
  {                                                                                                                  \
    if (outTileIdx != NULL)                                                                                          \
    {                                                                                                                \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileIdx) == numVal, XAI_ERR_DATASIZE,                                   \
                      "Output index tile size is incorrect");                                                        \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileIdx) == XAI_TILE3D_GET_DIM2(inTileVal), XAI_ERR_DATASIZE,           \
                      "Output index tile size is incorrect");                                                        \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileIdx) == XAI_TILE3D_GET_DIM3(inTileVal), XAI_ERR_DATASIZE,           \
                      "Output index tile size is incorrect");                                                        \
      XAI_CHECK_TILE3D_S32(outTileIdx);                                                                              \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileIdx);                                                                 \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTileIdx, outTileIdx);                                                      \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTileVal, outTileIdx);                                                      \
    }                                                                                                                \
    if (outTileVal != NULL)                                                                                          \
    {                                                                                                                \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileVal) == numVal, XAI_ERR_DATASIZE,                                   \
                      "Output tile size is incorrect");                                                              \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileVal) == XAI_TILE3D_GET_DIM2(inTileVal), XAI_ERR_DATASIZE,           \
                      "Output tile size is incorrect");                                                              \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileVal) == XAI_TILE3D_GET_DIM3(inTileVal), XAI_ERR_DATASIZE,           \
                      "Output tile size is incorrect");                                                              \
      XAI_CHECK_TILE3D(outTileVal);                                                                                  \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_TYPE(inTileVal) == XAI_TILE3D_GET_TYPE(outTileVal), XAI_ERR_DATATYPE,           \
                      "Data type of output tile must be same as input tile");                                        \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileVal);                                                                 \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTileVal, outTileVal);                                                      \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTileIdx, outTileVal);                                                      \
    }                                                                                                                \
    if ((outTileVal != NULL) && (outTileIdx != NULL))                                                                \
    {                                                                                                                \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(outTileIdx, outTileVal);                                                     \
    }                                                                                                                \
  }
#define XAI_CHECK_CONSISTENCY_ARGMAX_3D_DIM2(inTile, outTileIdx, outTileVal, numLargestVal)             \
  {                                                                                                     \
    if (outTileIdx != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D_U16(outTileIdx);                                                                 \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileIdx);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileIdx);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileIdx) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileIdx) == XAI_TILE3D_GET_DIM1(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileIdx) == XAI_TILE3D_GET_DIM3(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
    }                                                                                                   \
    if (outTileVal != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D(outTileVal);                                                                     \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_TYPE(inTile) == XAI_TILE3D_GET_TYPE(outTileVal), XAI_ERR_DATATYPE, \
                      "Data type of output tile must be same as input tile");                           \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileVal);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileVal);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileVal) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileVal) == XAI_TILE3D_GET_DIM1(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileVal) == XAI_TILE3D_GET_DIM3(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
    }                                                                                                   \
    if ((outTileVal != NULL) && (outTileIdx != NULL))                                                   \
    {                                                                                                   \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(outTileIdx, outTileVal);                                        \
    }                                                                                                   \
  }

#define XAI_CHECK_CONSISTENCY_ARGMAX_3D_DIM3(inTile, outTileIdx, outTileVal, numLargestVal)             \
  {                                                                                                     \
    if (outTileIdx != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D_U16(outTileIdx);                                                                 \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileIdx);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileIdx);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileIdx) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output index tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileIdx) == XAI_TILE3D_GET_DIM1(inTile), XAI_ERR_DATASIZE, \
                      "Output index tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileIdx) == XAI_TILE3D_GET_DIM2(inTile), XAI_ERR_DATASIZE, \
                      "Output index tile size is incorrect");                                           \
    }                                                                                                   \
    if (outTileVal != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D(outTileVal);                                                                     \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_TYPE(inTile) == XAI_TILE3D_GET_TYPE(outTileVal), XAI_ERR_DATATYPE, \
                      "Data type of output tile must be same as input tile");                           \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileVal);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileVal);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileVal) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output value tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileVal) == XAI_TILE3D_GET_DIM1(inTile), XAI_ERR_DATASIZE, \
                      "Output value tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileVal) == XAI_TILE3D_GET_DIM2(inTile), XAI_ERR_DATASIZE, \
                      "Output value tile size is incorrect");                                           \
    }                                                                                                   \
    if ((outTileVal != NULL) && (outTileIdx != NULL))                                                   \
    {                                                                                                   \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(outTileIdx, outTileVal);                                        \
    }                                                                                                   \
  }

#define XAI_CHECK_CONSISTENCY_ARGMAX_3D_DIM1_F32(inTile, outTileIdx, outTileVal, numLargestVal)         \
  {                                                                                                     \
    if (outTileIdx != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D_U32(outTileIdx);                                                                 \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileIdx);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileIdx);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileIdx) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output index tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileIdx) == XAI_TILE3D_GET_DIM2(inTile), XAI_ERR_DATASIZE, \
                      "Output index tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileIdx) == XAI_TILE3D_GET_DIM3(inTile), XAI_ERR_DATASIZE, \
                      "Output index tile size is incorrect");                                           \
    }                                                                                                   \
    if (outTileVal != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D(outTileVal);                                                                     \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_TYPE(inTile) == XAI_TILE3D_GET_TYPE(outTileVal), XAI_ERR_DATATYPE, \
                      "Data type of output tile must be same as input tile");                           \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileVal);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileVal);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileVal) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileVal) == XAI_TILE3D_GET_DIM2(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileVal) == XAI_TILE3D_GET_DIM3(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
    }                                                                                                   \
    if ((outTileVal != NULL) && (outTileIdx != NULL))                                                   \
    {                                                                                                   \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(outTileIdx, outTileVal);                                        \
    }                                                                                                   \
  }

#define XAI_CHECK_CONSISTENCY_ARGMAX_3D_DIM2_F32(inTile, outTileIdx, outTileVal, numLargestVal)         \
  {                                                                                                     \
    if (outTileIdx != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D_U32(outTileIdx);                                                                 \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileIdx);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileIdx);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileIdx) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileIdx) == XAI_TILE3D_GET_DIM1(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileIdx) == XAI_TILE3D_GET_DIM3(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
    }                                                                                                   \
    if (outTileVal != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D(outTileVal);                                                                     \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_TYPE(inTile) == XAI_TILE3D_GET_TYPE(outTileVal), XAI_ERR_DATATYPE, \
                      "Data type of output tile must be same as input tile");                           \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileVal);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileVal);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileVal) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileVal) == XAI_TILE3D_GET_DIM1(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileVal) == XAI_TILE3D_GET_DIM3(inTile), XAI_ERR_DATASIZE, \
                      "Output tile size is incorrect");                                                 \
    }                                                                                                   \
    if ((outTileVal != NULL) && (outTileIdx != NULL))                                                   \
    {                                                                                                   \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(outTileIdx, outTileVal);                                        \
    }                                                                                                   \
  }

#define XAI_CHECK_CONSISTENCY_ARGMAX_3D_DIM3_F32(inTile, outTileIdx, outTileVal, numLargestVal)         \
  {                                                                                                     \
    if (outTileIdx != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D_U32(outTileIdx);                                                                 \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileIdx);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileIdx);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileIdx) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output index tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileIdx) == XAI_TILE3D_GET_DIM1(inTile), XAI_ERR_DATASIZE, \
                      "Output index tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileIdx) == XAI_TILE3D_GET_DIM2(inTile), XAI_ERR_DATASIZE, \
                      "Output index tile size is incorrect");                                           \
    }                                                                                                   \
    if (outTileVal != NULL)                                                                             \
    {                                                                                                   \
      XAI_CHECK_TILE3D(outTileVal);                                                                     \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_TYPE(inTile) == XAI_TILE3D_GET_TYPE(outTileVal), XAI_ERR_DATATYPE, \
                      "Data type of output tile must be same as input tile");                           \
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTileVal);                                                    \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTileVal);                                            \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTileVal) == numLargestVal, XAI_ERR_DATASIZE,               \
                      "Output value tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTileVal) == XAI_TILE3D_GET_DIM1(inTile), XAI_ERR_DATASIZE, \
                      "Output value tile size is incorrect");                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTileVal) == XAI_TILE3D_GET_DIM2(inTile), XAI_ERR_DATASIZE, \
                      "Output value tile size is incorrect");                                           \
    }                                                                                                   \
    if ((outTileVal != NULL) && (outTileIdx != NULL))                                                   \
    {                                                                                                   \
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(outTileIdx, outTileVal);                                        \
    }                                                                                                   \
  }

#define XAI_CHECK_DIM_IN128DWH(coeffIn, coeffOut)                                                                     \
  {                                                                                                                   \
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffTileOut) % 128) == 0, XAI_ERR_DATASIZE,                                 \
                    "The dimension 1 of the output tile should be a multiple of 128");                                \
                                                                                                                      \
    if ((XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_WHDN) ||                                                       \
        (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_DWHN))                                                         \
    {                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) << 7) ==                                                     \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM4(coeffTileIn), 2 * XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The allocated output channels size in the IN128DWH tile is a multiple of 128");                \
    }                                                                                                                 \
    else                                                                                                              \
    {                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) << 7) ==                                                     \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM1(coeffTileIn), 2 * XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The dimension 2 of the output tile should be a multiple of 128");                              \
    }                                                                                                                 \
  }
#if (XCHAL_IVPN_SIMD_WIDTH == 64)
#define XAI_CHECK_DIM_IN64DWH(coeffIn, coeffOut)                                                                  \
  {                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffTileOut) % 64) == 0, XAI_ERR_DATASIZE,                              \
                    "The dimension 1 of the output tile should be a multiple of 64");                             \
                                                                                                                  \
    if ((XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_WHDN) ||                                                   \
        (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_DWHN))                                                     \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) << 6) ==                                                 \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM4(coeffTileIn), XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The allocated output channels size in the IN64DWH tile is a multiple of 64");              \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) << 6) ==                                                 \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM1(coeffTileIn), XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The dimension 2 of the output tile should be a multiple of 64");                           \
    }                                                                                                             \
  }

#define XAI_CHECK_DIM_IN32DWH(coeffIn, coeffOut)                                                                  \
  {                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffTileOut) % 32) == 0, XAI_ERR_DATASIZE,                              \
                    "The dimension 1 of the output tile should be a multiple of 32");                             \
                                                                                                                  \
    if ((XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_WHDN) ||                                                   \
        (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_DWHN))                                                     \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) << 5) ==                                                 \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM4(coeffTileIn), XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The allocated output channels size in the IN32DWH tile is a multiple of 32");              \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) << 5) ==                                                 \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM1(coeffTileIn), XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The dimension 2 of the output tile should be a multiple of 32");                           \
    }                                                                                                             \
  }

#else
#define XAI_CHECK_DIM_IN64DWH(coeffIn, coeffOut)                                                                      \
  {                                                                                                                   \
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffTileOut) % (2 * XCHAL_IVPN_SIMD_WIDTH)) == 0, XAI_ERR_DATASIZE,         \
                    "The dimension 1 of the output tile should be a multiple of 64");                                 \
                                                                                                                      \
    if ((XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_WHDN) ||                                                       \
        (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_DWHN))                                                         \
    {                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) * (2 * XCHAL_IVPN_SIMD_WIDTH)) ==                            \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM4(coeffTileIn), 2 * XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The allocated output channels size in the IN64DWH tile is a multiple of 64");                  \
    }                                                                                                                 \
    else                                                                                                              \
    {                                                                                                                 \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) * (2 * XCHAL_IVPN_SIMD_WIDTH)) ==                            \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM1(coeffTileIn), 2 * XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The dimension 2 of the output tile should be a multiple of 64");                               \
    }                                                                                                                 \
  }

#define XAI_CHECK_DIM_IN32DWH(coeffIn, coeffOut)                                                                  \
  {                                                                                                               \
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffTileOut) % XCHAL_IVPN_SIMD_WIDTH) == 0, XAI_ERR_DATASIZE,           \
                    "The dimension 1 of the output tile should be a multiple of 32");                             \
                                                                                                                  \
    if ((XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_WHDN) ||                                                   \
        (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_DWHN))                                                     \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) * XCHAL_IVPN_SIMD_WIDTH) ==                              \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM4(coeffTileIn), XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The allocated output channels size in the IN32DWH tile is a multiple of 32");              \
    }                                                                                                             \
    else                                                                                                          \
    {                                                                                                             \
      XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM2(coeffTileOut) * XCHAL_IVPN_SIMD_WIDTH) ==                              \
                      (XAI_ALIGN_VAL(XAI_TILE4D_GET_DIM1(coeffTileIn), XCHAL_IVPN_SIMD_WIDTH)), XAI_ERR_DATASIZE, \
                      "The dimension 2 of the output tile should be a multiple of 32");                           \
    }                                                                                                             \
  }
#endif

#define XAI_CHECK_COEFF_IN_DATA_ORDER_FC(coeffIn)                                     \
  {                                                                                   \
    XAI_CHECK_ERROR(((XAI_TILE3D_GET_DATA_ORDER(coeffIn) == XAI_NWHD) ||              \
                     (XAI_TILE3D_GET_DATA_ORDER(coeffIn) == XAI_NDWH)),               \
                    XAI_ERR_BADARG, "\nData Order of the given tiles not supported"); \
  }

/* To set appropriate pitch size for broadcast/normal elementwise operations */
#define  XAI_TILE3D_GET_BCAST23_PITCH(inTile1, inTile2, outTile, in1Stride, in2Stride, \
                                      in1Pitch1, in1Pitch2, in2Pitch1, in2Pitch2)      \
  {                                                                                    \
    int32_t m_in1Dim2, m_in1Dim3, m_in2Dim2, m_in2Dim3;                                \
    m_in1Dim2 = (XAI_TILE3D_GET_DIM2(inTile1) + in1Stride - 1) / in1Stride;            \
    m_in1Dim3 = (XAI_TILE3D_GET_DIM3(inTile1) + in1Stride - 1) / in1Stride;            \
    m_in2Dim2 = (XAI_TILE3D_GET_DIM2(inTile2) + in2Stride - 1) / in2Stride;            \
    m_in2Dim3 = (XAI_TILE3D_GET_DIM3(inTile2) + in2Stride - 1) / in2Stride;            \
    in1Pitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile1);                                    \
    in1Pitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile1);                                    \
    in2Pitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile2);                                    \
    in2Pitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile2);                                    \
    in1Pitch1 = m_in1Dim2 == XAI_TILE3D_GET_DIM2(outTile) ? in1Pitch1 : 0;             \
    in1Pitch2 = m_in1Dim3 == XAI_TILE3D_GET_DIM3(outTile) ? in1Pitch2 : 0;             \
    in2Pitch1 = m_in2Dim2 == XAI_TILE3D_GET_DIM2(outTile) ? in2Pitch1 : 0;             \
    in2Pitch2 = m_in2Dim3 == XAI_TILE3D_GET_DIM3(outTile) ? in2Pitch2 : 0;             \
  }

#define XAI_CHECK_REDUCE_DIM(inTile, outTile, params)                                                                                   \
  {                                                                                                                                     \
    if ((XAI_CNN_REDUCE_GET_CONFIG(params) & XAI_CNN_REDUCE_DIM1) != XAI_CNN_REDUCE_DIM1)                                               \
    {                                                                                                                                   \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1(outTile), XAI_ERR_DATASIZE,                                    \
                      "\nInput tile dim1size = %d, Output tile dim1size = %d\nFirst dimension of input and output tile must be equal",  \
                      XAI_TILE3D_GET_DIM1(inTile), XAI_TILE3D_GET_DIM1(outTile));                                                       \
    }                                                                                                                                   \
    else                                                                                                                                \
    {                                                                                                                                   \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(outTile) == 1, XAI_ERR_DATASIZE,                                                              \
                      "\nOutput tile dim1size = %d, size should be equal to 1", XAI_TILE3D_GET_DIM1(outTile));                          \
    }                                                                                                                                   \
    if ((XAI_CNN_REDUCE_GET_CONFIG(params) & XAI_CNN_REDUCE_DIM2) != XAI_CNN_REDUCE_DIM2)                                               \
    {                                                                                                                                   \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(inTile) == XAI_TILE3D_GET_DIM2(outTile), XAI_ERR_DATASIZE,                                    \
                      "\nInput tile dim2size = %d, Output tile dim2size = %d\nSecond dimension of input and output tile must be equal", \
                      XAI_TILE3D_GET_DIM2(inTile), XAI_TILE3D_GET_DIM2(outTile));                                                       \
    }                                                                                                                                   \
    else                                                                                                                                \
    {                                                                                                                                   \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2(outTile) == 1, XAI_ERR_DATASIZE,                                                              \
                      "\nOutput tile dim2size = %d, size should be equal to 1", XAI_TILE3D_GET_DIM2(outTile));                          \
    }                                                                                                                                   \
    if ((XAI_CNN_REDUCE_GET_CONFIG(params) & XAI_CNN_REDUCE_DIM3) != XAI_CNN_REDUCE_DIM3)                                               \
    {                                                                                                                                   \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inTile) == XAI_TILE3D_GET_DIM3(outTile), XAI_ERR_DATASIZE,                                    \
                      "\nInput tile dim3size = %d, Output tile dim3size = %d\nThird dimension of input and output tile must be equal",  \
                      XAI_TILE3D_GET_DIM3(inTile), XAI_TILE3D_GET_DIM3(outTile));                                                       \
    }                                                                                                                                   \
    else                                                                                                                                \
    {                                                                                                                                   \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(outTile) == 1, XAI_ERR_DATASIZE,                                                              \
                      "\nOutput tile dim3size = %d, size should be equal to 1", XAI_TILE3D_GET_DIM3(outTile));                          \
    }                                                                                                                                   \
  }

#define XAI_CHECK_REDUCE_DIM4D(inTile, outTile, params)                                                                  \
  {                                                                                                                      \
    if ((XAI_CNN_REDUCE_GET_CONFIG(params) & XAI_CNN_REDUCE_DIM1) != XAI_CNN_REDUCE_DIM1)                                \
    {                                                                                                                    \
      XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM1(inTile) == XAI_TILE4D_GET_DIM1(outTile), XAI_ERR_DATASIZE,                     \
                      "\nInput tile dim1size = %d, Output tile dim1size = %d\nInequality in first dimension",            \
                      XAI_TILE4D_GET_DIM1(inTile), XAI_TILE4D_GET_DIM1(outTile));                                        \
    }                                                                                                                    \
    else                                                                                                                 \
    {                                                                                                                    \
      XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM1(outTile) == 1, XAI_ERR_DATASIZE,                                               \
                      "\nOutput tile dim1size = %d, output first dimension should be 1", XAI_TILE4D_GET_DIM1(outTile));  \
    }                                                                                                                    \
    if ((XAI_CNN_REDUCE_GET_CONFIG(params) & XAI_CNN_REDUCE_DIM2) != XAI_CNN_REDUCE_DIM2)                                \
    {                                                                                                                    \
      XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM2(inTile) == XAI_TILE4D_GET_DIM2(outTile), XAI_ERR_DATASIZE,                     \
                      "\nInput tile dim2size = %d, Output tile dim2size = %d\nInequality in second dimension",           \
                      XAI_TILE4D_GET_DIM2(inTile), XAI_TILE4D_GET_DIM2(outTile));                                        \
    }                                                                                                                    \
    else                                                                                                                 \
    {                                                                                                                    \
      XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM2(outTile) == 1, XAI_ERR_DATASIZE,                                               \
                      "\nOutput tile dim2size = %d, output second dimension should be 1", XAI_TILE4D_GET_DIM2(outTile)); \
    }                                                                                                                    \
    if ((XAI_CNN_REDUCE_GET_CONFIG(params) & XAI_CNN_REDUCE_DIM3) != XAI_CNN_REDUCE_DIM3)                                \
    {                                                                                                                    \
      XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM3(inTile) == XAI_TILE4D_GET_DIM3(outTile), XAI_ERR_DATASIZE,                     \
                      "\nInput tile dim3size = %d, Output tile dim3size = %d\nInequality in third dimension",            \
                      XAI_TILE4D_GET_DIM3(inTile), XAI_TILE4D_GET_DIM3(outTile));                                        \
    }                                                                                                                    \
    else                                                                                                                 \
    {                                                                                                                    \
      XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM3(outTile) == 1, XAI_ERR_DATASIZE,                                               \
                      "\nOutput tile dim3size = %d, output third dimension should be 1", XAI_TILE4D_GET_DIM3(outTile));  \
    }                                                                                                                    \
    if ((XAI_CNN_REDUCE_GET_CONFIG(params) & XAI_CNN_REDUCE_DIM4) != XAI_CNN_REDUCE_DIM4)                                \
    {                                                                                                                    \
      XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM4(inTile) == XAI_TILE4D_GET_DIM4(outTile), XAI_ERR_DATASIZE,                     \
                      "\nInput tile dim4size = %d, Output tile dim4size = %d\nInequality in fourth dimension",           \
                      XAI_TILE4D_GET_DIM4(inTile), XAI_TILE4D_GET_DIM4(outTile));                                        \
    }                                                                                                                    \
    else                                                                                                                 \
    {                                                                                                                    \
      XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM4(outTile) == 1, XAI_ERR_DATASIZE,                                               \
                      "\nOutput tile dim3size = %d, output fourth dimension should be 1", XAI_TILE4D_GET_DIM4(outTile)); \
    }                                                                                                                    \
    XAI_CHECK_ERROR(XAI_CNN_REDUCE_GET_TILEFLAG(params) <= XAI_CNN_REDUCE_FIRST_LAST_TILE, XAI_ERR_BADARG,               \
                    "\nTile Flag = %hhu, Incorrect Tile Flag", XAI_CNN_REDUCE_GET_TILEFLAG(params));                     \
  }

#define XAI_CHECK_TILE3D_SIZE_BCAST_EQ(in, out, inStride)                                                                       \
  if (XAI_TILE3D_GET_DATA_ORDER(in) == XAI_WHD)                                                                                 \
  {                                                                                                                             \
    XAI_CHECK_ERROR(                                                                                                            \
      ((((XAI_TILE3D_GET_DIM1(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM1(out)) || (XAI_TILE3D_GET_DIM1(in) == 1)) && \
       (((XAI_TILE3D_GET_DIM2(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM2(out)) || (XAI_TILE3D_GET_DIM2(in) == 1)) && \
       ((XAI_TILE3D_GET_DIM3(in) == XAI_TILE3D_GET_DIM3(out)) || XAI_TILE3D_GET_DIM3(in) == 1)), XAI_ERR_DATASIZE,              \
      "Invalid dimension in (" #in ") or (" #out ") to perform Elementwise broadcast operation");                               \
  }                                                                                                                             \
  else if (XAI_TILE3D_GET_DATA_ORDER(in) == XAI_DWH)                                                                            \
  {                                                                                                                             \
    XAI_CHECK_ERROR(                                                                                                            \
      ((((XAI_TILE3D_GET_DIM2(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM2(out)) || (XAI_TILE3D_GET_DIM2(in) == 1)) && \
       (((XAI_TILE3D_GET_DIM3(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM3(out)) || (XAI_TILE3D_GET_DIM3(in) == 1)) && \
       ((XAI_TILE3D_GET_DIM1(in) == XAI_TILE3D_GET_DIM1(out)) || XAI_TILE3D_GET_DIM1(in) == 1)), XAI_ERR_DATASIZE,              \
      "Invalid dimension in (" #in ") or (" #out ") to perform Elementwise broadcast operation");                               \
  }

#define XAI_CHECK_TILE3D_SIZE_EQ_OR_BCAST(in, out, inStride)                                                 \
  if (XAI_TILE3D_GET_DATA_ORDER(in) == XAI_WHD)                                                              \
  {                                                                                                          \
    if ((XAI_TILE3D_GET_DIM3(in) == XAI_TILE3D_GET_DIM3(out)) &&                                             \
        ((XAI_TILE3D_GET_DIM1(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM1(out)) &&                 \
        ((XAI_TILE3D_GET_DIM2(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM2(out)))                   \
    {                                                                                                        \
      if (XAI_TILE3D_GET_DATA_PTR(in) == XAI_TILE3D_GET_DATA_PTR(out))                                       \
      {                                                                                                      \
        XAI_CHECK_ERROR(XAI_TILE3D_PITCH_EQ(in, out), XAI_ERR_INPLACE, "Inplace operation not "              \
                        "supported when pitch of ("#in ") and ("#out ") are not same");                      \
      }                                                                                                      \
    }                                                                                                        \
    else                                                                                                     \
    {                                                                                                        \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_PTR(in) != XAI_TILE3D_GET_DATA_PTR(out), XAI_ERR_INPLACE,          \
                      "Inplace operation not supported for Broadcast Operation for ("#in ") and ("#out ")"); \
      XAI_CHECK_TILE3D_SIZE_BCAST_EQ(in, out, inStride);                                                     \
    }                                                                                                        \
  }                                                                                                          \
  else if (XAI_TILE3D_GET_DATA_ORDER(in) == XAI_DWH)                                                         \
  {                                                                                                          \
    if ((XAI_TILE3D_GET_DIM1(in) == XAI_TILE3D_GET_DIM1(out)) &&                                             \
        ((XAI_TILE3D_GET_DIM2(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM2(out)) &&                 \
        ((XAI_TILE3D_GET_DIM3(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM3(out)))                   \
    {                                                                                                        \
      if (XAI_TILE3D_GET_DATA_PTR(in) == XAI_TILE3D_GET_DATA_PTR(out))                                       \
      {                                                                                                      \
        XAI_CHECK_ERROR(XAI_TILE3D_PITCH_EQ(in, out), XAI_ERR_INPLACE, "Inplace operation not "              \
                        "supported when pitch of ("#in ") and ("#out ") are not same");                      \
      }                                                                                                      \
    }                                                                                                        \
    else                                                                                                     \
    {                                                                                                        \
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_PTR(in) != XAI_TILE3D_GET_DATA_PTR(out), XAI_ERR_INPLACE,          \
                      "Inplace operation not supported for Broadcast Operation for ("#in ") and ("#out ")"); \
      XAI_CHECK_TILE3D_SIZE_BCAST_EQ(in, out, inStride);                                                     \
    }                                                                                                        \
  }

#define XAI_CHECK_TILE3D_BCAST_DIMENSIONS(inTile1, inTile2, outTile, in1Stride, in2Stride)                                \
  XAI_CHECK_TILE3D_SIZE_EQ_OR_BCAST(inTile1, outTile, in1Stride);                                                         \
  XAI_CHECK_TILE3D_SIZE_EQ_OR_BCAST(inTile2, outTile, in2Stride);                                                         \
  if (XAI_TILE3D_GET_DATA_ORDER(outTile) == XAI_WHD)                                                                      \
  {                                                                                                                       \
    XAI_CHECK_ERROR((MAX2(XAI_TILE3D_GET_DIM3(inTile1), XAI_TILE3D_GET_DIM3(inTile2)) == XAI_TILE3D_GET_DIM3(outTile)) && \
                    (MAX2((XAI_TILE3D_GET_DIM1(inTile1) + in1Stride - 1) / in1Stride,                                     \
                          (XAI_TILE3D_GET_DIM1(inTile2) + in2Stride - 1) / in2Stride) == XAI_TILE3D_GET_DIM1(outTile)) && \
                    (MAX2((XAI_TILE3D_GET_DIM2(inTile1) + in1Stride - 1) / in1Stride,                                     \
                          (XAI_TILE3D_GET_DIM2(inTile2) + in2Stride - 1) / in2Stride) == XAI_TILE3D_GET_DIM2(outTile)),   \
                    XAI_ERR_DATASIZE, "Invalid dimension to perform BroadCast/ElementWise operations");                   \
  }                                                                                                                       \
  else if (XAI_TILE3D_GET_DATA_ORDER(outTile) == XAI_DWH)                                                                 \
  {                                                                                                                       \
    XAI_CHECK_ERROR((MAX2(XAI_TILE3D_GET_DIM1(inTile1), XAI_TILE3D_GET_DIM1(inTile2)) == XAI_TILE3D_GET_DIM1(outTile)) && \
                    (MAX2((XAI_TILE3D_GET_DIM2(inTile1) + in1Stride - 1) / in1Stride,                                     \
                          (XAI_TILE3D_GET_DIM2(inTile2) + in2Stride - 1) / in2Stride) == XAI_TILE3D_GET_DIM2(outTile)) && \
                    (MAX2((XAI_TILE3D_GET_DIM3(inTile1) + in1Stride - 1) / in1Stride,                                     \
                          (XAI_TILE3D_GET_DIM3(inTile2) + in2Stride - 1) / in2Stride) == XAI_TILE3D_GET_DIM3(outTile)),   \
                    XAI_ERR_DATASIZE, "Invalid dimension to perform BroadCast/ElementWise operations");                   \
  }

#define XAI_TILE3D_GET_BCAST123_PITCH(inTile1, inTile2, inTile1Pitch0, inTile2Pitch0, inTile1Pitch1, \
                                      inTile2Pitch1, inTile1Pitch2, inTile2Pitch2)                   \
  int32_t inTile1Pitch0 = 1;                                                                         \
  int32_t inTile1Pitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile1);                                        \
  int32_t inTile1Pitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile1);                                        \
  int32_t inTile2Pitch0 = 1;                                                                         \
  int32_t inTile2Pitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile2);                                        \
  int32_t inTile2Pitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile2);                                        \
  if (XAI_TILE3D_GET_DIM1(inTile1) == 1) {                                                           \
    inTile1Pitch0 = 0; }                                                                             \
  else if (XAI_TILE3D_GET_DIM1(inTile2) == 1) {                                                      \
    inTile2Pitch0 = 0; }                                                                             \
  if (XAI_TILE3D_GET_DIM2(inTile1) == 1) {                                                           \
    inTile1Pitch1 = 0; }                                                                             \
  else if (XAI_TILE3D_GET_DIM2(inTile2) == 1) {                                                      \
    inTile2Pitch1 = 0; }                                                                             \
  if (XAI_TILE3D_GET_DIM3(inTile1) == 1) {                                                           \
    inTile1Pitch2 = 0; }                                                                             \
  else if (XAI_TILE3D_GET_DIM3(inTile2) == 1) {                                                      \
    inTile2Pitch2 = 0; }

#define XAI_TILE3D_SIZE_BCAST23_EQ(in, out, inStride)                                                                       \
  ((((XAI_TILE3D_GET_DIM2(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM2(out)) || (XAI_TILE3D_GET_DIM2(in) == 1)) && \
   (((XAI_TILE3D_GET_DIM3(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM3(out)) || (XAI_TILE3D_GET_DIM3(in) == 1)) && \
   (XAI_TILE3D_GET_DIM1(in) == XAI_TILE3D_GET_DIM1(out)))

#define XAI_CHECK_TILE3D_SIZE_EQ_OR_BCAST23(in, out, inStride)                                                  \
  if ((XAI_TILE3D_GET_DIM1(in) == XAI_TILE3D_GET_DIM1(out)) &&                                                  \
      ((XAI_TILE3D_GET_DIM2(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM2(out)) &&                      \
      ((XAI_TILE3D_GET_DIM3(in) + inStride - 1) / inStride == XAI_TILE3D_GET_DIM3(out)))                        \
  {                                                                                                             \
    if (XAI_TILE3D_GET_DATA_PTR(in) == XAI_TILE3D_GET_DATA_PTR(out))                                            \
    {                                                                                                           \
      XAI_CHECK_ERROR(XAI_TILE3D_PITCH_EQ(in, out), XAI_ERR_INPLACE, "Inplace operation not "                   \
                      "supported when pitch of ("#in ") and ("#out ") are not same");                           \
    }                                                                                                           \
  }                                                                                                             \
  else                                                                                                          \
  {                                                                                                             \
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_PTR(in) != XAI_TILE3D_GET_DATA_PTR(out), XAI_ERR_INPLACE,               \
                    "Inplace operation not supported for Broadcast Operation for ("#in ") and ("#out ")");      \
    XAI_CHECK_ERROR(XAI_TILE3D_SIZE_BCAST23_EQ(in, out, inStride), XAI_ERR_DATASIZE,                            \
                    "Invalid dimension in (" #in ") or (" #out ") to perform Elementwise broadcast operation"); \
  }

#define XAI_CHECK_TILE3D_BCAST23_DIMENSIONS(inTile1, inTile2, outTile, in1Stride, in2Stride)                            \
  XAI_CHECK_TILE3D_SIZE_EQ_OR_BCAST23(inTile1, outTile, in1Stride)                                                      \
  XAI_CHECK_TILE3D_SIZE_EQ_OR_BCAST23(inTile2, outTile, in2Stride)                                                      \
  XAI_CHECK_ERROR((MAX2(XAI_TILE3D_GET_DIM1(inTile1), XAI_TILE3D_GET_DIM1(inTile2)) ==                                  \
                   XAI_TILE3D_GET_DIM1(outTile)) &&                                                                     \
                  (MAX2((XAI_TILE3D_GET_DIM2(inTile1) + in1Stride - 1) / in1Stride,                                     \
                        (XAI_TILE3D_GET_DIM2(inTile2) + in2Stride - 1) / in2Stride) == XAI_TILE3D_GET_DIM2(outTile)) && \
                  (MAX2((XAI_TILE3D_GET_DIM3(inTile1) + in1Stride - 1) / in1Stride,                                     \
                        (XAI_TILE3D_GET_DIM3(inTile2) + in2Stride - 1) / in2Stride) == XAI_TILE3D_GET_DIM3(outTile)),   \
                  XAI_ERR_DATASIZE, "Invalid dimension to perform BroadCast/ElementWise operations")

#define XAI_CHECK_LSTM_BLOCK(functionCall)                                             \
  {                                                                                    \
    int32_t retVal = (functionCall);                                                   \
    (void) retVal;                                                                     \
    XAI_ERROR_CHECKS_CONTINUE()                                                        \
    {                                                                                  \
      XAI_CHECK_ERROR((retVal == XAI_ERR_OK), retVal,                                  \
                      "\nError in file: %s, function: %s, LSTM block: %s, line: %d\n", \
                      __FILE__, __func__, #functionCall, __LINE__);                    \
    }                                                                                  \
  }                                                                                    \

