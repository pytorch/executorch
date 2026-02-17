/*
 * Copyright (c) 2022 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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

#include <string.h>
#include "xai_cnn.h"
#include "xai_intrin.h"

#if ((XCHAL_VISION_TYPE >= 6))

#undef INPUT_DATA_TYPE
#undef OUTPUT_DATA_TYPE

#define INPUT_DATA_TYPE  INTEGER8BIT
#include "cnn_fill_tile.h"
#undef INPUT_DATA_TYPE

#define INPUT_DATA_TYPE  INTEGER16BIT
#include "cnn_fill_tile.h"
#undef INPUT_DATA_TYPE

#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
#define INPUT_DATA_TYPE  FLOAT16BIT
#include "cnn_fill_tile.h"
#undef INPUT_DATA_TYPE
#endif

#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
#define INPUT_DATA_TYPE  FLOAT32BIT
#include "cnn_fill_tile.h"
#undef INPUT_DATA_TYPE
#endif

#define INPUT_DATA_TYPE  INTEGER8BIT
#include "cnn_extend_edge.h"
#undef INPUT_DATA_TYPE

#define INPUT_DATA_TYPE  INTEGER16BIT
#include "cnn_extend_edge.h"
#undef INPUT_DATA_TYPE

#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
#define INPUT_DATA_TYPE  FLOAT16BIT
#include "cnn_extend_edge.h"
#undef INPUT_DATA_TYPE
#endif

#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
#define INPUT_DATA_TYPE  FLOAT32BIT
#include "cnn_extend_edge.h"
#undef INPUT_DATA_TYPE
#endif

#define INPUT_DATA_TYPE  SIGNED16BIT
#include "cnn_dataConversion3D_I16I8.h"
#undef INPUT_DATA_TYPE

#define INPUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_dataConversion3D_I16I8.h"
#undef INPUT_DATA_TYPE

#define INPUT_DATA_TYPE  SIGNED8BIT
#include "cnn_dataConversion3D_I8I32.h"
#undef INPUT_DATA_TYPE

#define INPUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_dataConversion3D_I8I32.h"
#undef INPUT_DATA_TYPE

#define INPUT_DATA_TYPE   SIGNED32BIT
#define OUTPUT_DATA_TYPE  SIGNED8BIT
#include "cnn_dataConversion3D_S32IX.h"
#undef INPUT_DATA_TYPE
#undef OUTPUT_DATA_TYPE

#define INPUT_DATA_TYPE   SIGNED32BIT
#define OUTPUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_dataConversion3D_S32IX.h"
#undef INPUT_DATA_TYPE
#undef OUTPUT_DATA_TYPE

#define INPUT_DATA_TYPE   SIGNED32BIT
#define OUTPUT_DATA_TYPE  SIGNED16BIT
#include "cnn_dataConversion3D_S32IX.h"
#undef INPUT_DATA_TYPE
#undef OUTPUT_DATA_TYPE

#define INPUT_DATA_TYPE   SIGNED32BIT
#define OUTPUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_dataConversion3D_S32IX.h"
#undef INPUT_DATA_TYPE
#undef OUTPUT_DATA_TYPE

#define OUTPUT_DATA_TYPE  SIGNED8BIT
#include "cnn_dataConversion3D_AsymQ_S8IX.h"
#undef OUTPUT_DATA_TYPE

#define OUTPUT_DATA_TYPE  UNSIGNED8BIT
#include "cnn_dataConversion3D_AsymQ_S8IX.h"
#undef OUTPUT_DATA_TYPE

#define OUTPUT_DATA_TYPE  SIGNED16BIT
#include "cnn_dataConversion3D_AsymQ_S8IX.h"
#undef OUTPUT_DATA_TYPE

#define OUTPUT_DATA_TYPE  UNSIGNED16BIT
#include "cnn_dataConversion3D_AsymQ_S8IX.h"
#undef OUTPUT_DATA_TYPE

#define PACK_ROUND_U16(vecOut1, vecInData1, Scale, Shift)  {                                           \
    xb_vecNx48 acc          = IVP_MULUSNX16((xb_vecNx16U) Scale, vecInData1);                          \
    xb_vecN_2x32v m_outEven = IVP_PACKVRNX48_0(acc, Shift);                                            \
    xb_vecN_2x32v m_outOdd  = IVP_PACKVRNX48_1(acc, Shift);                                            \
    m_outEven = IVP_MAXN_2X32(IVP_MINN_2X32(m_outEven, (xb_vecN_2x32v) USHRT_MAX), (xb_vecN_2x32v) 0); \
    m_outOdd  = IVP_MAXN_2X32(IVP_MINN_2X32(m_outOdd, (xb_vecN_2x32v) USHRT_MAX), (xb_vecN_2x32v) 0);  \
    xb_vecNx16U temp1 = IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROMN_2X32(m_outEven));                      \
    xb_vecNx16U temp2 = IVP_MOVNX16U_FROMNX16(IVP_MOVNX16_FROMN_2X32(m_outOdd));                       \
    vecOut1 = IVP_SELNX16UI(temp2, temp1, IVP_SELI_16B_INTERLEAVE_1_EVEN);                             \
}

/*************************** xaiFillTile3D ***********************************/
/* Description : General API for FillTile3D optimized implementation        */
/*               Calls one of the FillTile3D functions based                */
/*               on the parameters                                          */
/* Inputs      : constant value to fill, fillEdgeExtension                  */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Destination Tile                                           */
/****************************************************************************/
XAI_ERR_TYPE xaiFillTile3D(xai_pTile3D dstTile,
                           const int32_t value,
                           xai_bool fillEdgeExtension)
{
  if (!dstTile)
  {
    return(XAI_ERR_NULLARG);
  }

  if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_S8) || XAI_TILE3D_CHECK_TYPE(dstTile, XAI_U8))
  {
    return(xaiFillTile3D_I8(dstTile, value, fillEdgeExtension));
  }
  else if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_S16) || XAI_TILE3D_CHECK_TYPE(dstTile, XAI_U16))
  {
    return(xaiFillTile3D_I16(dstTile, value, fillEdgeExtension));
  }
#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
  else if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_F16))
  {
    return(xaiFillTile3D_F16(dstTile, value, fillEdgeExtension));
  }
#endif
#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
  else if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_F32))
  {
    return(xaiFillTile3D_F32(dstTile, value, fillEdgeExtension));
  }
#endif
  return(XAI_ERR_NO_VARIANT);
}

/************************* xaiExtendEdgesConst3D *****************************/
/* Description : General API for ExtendEdgesConst3D optimized implementation*/
/*               Calls one of the ExtendEdgesConst3D functions based        */
/*               on the parameters                                          */
/* Inputs      : constant value to fill the edges                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Destination Tile                                           */
/****************************************************************************/
XAI_ERR_TYPE xaiExtendEdgesConst3D(xai_pTile3D dstTile,
                                   const int32_t value,
                                   xai_size3D frame3DSize)
{
  if (!dstTile)
  {
    return(XAI_ERR_NULLARG);
  }

  if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_S8) || XAI_TILE3D_CHECK_TYPE(dstTile, XAI_U8))
  {
    return(xaiExtendEdgesConst3D_I8(dstTile, value, frame3DSize));
  }
  else if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_S16) || XAI_TILE3D_CHECK_TYPE(dstTile, XAI_U16))
  {
    return(xaiExtendEdgesConst3D_I16(dstTile, value, frame3DSize));
  }
#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
  else if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_F16))
  {
    int16_t valueS16 = (int16_t) value;
#if defined(__XTENSA__)
    xb_f16 valueF16;
    memcpy(&valueF16, &valueS16, sizeof(int16_t));
#else
    xb_f16 valueF16 = *(xb_f16 *) (&valueS16);
#endif
    return(xaiExtendEdgesConst3D_F16(dstTile, valueF16, frame3DSize));
  }
#endif
#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
  else if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_F32))
  {
    int32_t valueS32 = (int32_t) value;
    float valueF32;
    memcpy(&valueF32, &valueS32, sizeof(int32_t));
    return(xaiExtendEdgesConst3D_F32(dstTile, valueF32, frame3DSize));
  }
#endif
  return(XAI_ERR_NO_VARIANT);
}

/***********************   xaiExtendEdges3D   ********************************/
/* Description : General API for ExtendEdges3D optimized implementation     */
/*               Calls one of the ExtendEdges3D functions based             */
/*               on the parameters                                          */
/* Inputs      : pArray, frame3DSize                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Input Tile                                                 */
/****************************************************************************/
XAI_ERR_TYPE xaiExtendEdges3D(xai_pTile3D dstTile,
                              const xai_pArray pArray,
                              xai_size3D frame3DSize)
{
  if (!dstTile)
  {
    return(XAI_ERR_NULLARG);
  }

  if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_S8) || XAI_TILE3D_CHECK_TYPE(dstTile, XAI_U8))
  {
    return(xaiExtendEdges3D_I8(dstTile, pArray, frame3DSize));
  }
  else if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_S16) || XAI_TILE3D_CHECK_TYPE(dstTile, XAI_U16))
  {
    return(xaiExtendEdges3D_I16(dstTile, pArray, frame3DSize));
  }
#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
  else if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_F16))
  {
    return(xaiExtendEdges3D_F16(dstTile, pArray, frame3DSize));
  }
#endif
#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
  else if (XAI_TILE3D_CHECK_TYPE(dstTile, XAI_F32))
  {
    return(xaiExtendEdges3D_F32(dstTile, pArray, frame3DSize));
  }
#endif
  return(XAI_ERR_NO_VARIANT);
}

/************************** xaiCopyTile3D  ***********************************/
/* Description : P6 optimized implementation for copying the contents of a  */
/*               3D tile to another 3D tile. This function supports copying */
/*               of 8/16/32/64 bit input tile data based on data type of    */
/*               tile data elements. copy_edge_extension flag is used to    */
/*               control copy of edges. If edge sizes are different, then   */
/*               minimum of input & output edge size number of elements is  */
/*               copied from edges.                                         */
/* Inputs      : Input Tile data, copy_edge_extension,                      */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : Active data size of input & output tiles are the same      */
/****************************************************************************/

XAI_ERR_TYPE xaiCopyTile3D(const xai_pTile3D inTile,
                           xai_pTile3D outTile,
                           xai_bool copy_edge_extension)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D(inTile);
    XAI_CHECK_TILE3D(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR((((XAI_TILE3D_GET_ELEMENT_SIZE(inTile) == 1) || (XAI_TILE3D_GET_ELEMENT_SIZE(inTile) == 2)) ||                \
                     (XAI_TILE3D_GET_ELEMENT_SIZE(inTile) == 4) || (XAI_TILE3D_GET_ELEMENT_SIZE(inTile) == 8)), XAI_ERR_DATATYPE, \
                    "Element size of Input tile = %d, The argument of input tile has unsupported data type",                      \
                    XAI_TILE3D_GET_ELEMENT_SIZE(inTile));
    XAI_CHECK_TILE3D_ELEMENT_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Getting parameters from the tile structures                               */
  /* Tile size across first dimension of input tile and output tile is scaled  */
  /* based on input data type of tile data elements                            */

  const int32_t element_size  = XAI_TILE3D_GET_ELEMENT_SIZE(inTile);
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile) * element_size;
  const int32_t inDim1Edge1   = XAI_TILE3D_GET_DIM1_EDGE1(inTile) * element_size;
  const int32_t inDim1Edge2   = XAI_TILE3D_GET_DIM1_EDGE2(inTile) * element_size;
  const int32_t outDim1Edge1  = XAI_TILE3D_GET_DIM1_EDGE1(outTile) * element_size;
  const int32_t outDim1Edge2  = XAI_TILE3D_GET_DIM1_EDGE2(outTile) * element_size;
  const int32_t inDataPitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile) * element_size;
  const int32_t inDataPitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile) * element_size;
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile) * element_size;
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile) * element_size;
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inDim2Edge1   = XAI_TILE3D_GET_DIM2_EDGE1(inTile);
  const int32_t inDim2Edge2   = XAI_TILE3D_GET_DIM2_EDGE2(inTile);
  const int32_t inDim3Edge1   = XAI_TILE3D_GET_DIM3_EDGE1(inTile);
  const int32_t inDim3Edge2   = XAI_TILE3D_GET_DIM3_EDGE2(inTile);
  const int32_t outDim2Edge1  = XAI_TILE3D_GET_DIM2_EDGE1(outTile);
  const int32_t outDim2Edge2  = XAI_TILE3D_GET_DIM2_EDGE2(outTile);
  const int32_t outDim3Edge1  = XAI_TILE3D_GET_DIM3_EDGE1(outTile);
  const int32_t outDim3Edge2  = XAI_TILE3D_GET_DIM3_EDGE2(outTile);
  /* Vectorization for xaiCopyTile3D function is always done across the first dimension */
  int32_t vectorizationWidth   = 2 * XCHAL_IVPN_SIMD_WIDTH;
  int32_t vectorizationWidth2X = vectorizationWidth * 2;
  int32_t vectorizationWidth3X = vectorizationWidth * 3;
  int32_t vectorizationWidth4X = vectorizationWidth * 4;

  int8_t *pInput  = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutput = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  int32_t z, x, y;
  int32_t dim1CopySize = dim1Size;
  int32_t dim2CopySize = dim2Size;
  int32_t dim3CopySize = dim3Size;
  int32_t dim1CopyEdge1Size;
  int32_t dim2CopyEdge1Size;
  int32_t dim3CopyEdge1Size;
  int32_t dim1CopyEdge2Size;
  int32_t dim2CopyEdge2Size;
  int32_t dim3CopyEdge2Size;
  int32_t maxLoopCount;
  valign vaInData;
  valign vaOutData = IVP_ZALIGN();
  xb_vec2Nx8* restrict pdvecIn;
  xb_vec2Nx8* restrict pdvecOut;
  xb_vec2Nx8 vecValue;

  /* If copy_edge_extension flag is enabled update input and output data pointer  */
  /* and data copy size across all 3 dimensions.                                 */

  if (copy_edge_extension)
  {
    dim1CopyEdge1Size = XT_MIN(inDim1Edge1, outDim1Edge1);
    dim2CopyEdge1Size = XT_MIN(inDim2Edge1, outDim2Edge1);
    dim3CopyEdge1Size = XT_MIN(inDim3Edge1, outDim3Edge1);
    dim1CopyEdge2Size = XT_MIN(inDim1Edge2, outDim1Edge2);
    dim2CopyEdge2Size = XT_MIN(inDim2Edge2, outDim2Edge2);
    dim3CopyEdge2Size = XT_MIN(inDim3Edge2, outDim3Edge2);
    dim1CopySize      = dim1Size + dim1CopyEdge1Size + dim1CopyEdge2Size;
    dim2CopySize      = dim2Size + dim2CopyEdge1Size + dim2CopyEdge2Size;
    dim3CopySize      = dim3Size + dim3CopyEdge1Size + dim3CopyEdge2Size;
    pInput            = &pInput[-dim1CopyEdge1Size + ((-dim2CopyEdge1Size) * inDataPitch1) \
                                + ((-dim3CopyEdge1Size) * inDataPitch2)];
    pOutput = &pOutput[-dim1CopyEdge1Size + ((-dim2CopyEdge1Size) * outDataPitch1) \
                       + ((-dim3CopyEdge1Size) * outDataPitch2)];
  }

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When output tile pitch is equal to output tile copy size.               */
  /*    - If above condition holds good, memory location to be copied           */
  /*      from inTile to outTile is contiguous. Hence vectorization can be      */
  /*      utilized effectively                                                  */
  /* 2. When output tile pitch is greater than output tile copy size.           */
  /*    - If above condition holds good, memory location to be copied           */
  /*      from inTile to outTile is contiguous. In order to do                  */
  /*      vectorization across first dimension, output data pointers            */
  /*      need to be updated based on output tile copy size and                 */
  /*      output tile pitch                                                     */
  /******************************************************************************/

  if ((inDataPitch1 == dim1CopySize) && (outDataPitch1 == dim1CopySize))
  {
    /* Data to be copied exist in contiguous memory location with respect to */
    /* first dimension                                                       */

    /* Initialize max loop counter */
    int32_t maxdim3LoopCount = dim3CopySize;
    maxLoopCount = dim1CopySize * dim2CopySize;

    if ((inDataPitch2 == maxLoopCount) && (outDataPitch2 == maxLoopCount))
    {
      /* Data to be filled exist in contiguous memory location with respect to */
      /* first and second dimension                                            */

      /* Update max loop counter */
      maxdim3LoopCount = 1;
      maxLoopCount    *= dim3CopySize;
    }
    for (z = 0; z < maxdim3LoopCount; z++)
    {
      /* initialize input and output data pointer */
      pdvecIn  = (xb_vec2Nx8 *) (pInput + (z * inDataPitch2));
      pdvecOut = (xb_vec2Nx8 *) (pOutput + (z * outDataPitch2));
      vaInData = IVP_LA2NX8_PP(pdvecIn);
      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Read vector input data */
        IVP_LA2NX8_IP(vecValue, vaInData, pdvecIn);
        /* Store vector output data */
        IVP_SA2NX8_IP(vecValue, vaOutData, pdvecOut);
      }

      IVP_LAV2NX8_XP(vecValue, vaInData, pdvecIn, maxLoopCount - x);
      IVP_SAV2NX8_XP(vecValue, vaOutData, pdvecOut, maxLoopCount - x);
      IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
    }
  }
  else
  {
    /* else block execute, if output tile pitch is  greater than output tile copy size   */
    /* or input tile pitch in not equal to output tile pitch                             */

    for (z = 0; z < dim3CopySize; z++) /* Loop across dim3 */
    {
      x = 0;
      /* Loop across dimension 1 */

      /* Condition check added to maximize vectorization across dimension 1*/
      /* Loop across dim1 */
      for (; x < (dim1CopySize - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* initialize input and output data pointer */
        int8_t *pInput1  = pInput + x + (z * inDataPitch2);
        int8_t *pOutput1 = pOutput + x + (z * outDataPitch2);
        int32_t varLen   = dim1CopySize - (x + vectorizationWidth3X);

        for (y = 0; y < dim2CopySize; y++)
        {
          pdvecIn  = (xb_vec2Nx8 *) (pInput1 + (y * inDataPitch1));
          pdvecOut = (xb_vec2Nx8 *) (pOutput1 + (y * outDataPitch1));
          vaInData = IVP_LA2NX8_PP(pdvecIn);

          /* Read vector data from inTile and copy vector data to outTile */
          IVP_LA2NX8_IP(vecValue, vaInData, pdvecIn);
          IVP_SA2NX8_IP(vecValue, vaOutData, pdvecOut);
          IVP_LA2NX8_IP(vecValue, vaInData, pdvecIn);
          IVP_SA2NX8_IP(vecValue, vaOutData, pdvecOut);
          IVP_LA2NX8_IP(vecValue, vaInData, pdvecIn);
          IVP_SA2NX8_IP(vecValue, vaOutData, pdvecOut);
          IVP_LAV2NX8_XP(vecValue, vaInData, pdvecIn, varLen);
          IVP_SAV2NX8_XP(vecValue, vaOutData, pdvecOut, varLen);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
      }
      if (x < (dim1CopySize - vectorizationWidth2X)) /* Loop unrolling across dim2 */
      {
        /* initialize input and output data pointer */
        int8_t *pInput1  = pInput + x + (z * inDataPitch2);
        int8_t *pOutput1 = pOutput + x + (z * outDataPitch2);
        int32_t varLen   = dim1CopySize - (x + vectorizationWidth2X);
        for (y = 0; y < dim2CopySize; y++)
        {
          pdvecIn  = (xb_vec2Nx8 *) (pInput1 + (y * inDataPitch1));
          pdvecOut = (xb_vec2Nx8 *) (pOutput1 + (y * outDataPitch1));
          vaInData = IVP_LA2NX8_PP(pdvecIn);

          /* Read vector data from inTile and copy vector data to outTile */
          IVP_LA2NX8_IP(vecValue, vaInData, pdvecIn);
          IVP_SA2NX8_IP(vecValue, vaOutData, pdvecOut);
          IVP_LA2NX8_IP(vecValue, vaInData, pdvecIn);
          IVP_SA2NX8_IP(vecValue, vaOutData, pdvecOut);
          IVP_LAV2NX8_XP(vecValue, vaInData, pdvecIn, varLen);
          IVP_SAV2NX8_XP(vecValue, vaOutData, pdvecOut, varLen);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
      }
      else if (x < (dim1CopySize - vectorizationWidth))
      {
        /* initialize input and output data pointer */
        int8_t *pInput1  = pInput + x + (z * inDataPitch2);
        int8_t *pOutput1 = pOutput + x + (z * outDataPitch2);
        int32_t varLen   = dim1CopySize - (x + vectorizationWidth);
        for (y = 0; y < dim2CopySize; y++)
        {
          pdvecIn  = (xb_vec2Nx8 *) (pInput1 + (y * inDataPitch1));
          pdvecOut = (xb_vec2Nx8 *) (pOutput1 + (y * outDataPitch1));
          vaInData = IVP_LA2NX8_PP(pdvecIn);

          /* Read vector data from inTile and copy vector data to outTile */
          IVP_LA2NX8_IP(vecValue, vaInData, pdvecIn);
          IVP_SA2NX8_IP(vecValue, vaOutData, pdvecOut);
          IVP_LAV2NX8_XP(vecValue, vaInData, pdvecIn, varLen);
          IVP_SAV2NX8_XP(vecValue, vaOutData, pdvecOut, varLen);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
      }
      else if (x < dim1CopySize)
      {
        /* initialize input and output data pointer */
        int8_t *pInput1  = pInput + x + (z * inDataPitch2);
        int8_t *pOutput1 = pOutput + x + (z * outDataPitch2);
        int32_t varLen   = dim1CopySize - x;
        for (y = 0; y < dim2CopySize; y++)
        {
          pdvecIn  = (xb_vec2Nx8 *) (pInput1 + (y * inDataPitch1));
          pdvecOut = (xb_vec2Nx8 *) (pOutput1 + (y * outDataPitch1));
          vaInData = IVP_LA2NX8_PP(pdvecIn);

          /* Read vector data from inTile and copy vector data */
          IVP_LAV2NX8_XP(vecValue, vaInData, pdvecIn, varLen);
          IVP_SAV2NX8_XP(vecValue, vaOutData, pdvecOut, varLen);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/************************ xaiUnsignedToSigned3D_U8S8 ******************************/
/* Description : P6 optimized implementation for converting the tile data from   */
/*               unsigned 8bit to signed 8bit. This function can operate         */
/*               in-place. Applications needing this function to operate         */
/*               in-place can provide the same Input and Output Tiles.           */
/* Inputs      : Input Tile                                                      */
/* Outputs     : XI Error Code                                                   */
/* InOuts      : Output Tile                                                     */
/* Assumptions : InData is unsigned 8bit                                         */
/*               Unsigned to Signed 8bit conversion not performed on tile edges  */
/*********************************************************************************/
XAI_ERR_TYPE xaiUnsignedToSigned3D_U8S8(xai_pTile3D inTile, xai_pTile3D outTile)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_TILE3D_S8(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Getting parameters from the tile structures */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t inDataPitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);

  /* Input and Output Data Pointers */
  uint8_t *pInput = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutput = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int32_t maxLoopCount;

  /*  xaiUnsignedToSigned3D_U8S8 function support in-place unsigned to signed 8bit */
  /*  conversion. In a such a scenario inTile and outTile will be overlapping.    */
  /*  Hence restrict keyword is not used for input and output data pointers       */
  xb_vec2Nx8U* restrict pdvecIn;
  xb_vec2Nx8* restrict pdvecOut;
  valign vaInData;
  valign vaOutData = IVP_ZALIGN();
  xb_vec2Nx8U vecValue1, vecValue2, vecValue3, vecValue4;
  xb_vec2Nx8 vecValueSigned1, vecValueSigned2, vecValueSigned3, vecValueSigned4;
  const xb_vec2Nx8 signedCharMax = SCHAR_MAX;

  /* Vectorization for xaiUnsignedToSigned3D_U8S8 function */
  /* is always done across the first dimension            */
  int32_t vectorizationWidth   = 2 * XCHAL_IVPN_SIMD_WIDTH;
  int32_t vectorizationWidth2X = 2 * vectorizationWidth;
  int32_t vectorizationWidth3X = 3 * vectorizationWidth;
  int32_t vectorizationWidth4X = 4 * vectorizationWidth;
  int32_t x, y, z;

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which unsigned       */
  /*      8 bit to signed 8 bit conversion need to done present in contiguous   */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - If above condition holds good, data elements for which unsigned       */
  /*      8 bit to signed 8 bit conversion need to done exist in non-contiguous */
  /*      memory location. In order to do vectorization across first dimension, */
  /*      output data pointers need to be updated based on output tile size     */
  /*      and output tile pitch                                                 */
  /******************************************************************************/

  if ((inDataPitch1 == dim1Size) && (outDataPitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    maxLoopCount = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inDataPitch2 == maxLoopCount) && (outDataPitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/
      dim3MaxLoopCount = 1;       /* Update max loop counter */
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input data pointer */
      pdvecIn = (xb_vec2Nx8U *) (pInput + (z * inDataPitch2));
      /* initialize output data pointer */
      pdvecOut = (xb_vec2Nx8 *) (pOutput + (z * outDataPitch2));
      vaInData = IVP_LA2NX8U_PP(pdvecIn);

      for (x = 0; x < maxLoopCount - vectorizationWidth4X; x += vectorizationWidth4X)
      {
        /* Load Data */
        IVP_LA2NX8U_IP(vecValue1, vaInData, pdvecIn);
        IVP_LA2NX8U_IP(vecValue2, vaInData, pdvecIn);
        IVP_LA2NX8U_IP(vecValue3, vaInData, pdvecIn);
        IVP_LA2NX8U_IP(vecValue4, vaInData, pdvecIn);

        /* Perform unsigned to signed conversion and rounding off operation */
        vecValue1 = IVP_AVGRU2NX8(vecValue1, 0);
        vecValue2 = IVP_AVGRU2NX8(vecValue2, 0);
        vecValue3 = IVP_AVGRU2NX8(vecValue3, 0);
        vecValue4 = IVP_AVGRU2NX8(vecValue4, 0);

        /* Perform saturation of signed max value */
        vecValueSigned1 = IVP_MINU2NX8U(signedCharMax, vecValue1);
        vecValueSigned2 = IVP_MINU2NX8U(signedCharMax, vecValue2);
        vecValueSigned3 = IVP_MINU2NX8U(signedCharMax, vecValue3);
        vecValueSigned4 = IVP_MINU2NX8U(signedCharMax, vecValue4);

        /* Store Data */
        IVP_SA2NX8_IP(vecValueSigned1, vaOutData, pdvecOut);
        IVP_SA2NX8_IP(vecValueSigned2, vaOutData, pdvecOut);
        IVP_SA2NX8_IP(vecValueSigned3, vaOutData, pdvecOut);
        IVP_SA2NX8_IP(vecValueSigned4, vaOutData, pdvecOut);
      }
      /* Load remaining data */
      IVP_LAV2NX8U_XP(vecValue1, vaInData, pdvecIn, maxLoopCount - (x + vectorizationWidth3X));
      IVP_LAV2NX8U_XP(vecValue2, vaInData, pdvecIn, maxLoopCount - (x + vectorizationWidth2X));
      IVP_LAV2NX8U_XP(vecValue3, vaInData, pdvecIn, maxLoopCount - (x + vectorizationWidth));
      IVP_LAV2NX8U_XP(vecValue4, vaInData, pdvecIn, maxLoopCount - x);

      /* Perform unsigned to signed conversion and rounding off operation */
      vecValue1 = IVP_AVGRU2NX8(vecValue1, 0);
      vecValue2 = IVP_AVGRU2NX8(vecValue2, 0);
      vecValue3 = IVP_AVGRU2NX8(vecValue3, 0);
      vecValue4 = IVP_AVGRU2NX8(vecValue4, 0);

      /* Perform saturation of signed max value */
      vecValueSigned1 = IVP_MINU2NX8U(signedCharMax, vecValue1);
      vecValueSigned2 = IVP_MINU2NX8U(signedCharMax, vecValue2);
      vecValueSigned3 = IVP_MINU2NX8U(signedCharMax, vecValue3);
      vecValueSigned4 = IVP_MINU2NX8U(signedCharMax, vecValue4);

      /* Variable stores */
      IVP_SAV2NX8_XP(vecValueSigned1, vaOutData, pdvecOut,
                     maxLoopCount - (x + vectorizationWidth3X));
      IVP_SAV2NX8_XP(vecValueSigned2, vaOutData, pdvecOut,
                     maxLoopCount - (x + vectorizationWidth2X));
      IVP_SAV2NX8_XP(vecValueSigned3, vaOutData, pdvecOut, maxLoopCount - (x + vectorizationWidth));
      IVP_SAV2NX8_XP(vecValueSigned4, vaOutData, pdvecOut, maxLoopCount - x);
      IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++) /* Loop across dim3 */
    {
      x = 0;
      /* Loop across dimension 1 */
      /* Condition check added to maximize vectorization across dimension 1*/
      /* Loop across dim1 */
      for (; x < (dim1Size - 3 * vectorizationWidth); x += 4 * vectorizationWidth)
      {
        /* initialize input and output data pointer */
        uint8_t *pInput1 = pInput + x + (z * inDataPitch2);
        int8_t *pOutput1 = pOutput + x + (z * outDataPitch2);
        int32_t varLen   = dim1Size - (x + 3 * vectorizationWidth);

        for (y = 0; y < dim2Size; y++)
        {
          pdvecIn  = (xb_vec2Nx8U *) (pInput1 + (y * inDataPitch1));
          pdvecOut = (xb_vec2Nx8 *) (pOutput1 + (y * outDataPitch1));
          vaInData = IVP_LA2NX8U_PP(pdvecIn);

          /* Load Input Data */
          IVP_LA2NX8U_IP(vecValue1, vaInData, pdvecIn);
          IVP_LA2NX8U_IP(vecValue2, vaInData, pdvecIn);
          IVP_LA2NX8U_IP(vecValue3, vaInData, pdvecIn);
          IVP_LAV2NX8U_XP(vecValue4, vaInData, pdvecIn, varLen);

          /* Perform unsigned to signed conversion and rounding off operation */
          vecValue1 = IVP_AVGRU2NX8(vecValue1, 0);
          vecValue2 = IVP_AVGRU2NX8(vecValue2, 0);
          vecValue3 = IVP_AVGRU2NX8(vecValue3, 0);
          vecValue4 = IVP_AVGRU2NX8(vecValue4, 0);

          /* Perform saturation of signed max value */
          vecValueSigned1 = IVP_MINU2NX8U(signedCharMax, vecValue1);
          vecValueSigned2 = IVP_MINU2NX8U(signedCharMax, vecValue2);
          vecValueSigned3 = IVP_MINU2NX8U(signedCharMax, vecValue3);
          vecValueSigned4 = IVP_MINU2NX8U(signedCharMax, vecValue4);

          /* Store */
          IVP_SA2NX8_IP(vecValueSigned1, vaOutData, pdvecOut);
          IVP_SA2NX8_IP(vecValueSigned2, vaOutData, pdvecOut);
          IVP_SA2NX8_IP(vecValueSigned3, vaOutData, pdvecOut);
          IVP_SAV2NX8_XP(vecValueSigned4, vaOutData, pdvecOut, varLen);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
      }
      if (x < (dim1Size - 2 * vectorizationWidth)) /* Loop unrolling across dim2 */
      {
        /* initialize input and output data pointer */
        uint8_t *pInput1 = pInput + x + (z * inDataPitch2);
        int8_t *pOutput1 = pOutput + x + (z * outDataPitch2);
        int32_t varLen   = dim1Size - (x + 2 * vectorizationWidth);

        for (y = 0; y < dim2Size; y++)
        {
          pdvecIn  = (xb_vec2Nx8U *) (pInput1 + (y * inDataPitch1));
          pdvecOut = (xb_vec2Nx8 *) (pOutput1 + (y * outDataPitch1));
          vaInData = IVP_LA2NX8U_PP(pdvecIn);

          /* Load Input Data */
          IVP_LA2NX8U_IP(vecValue1, vaInData, pdvecIn);
          IVP_LA2NX8U_IP(vecValue2, vaInData, pdvecIn);
          IVP_LAV2NX8U_XP(vecValue3, vaInData, pdvecIn, varLen);

          /* Perform unsigned to signed conversion and rounding off operation */
          vecValue1 = IVP_AVGRU2NX8(vecValue1, 0);
          vecValue2 = IVP_AVGRU2NX8(vecValue2, 0);
          vecValue3 = IVP_AVGRU2NX8(vecValue3, 0);

          /* Perform saturation of signed max value */
          vecValueSigned1 = IVP_MINU2NX8U(signedCharMax, vecValue1);
          vecValueSigned2 = IVP_MINU2NX8U(signedCharMax, vecValue2);
          vecValueSigned3 = IVP_MINU2NX8U(signedCharMax, vecValue3);

          /* Store */
          IVP_SA2NX8_IP(vecValueSigned1, vaOutData, pdvecOut);
          IVP_SA2NX8_IP(vecValueSigned2, vaOutData, pdvecOut);
          IVP_SAV2NX8_XP(vecValueSigned3, vaOutData, pdvecOut, varLen);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* initialize input and output data pointer */
        uint8_t *pInput1 = pInput + x + (z * inDataPitch2);
        int8_t *pOutput1 = pOutput + x + (z * outDataPitch2);
        int32_t varLen   = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++)
        {
          pdvecIn  = (xb_vec2Nx8U *) (pInput1 + (y * inDataPitch1));
          pdvecOut = (xb_vec2Nx8 *) (pOutput1 + (y * outDataPitch1));
          vaInData = IVP_LA2NX8U_PP(pdvecIn);

          /* Load Input Data */
          IVP_LA2NX8U_IP(vecValue1, vaInData, pdvecIn);
          IVP_LAV2NX8U_XP(vecValue2, vaInData, pdvecIn, varLen);

          /* Perform unsigned to signed conversion and rounding off operation */
          vecValue1 = IVP_AVGRU2NX8(vecValue1, 0);
          vecValue2 = IVP_AVGRU2NX8(vecValue2, 0);

          /* Perform saturation of signed max value */
          vecValueSigned1 = IVP_MINU2NX8U(signedCharMax, vecValue1);
          vecValueSigned2 = IVP_MINU2NX8U(signedCharMax, vecValue2);

          /* Store */
          IVP_SA2NX8_IP(vecValueSigned1, vaOutData, pdvecOut);
          IVP_SAV2NX8_XP(vecValueSigned2, vaOutData, pdvecOut, dim1Size - (x + vectorizationWidth));
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* initialize input and output data pointer */
        uint8_t *pInput1 = pInput + x + (z * inDataPitch2);
        int8_t *pOutput1 = pOutput + x + (z * outDataPitch2);
        int32_t varLen   = dim1Size - x;

        for (y = 0; y < dim2Size; y++)
        {
          pdvecIn  = (xb_vec2Nx8U *) (pInput1 + (y * inDataPitch1));
          pdvecOut = (xb_vec2Nx8 *) (pOutput1 + (y * outDataPitch1));
          vaInData = IVP_LA2NX8U_PP(pdvecIn);

          /* Load Input Data */
          IVP_LAV2NX8U_XP(vecValue1, vaInData, pdvecIn, varLen);

          /* Perform unsigned to signed conversion and rounding off operation */
          vecValue1 = IVP_AVGRU2NX8(vecValue1, 0);

          /* Perform saturation of signed max value */
          vecValueSigned1 = IVP_MINU2NX8U(signedCharMax, vecValue1);

          /* Store */
          IVP_SAV2NX8_XP(vecValueSigned1, vaOutData, pdvecOut, varLen);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_S8S16 ************************/
/* Description : P6 implementation for conversion from S8 to S16       */
/* Inputs      : Input Tile, scale, shift                              */
/* Outputs     : XI Error Code                                         */
/* InOuts      : Output Tile                                           */
/* Assumptions : InData is signed 8bit                                 */
/***********************************************************************/

XAI_ERR_TYPE xaiDataConversion3D_S8S16(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const uint16_t scale,
                                       const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_TILE3D_S16(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  int8_t *pInput   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t *pOutput = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx8 * restrict pvecIn;
  xb_vecNx16 * restrict pvecOut;

  /* input and output data vectors */
  xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
  xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from S8 bit to S16 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from S8 bit */
  /*      S16 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    xb_vecNx16 vecInData;
    xb_vecNx16 vecOut;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx8 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx16 *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8S_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied, so final result is
         * 32 way 16 bit.
         */
        vecOut = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData), shift);

        /* store output data */
        IVP_SANX16_IP(vecOut, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied, so final result is
       * 32 way 16 bit.
       */
      vecOut = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData), shift);

      /* store output data */
      IVP_SAVNX16_XP(vecOut, vaOut, pvecOut, (varlen << 1));
      IVP_SAPOSNX16_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1), shift);

          vecOut2 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2), shift);

          vecOut3 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData3), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut3, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1), shift);

          vecOut2 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut2, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);

          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut1, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);

          /* Store output data */
          IVP_SAVNX16_XP(vecOut0, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_U8S8 ***********************/
/* Description : P6 implementation for conversion from U8 to S8      */
/* Inputs      : Input Tile, scale, shift                            */
/* Outputs     : XI Error Code                                       */
/* InOuts      : Output Tile                                         */
/* Assumptions : InData is unsigned 8bit                             */
/*********************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_U8S8(const xai_pTile3D inTile,
                                      xai_pTile3D outTile,
                                      const uint16_t scale,
                                      const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_TILE3D_S8(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }
  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  uint8_t *pInput = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutput = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx8U * restrict pvecIn;
  xb_vecNx8 * restrict pvecOut;

  /* input and output data vectors */
  xb_vecNx16U vecInData0, vecInData1, vecInData2, vecInData3;
  xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

  /********************************************************************************/
  /* The overall design approach is split into 2 parts                            */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch   */
  /*    is equal to output tile pitch                                             */
  /*    - If above condition holds good, data elements for which data             */
  /*      conversion from U8 bit to S8 bit need to done is present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively        */
  /*                                                                              */
  /* 2. When input tile pitch is not equal to input tile size or input tile       */
  /*    pitch is not equal to output tile pitch                                   */
  /*    - In this scenario, data elements for which data conversion from U8 bit   */
  /*      S8 bit need to done exist in non-contiguous memory location.            */
  /*      In order to do vectorization across first dimension, output data        */
  /*      pointers need to be updated based on output tile size and output tile   */
  /*      pitch.                                                                  */
  /********************************************************************************/
  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/
    /*Input and output vectors*/
    xb_vecNx16U vecInData;
    xb_vecNx16 vecOut;

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx8U *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx8 *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8U_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied and data is truncated
         * in the 8 bit range 0 to SCHAR_MAX. So the final result
         * is 32-way, 8-bit.
         */
        vecOut = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData), shift);

        /* store output data */
        IVP_SANX8S_IP(vecOut, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied and data is truncated
       * in the 8 bit range 0 to SCHAR_MAX. So the final result
       * is 32-way, 8-bit.
       */
      vecOut = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData), shift);

      /* store output data */
      IVP_SAVNX8S_XP(vecOut, vaOut, pvecOut, varlen);
      IVP_SAPOSNX8S_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied and data is truncated
           * in the 8 bit range 0 to SCHAR_MAX. So the final result
           * is 32-way, 8-bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1), shift);

          vecOut2 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2), shift);

          vecOut3 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData3), shift);

          /* Store output data */
          IVP_SANX8S_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8S_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX8S_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX8S_XP(vecOut3, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8S_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied and data is truncated
           * in the 8 bit range 0 to SCHAR_MAX. So the final result
           * is 32-way, 8-bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1), shift);

          vecOut2 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2), shift);

          /* Store output data */
          IVP_SANX8S_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8S_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX8S_XP(vecOut2, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8S_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied and data is truncated
           * in the 8 bit range 0 to SCHAR_MAX. So the final result
           * is 32-way, 8-bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1), shift);

          /* Store output data */
          IVP_SANX8S_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX8S_XP(vecOut1, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8S_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to shift is applied and data is truncated
           * in the 8 bit range 0 to SCHAR_MAX. So the final result
           * is 32-way, 8-bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);

          /* Store output data */
          IVP_SAVNX8S_XP(vecOut0, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8S_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_U8S16 ***********************/
/* Description : P6 implementation for conversion from U8 to S16      */
/* Inputs      : Input Tile, scale, shift                             */
/* Outputs     : XI Error Code                                        */
/* InOuts      : Output Tile                                          */
/* Assumptions : InData is unsigned 8bit                              */
/**********************************************************************/

XAI_ERR_TYPE xaiDataConversion3D_U8S16(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const uint16_t scale,
                                       const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_TILE3D_S16(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  uint8_t *pInput  = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t *pOutput = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx8U * restrict pvecIn;
  xb_vecNx16 * restrict pvecOut;

  /* input and output data vectors */
  xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
  xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from U8 bit to S16 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from U8 bit */
  /*      S16 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    xb_vecNx16U vecInData;
    xb_vecNx16 vecOut;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx8U *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx16 *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8U_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied, so final result is
         * 32 way 16 bit.
         */
        vecOut = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData), shift);

        /* store output data */
        IVP_SANX16_IP(vecOut, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied, so final result is
       * 32 way 16 bit.
       */
      vecOut = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData), shift);

      /* store output data */
      IVP_SAVNX16_XP(vecOut, vaOut, pvecOut, (varlen << 1));
      IVP_SAPOSNX16_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);
          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1), shift);

          vecOut2 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2), shift);

          vecOut3 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData3), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut3, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8U_PP(pvecIn);
          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1), shift);

          vecOut2 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut2, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut1, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8U_PP(pvecIn);
          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);

          /* Store output data */
          IVP_SAVNX16_XP(vecOut0, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_U8U16 ***********************/
/* Description : P6 implementation for conversion from U8 to U16      */
/* Inputs      : Input Tile, scale, shift                             */
/* Outputs     : XI Error Code                                        */
/* InOuts      : Output Tile                                          */
/* Assumptions : InData is unsigned 8bit                              */
/**********************************************************************/

XAI_ERR_TYPE xaiDataConversion3D_U8U16(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const uint16_t scale,
                                       const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_TILE3D_U16(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  uint8_t *pInput   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  uint16_t *pOutput = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx8U * restrict pvecIn;
  xb_vecNx16U * restrict pvecOut;

  /* input and output data vectors */
  xb_vecNx16 vecInData0, vecInData1;

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from U8 bit to U16 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from U8 bit */
  /*      U16 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    xb_vecNx16U vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx8U *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx16U *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8U_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied, so final result is
         * 32 way 16 bit.
         */
        xb_vecN_2x32v hvecEven = IVP_PACKVRNX48_0(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData), shift);
        xb_vecN_2x32v hvecOdd  = IVP_PACKVRNX48_1(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData), shift);
        xb_vecNx16U vecOut     = IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecOdd, USHRT_MAX)), \
                                              IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecEven, USHRT_MAX)), IVP_SELI_INTERLEAVE_1_EVEN);

        /* store output data */
        IVP_SANX16U_IP(vecOut, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied, so final result is
       * 32 way 16 bit.
       */
      xb_vecN_2x32v hvecEven = IVP_PACKVRNX48_0(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData), shift);
      xb_vecN_2x32v hvecOdd  = IVP_PACKVRNX48_1(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData), shift);
      xb_vecNx16U vecOut     = IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecOdd, USHRT_MAX)), \
                                            IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecEven, USHRT_MAX)), IVP_SELI_INTERLEAVE_1_EVEN);

      /* store output data */
      IVP_SAVNX16U_XP(vecOut, vaOut, pvecOut, (varlen << 1));
      IVP_SAPOSNX16U_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;

      for (; x < (dim1Size - vectorizationWidth2X); x += vectorizationWidth2X)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        uint16_t *pOut = &pOutput[z * outTilePitch2 + x];

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16U *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecN_2x32v hvecEven0 = IVP_PACKVRNX48_0(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData0), shift);
          xb_vecN_2x32v hvecOdd0  = IVP_PACKVRNX48_1(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData0), shift);
          xb_vecNx16U vecOut0     = IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecOdd0, USHRT_MAX)), \
                                                 IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecEven0, USHRT_MAX)), IVP_SELI_INTERLEAVE_1_EVEN);

          xb_vecN_2x32v hvecEven1 = IVP_PACKVRNX48_0(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData1), shift);
          xb_vecN_2x32v hvecOdd1  = IVP_PACKVRNX48_1(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData1), shift);
          xb_vecNx16U vecOut1     = IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecOdd1, USHRT_MAX)), \
                                                 IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecEven1, USHRT_MAX)), IVP_SELI_INTERLEAVE_1_EVEN);

          /* Store output data */
          IVP_SANX16U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16U_IP(vecOut1, vaOut, pvecOut);
          IVP_SAPOSNX16U_FP(vaOut, pvecOut);
        }
      }
      if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        uint16_t *pOut = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LAVNX8U_XP(vecInData1, vaInData, pvecIn, varLen - vectorizationWidth);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecN_2x32v hvecEven0 = IVP_PACKVRNX48_0(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData0), shift);
          xb_vecN_2x32v hvecOdd0  = IVP_PACKVRNX48_1(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData0), shift);
          xb_vecNx16U vecOut0     = IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecOdd0, USHRT_MAX)), \
                                                 IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecEven0, USHRT_MAX)), IVP_SELI_INTERLEAVE_1_EVEN);

          xb_vecN_2x32v hvecEven1 = IVP_PACKVRNX48_0(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData1), shift);
          xb_vecN_2x32v hvecOdd1  = IVP_PACKVRNX48_1(IVP_MULUUNX16U((xb_vecNx16U) scale, vecInData1), shift);
          xb_vecNx16U vecOut1     = IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecOdd1, USHRT_MAX)), \
                                                 IVP_MOVNX16_FROMN_2X32U(IVP_MINN_2X32(hvecEven1, USHRT_MAX)), IVP_SELI_INTERLEAVE_1_EVEN);

          /* Store output data */
          IVP_SAVNX16U_XP(vecOut0, vaOut, pvecOut, (varLen << 1));
          IVP_SAVNX16U_XP(vecOut1, vaOut, pvecOut, ((varLen - vectorizationWidth) << 1));
          IVP_SAPOSNX16U_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_S8U8 ***********************/
/* Description : P6 implementation for conversion from S8 to U8      */
/* Inputs      : Input Tile, scale, shift                             */
/* Outputs     : XI Error Code                                        */
/* InOuts      : Output Tile                                          */
/* Assumptions : InData is signed 8bit                              */
/**********************************************************************/

XAI_ERR_TYPE xaiDataConversion3D_S8U8(const xai_pTile3D inTile,
                                      xai_pTile3D outTile,
                                      const uint16_t scale,
                                      const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_TILE3D_U8(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }
  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);
  const int16_t minLim        = 0;
  const int16_t maxLim        = UCHAR_MAX;

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  int8_t *pInput   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  uint8_t *pOutput = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx8 * restrict pvecIn;
  xb_vecNx8U * restrict pvecOut;

  /* input and output data vectors */
  xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
  xb_vecNx16U vecOut0, vecOut1, vecOut2, vecOut3;

  /********************************************************************************/
  /* The overall design approach is split into 2 parts                            */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch   */
  /*    is equal to output tile pitch                                             */
  /*    - If above condition holds good, data elements for which data             */
  /*      conversion from S8 bit to U8 bit need to done is present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively        */
  /*                                                                              */
  /* 2. When input tile pitch is not equal to input tile size or input tile       */
  /*    pitch is not equal to output tile pitch                                   */
  /*    - In this scenario, data elements for which data conversion from U8 bit   */
  /*      S8 bit need to done exist in non-contiguous memory location.            */
  /*      In order to do vectorization across first dimension, output data        */
  /*      pointers need to be updated based on output tile size and output tile   */
  /*      pitch.                                                                  */
  /********************************************************************************/
  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/
    /*Input and Output vectors*/
    xb_vecNx16 vecInData;
    xb_vecNx16 vecOut;

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx8 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx8U *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LANX8S_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied and data is truncated
         * in the 8 bit range 0 to UCHAR_MAX. So the final result
         * is 32-way, 8-bit.
         */
        vecOut = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData), shift);
        vecOut = IVP_MAXNX16(IVP_MINNX16(vecOut, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

        /* store output data */
        IVP_SANX8U_IP(vecOut, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied and data is truncated
       * in the 8 bit range 0 to UCHAR_MAX. So the final result
       * is 32-way, 8-bit.
       */
      vecOut = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData), shift);
      vecOut = IVP_MAXNX16(IVP_MINNX16(vecOut, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

      /* store output data */
      IVP_SAVNX8U_XP(vecOut, vaOut, pvecOut, varlen);
      IVP_SAPOSNX8U_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */
    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        uint8_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);

          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied and data is truncated
           * in the 8 bit range 0 to UCHAR_MAX. So the final result
           * is 32-way, 8-bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1), shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut2 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2), shift);
          vecOut2 = IVP_MAXNX16(IVP_MINNX16(vecOut2, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut3 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData3), shift);
          vecOut3 = IVP_MAXNX16(IVP_MINNX16(vecOut3, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut3, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        uint8_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);

          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied and data is truncated
           * in the 8 bit range 0 to UCHAR_MAX. So the final result
           * is 32-way, 8-bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1), shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut2 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2), shift);
          vecOut2 = IVP_MAXNX16(IVP_MINNX16(vecOut2, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut2, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        uint8_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);

          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied and data is truncated
           * in the 8 bit range 0 to UCHAR_MAX. So the final result
           * is 32-way, 8-bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          vecOut1 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1), shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut1, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        uint8_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);

          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to shift is applied and data is truncated
           * in the 8 bit range 0 to UCHAR_MAX. So the final result
           * is 32-way, 8-bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SAVNX8U_XP(vecOut0, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_S16 *****************************/
/* Description : P6 implementation for conversion  S16 to S16             */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, scale, shift                                 */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is signed 16bit                                   */
/**************************************************************************/

XAI_ERR_TYPE xaiDataConversion3D_S16(const xai_pTile3D inTile,
                                     xai_pTile3D outTile,
                                     const uint16_t scale,
                                     const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S16(inTile);
    XAI_CHECK_TILE3D_S16(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 32", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Get Data Pointers */
  int16_t *pInput  = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t *pOutput = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  valign vaOut     = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx16 * restrict pvecIn;
  xb_vecNx16 * restrict pvecOut;

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from signed 16 bit to S16 bit need to done present in      */
  /*      in contiguous memory location. Hence vectorization can be utilized    */
  /*      effectively                                                           */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from signed */
  /*      16 bit to S16 bit need to done exist in non-contiguous memory         */
  /*      location. In order to do vectorization across first dimension,        */
  /*      output data pointers need to be updated based on output tile size     */
  /*      and output tile pitch                                                 */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input data vectors */
    xb_vecNx16 vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx16 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx16 *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LANX16_PP(pvecIn);
      xb_vecNx16 vecOut;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* load input data */
        IVP_LANX16_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied, so final result is
         * 32 way 16 bit.
         */
        vecOut = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData), shift);

        /* store output data */
        IVP_SANX16_IP(vecOut, vaOut, pvecOut);
      }
      int32_t varLen = (maxLoopCount - x);
      /* load input data */
      IVP_LANX16_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied, so final result is
       * 32 way 16 bit.
       */
      vecOut = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData), shift);

      /* store output data */
      IVP_SAVNX16_XP(vecOut, vaOut, pvecOut, (varLen << 1));
      IVP_SAPOSNX16_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
          xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);
          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut1 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1), shift);
          vecOut2 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2), shift);
          vecOut3 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData3), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut3, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2;
          xb_vecNx16 vecOut0, vecOut1, vecOut2;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut1 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1), shift);
          vecOut2 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut2, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1;
          xb_vecNx16 vecOut0, vecOut1;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut1 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut1, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0;
          xb_vecNx16 vecOut0;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0), shift);

          /* Store output data */
          IVP_SAVNX16_XP(vecOut0, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_S16I32 *****************************/
/* Description : P6 implementation for conversion  S16 to I32             */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, scale, shift                                 */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is signed 16bit                                   */
/**************************************************************************/

XAI_ERR_TYPE xaiDataConversion3D_S16I32(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const uint16_t scale,
                                        const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S16(inTile);
    XAI_CHECK_TILE3D_I32(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 32", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Get Data Pointers */
  int16_t *pInput  = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int32_t *pOutput = (int32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  valign vaOut     = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  const int32_t minLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32)) ? INT_MIN : 0;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx16 * restrict pvecIn;
  xb_vecN_2x32v * restrict pvecOut;

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from signed 16 bit to I32 bit need to done present in      */
  /*      in contiguous memory location. Hence vectorization can be utilized    */
  /*      effectively                                                           */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from signed */
  /*      16 bit to I32 bit need to done exist in non-contiguous memory         */
  /*      location. In order to do vectorization across first dimension,        */
  /*      output data pointers need to be updated based on output tile size     */
  /*      and output tile pitch                                                 */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input data vectors */
    xb_vecNx16 vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx16 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecN_2x32v *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LANX16_PP(pvecIn);
      xb_vecN_2x32v vecOutL, vecOutH;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* load input data */
        IVP_LANX16_IP(vecInData, vaInData, pvecIn);

        xb_vecNx48 vecOutIntm1    = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData);
        xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
        vecOutL = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
        vecOutL = IVP_MAXN_2X32(vecOutL, (xb_vecN_2x32v) minLim);

        vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
        vecOutH     = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
        vecOutH     = IVP_MAXN_2X32(vecOutH, (xb_vecN_2x32v) minLim);
        /* store output data */
        IVP_SAN_2X32_IP(vecOutL, vaOut, pvecOut);
        IVP_SAN_2X32_IP(vecOutH, vaOut, pvecOut);
      }
      int32_t varLen = (maxLoopCount - x);
      /* load input data */
      IVP_LANX16_IP(vecInData, vaInData, pvecIn);

      xb_vecNx48 vecOutIntm1    = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData);
      xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
      vecOutL = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
      vecOutL = IVP_MAXN_2X32(vecOutL, (xb_vecN_2x32v) minLim);

      vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
      vecOutH     = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
      vecOutH     = IVP_MAXN_2X32(vecOutH, (xb_vecN_2x32v) minLim);

      /* store output data */
      IVP_SAVN_2X32_XP(vecOutL, vaOut, pvecOut, (varLen << 2));
      IVP_SAVN_2X32_XP(vecOutH, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
      IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
          xb_vecN_2x32v vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H, vecOut3L, vecOut3H;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);
          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData3, vaInData, pvecIn);


          xb_vecNx48 vecOutIntm1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecOutIntm2 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecOutIntm3 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2);
          xb_vecNx48 vecOutIntm4 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData3);

          xb_vecN_2x64w vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
          vecOut0L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut0L   = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
          vecOut0H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut0H   = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm2), IVP_CVT64SNX48LL(vecOutIntm2));
          vecOut1L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut1L   = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm2), IVP_CVT64SNX48HL(vecOutIntm2));
          vecOut1H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut1H   = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm3), IVP_CVT64SNX48LL(vecOutIntm3));
          vecOut2L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut2L   = IVP_MAXN_2X32(vecOut2L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm3), IVP_CVT64SNX48HL(vecOutIntm3));
          vecOut2H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut2H   = IVP_MAXN_2X32(vecOut2H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm4), IVP_CVT64SNX48LL(vecOutIntm4));
          vecOut3L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut3L   = IVP_MAXN_2X32(vecOut3L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm4), IVP_CVT64SNX48HL(vecOutIntm4));
          vecOut3H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut3H   = IVP_MAXN_2X32(vecOut3H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut2L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut2H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut3L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut3H, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2;
          xb_vecN_2x32v vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);

          xb_vecNx48 vecOutIntm1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecOutIntm2 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecOutIntm3 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2);

          xb_vecN_2x64w vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
          vecOut0L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut0L   = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
          vecOut0H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut0H   = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm2), IVP_CVT64SNX48LL(vecOutIntm2));
          vecOut1L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut1L   = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm2), IVP_CVT64SNX48HL(vecOutIntm2));
          vecOut1H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut1H   = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm3), IVP_CVT64SNX48LL(vecOutIntm3));
          vecOut2L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut2L   = IVP_MAXN_2X32(vecOut2L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm3), IVP_CVT64SNX48HL(vecOutIntm3));
          vecOut2H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut2H   = IVP_MAXN_2X32(vecOut2H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut2L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut2H, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1;
          xb_vecN_2x32v vecOut0L, vecOut0H, vecOut1L, vecOut1H;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);


          xb_vecNx48 vecOutIntm1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecOutIntm2 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1);

          xb_vecN_2x64w vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
          vecOut0L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut0L   = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
          vecOut0H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut0H   = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm2), IVP_CVT64SNX48LL(vecOutIntm2));
          vecOut1L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut1L   = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm2), IVP_CVT64SNX48HL(vecOutIntm2));
          vecOut1H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut1H   = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut1L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut1H, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0;
          xb_vecN_2x32v vecOut0L, vecOut0H;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);

          xb_vecNx48 vecOutIntm1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);

          xb_vecN_2x64w vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
          vecOut0L   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut0L   = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
          vecOut0H   = IVP_PACKVRN_2X64W(vecOutIntm, shift);
          vecOut0H   = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAVN_2X32_XP(vecOut0L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut0H, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_U16I32 *****************************/
/* Description : P6 implementation for conversion  U16 to I32             */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, scale, shift                                 */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is un-signed 16bit                                   */
/**************************************************************************/

XAI_ERR_TYPE xaiDataConversion3D_U16I32(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const uint16_t scale,
                                        const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U16(inTile);
    XAI_CHECK_TILE3D_I32(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 32", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Get Data Pointers */
  uint16_t *pInput = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int32_t *pOutput = (int32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  valign vaOut     = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;
  const uint32_t rndVal              = (1 << (shift - 1));
  const int32_t minLim               = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32)) ? 0 : INT_MIN;
  /******************************************************************************************************/
  /*usage of minLim																				                                              */
  /*U16 x U16 = U32 - result is in U32. We have two output variants S32 and U32				                  */
  /*For S32 output we need to clamp i.e.,(MIN(res,INT_MAX)) result using S32_MAX				                */
  /*For U32 output we need to clamp i.e., (MIN(res,UINT_MAX)) result using U32_MAX				              */
  /*PACK ISA available (IVP_PACKVRN_2X64W) will clamp the result to S32 range only				              */
  /*one option to implement this is to write two APIs with change only in clamping operation -	        */
  /*Note : we don't prefer using an if inside loop                                                      */
  /*To avoid above condition below code uses a hack - Final res is in S32 container so -		            */
  /* U32 to S32 can be done by MAX(0,res) and U32 to U32 can be done by MAX(INT_MIN,res)		            */
  /* MAX(0,res) will work because all values above S32_MAX will be interpretted as < 0 in S32 container	*/
  /******************************************************************************************************/

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx16U * restrict pvecIn;
  xb_vecN_2x32v * restrict pvecOut;

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from unsigned 16 bit to I32 bit need to done present in      */
  /*      in contiguous memory location. Hence vectorization can be utilized    */
  /*      effectively                                                           */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from unsigned */
  /*      16 bit to I32 bit need to done exist in non-contiguous memory         */
  /*      location. In order to do vectorization across first dimension,        */
  /*      output data pointers need to be updated based on output tile size     */
  /*      and output tile pitch                                                 */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input data vectors */
    xb_vecNx16U vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx16U *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecN_2x32v *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LANX16U_PP(pvecIn);
      xb_vecN_2x32v vecOutL, vecOutH;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* load input data */
        IVP_LANX16_IP(vecInData, vaInData, pvecIn);

        xb_vecNx48 vecOutIntm1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData);

        xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
        IVP_MULUUAN_2X16X32_0(vecOutIntm2, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
        vecOutL = IVP_PACKVRNRN_2X64W(vecOutIntm2, shift);
        vecOutL = IVP_MAXN_2X32(vecOutL, (xb_vecN_2x32v) minLim);

        vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
        IVP_MULUUAN_2X16X32_0(vecOutIntm2, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
        vecOutH = IVP_PACKVRNRN_2X64W(vecOutIntm2, shift);
        vecOutH = IVP_MAXN_2X32(vecOutH, (xb_vecN_2x32v) minLim);
        /* store output data */
        IVP_SAN_2X32_IP(vecOutL, vaOut, pvecOut);
        IVP_SAN_2X32_IP(vecOutH, vaOut, pvecOut);
      }
      int32_t varLen = (maxLoopCount - x);
      /* load input data */
      IVP_LANX16_IP(vecInData, vaInData, pvecIn);

      xb_vecNx48 vecOutIntm1    = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData);
      xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
      IVP_MULUUAN_2X16X32_0(vecOutIntm2, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
      vecOutL = IVP_PACKVRNRN_2X64W(vecOutIntm2, shift);
      vecOutL = IVP_MAXN_2X32(vecOutL, (xb_vecN_2x32v) minLim);


      vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
      IVP_MULUUAN_2X16X32_0(vecOutIntm2, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
      vecOutH = IVP_PACKVRNRN_2X64W(vecOutIntm2, shift);
      vecOutH = IVP_MAXN_2X32(vecOutH, (xb_vecN_2x32v) minLim);

      /* store output data */
      IVP_SAVN_2X32_XP(vecOutL, vaOut, pvecOut, (varLen << 2));
      IVP_SAVN_2X32_XP(vecOutH, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
      IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
          xb_vecN_2x32v vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H, vecOut3L, vecOut3H;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);
          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData3, vaInData, pvecIn);


          xb_vecNx48 vecOutIntm1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecOutIntm2 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecOutIntm3 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2);
          xb_vecNx48 vecOutIntm4 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData3);

          xb_vecN_2x64w vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut0L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut0L   = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut0H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut0H = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm2), IVP_CVT64SNX48LL(vecOutIntm2));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut1L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut1L   = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm2), IVP_CVT64SNX48HL(vecOutIntm2));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut1H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut1H = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm3), IVP_CVT64SNX48LL(vecOutIntm3));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut2L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut2L   = IVP_MAXN_2X32(vecOut2L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm3), IVP_CVT64SNX48HL(vecOutIntm3));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut2H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut2H = IVP_MAXN_2X32(vecOut2H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm4), IVP_CVT64SNX48LL(vecOutIntm4));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut3L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut3L   = IVP_MAXN_2X32(vecOut3L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm4), IVP_CVT64SNX48HL(vecOutIntm4));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut3H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut3H = IVP_MAXN_2X32(vecOut3H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut2L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut2H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut3L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut3H, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2;
          xb_vecN_2x32v vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);

          xb_vecNx48 vecOutIntm1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecOutIntm2 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecOutIntm3 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2);

          xb_vecN_2x64w vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut0L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut0L   = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut0H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut0H = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm2), IVP_CVT64SNX48LL(vecOutIntm2));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut1L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut1L   = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm2), IVP_CVT64SNX48HL(vecOutIntm2));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut1H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut1H = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm3), IVP_CVT64SNX48LL(vecOutIntm3));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut2L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut2L   = IVP_MAXN_2X32(vecOut2L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm3), IVP_CVT64SNX48HL(vecOutIntm3));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut2H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut2H = IVP_MAXN_2X32(vecOut2H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut2L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut2H, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1;
          xb_vecN_2x32v vecOut0L, vecOut0H, vecOut1L, vecOut1H;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);


          xb_vecNx48 vecOutIntm1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecOutIntm2 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1);

          xb_vecN_2x64w vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut0L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut0L   = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut0H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut0H = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm2), IVP_CVT64SNX48LL(vecOutIntm2));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut1L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut1L   = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm2), IVP_CVT64SNX48HL(vecOutIntm2));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut1H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut1H = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut1L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut1H, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0;
          xb_vecN_2x32v vecOut0L, vecOut0H;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);

          xb_vecNx48 vecOutIntm1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);

          xb_vecN_2x64w vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecOutIntm1), IVP_CVT64SNX48LL(vecOutIntm1));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut0L   = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut0L   = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOutIntm = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecOutIntm1), IVP_CVT64SNX48HL(vecOutIntm1));
          IVP_MULUUAN_2X16X32_0(vecOutIntm, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal);
          vecOut0H = IVP_PACKVRNRN_2X64W(vecOutIntm, shift);
          vecOut0H = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);

          /* Store output data */
          IVP_SAVN_2X32_XP(vecOut0L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut0H, vaOut, pvecOut, (varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_U16S16 **************************/
/* Description : P6 implementation for conversion  U16 to S16             */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, scale, shift                                 */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is unsigned 16bit                                 */
/**************************************************************************/

XAI_ERR_TYPE xaiDataConversion3D_U16S16(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const uint16_t scale,
                                        const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U16(inTile);
    XAI_CHECK_TILE3D_S16(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 32", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Get Data Pointers */
  uint16_t *pInput = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int16_t *pOutput = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  valign vaOut     = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx16U * restrict pvecIn;
  xb_vecNx16 * restrict pvecOut;

  /********************************************************************************/
  /* The overall design approach is split into 2 parts                            */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch   */
  /*    is equal to output tile pitch                                             */
  /*    - If above condition holds good, data elements for which data             */
  /*      conversion from unsigned 16 bit to S16 bit need to done present in      */
  /*      in contiguous memory location. Hence vectorization can be utilized      */
  /*      effectively                                                             */
  /*                                                                              */
  /* 2. When input tile pitch is not equal to input tile size or input tile       */
  /*    pitch is not equal to output tile pitch                                   */
  /*    - In this scenario, data elements for which data conversion from unsigned */
  /*      16 bit to S16 bit need to done exist in non-contiguous memory           */
  /*      location. In order to do vectorization across first dimension,          */
  /*      output data pointers need to be updated based on output tile size       */
  /*      and output tile pitch                                                   */
  /********************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input data vectors */
    xb_vecNx16U vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx16U *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx16 *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LANX16U_PP(pvecIn);
      xb_vecNx16 vecOut;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* load input data */
        IVP_LANX16U_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied, so final result is
         * 32 way 16 bit.
         */
        vecOut = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData), shift);

        /* store output data */
        IVP_SANX16_IP(vecOut, vaOut, pvecOut);
      }
      int32_t varLen = (maxLoopCount - x);
      /* load input data */
      IVP_LANX16U_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied, so final result is
       * 32 way 16 bit.
       */
      vecOut = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData), shift);

      /* store output data */
      IVP_SAVNX16_XP(vecOut, vaOut, pvecOut, (varLen << 1));
      IVP_SAPOSNX16_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
          xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);
          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut1 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1), shift);
          vecOut2 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2), shift);
          vecOut3 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData3), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut3, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16U vecInData0, vecInData1, vecInData2;
          xb_vecNx16 vecOut0, vecOut1, vecOut2;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);

          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut1 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1), shift);
          vecOut2 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut2, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16U vecInData0, vecInData1;
          xb_vecNx16 vecOut0, vecOut1;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);
          vecOut1 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1), shift);

          /* Store output data */
          IVP_SANX16_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX16_XP(vecOut1, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int16_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0;
          xb_vecNx16 vecOut0;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16 *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          vecOut0 = IVP_PACKVRNX48(IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0), shift);

          /* Store output data */
          IVP_SAVNX16_XP(vecOut0, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_S16U16 **************************/
/* Description : P6 implementation for conversion  S16 to U16             */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, scale, shift                                 */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is unsigned 16bit                                 */
/**************************************************************************/

XAI_ERR_TYPE xaiDataConversion3D_S16U16(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const uint16_t scale,
                                        const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S16(inTile);
    XAI_CHECK_TILE3D_U16(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 32", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Get Data Pointers */
  int16_t *pInput   = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  uint16_t *pOutput = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  valign vaOut      = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;
  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx16 * restrict pvecIn;
  xb_vecNx16U * restrict pvecOut;

  /********************************************************************************/
  /* The overall design approach is split into 2 parts                            */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch   */
  /*    is equal to output tile pitch                                             */
  /*    - If above condition holds good, data elements for which data             */
  /*      conversion from unsigned 16 bit to S16 bit need to done present in      */
  /*      in contiguous memory location. Hence vectorization can be utilized      */
  /*      effectively                                                             */
  /*                                                                              */
  /* 2. When input tile pitch is not equal to input tile size or input tile       */
  /*    pitch is not equal to output tile pitch                                   */
  /*    - In this scenario, data elements for which data conversion from unsigned */
  /*      16 bit to S16 bit need to done exist in non-contiguous memory           */
  /*      location. In order to do vectorization across first dimension,          */
  /*      output data pointers need to be updated based on output tile size       */
  /*      and output tile pitch                                                   */
  /********************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input data vectors */
    xb_vecNx16 vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx16 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx16U *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LANX16_PP(pvecIn);
      xb_vecNx16 vecOut;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* load input data */
        IVP_LANX16_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied, so final result is
         * 32 way 16 bit.
         */
        PACK_ROUND_U16(vecOut, vecInData, scale, shift);
        /* store output data */
        IVP_SANX16U_IP(vecOut, vaOut, pvecOut);
      }
      int32_t varLen = (maxLoopCount - x);
      /* load input data */
      IVP_LANX16_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied, so final result is
       * 32 way 16 bit.
       */
      PACK_ROUND_U16(vecOut, vecInData, scale, shift);
      /* store output data */
      IVP_SAVNX16U_XP(vecOut, vaOut, pvecOut, (varLen << 1));
      IVP_SAPOSNX16U_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        uint16_t *pOut = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
          xb_vecNx16U vecOut0, vecOut1, vecOut2, vecOut3;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);
          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */

          PACK_ROUND_U16(vecOut0, vecInData0, scale, shift);
          PACK_ROUND_U16(vecOut1, vecInData1, scale, shift);
          PACK_ROUND_U16(vecOut2, vecInData2, scale, shift);
          PACK_ROUND_U16(vecOut3, vecInData3, scale, shift);

          /* Store output data */
          IVP_SANX16U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16U_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX16U_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX16U_XP(vecOut3, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16U_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        uint16_t *pOut = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2;
          xb_vecNx16U vecOut0, vecOut1, vecOut2;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          PACK_ROUND_U16(vecOut0, vecInData0, scale, shift);
          PACK_ROUND_U16(vecOut1, vecInData1, scale, shift);
          PACK_ROUND_U16(vecOut2, vecInData2, scale, shift);

          /* Store output data */
          IVP_SANX16U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX16U_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX16U_XP(vecOut2, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16U_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        uint16_t *pOut = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1;
          xb_vecNx16U vecOut0, vecOut1;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          PACK_ROUND_U16(vecOut0, vecInData0, scale, shift);
          PACK_ROUND_U16(vecOut1, vecInData1, scale, shift);

          /* Store output data */
          IVP_SANX16U_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX16U_XP(vecOut1, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16U_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        uint16_t *pOut = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0;
          xb_vecNx16U vecOut0;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx16U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          PACK_ROUND_U16(vecOut0, vecInData0, scale, shift);
          /* Store output data */
          IVP_SAVNX16U_XP(vecOut0, vaOut, pvecOut, (varLen << 1));
          IVP_SAPOSNX16U_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_S8I64 *****************************/
/* Description : P6 implementation for conversion  S8 to I64             */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, scale, shift                                 */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is signed 8bit                                   */
/**************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_S8I64(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const uint16_t scale,
                                       const uint8_t shift)
{
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_TILE3D_I64(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  const int32_t dim1Size             = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size             = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size             = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1         = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2         = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1        = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2        = XAI_TILE3D_GET_DIM2_PITCH(outTile);
  valign vaOut                       = IVP_ZALIGN();
  int8_t *pInput                     = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int64_t *pOutput                   = (int64_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;
  const int32_t minLim               = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64)) ? INT_MIN : 0;
  //S16 x U16 = S32 , rounded and shifted back to S32.
  //even though the output type is S64. Data is within S32 range so INT_MIN is sufficient.
  int32_t x, y, z;
  xb_vecNx8 *restrict pvecIn;
  xb_vecN_2x64w *restrict pvecOut;


  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile dim1 and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from S8 bit to I64 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from S8 bit */
  /*      I64 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/
  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    xb_vecNx16 vecInData;
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      xb_vecN_2x32v vecOutTempL, vecOutTempH;
      xb_vecN_2x64w vecOutL, vecOutH;
      pvecIn  = (xb_vecNx8 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecN_2x64w *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8S_PP(pvecIn);
      int32_t varlen;
      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        IVP_LANX8S_IP(vecInData, vaInData, pvecIn);
        xb_vecNx48 vecIntRes      = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData);
        xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
        vecOutTempL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
        vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
        vecOutL     = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
        xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
        vecOutTempH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
        vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
        vecOutH     = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
        IVP_SAN_2X64W_IP(vecOutL, vaOut, pvecOut);
        IVP_SAN_2X64W_IP(vecOutH, vaOut, pvecOut);
      }

      varlen = (maxLoopCount - x);
      IVP_LANX8S_IP(vecInData, vaInData, pvecIn);
      xb_vecNx48 vecIntRes      = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData);
      xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
      vecOutTempL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
      vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
      vecOutL     = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
      xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
      vecOutTempH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
      vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
      vecOutH     = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
      IVP_SAVN_2X64W_XP(vecOutL, vaOut, pvecOut, (varlen << 3));
      IVP_SAVN_2X64W_XP(vecOutH, vaOut, pvecOut, ((varlen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
      IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
    }
  }
  else
  {
    xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
    xb_vecN_2x64w vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H, vecOut3L, vecOut3H;
    xb_vecN_2x32v vecOutTempL, vecOutTempH;
    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);
        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData3, vaInData, pvecIn);
          xb_vecNx48 vecIntRes0 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecIntRes2 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2);
          xb_vecNx48 vecIntRes3 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData3);
          vecOut0L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut2L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));
          vecOut2H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));
          vecOut3L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes3), IVP_CVT64SNX48LL(vecIntRes3));
          vecOut3H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes3), IVP_CVT64SNX48HL(vecIntRes3));
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut0H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut1L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut1H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut2L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut2L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut2H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut2H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut3L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut3L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut3H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut3H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut3L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut3H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);
        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);
          xb_vecNx48 vecIntRes0 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecIntRes2 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2);
          vecOut0L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut2L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));
          vecOut2H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut0H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut1L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut1H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut2L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut2L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut2H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut2H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut2L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut2H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);
        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          xb_vecNx48 vecIntRes0 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1);
          vecOut0L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut0H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut1L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut1H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut1L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut1H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;
        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          xb_vecNx48 vecIntRes0 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          vecOut0L    = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H    = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          vecOut0H    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);
          IVP_SAVN_2X64W_XP(vecOut0L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut0H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_U8I64 *****************************/
/* Description : P6 implementation for conversion  U8 to I64             */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, scale, shift                                 */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is unsigned 8bit                                   */
/**************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_U8I64(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const uint16_t scale,
                                       const uint8_t shift)
{
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_TILE3D_I64(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);
  valign vaOut                = IVP_ZALIGN();

  /* Get Data Pointers */
  uint8_t *pInput  = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int64_t *pOutput = (int64_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx8U *restrict pvecIn;
  xb_vecN_2x64w *restrict pvecOut;


  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile dim1 and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from U8 bit to I64 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from U8 bit */
  /*      I64 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/
  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    xb_vecNx16U vecInData;

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      xb_vecN_2x32v vecOutTempL, vecOutTempH;
      xb_vecN_2x64w vecOutL, vecOutH;
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx8U *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecN_2x64w *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8U_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

        xb_vecNx48 vecIntRes      = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData);
        xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
        vecOutTempL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
        //sign extending to 64bit
        vecOutL = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

        xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
        vecOutTempH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
        //sign extending to 64bit
        vecOutH = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

        IVP_SAN_2X64W_IP(vecOutL, vaOut, pvecOut);
        IVP_SAN_2X64W_IP(vecOutH, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

      xb_vecNx48 vecIntRes      = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData);
      xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
      vecOutTempL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
      //sign extending to 64bit
      vecOutL = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

      xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
      vecOutTempH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
      vecOutH     = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

      /* store output data */
      IVP_SAVN_2X64W_XP(vecOutL, vaOut, pvecOut, (varlen << 3));
      IVP_SAVN_2X64W_XP(vecOutH, vaOut, pvecOut, ((varlen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
      IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */
    xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
    xb_vecN_2x64w vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H, vecOut3L, vecOut3H;
    xb_vecN_2x32v vecOutTempL, vecOutTempH;

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);
          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData3, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecIntRes2 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2);
          xb_vecNx48 vecIntRes3 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData3);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut2L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));
          vecOut2H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));
          vecOut3L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes3), IVP_CVT64SNX48LL(vecIntRes3));
          vecOut3H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes3), IVP_CVT64SNX48HL(vecIntRes3));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          //sign extending to 64bit
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          //sign extending to 64bit
          vecOut1L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          //sign extending to 64bit
          vecOut1H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut2L, shift);
          //sign extending to 64bit
          vecOut2L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut2H, shift);
          //sign extending to 64bit
          vecOut2H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut3L, shift);
          //sign extending to 64bit
          vecOut3L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut3H, shift);
          //sign extending to 64bit
          vecOut3H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);


          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut3L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut3H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);
          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData2, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecIntRes2 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut2L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));
          vecOut2H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          //sign extending to 64bit
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          //sign extending to 64bit
          vecOut1L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          //sign extending to 64bit
          vecOut1H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut2L, shift);
          //sign extending to 64bit
          vecOut2L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut2H, shift);
          //sign extending to 64bit
          vecOut2H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);


          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut2L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut2H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);
          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));


          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          //sign extending to 64bit
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          //sign extending to 64bit
          vecOut1L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          //sign extending to 64bit
          vecOut1H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);


          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut1L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut1H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);
          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          //sign extending to 64bit
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          /* Store output data */
          IVP_SAVN_2X64W_XP(vecOut0L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut0H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_S16I64 *****************************/
/* Description : P6 implementation for conversion  S16 to I64             */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, scale, shift                                 */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is signed 16bit                                   */
/**************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_S16I64(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const uint16_t scale,
                                        const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S16(inTile);
    XAI_CHECK_TILE3D_I64(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  int16_t *pInput  = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int64_t *pOutput = (int64_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;
  const int32_t minLim               = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64)) ? INT_MIN : 0;
  //S16 x U16 = S32 , rounded and shifted back to S32.
  //even though the output type is S64. Data is within S32 range so INT_MIN is sufficient.
  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx16 *restrict pvecIn;
  xb_vecN_2x64w *restrict pvecOut;


  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile dim1 and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from S16 bit to I64 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from S16 bit */
  /*      I64 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/
  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    xb_vecNx16 vecInData;

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      xb_vecN_2x32v vecOutTempL, vecOutTempH;
      xb_vecN_2x64w vecOutL, vecOutH;
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx16 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecN_2x64w *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX16_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX16_IP(vecInData, vaInData, pvecIn);

        xb_vecNx48 vecIntRes      = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData);
        xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
        vecOutTempL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
        vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);

        //sign extending to 64bit
        vecOutL = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

        xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
        vecOutTempH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
        vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
        //sign extending to 64bit
        vecOutH = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

        IVP_SAN_2X64W_IP(vecOutL, vaOut, pvecOut);
        IVP_SAN_2X64W_IP(vecOutH, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX16_IP(vecInData, vaInData, pvecIn);

      xb_vecNx48 vecIntRes      = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData);
      xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
      vecOutTempL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
      vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
      //sign extending to 64bit
      vecOutL = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

      xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
      vecOutTempH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
      vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
      vecOutH     = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

      /* store output data */
      IVP_SAVN_2X64W_XP(vecOutL, vaOut, pvecOut, (varlen << 3));
      IVP_SAVN_2X64W_XP(vecOutH, vaOut, pvecOut, ((varlen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
      IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */
    xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
    xb_vecN_2x64w vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H, vecOut3L, vecOut3H;
    xb_vecN_2x32v vecOutTempL, vecOutTempH;

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX16_PP(pvecIn);
          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData3, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecIntRes2 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2);
          xb_vecNx48 vecIntRes3 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData3);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut2L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));
          vecOut2H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));
          vecOut3L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes3), IVP_CVT64SNX48LL(vecIntRes3));
          vecOut3H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes3), IVP_CVT64SNX48HL(vecIntRes3));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut2L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut2L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut2H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut2H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut3L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut3L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut3H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut3H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);


          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut3L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut3H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX16_PP(pvecIn);
          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecIntRes2 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData2);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut2L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));
          vecOut2H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut2L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut2L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut2H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut2H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);


          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut2L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut2H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX16_PP(pvecIn);
          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData1);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));


          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);


          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut1L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut1H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX16_PP(pvecIn);
          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUSNX16((xb_vecNx16U) scale, vecInData0);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0L    = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          /* Store output data */
          IVP_SAVN_2X64W_XP(vecOut0L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut0H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_U16I64 *****************************/
/* Description : P6 implementation for conversion  U16 to I64             */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, scale, shift                                 */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is unsigned 16bit                                   */
/**************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_U16I64(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const uint16_t scale,
                                        const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U16(inTile);
    XAI_CHECK_TILE3D_I64(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  uint16_t *pInput = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int64_t *pOutput = (int64_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;
  const uint32_t rndVal              = (1 << (shift - 1));

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx16U *restrict pvecIn;
  xb_vecN_2x64w *restrict pvecOut;


  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile dim1 and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from U16 bit to I64 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from U16 bit */
  /*      I64 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/
  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    xb_vecNx16U vecInData;

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      xb_vecN_2x32Uv vecOutTempL, vecOutTempH;
      xb_vecN_2x64w vecOutL, vecOutH;
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx16U *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecN_2x64w *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX16U_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX16U_IP(vecInData, vaInData, pvecIn);

        xb_vecNx48 vecIntRes      = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData);
        xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
        IVP_MULUUAN_2X16X32_0(vecOutIntm1, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
        vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOutIntm1, shift));
        //sign extending to 64bit
        vecOutL = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

        xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
        IVP_MULUUAN_2X16X32_0(vecOutIntm2, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
        vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOutIntm2, shift));
        //sign extending to 64bit
        vecOutH = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

        IVP_SAN_2X64W_IP(vecOutL, vaOut, pvecOut);
        IVP_SAN_2X64W_IP(vecOutH, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX16_IP(vecInData, vaInData, pvecIn);

      xb_vecNx48 vecIntRes      = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData);
      xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes), IVP_CVT64SNX48LL(vecIntRes));
      IVP_MULUUAN_2X16X32_0(vecOutIntm1, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
      vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOutIntm1, shift));
      //sign extending to 64bit
      vecOutL = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

      xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes), IVP_CVT64SNX48HL(vecIntRes));
      IVP_MULUUAN_2X16X32_0(vecOutIntm2, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
      vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOutIntm2, shift));
      //sign extending to 64bit
      vecOutH = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

      /* store output data */
      IVP_SAVN_2X64W_XP(vecOutL, vaOut, pvecOut, (varlen << 3));
      IVP_SAVN_2X64W_XP(vecOutH, vaOut, pvecOut, ((varlen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
      IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */
    xb_vecNx16U vecInData0, vecInData1, vecInData2, vecInData3;
    xb_vecN_2x64w vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H, vecOut3L, vecOut3H;
    xb_vecN_2x32Uv vecOutTempL, vecOutTempH;

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX16U_PP(pvecIn);
          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData3, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecIntRes2 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2);
          xb_vecNx48 vecIntRes3 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData3);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut2L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));
          vecOut2H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));
          vecOut3L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes3), IVP_CVT64SNX48LL(vecIntRes3));
          vecOut3H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes3), IVP_CVT64SNX48HL(vecIntRes3));

          IVP_MULUUAN_2X16X32_0(vecOut0L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut0L, shift));
          //sign extending to 64bit
          vecOut0L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut0H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut0H, shift));
          //sign extending to 64bit
          vecOut0H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          IVP_MULUUAN_2X16X32_0(vecOut1L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut1L, shift));
          //sign extending to 64bit
          vecOut1L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut1H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut1H, shift));
          //sign extending to 64bit
          vecOut1H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          IVP_MULUUAN_2X16X32_0(vecOut2L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut2L, shift));
          //sign extending to 64bit
          vecOut2L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut2H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut2H, shift));
          //sign extending to 64bit
          vecOut2H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          IVP_MULUUAN_2X16X32_0(vecOut3L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut3L, shift));
          //sign extending to 64bit
          vecOut3L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut3H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut3H, shift));
          //sign extending to 64bit
          vecOut3H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);


          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut3L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut3H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX16U_PP(pvecIn);
          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData2, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1);
          xb_vecNx48 vecIntRes2 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData2);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));
          vecOut2L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes2), IVP_CVT64SNX48LL(vecIntRes2));
          vecOut2H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes2), IVP_CVT64SNX48HL(vecIntRes2));

          IVP_MULUUAN_2X16X32_0(vecOut0L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut0L, shift));
          //sign extending to 64bit
          vecOut0L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut0H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut0H, shift));
          //sign extending to 64bit
          vecOut0H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          IVP_MULUUAN_2X16X32_0(vecOut1L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut1L, shift));
          //sign extending to 64bit
          vecOut1L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut1H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut1H, shift));
          //sign extending to 64bit
          vecOut1H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          IVP_MULUUAN_2X16X32_0(vecOut2L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut2L, shift));
          //sign extending to 64bit
          vecOut2L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut2H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut2H, shift));
          //sign extending to 64bit
          vecOut2H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut2L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut2H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX16U_PP(pvecIn);
          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData1, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);
          xb_vecNx48 vecIntRes1 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData1);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes1), IVP_CVT64SNX48LL(vecIntRes1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes1), IVP_CVT64SNX48HL(vecIntRes1));


          IVP_MULUUAN_2X16X32_0(vecOut0L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut0L, shift));
          //sign extending to 64bit
          vecOut0L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut0H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut0H, shift));
          //sign extending to 64bit
          vecOut0H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          IVP_MULUUAN_2X16X32_0(vecOut1L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut1L, shift));
          //sign extending to 64bit
          vecOut1L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut1H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut1H, shift));
          //sign extending to 64bit
          vecOut1H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);


          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut1L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut1H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX16U_PP(pvecIn);
          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);

          xb_vecNx48 vecIntRes0 = IVP_MULUUNX16((xb_vecNx16U) scale, vecInData0);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(vecIntRes0), IVP_CVT64SNX48LL(vecIntRes0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(vecIntRes0), IVP_CVT64SNX48HL(vecIntRes0));

          IVP_MULUUAN_2X16X32_0(vecOut0L, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempL = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut0L, shift));
          //sign extending to 64bit
          vecOut0L = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);
          IVP_MULUUAN_2X16X32_0(vecOut0H, (xb_vecNx16U) 1, (xb_vecN_2x32Uv) rndVal); //rounding
          vecOutTempH = xb_vecN_2x32v_rtor_xb_vecN_2x32Uv(IVP_PACKVRNRN_2X64W(vecOut0H, shift));
          //sign extending to 64bit
          vecOut0H = IVP_MULUUN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          /* Store output data */
          IVP_SAVN_2X64W_XP(vecOut0L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut0H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D ****************************************/
/* Description : General API for DataConversion3D optimized implementation       */
/*               Calls one of the DataConversion3D functions based               */
/*               on the parameters                                               */
/* Inputs      : Input Tile, scale, shift                                        */
/* Outputs     : XI Error Code                                                   */
/* InOuts      : Output Tile                                                     */
/*********************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D(const xai_pTile3D inTile,
                                 xai_pTile3D outTile,
                                 const uint16_t scale,
                                 const uint8_t shift)
{
  if ((!inTile) || (!outTile))
  {
    return(XAI_ERR_NULLARG);
  }
  if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U16))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16))
    {
      return(xaiDataConversion3D_U16S16(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32) || XAI_TILE3D_CHECK_TYPE(outTile, XAI_U32))
    {
      return(xaiDataConversion3D_U16I32(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64) || XAI_TILE3D_CHECK_TYPE(outTile, XAI_U64))
    {
      return(xaiDataConversion3D_U16I64(inTile, outTile, scale, shift));
    }
    else
    {
      return(xaiDataConversion3D_U16I8(inTile, outTile, scale, shift));
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16))
    {
      return(xaiDataConversion3D_S16(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16))
    {
      return(xaiDataConversion3D_S16U16(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32) || XAI_TILE3D_CHECK_TYPE(outTile, XAI_U32))
    {
      return(xaiDataConversion3D_S16I32(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64) || XAI_TILE3D_CHECK_TYPE(outTile, XAI_U64))
    {
      return(xaiDataConversion3D_S16I64(inTile, outTile, scale, shift));
    }
    else
    {
      return(xaiDataConversion3D_S16I8(inTile, outTile, scale, shift));
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U8))
    {
      return(xaiDataConversion3D_S8U8(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16))
    {
      return(xaiDataConversion3D_S8S16(inTile, outTile, scale, shift));
    }
    else if ((XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U32)))
    {
      return(xaiDataConversion3D_S8I32(inTile, outTile, scale, shift));
    }
    else if ((XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U64)))
    {
      return(xaiDataConversion3D_S8I64(inTile, outTile, scale, shift));
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))
    {
      return(xaiDataConversion3D_U8S8(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16))
    {
      return(xaiDataConversion3D_U8S16(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16))
    {
      return(xaiDataConversion3D_U8U16(inTile, outTile, scale, shift));
    }
    else if ((XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U32)))
    {
      return(xaiDataConversion3D_U8I32(inTile, outTile, scale, shift));
    }
    else if ((XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U64)))
    {
      return(xaiDataConversion3D_U8I64(inTile, outTile, scale, shift));
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S32))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))
    {
      return(xaiDataConversion3D_S32S8(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U8))
    {
      return(xaiDataConversion3D_S32U8(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16))
    {
      return(xaiDataConversion3D_S32S16(inTile, outTile, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16))
    {
      return(xaiDataConversion3D_S32U16(inTile, outTile, scale, shift));
    }
  }
  return(XAI_ERR_NO_VARIANT);
}

/********************* xaiDataConversion3D_AsymQ_U8S8 ********************/
/* Description : P6 implementation for conversion from U8_SYM to S8_ASYM */
/* Inputs      : Input Tile, zeroOut, scale, shift                       */
/* Outputs     : XI Error Code                                           */
/* InOuts      : Output Tile                                             */
/* Assumptions : InData is unsigned 8bit                                 */
/*************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_AsymQ_U8S8(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            const int16_t zeroOut,
                                            const uint16_t scale,
                                            const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_TILE3D_S8(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR((zeroOut >= -128) && (zeroOut < 128), XAI_ERR_NORM, \
                    "\nzeroOut = %hi, value must be greater than or equal to -128 and less than 128", zeroOut);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }
  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  uint8_t *pInput = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutput = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx8U * restrict pvecIn;
  xb_vecNx8 * restrict pvecOut;

  /* input and output data vectors */
  xb_vecNx16U vecInData0, vecInData1, vecInData2, vecInData3;
  xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

  int32_t zeroOutParam    = zeroOut;
  xb_vecNx48 zeroOutShift = IVP_CVT48SNX32((xb_vecN_2x32v) (zeroOutParam << shift), (xb_vecN_2x32v) (zeroOutParam << shift));
  xb_vecNx16U vecScale    = (xb_vecNx16U) (scale);

  /********************************************************************************/
  /* The overall design approach is split into 2 parts                            */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch   */
  /*    is equal to output tile pitch                                             */
  /*    - If above condition holds good, data elements for which data             */
  /*      conversion from U8 bit to S8 bit need to done is present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively        */
  /*                                                                              */
  /* 2. When input tile pitch is not equal to input tile size or input tile       */
  /*    pitch is not equal to output tile pitch                                   */
  /*    - In this scenario, data elements for which data conversion from U8 bit   */
  /*      S8 bit need to done exist in non-contiguous memory location.            */
  /*      In order to do vectorization across first dimension, output data        */
  /*      pointers need to be updated based on output tile size and output tile   */
  /*      pitch.                                                                  */
  /********************************************************************************/
  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/
    /*Input and output vectors*/
    xb_vecNx16U vecInData;
    xb_vecNx16 vecOut;

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx8U *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx8 *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8U_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

        /* add zeroOut, apply scale and shift to input data.
         * To 48bit shifted zeroOut values, add scaled input which is 32 way 48-bit data,
         * then shift is applied and data is truncated in the 8 bit range
         * SCHAR_MIN to SCHAR_MAX. So the final result is 32-way, 8-bit.
         */
        xb_vecNx48 acc = zeroOutShift;
        IVP_MULUUANX16(acc, vecInData, vecScale);
        vecOut = IVP_PACKVRNX48(acc, shift);

        /* store output data */
        IVP_SANX8S_IP(vecOut, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX8U_IP(vecInData, vaInData, pvecIn);

      /* add zeroOut, apply scale and shift to input data.
       * To 48bit shifted zeroOut values, add scaled input which is 32 way 48-bit data,
       * then shift is applied and data is truncated in the 8 bit range
       * SCHAR_MIN to SCHAR_MAX. So the final result is 32-way, 8-bit.
       */
      xb_vecNx48 acc = zeroOutShift;
      IVP_MULUUANX16(acc, vecInData, vecScale);
      vecOut = IVP_PACKVRNX48(acc, shift);

      /* store output data */
      IVP_SAVNX8S_XP(vecOut, vaOut, pvecOut, varlen);
      IVP_SAPOSNX8S_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData3, vaInData, pvecIn);

          /* add zeroOut, apply scale and shift to input data.
           * To 48bit shifted zeroOut values, add scaled input which is 32 way 48-bit data,
           * then shift is applied and data is truncated in the 8 bit range
           * SCHAR_MIN to SCHAR_MAX. So the final result is 32-way, 8-bit.
           */
          xb_vecNx48 acc0, acc1, acc2, acc3;
          acc0 = zeroOutShift;
          acc1 = zeroOutShift;
          acc2 = zeroOutShift;
          acc3 = zeroOutShift;

          IVP_MULUUANX16(acc0, vecInData0, vecScale);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);

          IVP_MULUUANX16(acc1, vecInData1, vecScale);
          vecOut1 = IVP_PACKVRNX48(acc1, shift);

          IVP_MULUUANX16(acc2, vecInData2, vecScale);
          vecOut2 = IVP_PACKVRNX48(acc2, shift);

          IVP_MULUUANX16(acc3, vecInData3, vecScale);
          vecOut3 = IVP_PACKVRNX48(acc3, shift);

          /* Store output data */
          IVP_SANX8S_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8S_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX8S_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX8S_XP(vecOut3, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8S_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData2, vaInData, pvecIn);

          /* add zeroOut, apply scale and shift to input data.
           * To 48bit shifted zeroOut values, add scaled input which is 32 way 48-bit data,
           * then shift is applied and data is truncated in the 8 bit range
           * SCHAR_MIN to SCHAR_MAX. So the final result is 32-way, 8-bit.
           */
          xb_vecNx48 acc0, acc1, acc2;
          acc0 = zeroOutShift;
          acc1 = zeroOutShift;
          acc2 = zeroOutShift;

          IVP_MULUUANX16(acc0, vecInData0, vecScale);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);

          IVP_MULUUANX16(acc1, vecInData1, vecScale);
          vecOut1 = IVP_PACKVRNX48(acc1, shift);

          IVP_MULUUANX16(acc2, vecInData2, vecScale);
          vecOut2 = IVP_PACKVRNX48(acc2, shift);

          /* Store output data */
          IVP_SANX8S_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8S_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX8S_XP(vecOut2, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8S_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8U_IP(vecInData1, vaInData, pvecIn);

          /* add zeroOut, apply scale and shift to input data.
           * To 48bit shifted zeroOut values, add scaled input which is 32 way 48-bit data,
           * then shift is applied and data is truncated in the 8 bit range
           * SCHAR_MIN to SCHAR_MAX. So the final result is 32-way, 8-bit.
           */

          xb_vecNx48 acc0, acc1;
          acc0 = zeroOutShift;
          acc1 = zeroOutShift;

          IVP_MULUUANX16(acc0, vecInData0, vecScale);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);

          IVP_MULUUANX16(acc1, vecInData1, vecScale);
          vecOut1 = IVP_PACKVRNX48(acc1, shift);

          /* Store output data */
          IVP_SANX8S_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX8S_XP(vecOut1, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8S_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        uint8_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8 *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8U_PP(pvecIn);

          /* load input data */
          IVP_LANX8U_IP(vecInData0, vaInData, pvecIn);

          /* add zeroOut, apply scale and shift to input data.
           * To 48bit shifted zeroOut values, add scaled input which is 32 way 48-bit data,
           * then shift is applied and data is truncated in the 8 bit range
           * SCHAR_MIN to SCHAR_MAX. So the final result is 32-way, 8-bit.
           */
          xb_vecNx48 acc0;
          acc0 = zeroOutShift;

          IVP_MULUUANX16(acc0, vecInData0, vecScale);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);

          /* Store output data */
          IVP_SAVNX8S_XP(vecOut0, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8S_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_AsymQ_S16S8 ********************/
/* Description : P6 implementation for conversion from S16_SYM to S8_ASYM */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, zeroOut, scale, shift                        */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is signed 16bit                                   */
/**************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S16S8(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const int16_t zeroOut,
                                             const uint16_t scale,
                                             const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S16(inTile);
    XAI_CHECK_TILE3D_S8(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 32", shift);
    XAI_CHECK_ERROR((zeroOut >= -128) && (zeroOut < 128), XAI_ERR_NORM, \
                    "\nzeroOut = %hi, value must be greater than or equal to -128 and less than 128", zeroOut);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const int16_t minLim = SCHAR_MIN;
  const int16_t maxLim = SCHAR_MAX;

  /* Get Data Pointers */
  int16_t *pInput = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutput = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  valign vaOut    = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx16 * restrict pvecIn;
  xb_vecNx8U * restrict pvecOut;

  int64_t zerOutShifted   = (int64_t) zeroOut << shift;
  xb_vecN_2x32v hvecZeroL = (xb_vecN_2x32v) ((int32_t) (zerOutShifted & 0xFFFFFFFF));
  xb_vecN_2x32v hvecZeroH = (xb_vecN_2x32v) ((int32_t) ((zerOutShifted >> 32) & 0xFFFFFFFF));
  xb_vec2Nx8 dvecZeroSh   = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32 \
                                                   (IVP_SELN_2X32I(hvecZeroH, hvecZeroL, IVP_SELI_32B_INTERLEAVE_1_LO)));
  xb_vecNx48 zeroOutShift = IVP_CVT48UN_2X64L(dvecZeroSh, dvecZeroSh);
  IVP_CVT48UN_2X64H(zeroOutShift, dvecZeroSh, dvecZeroSh);

  xb_vecNx16U vecScale = (xb_vecNx16U) (scale);

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from signed 16 bit to S8/U8 bit need to done present in    */
  /*      in contiguous memory location. Hence vectorization can be utilized    */
  /*      effectively                                                  */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from signed */
  /*      16 bit to S8/U8 bit need to done exist in non-contiguous memory       */
  /*      location. In order to do vectorization across first dimension, */
  /*      output data pointers need to be updated based on output tile size     */
  /*      and output tile pitch                                                 */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input data vectors */
    xb_vecNx16 vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx16 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx8U *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LANX16_PP(pvecIn);
      xb_vecNx16 vecOut;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* load input data */
        IVP_LANX16_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied, so final result is
         * 32 way 16 bit.
         */
        xb_vecNx48 acc = zeroOutShift;
        IVP_MULUSANX16(acc, vecScale, vecInData);
        vecOut = IVP_PACKVRNX48(acc, shift);
        vecOut = IVP_MAXNX16(IVP_MINNX16(vecOut, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

        /* store output data */
        IVP_SANX8U_IP(vecOut, vaOut, pvecOut);
      }
      /* load input data */
      IVP_LANX16_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied, so final result is
       * 32 way 16 bit.
       */
      xb_vecNx48 acc = zeroOutShift;
      IVP_MULUSANX16(acc, vecScale, vecInData);
      vecOut = IVP_PACKVRNX48(acc, shift);
      vecOut = IVP_MAXNX16(IVP_MINNX16(vecOut, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

      /* store output data */
      IVP_SAVNX8U_XP(vecOut, vaOut, pvecOut, (maxLoopCount - x));
      IVP_SAPOSNX8U_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
          xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);
          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0, acc1, acc2, acc3;
          acc0 = zeroOutShift;
          acc1 = zeroOutShift;
          acc2 = zeroOutShift;
          acc3 = zeroOutShift;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUSANX16(acc1, vecScale, vecInData1);
          vecOut1 = IVP_PACKVRNX48(acc1, shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUSANX16(acc2, vecScale, vecInData2);
          vecOut2 = IVP_PACKVRNX48(acc2, shift);
          vecOut2 = IVP_MAXNX16(IVP_MINNX16(vecOut2, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUSANX16(acc3, vecScale, vecInData3);
          vecOut3 = IVP_PACKVRNX48(acc3, shift);
          vecOut3 = IVP_MAXNX16(IVP_MINNX16(vecOut3, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut3, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1, vecInData2;
          xb_vecNx16 vecOut0, vecOut1, vecOut2;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0, acc1, acc2;
          acc0 = zeroOutShift;
          acc1 = zeroOutShift;
          acc2 = zeroOutShift;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUSANX16(acc1, vecScale, vecInData1);
          vecOut1 = IVP_PACKVRNX48(acc1, shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUSANX16(acc2, vecScale, vecInData2);
          vecOut2 = IVP_PACKVRNX48(acc2, shift);
          vecOut2 = IVP_MAXNX16(IVP_MINNX16(vecOut2, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut2, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0, vecInData1;
          xb_vecNx16 vecOut0, vecOut1;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0, acc1;
          acc0 = zeroOutShift;
          acc1 = zeroOutShift;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUSANX16(acc1, vecScale, vecInData1);
          vecOut1 = IVP_PACKVRNX48(acc1, shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut1, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int16_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16 vecInData0;
          xb_vecNx16 vecOut0;

          pvecIn  = (xb_vecNx16 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16_PP(pvecIn);

          /* load input data */
          IVP_LANX16_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0 = zeroOutShift;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SAVNX8U_XP(vecOut0, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************** xaiDataConversion3D_AsymQ_U16S8 *******************/
/* Description : P6 implementation for conversion from U16_SYM to S8_ASYM */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, zeroOut, scale, shift                        */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is unsigned 16bit                                 */
/**************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_AsymQ_U16S8(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const int16_t zeroOut,
                                             const uint16_t scale,
                                             const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U16(inTile);
    XAI_CHECK_TILE3D_S8(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 32", shift);
    XAI_CHECK_ERROR((zeroOut >= -128) && (zeroOut < 128), XAI_ERR_NORM, \
                    "\nzeroOut = %hi, value must be greater than or equal to -128 and less than 128", zeroOut);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const int16_t minLim = SCHAR_MIN;
  const int16_t maxLim = SCHAR_MAX;

  /* Get Data Pointers */
  uint16_t *pInput = (uint16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutput  = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  valign vaOut     = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx16U * restrict pvecIn;
  xb_vecNx8U * restrict pvecOut;

  int64_t zerOutShifted   = (int64_t) zeroOut << shift;
  xb_vecN_2x32v hvecZeroL = (xb_vecN_2x32v) ((int32_t) (zerOutShifted & 0xFFFFFFFF));
  xb_vecN_2x32v hvecZeroH = (xb_vecN_2x32v) ((int32_t) ((zerOutShifted >> 32) & 0xFFFFFFFF));
  xb_vec2Nx8 dvecZeroSh   = IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32 \
                                                   (IVP_SELN_2X32I(hvecZeroH, hvecZeroL, IVP_SELI_32B_INTERLEAVE_1_LO)));
  xb_vecNx48 zeroOutShift = IVP_CVT48UN_2X64L(dvecZeroSh, dvecZeroSh);
  IVP_CVT48UN_2X64H(zeroOutShift, dvecZeroSh, dvecZeroSh);

  xb_vecNx16U vecScale = (xb_vecNx16U) (scale);

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from signed 16 bit to S8/U8 bit need to done present in    */
  /*      in contiguous memory location. Hence vectorization can be utilized    */
  /*      effectively                                                  */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from signed */
  /*      16 bit to S8/U8 bit need to done exist in non-contiguous memory       */
  /*      location. In order to do vectorization across first dimension, */
  /*      output data pointers need to be updated based on output tile size     */
  /*      and output tile pitch                                                 */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input data vectors */
    xb_vecNx16U vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx16U *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx8U *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LANX16U_PP(pvecIn);
      xb_vecNx16 vecOut;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* load input data */
        IVP_LANX16U_IP(vecInData, vaInData, pvecIn);

        /* apply scale and shift to input data.
         * multiplying with scale results in 32 way 48-bit
         * data to which shift is applied, so final result is
         * 32 way 16 bit.
         */
        xb_vecNx48 acc = zeroOutShift;
        IVP_MULUUANX16(acc, vecScale, vecInData);
        vecOut = IVP_PACKVRNX48(acc, shift);
        vecOut = IVP_MAXNX16(IVP_MINNX16(vecOut, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

        /* store output data */
        IVP_SANX8U_IP(vecOut, vaOut, pvecOut);
      }
      /* load input data */
      IVP_LANX16U_IP(vecInData, vaInData, pvecIn);

      /* apply scale and shift to input data.
       * multiplying with scale results in 32 way 48-bit
       * data to which shift is applied, so final result is
       * 32 way 16 bit.
       */
      xb_vecNx48 acc = zeroOutShift;
      IVP_MULUUANX16(acc, vecScale, vecInData);
      vecOut = IVP_PACKVRNX48(acc, shift);
      vecOut = IVP_MAXNX16(IVP_MINNX16(vecOut, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

      /* store output data */
      IVP_SAVNX8U_XP(vecOut, vaOut, pvecOut, (maxLoopCount - x));
      IVP_SAPOSNX8U_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16U vecInData0, vecInData1, vecInData2, vecInData3;
          xb_vecNx16 vecOut0, vecOut1, vecOut2, vecOut3;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);
          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0, acc1, acc2, acc3;
          acc0 = zeroOutShift;
          acc1 = zeroOutShift;
          acc2 = zeroOutShift;
          acc3 = zeroOutShift;

          IVP_MULUUANX16(acc0, vecScale, vecInData0);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUUANX16(acc1, vecScale, vecInData1);
          vecOut1 = IVP_PACKVRNX48(acc1, shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUUANX16(acc2, vecScale, vecInData2);
          vecOut2 = IVP_PACKVRNX48(acc2, shift);
          vecOut2 = IVP_MAXNX16(IVP_MINNX16(vecOut2, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUUANX16(acc3, vecScale, vecInData3);
          vecOut3 = IVP_PACKVRNX48(acc3, shift);
          vecOut3 = IVP_MAXNX16(IVP_MINNX16(vecOut3, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut1, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut2, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut3, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16U vecInData0, vecInData1, vecInData2;
          xb_vecNx16 vecOut0, vecOut1, vecOut2;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);

          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX16U_IP(vecInData2, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0, acc1, acc2;
          acc0 = zeroOutShift;
          acc1 = zeroOutShift;
          acc2 = zeroOutShift;

          IVP_MULUUANX16(acc0, vecScale, vecInData0);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUUANX16(acc1, vecScale, vecInData1);
          vecOut1 = IVP_PACKVRNX48(acc1, shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUUANX16(acc2, vecScale, vecInData2);
          vecOut2 = IVP_PACKVRNX48(acc2, shift);
          vecOut2 = IVP_MAXNX16(IVP_MINNX16(vecOut2, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SANX8U_IP(vecOut1, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut2, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16U vecInData0, vecInData1;
          xb_vecNx16 vecOut0, vecOut1;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);

          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX16_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0, acc1;
          acc0 = zeroOutShift;
          acc1 = zeroOutShift;

          IVP_MULUUANX16(acc0, vecScale, vecInData0);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          IVP_MULUUANX16(acc1, vecScale, vecInData1);
          vecOut1 = IVP_PACKVRNX48(acc1, shift);
          vecOut1 = IVP_MAXNX16(IVP_MINNX16(vecOut1, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SANX8U_IP(vecOut0, vaOut, pvecOut);
          IVP_SAVNX8U_XP(vecOut1, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        uint16_t * pIn = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecNx16U vecInData0;
          xb_vecNx16 vecOut0;

          pvecIn  = (xb_vecNx16U *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8U *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX16U_PP(pvecIn);

          /* load input data */
          IVP_LANX16U_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0 = zeroOutShift;

          IVP_MULUUANX16(acc0, vecScale, vecInData0);
          vecOut0 = IVP_PACKVRNX48(acc0, shift);
          vecOut0 = IVP_MAXNX16(IVP_MINNX16(vecOut0, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

          /* Store output data */
          IVP_SAVNX8U_XP(vecOut0, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8U_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

// Temporary wrapper, to be removed later
XAI_ERR_TYPE xaiDataConversion3D_U16AS8(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const int16_t zeroOut,
                                        const uint16_t scale,
                                        const uint8_t shift)
{
  return(xaiDataConversion3D_AsymQ_U16S8(inTile, outTile, zeroOut, scale, shift));
}

/********************* xaiDataConversion3D_AsymQ_S32S8 ********************/
/* Description : P6 implementation for conversion from S32_SYM to S8_ASYM */
/*               depending on Output Tile type                            */
/* Inputs      : Input Tile, zeroOut, scale, shift                        */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is signed 32bit                                   */
/**************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S32S8(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const int16_t zeroOut,
                                             const uint16_t scale,
                                             const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S32(inTile);
    XAI_CHECK_TILE3D_S8(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 32, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 32", shift);
    XAI_CHECK_ERROR((zeroOut >= -128) && (zeroOut < 128), XAI_ERR_NORM, \
                    "\nzeroOut = %hi, value must be greater than or equal to -128 and less than 128", zeroOut);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const int16_t minLim = SCHAR_MIN;
  const int16_t maxLim = SCHAR_MAX;

  /* Get Data Pointers */
  int32_t *pInput = (int32_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutput = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  valign vaOut = IVP_ZALIGN();

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH / 2;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecN_2x32v * restrict pvecIn;
  xb_vecNx8 * restrict pvecOut;

  xb_vecN_2x64w vec0scaledIn64B, vec1scaledIn64B;

  /* SCALE*/
  xb_vecNx16U vecScale = (xb_vecNx16U) (scale);

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile width and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from signed 32 bit to S8/U8 bit need to done present in    */
  /*      in contiguous memory location. Hence vectorization can be utilized    */
  /*      effectively                                                  */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from signed */
  /*      32 bit to S8/U8 bit need to done exist in non-contiguous memory       */
  /*      location. In order to do vectorization across first dimension, */
  /*      output data pointers need to be updated based on output tile size     */
  /*      and output tile pitch                                                 */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input data vectors */
    xb_vecN_2x32v vecInData0, vecInData1;

    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecN_2x32v *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecNx8 *) (pOutput + (z * outTilePitch2));

      valign vaInData = IVP_LAN_2X32_PP(pvecIn);
      xb_vecNx16 vecOut, vecOut0, vecOut1;
      x = 0;
      for (; x < maxLoopCount - vectorizationWidth2X; x += vectorizationWidth2X)
      {
        /* Load input data */
        IVP_LAN_2X32_IP(vecInData0, vaInData, pvecIn);
        IVP_LAN_2X32_IP(vecInData1, vaInData, pvecIn);

        /* Initialize the 64-bit wide vector with (zeroOut << shift)*/
        vec0scaledIn64B = IVP_MULSUN_2X16X32_0(zeroOut, (1 << shift));
        vec1scaledIn64B = IVP_MULSUN_2X16X32_0(zeroOut, (1 << shift));

        /* Multiply U16 scale with S32 input and ACCUMULATE in 64-bit wide vector */
        IVP_MULUSAN_2X16X32_0(vec0scaledIn64B, vecScale, vecInData0);
        IVP_MULUSAN_2X16X32_0(vec1scaledIn64B, vecScale, vecInData1);

        /* Pack the 64-bit wide vector in 32 bit vecotr, by applying shift */
        xb_vecN_2x32v vec0scaledIn32B = IVP_PACKVRN_2X64W(vec0scaledIn64B, shift);
        xb_vecN_2x32v vec1scaledIn32B = IVP_PACKVRN_2X64W(vec1scaledIn64B, shift);

        /* CLAMP the 32bit scaled-shift data to minLim and maxLim, & store it
         * in 16-bit vector, whose odd lanes (1, 3, 5...) are zeroes.*/
        vecOut0 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec0scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));
        vecOut1 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec1scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));

        /* Select the actual data present at even lanes, i.e. 0, 2, 4,...  */
        vecOut = IVP_SELNX16I(vecOut1, vecOut0, IVP_SELI_EXTRACT_1_OF_2_OFF_0);

        /* Store output data */
        IVP_SANX8S_IP(vecOut, vaOut, pvecOut);
      }

      /* Load remaining input data */
      IVP_LAVN_2X32_XP(vecInData0, vaInData, pvecIn, (maxLoopCount - x) * 4);
      IVP_LAVN_2X32_XP(vecInData1, vaInData, pvecIn, ((maxLoopCount - x) - (vectorizationWidth >> 1)) * 4);

      /* Initialize the 64-bit wide vector with (zeroOut << shift)*/
      vec0scaledIn64B = IVP_MULSUN_2X16X32_0(zeroOut, (1 << shift));
      vec1scaledIn64B = IVP_MULSUN_2X16X32_0(zeroOut, (1 << shift));

      /* Multiply U16 scale with S32 input and ACCUMULATE in 64-bit wide vector */
      IVP_MULUSAN_2X16X32_0(vec0scaledIn64B, vecScale, vecInData0);
      IVP_MULUSAN_2X16X32_0(vec1scaledIn64B, vecScale, vecInData1);

      /* Pack the 64-bit wide vector in 32 bit vecotr, by applying shift */
      xb_vecN_2x32v vec0scaledIn32B = IVP_PACKVRN_2X64W(vec0scaledIn64B, shift);
      xb_vecN_2x32v vec1scaledIn32B = IVP_PACKVRN_2X64W(vec1scaledIn64B, shift);

      /* CLAMP the 32bit scaled-shift data to minLim and maxLim, & store it
       * in 16-bit vector, whose odd lanes (1, 3, 5...) are zeroes.*/
      vecOut0 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec0scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));
      vecOut1 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec1scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));

      /* Select the actual data present at even lanes, i.e. 0, 2, 4,...  */
      vecOut = IVP_SELNX16I(vecOut1, vecOut0, IVP_SELI_EXTRACT_1_OF_2_OFF_0);

      /* Store output data */
      IVP_SAVNX8S_XP(vecOut, vaOut, pvecOut, (maxLoopCount - x));
      IVP_SAPOSNX8S_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      for (; x < dim1Size; x += vectorizationWidth2X)
      {
        /* Initialize input and output data pointer */
        int32_t * pIn  = &pInput[z * inTilePitch2 + x];
        int8_t *pOut   = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          /* input and output data vectors */
          xb_vecN_2x32v vecInData0, vecInData1;
          xb_vecNx16 vecOut0, vecOut1, vecOut;

          pvecIn  = (xb_vecN_2x32v *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecNx8 *) (pOut + (y * outTilePitch1));

          /* Load input data */
          valign vaInData = IVP_LAN_2X32_PP(pvecIn);
          IVP_LAVN_2X32_XP(vecInData0, vaInData, pvecIn, varLen * 4);
          IVP_LAVN_2X32_XP(vecInData1, vaInData, pvecIn, (varLen - (vectorizationWidth >> 1)) * 4);

          /* Initialize the 64-bit wide vector with (zeroOut << shift)*/
          vec0scaledIn64B = IVP_MULSUN_2X16X32_0(zeroOut, (1 << shift));
          vec1scaledIn64B = IVP_MULSUN_2X16X32_0(zeroOut, (1 << shift));

          /* Multiply U16 scale with S32 input and ACCUMULATE in 64-bit wide vector */
          IVP_MULUSAN_2X16X32_0(vec0scaledIn64B, vecScale, vecInData0);
          IVP_MULUSAN_2X16X32_0(vec1scaledIn64B, vecScale, vecInData1);

          /* Pack the 64-bit wide vector in 32 bit vecotr, by applying shift */
          xb_vecN_2x32v vec0scaledIn32B = IVP_PACKVRN_2X64W(vec0scaledIn64B, shift);
          xb_vecN_2x32v vec1scaledIn32B = IVP_PACKVRN_2X64W(vec1scaledIn64B, shift);

          /* CLAMP the 32bit scaled-shift data to minLim and maxLim, & store it
           * in 16-bit vector, whose odd lanes (1, 3, 5...) are zeroes.*/
          vecOut0 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec0scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));
          vecOut1 = IVP_MOVNX16_FROMN_2X32(IVP_MAXN_2X32(IVP_MINN_2X32(vec1scaledIn32B, (xb_vecN_2x32v) maxLim), (xb_vecN_2x32v) minLim));

          /* Select the actual data present at even lanes, i.e. 0, 2, 4,...  */
          vecOut = IVP_SELNX16I(vecOut1, vecOut0, IVP_SELI_EXTRACT_1_OF_2_OFF_0);

          /* Store output data */
          IVP_SAVNX8S_XP(vecOut, vaOut, pvecOut, varLen);
          IVP_SAPOSNX8S_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_AsymQ_S8I32 ********************/
/* Description : P6 implementation for conversion from S8_ASYM to I32_SYM */
/* Inputs      : Input Tile, zeroIn, scale, shift                         */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is signed 8bit                                    */
/**************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S8I32(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const int16_t zeroIn,
                                             const uint16_t scale,
                                             const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_TILE3D_I32(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR((zeroIn >= -128) && (zeroIn < 128), XAI_ERR_NORM, \
                    "\nzeroIn = %hi, value must be greater than or equal to -128 and less than 128", zeroIn);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  int8_t *pInput   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int32_t *pOutput = (int32_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  const int32_t minLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32)) ? INT_MIN : 0;

  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx8 * restrict pvecIn;
  xb_vecN_2x32v * restrict pvecOut;

  /* input and output data vectors */
  xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
  xb_vecN_2x32v vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H, vecOut3L, vecOut3H;

  xb_vecNx16U vecScale   = (xb_vecNx16U) (scale);
  xb_vecNx16 vecZeroIn   = (xb_vecNx16) (-zeroIn);
  xb_vecNx48 zeroInScale = IVP_MULUSNX16(vecScale, vecZeroIn);

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile dim1 and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from S8 bit to S16 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from S8 bit */
  /*      S16 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    xb_vecNx16 vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx8 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecN_2x32v *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8S_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

        xb_vecNx48 acc = zeroInScale;
        IVP_MULUSANX16(acc, vecScale, vecInData);
        //vecOut = IVP_PACKVRNX48(acc, shift);
        xb_vecN_2x32v vecIntResL = IVP_CVT32SNX48L(acc);
        xb_vecN_2x32v vecIntResH = IVP_CVT32SNX48H(acc);
        vecIntResL = IVP_ADDN_2X32(vecIntResL, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
        vecIntResH = IVP_ADDN_2X32(vecIntResH, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
        vecIntResL = IVP_SRAN_2X32(vecIntResL, (xb_vecN_2x32v) (shift));
        vecIntResH = IVP_SRAN_2X32(vecIntResH, (xb_vecN_2x32v) (shift));
        vecOut0L   = IVP_MAXN_2X32(vecIntResL, (xb_vecN_2x32v) minLim);
        vecOut0H   = IVP_MAXN_2X32(vecIntResH, (xb_vecN_2x32v) minLim);


        /* store output data */
        IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
        IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

      xb_vecNx48 acc = zeroInScale;
      IVP_MULUSANX16(acc, vecScale, vecInData);
      xb_vecN_2x32v vecIntResL = IVP_CVT32SNX48L(acc);
      xb_vecN_2x32v vecIntResH = IVP_CVT32SNX48H(acc);

      vecIntResL = IVP_ADDN_2X32(vecIntResL, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
      vecIntResH = IVP_ADDN_2X32(vecIntResH, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
      vecIntResL = IVP_SRAN_2X32(vecIntResL, (xb_vecN_2x32v) (shift));
      vecIntResH = IVP_SRAN_2X32(vecIntResH, (xb_vecN_2x32v) (shift));
      vecOut0L   = IVP_MAXN_2X32(vecIntResL, (xb_vecN_2x32v) minLim);
      vecOut0H   = IVP_MAXN_2X32(vecIntResH, (xb_vecN_2x32v) minLim);

      /* store output data */
      IVP_SAVN_2X32_XP(vecOut0L, vaOut, pvecOut, (varlen << 2));
      IVP_SAVN_2X32_XP(vecOut0H, vaOut, pvecOut, ((varlen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
      IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0, acc1, acc2, acc3;
          acc0 = zeroInScale;
          acc1 = zeroInScale;
          acc2 = zeroInScale;
          acc3 = zeroInScale;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          IVP_MULUSANX16(acc1, vecScale, vecInData1);
          IVP_MULUSANX16(acc2, vecScale, vecInData2);
          IVP_MULUSANX16(acc3, vecScale, vecInData3);

          xb_vecN_2x32v vecIntRes0L = IVP_CVT32SNX48L(acc0);
          xb_vecN_2x32v vecIntRes0H = IVP_CVT32SNX48H(acc0);
          xb_vecN_2x32v vecIntRes1L = IVP_CVT32SNX48L(acc1);
          xb_vecN_2x32v vecIntRes1H = IVP_CVT32SNX48H(acc1);
          xb_vecN_2x32v vecIntRes2L = IVP_CVT32SNX48L(acc2);
          xb_vecN_2x32v vecIntRes2H = IVP_CVT32SNX48H(acc2);
          xb_vecN_2x32v vecIntRes3L = IVP_CVT32SNX48L(acc3);
          xb_vecN_2x32v vecIntRes3H = IVP_CVT32SNX48H(acc3);

          vecIntRes0L = IVP_ADDN_2X32(vecIntRes0L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes0H = IVP_ADDN_2X32(vecIntRes0H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut0L    = IVP_SRAN_2X32(vecIntRes0L, (xb_vecN_2x32v) (shift));
          vecOut0H    = IVP_SRAN_2X32(vecIntRes0H, (xb_vecN_2x32v) (shift));
          vecOut0L    = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOut0H    = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);


          vecIntRes1L = IVP_ADDN_2X32(vecIntRes1L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes1H = IVP_ADDN_2X32(vecIntRes1H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut1L    = IVP_SRAN_2X32(vecIntRes1L, (xb_vecN_2x32v) (shift));
          vecOut1H    = IVP_SRAN_2X32(vecIntRes1H, (xb_vecN_2x32v) (shift));
          vecOut1L    = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOut1H    = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);


          vecIntRes2L = IVP_ADDN_2X32(vecIntRes2L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes2H = IVP_ADDN_2X32(vecIntRes2H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut2L    = IVP_SRAN_2X32(vecIntRes2L, (xb_vecN_2x32v) (shift));
          vecOut2H    = IVP_SRAN_2X32(vecIntRes2H, (xb_vecN_2x32v) (shift));
          vecOut2L    = IVP_MAXN_2X32(vecOut2L, (xb_vecN_2x32v) minLim);
          vecOut2H    = IVP_MAXN_2X32(vecOut2H, (xb_vecN_2x32v) minLim);


          vecIntRes3L = IVP_ADDN_2X32(vecIntRes3L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes3H = IVP_ADDN_2X32(vecIntRes3H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut3L    = IVP_SRAN_2X32(vecIntRes3L, (xb_vecN_2x32v) (shift));
          vecOut3H    = IVP_SRAN_2X32(vecIntRes3H, (xb_vecN_2x32v) (shift));
          vecOut3L    = IVP_MAXN_2X32(vecOut3L, (xb_vecN_2x32v) minLim);
          vecOut3H    = IVP_MAXN_2X32(vecOut3H, (xb_vecN_2x32v) minLim);


          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut2L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut2H, vaOut, pvecOut);

          IVP_SAVN_2X32_XP(vecOut3L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut3H, vaOut, pvecOut, ((varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);

          xb_vecNx48 acc0, acc1, acc2;
          acc0 = zeroInScale;
          acc1 = zeroInScale;
          acc2 = zeroInScale;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          IVP_MULUSANX16(acc1, vecScale, vecInData1);
          IVP_MULUSANX16(acc2, vecScale, vecInData2);

          xb_vecN_2x32v vecIntRes0L = IVP_CVT32SNX48L(acc0);
          xb_vecN_2x32v vecIntRes0H = IVP_CVT32SNX48H(acc0);
          xb_vecN_2x32v vecIntRes1L = IVP_CVT32SNX48L(acc1);
          xb_vecN_2x32v vecIntRes1H = IVP_CVT32SNX48H(acc1);
          xb_vecN_2x32v vecIntRes2L = IVP_CVT32SNX48L(acc2);
          xb_vecN_2x32v vecIntRes2H = IVP_CVT32SNX48H(acc2);


          vecIntRes0L = IVP_ADDN_2X32(vecIntRes0L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes0H = IVP_ADDN_2X32(vecIntRes0H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut0L    = IVP_SRAN_2X32(vecIntRes0L, (xb_vecN_2x32v) (shift));
          vecOut0H    = IVP_SRAN_2X32(vecIntRes0H, (xb_vecN_2x32v) (shift));
          vecOut0L    = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOut0H    = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);


          vecIntRes1L = IVP_ADDN_2X32(vecIntRes1L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes1H = IVP_ADDN_2X32(vecIntRes1H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut1L    = IVP_SRAN_2X32(vecIntRes1L, (xb_vecN_2x32v) (shift));
          vecOut1H    = IVP_SRAN_2X32(vecIntRes1H, (xb_vecN_2x32v) (shift));
          vecOut1L    = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOut1H    = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);


          vecIntRes2L = IVP_ADDN_2X32(vecIntRes2L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes2H = IVP_ADDN_2X32(vecIntRes2H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut2L    = IVP_SRAN_2X32(vecIntRes2L, (xb_vecN_2x32v) (shift));
          vecOut2H    = IVP_SRAN_2X32(vecIntRes2H, (xb_vecN_2x32v) (shift));
          vecOut2L    = IVP_MAXN_2X32(vecOut2L, (xb_vecN_2x32v) minLim);
          vecOut2H    = IVP_MAXN_2X32(vecOut2H, (xb_vecN_2x32v) minLim);



          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut2L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut2H, vaOut, pvecOut, ((varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);

          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0, acc1;
          acc0 = zeroInScale;
          acc1 = zeroInScale;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          IVP_MULUSANX16(acc1, vecScale, vecInData1);

          xb_vecN_2x32v vecIntRes0L = IVP_CVT32SNX48L(acc0);
          xb_vecN_2x32v vecIntRes0H = IVP_CVT32SNX48H(acc0);
          xb_vecN_2x32v vecIntRes1L = IVP_CVT32SNX48L(acc1);
          xb_vecN_2x32v vecIntRes1H = IVP_CVT32SNX48H(acc1);



          vecIntRes0L = IVP_ADDN_2X32(vecIntRes0L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes0H = IVP_ADDN_2X32(vecIntRes0H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut0L    = IVP_SRAN_2X32(vecIntRes0L, (xb_vecN_2x32v) (shift));
          vecOut0H    = IVP_SRAN_2X32(vecIntRes0H, (xb_vecN_2x32v) (shift));
          vecOut0L    = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOut0H    = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);


          vecIntRes1L = IVP_ADDN_2X32(vecIntRes1L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes1H = IVP_ADDN_2X32(vecIntRes1H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut1L    = IVP_SRAN_2X32(vecIntRes1L, (xb_vecN_2x32v) (shift));
          vecOut1H    = IVP_SRAN_2X32(vecIntRes1H, (xb_vecN_2x32v) (shift));
          vecOut1L    = IVP_MAXN_2X32(vecOut1L, (xb_vecN_2x32v) minLim);
          vecOut1H    = IVP_MAXN_2X32(vecOut1H, (xb_vecN_2x32v) minLim);



          /* Store output data */
          IVP_SAN_2X32_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X32_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAVN_2X32_XP(vecOut1L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut1H, vaOut, pvecOut, ((varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int32_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x32v *) (pOut + (y * outTilePitch1));
          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0 = zeroInScale;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);

          xb_vecN_2x32v vecIntRes0L = IVP_CVT32SNX48L(acc0);
          xb_vecN_2x32v vecIntRes0H = IVP_CVT32SNX48H(acc0);

          vecIntRes0L = IVP_ADDN_2X32(vecIntRes0L, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecIntRes0H = IVP_ADDN_2X32(vecIntRes0H, IVP_SLAN_2X32((xb_vecN_2x32v) 1, (xb_vecN_2x32v) (shift - 1)));
          vecOut0L    = IVP_SRAN_2X32(vecIntRes0L, (xb_vecN_2x32v) (shift));
          vecOut0H    = IVP_SRAN_2X32(vecIntRes0H, (xb_vecN_2x32v) (shift));
          vecOut0L    = IVP_MAXN_2X32(vecOut0L, (xb_vecN_2x32v) minLim);
          vecOut0H    = IVP_MAXN_2X32(vecOut0H, (xb_vecN_2x32v) minLim);


          /* Store output data */
          IVP_SAVN_2X32_XP(vecOut0L, vaOut, pvecOut, (varLen << 2));
          IVP_SAVN_2X32_XP(vecOut0H, vaOut, pvecOut, ((varLen << 2) - (XCHAL_IVPN_SIMD_WIDTH << 1)));
          IVP_SAPOSN_2X32_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_AsymQ_S8I64 ********************/
/* Description : Q8 implementation for conversion from S8_ASYM to I64_SYM */
/* Inputs      : Input Tile, zeroIn, scale, shift                         */
/* Outputs     : XI Error Code                                            */
/* InOuts      : Output Tile                                              */
/* Assumptions : InData is signed 8bit                                    */
/**************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S8I64(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const int16_t zeroIn,
                                             const uint16_t scale,
                                             const uint8_t shift)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_TILE3D_I64(outTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE3D_SIZE_EQ(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_ERROR(shift < 24, XAI_ERR_NORM, \
                    "Shift = %hhu, value should be less than 24", shift);
    XAI_CHECK_ERROR((zeroIn >= -128) && (zeroIn < 128), XAI_ERR_NORM, \
                    "\nzeroIn = %hi, value must be greater than or equal to -128 and less than 128", zeroIn);
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile), XAI_ERR_BADARG,                  \
                    "\nData Order of InputTile = %d, OutputTile = %d\nData Order of InputTile and OutputTile should be same", \
                    XAI_TILE3D_GET_DATA_ORDER(inTile), XAI_TILE3D_GET_DATA_ORDER(outTile));
  }

  /* Get Tile Parameters */
  const int32_t dim1Size      = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t dim2Size      = XAI_TILE3D_GET_DIM2(inTile);
  const int32_t dim3Size      = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t inTilePitch1  = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inTilePitch2  = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outTilePitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outTilePitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  valign vaOut = IVP_ZALIGN();

  /* Get Data Pointers */
  int8_t *pInput   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int64_t *pOutput = (int64_t *) XAI_TILE3D_GET_DATA_PTR(outTile);

  /* vectorization width */
  const int32_t vectorizationWidth   = XCHAL_IVPN_SIMD_WIDTH;
  const int32_t vectorizationWidth2X = vectorizationWidth * 2;
  const int32_t vectorizationWidth3X = vectorizationWidth * 3;
  const int32_t vectorizationWidth4X = vectorizationWidth * 4;

  const int32_t minLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64)) ? INT_MIN : 0;
  //S16 x U16 = S32 , rounded and shifted back to S32.
  //even though the output type is S64. Data is within S32 range so INT_MIN is sufficient.
  /* loop variables */
  int32_t x, y, z;

  /* input and output pointers */
  xb_vecNx8 * restrict pvecIn;
  xb_vecN_2x64w * restrict pvecOut;

  /* input and output data vectors */
  xb_vecNx16 vecInData0, vecInData1, vecInData2, vecInData3;
  xb_vecN_2x64w vecOut0L, vecOut0H, vecOut1L, vecOut1H, vecOut2L, vecOut2H, vecOut3L, vecOut3H;

  xb_vecNx16U vecScale   = (xb_vecNx16U) (scale);
  xb_vecNx16 vecZeroIn   = (xb_vecNx16) (-zeroIn);
  xb_vecNx48 zeroInScale = IVP_MULUSNX16(vecScale, vecZeroIn);

  /******************************************************************************/
  /* The overall design approach is split into 2 parts                          */
  /* 1. When input tile pitch is equal to input tile dim1 and input tile pitch */
  /*    is equal to output tile pitch                                           */
  /*    - If above condition holds good, data elements for which data           */
  /*      conversion from S8 bit to S64 bit need to done present in contiguous  */
  /*      memory location. Hence vectorization can be utilized effectively      */
  /*                                                                            */
  /* 2. When input tile pitch is not equal to input tile size or input tile     */
  /*    pitch is not equal to output tile pitch                                 */
  /*    - In this scenario, data elements for which data conversion from S8 bit */
  /*      S64 bit need to done exist in non-contiguous memory location.         */
  /*      In order to do vectorization across first dimension, output data      */
  /*      pointers need to be updated based on output tile size and output tile */
  /*      pitch.                                                                */
  /******************************************************************************/

  if ((inTilePitch1 == dim1Size) && (outTilePitch1 == dim1Size))
  {
    /******************************************************************************/
    /* Data exist in contiguous memory location with respect to first dimension   */
    /******************************************************************************/

    /* input and output vectors */
    xb_vecNx16 vecInData;
    /* Initialize max loop counter */
    int32_t dim3MaxLoopCount = dim3Size;
    int32_t maxLoopCount     = dim1Size * dim2Size;

    /* Updated Loop count based on tile dimension configuration */
    if ((inTilePitch2 == maxLoopCount) && (outTilePitch2 == maxLoopCount))
    {
      /**********************************************************************/
      /* Data exist in contiguous memory location with respect to first and */
      /* second dimension                                                   */
      /**********************************************************************/

      /* Update max loop counter */
      dim3MaxLoopCount = 1;
      maxLoopCount    *= dim3Size;
    }
    for (z = 0; z < dim3MaxLoopCount; z++)
    {
      xb_vecN_2x32v vecOutTempL, vecOutTempH;
      /* initialize input and output data pointer */
      pvecIn  = (xb_vecNx8 *) (pInput + (z * inTilePitch2));
      pvecOut = (xb_vecN_2x64w *) (pOutput + (z * outTilePitch2));
      valign vaInData = IVP_LANX8S_PP(pvecIn);
      int32_t varlen;

      for (x = 0; x < maxLoopCount - vectorizationWidth; x += vectorizationWidth)
      {
        /* Load input data */
        IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

        xb_vecNx48 acc = zeroInScale;
        IVP_MULUSANX16(acc, vecScale, vecInData);

        xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc), IVP_CVT64SNX48LL(acc));
        vecOutTempL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
        vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
        //sign extending to 64bit
        vecOut0L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

        xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc), IVP_CVT64SNX48HL(acc));
        vecOutTempH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
        vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
        //sign extending to 64bit
        vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

        /* store output data */
        IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
        IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
      }
      varlen = (maxLoopCount - x);
      IVP_LANX8S_IP(vecInData, vaInData, pvecIn);

      xb_vecNx48 acc = zeroInScale;
      IVP_MULUSANX16(acc, vecScale, vecInData);

      xb_vecN_2x64w vecOutIntm1 = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc), IVP_CVT64SNX48LL(acc));
      vecOutTempL = IVP_PACKVRN_2X64W(vecOutIntm1, shift);
      vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
      //sign extending to 64bit
      vecOut0L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

      xb_vecN_2x64w vecOutIntm2 = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc), IVP_CVT64SNX48HL(acc));
      vecOutTempH = IVP_PACKVRN_2X64W(vecOutIntm2, shift);
      vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
      //sign extending to 64bit
      vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

      /* store output data */
      IVP_SAVN_2X64W_XP(vecOut0L, vaOut, pvecOut, (varlen << 3));
      IVP_SAVN_2X64W_XP(vecOut0H, vaOut, pvecOut, ((varlen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
      IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
    }
  }
  else
  {
    /* else block is executed if input tile pitch is not equal to input tile width or input tile */
    /* pitch is not equal to output tile pitch                                                   */

    for (z = 0; z < dim3Size; z++)     /* along 3rd dimension */
    {
      xb_vecN_2x32v vecOutTempL, vecOutTempH;
      x = 0;
      /* Loop Unroll=4 along 1st dimension */
      for (; x < (dim1Size - vectorizationWidth3X); x += vectorizationWidth4X)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth3X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData3, vaInData, pvecIn);

          /* apply scale and shift to input data.
           * multiplying with scale results in 32 way 48-bit
           * data to which shift is applied, so final result is
           * 32 way 16 bit.
           */
          xb_vecNx48 acc0, acc1, acc2, acc3;
          acc0 = zeroInScale;
          acc1 = zeroInScale;
          acc2 = zeroInScale;
          acc3 = zeroInScale;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          IVP_MULUSANX16(acc1, vecScale, vecInData1);
          IVP_MULUSANX16(acc2, vecScale, vecInData2);
          IVP_MULUSANX16(acc3, vecScale, vecInData3);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc0), IVP_CVT64SNX48LL(acc0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc0), IVP_CVT64SNX48HL(acc0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc1), IVP_CVT64SNX48LL(acc1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc1), IVP_CVT64SNX48HL(acc1));
          vecOut2L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc2), IVP_CVT64SNX48LL(acc2));
          vecOut2H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc2), IVP_CVT64SNX48HL(acc2));
          vecOut3L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc3), IVP_CVT64SNX48LL(acc3));
          vecOut3H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc3), IVP_CVT64SNX48HL(acc3));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut2L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut2L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut2H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut2H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut3L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut3L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut3H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut3H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut2H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut3L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut3H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      if (x < (dim1Size - vectorizationWidth2X))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth2X);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData2, vaInData, pvecIn);

          xb_vecNx48 acc0, acc1, acc2;
          acc0 = zeroInScale;
          acc1 = zeroInScale;
          acc2 = zeroInScale;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          IVP_MULUSANX16(acc1, vecScale, vecInData1);
          IVP_MULUSANX16(acc2, vecScale, vecInData2);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc0), IVP_CVT64SNX48LL(acc0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc0), IVP_CVT64SNX48HL(acc0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc1), IVP_CVT64SNX48LL(acc1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc1), IVP_CVT64SNX48HL(acc1));
          vecOut2L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc2), IVP_CVT64SNX48LL(acc2));
          vecOut2H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc2), IVP_CVT64SNX48HL(acc2));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut2L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut2L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut2H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut2H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);


          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut1H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut2L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut2H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < (dim1Size - vectorizationWidth))
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - (x + vectorizationWidth);

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);
          IVP_LANX8S_IP(vecInData1, vaInData, pvecIn);

          xb_vecNx48 acc0, acc1;
          acc0 = zeroInScale;
          acc1 = zeroInScale;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);
          IVP_MULUSANX16(acc1, vecScale, vecInData1);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc0), IVP_CVT64SNX48LL(acc0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc0), IVP_CVT64SNX48HL(acc0));
          vecOut1L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc1), IVP_CVT64SNX48LL(acc1));
          vecOut1H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc1), IVP_CVT64SNX48HL(acc1));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut1L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut1H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut1H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          /* Store output data */
          IVP_SAN_2X64W_IP(vecOut0L, vaOut, pvecOut);
          IVP_SAN_2X64W_IP(vecOut0H, vaOut, pvecOut);
          IVP_SAVN_2X64W_XP(vecOut1L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut1H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
      else if (x < dim1Size)
      {
        /* Initialize input and output data pointer */
        int8_t * pIn   = &pInput[z * inTilePitch2 + x];
        int64_t *pOut  = &pOutput[z * outTilePitch2 + x];
        int32_t varLen = dim1Size - x;

        for (y = 0; y < dim2Size; y++) /* along 2nd dimension */
        {
          pvecIn  = (xb_vecNx8 *) (pIn + (y * inTilePitch1));
          pvecOut = (xb_vecN_2x64w *) (pOut + (y * outTilePitch1));

          valign vaInData = IVP_LANX8S_PP(pvecIn);
          /* load input data */
          IVP_LANX8S_IP(vecInData0, vaInData, pvecIn);

          xb_vecNx48 acc0;
          acc0 = zeroInScale;

          IVP_MULUSANX16(acc0, vecScale, vecInData0);

          vecOut0L = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(acc0), IVP_CVT64SNX48LL(acc0));
          vecOut0H = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(acc0), IVP_CVT64SNX48HL(acc0));

          vecOutTempL = IVP_PACKVRN_2X64W(vecOut0L, shift);
          vecOutTempL = IVP_MAXN_2X32(vecOutTempL, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0L = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempL);

          vecOutTempH = IVP_PACKVRN_2X64W(vecOut0H, shift);
          vecOutTempH = IVP_MAXN_2X32(vecOutTempH, (xb_vecN_2x32v) minLim);
          //sign extending to 64bit
          vecOut0H = IVP_MULUSN_2X16X32_0((xb_vecNx16U) 1, vecOutTempH);

          /* Store output data */
          IVP_SAVN_2X64W_XP(vecOut0L, vaOut, pvecOut, (varLen << 3));
          IVP_SAVN_2X64W_XP(vecOut0H, vaOut, pvecOut, ((varLen << 3) - (XCHAL_IVPN_SIMD_WIDTH << 2)));
          IVP_SAPOSN_2X64W_FP(vaOut, pvecOut);
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/********************* xaiDataConversion3D_AsymQ *********************************/
/* Description : General API for DataConversion3D_AsymQ optimized implementation */
/*               Calls one of the DataConversion3D_AsymQ functions based         */
/*               on the parameters                                               */
/* Inputs      : Input Tile, zeroPoint, scale, shift                             */
/* Outputs     : XI Error Code                                                   */
/* InOuts      : Output Tile                                                     */
/*********************************************************************************/
XAI_ERR_TYPE xaiDataConversion3D_AsymQ(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const int16_t zeroPoint,
                                       const uint16_t scale,
                                       const uint8_t shift)
{
  if ((!inTile) || (!outTile))
  {
    return(XAI_ERR_NULLARG);
  }

  if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))
    {
      // Converts S8_SYM/S8_ASYM input to S8_SYM/S8_ASYM output (The "zeroPoint" used here serves as "fixUp" for the API)
      return(xaiDataConversion3D_AsymQ_S8S8(inTile, outTile, zeroPoint, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U8))
    {
      // Converts S8_ASYM input to U8_SYM output (The "zeroPoint" used here serves as "fixUp" for the API)
      return(xaiDataConversion3D_AsymQ_S8U8(inTile, outTile, zeroPoint, scale, shift));
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16))
    {
      if (zeroPoint == 0)
      {
        return(xaiDataConversion3D_S8S16(inTile, outTile, scale, shift));
      }
      else
      {
        // Converts S8_ASYM input to S16_SYM output (The "zeroPoint" used here serves as "fixUp" for the API)
        return(xaiDataConversion3D_AsymQ_S8S16(inTile, outTile, zeroPoint, scale, shift));
      }
    }
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16))
    {
      // Converts S8_ASYM input to U16_SYM output (The "zeroPoint" used here serves as "fixUp" for the API)
      return(xaiDataConversion3D_AsymQ_S8U16(inTile, outTile, zeroPoint, scale, shift));
    }
    else if ((XAI_TILE3D_CHECK_TYPE(outTile, XAI_S32)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U32)))
    {
      // Converts S8_ASYM input to I32 output (The "zeroPoint" used here serves as "ZeroIn" for the API)
      return(xaiDataConversion3D_AsymQ_S8I32(inTile, outTile, zeroPoint, scale, shift));
    }
    else if ((XAI_TILE3D_CHECK_TYPE(outTile, XAI_S64)) || (XAI_TILE3D_CHECK_TYPE(outTile, XAI_U64)))
    {
      // Converts S8_ASYM input to I64 output (The "zeroPoint" used here serves as "ZeroIn" for the API)
      return(xaiDataConversion3D_AsymQ_S8I64(inTile, outTile, zeroPoint, scale, shift));
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))
    {
      if (zeroPoint == 0)
      {
        return(xaiDataConversion3D_U8S8(inTile, outTile, scale, shift));
      }
      else
      {
        // Converts U8_SYM input to S8_ASYM output (The "zeroPoint" used here serves as "ZeroOut" for the API)
        return(xaiDataConversion3D_AsymQ_U8S8(inTile, outTile, zeroPoint, scale, shift));
      }
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))
    {
      if (zeroPoint == 0)
      {
        return(xaiDataConversion3D_S16I8(inTile, outTile, scale, shift));
      }
      else
      {
        // Converts S16_SYM input to S8_ASYM output (The "zeroPoint" used here serves as "ZeroOut" for the API)
        return(xaiDataConversion3D_AsymQ_S16S8(inTile, outTile, zeroPoint, scale, shift));
      }
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U16))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))
    {
      if (zeroPoint == 0)
      {
        return(xaiDataConversion3D_U16I8(inTile, outTile, scale, shift));
      }
      else
      {
        // Converts U16_SYM input to S8_ASYM output (The "zeroPoint" used here serves as "ZeroOut" for the API)
        // return(xaiDataConversion3D_U16AS8(inTile, outTile, zeroPoint, scale, shift));
        return(xaiDataConversion3D_AsymQ_U16S8(inTile, outTile, zeroPoint, scale, shift));
      }
    }
  }
  else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S32))
  {
    if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8))
    {
      if (zeroPoint == 0)
      {
        return(xaiDataConversion3D_S32S8(inTile, outTile, scale, shift));
      }
      else
      {
        // Converts S32_SYM input to S8_ASYM output (The "zeroPoint" used here serves as "ZeroOut" for the API)
        return(xaiDataConversion3D_AsymQ_S32S8(inTile, outTile, zeroPoint, scale, shift));
      }
    }
  }

  return(XAI_ERR_NO_VARIANT);
}
#endif //if ((XCHAL_VISION_TYPE >= 6))
