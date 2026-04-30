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

#include "xai_cnn.h"
#include "xai_intrin.h"

#if ((XCHAL_VISION_TYPE >= 6))

#define S24_MIN  (-(((int32_t) 1) << 23))
#define S24_MAX  ((((int32_t) 1) << 23) - 1)

/****************************************************************************/
/* Description : Implementation for getting the sub-kernel and              */
/*               super kernel related information.                          */
/*               If getNumKernelsFlag is passed as 1, function returns the  */
/*               number of sub-kernels.                                     */
/*               If getNumKernelsFlag is passed as 0, function returns the  */
/*               tile dimension for the sub-kernels.                        */
/* Inputs      : Input Coeff Tile, stride along X & Y directions,           */
/*               getNumKernelsFlag                                          */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Array of Coeff Sub, numSubKernels.                         */
/* Assumptions : Coeff is in WHDN format                                    */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvGetDim4D_WHDN(const xai_pTile4D coeffTile,
                                    xai_pTile4D subCoeffInfo[],
                                    uint16_t *numSubKernels,
                                    const uint8_t strideX,
                                    const uint8_t strideY,
                                    const uint8_t getNumKernelsFlag)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    if (getNumKernelsFlag)
    {
      XAI_CHECK_POINTER(numSubKernels);
    }
    XAI_CHECK_ERROR((strideX > 0) && (strideY > 0),                                          \
                    XAI_ERR_BADARG, "strideX = %hhu, strideY = %hhu\nStride has to be >= 1", \
                    strideX, strideY);
  }
  if (getNumKernelsFlag)
  {
    *numSubKernels = strideX * strideY;
    return(XAI_ERROR_STATUS());
  }

  int32_t kIdx, kIdy;
  int32_t kernelIdx;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE4D(coeffTile);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_WHDN);
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        XAI_CHECK_POINTER(subCoeffInfo[kernelIdx]);
      }
    }
  }

  const int32_t kWidth  = XAI_TILE4D_GET_DIM1(coeffTile);
  const int32_t kHeight = XAI_TILE4D_GET_DIM2(coeffTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_ERROR((strideX <= kWidth) && (strideY <= kHeight), XAI_ERR_BADARG,                                                                 \
                    "\nstrideX = %hhu, kWidth = %d and strideY = %hhu, kHeight = %d\nStride should be less than corresponding Kernel Dimension", \
                    strideX, kWidth, strideY, kHeight);
  }

  for (kIdy = 0; kIdy < strideY; kIdy++)
  {
    for (kIdx = 0; kIdx < strideX; kIdx++)
    {
      kernelIdx = kIdy * strideX + kIdx;

      XAI_TILE4D_SET_DIM1(subCoeffInfo[kernelIdx], \
                          (kWidth + strideX - kIdx - 1) / strideX);
      XAI_TILE4D_SET_DIM2(subCoeffInfo[kernelIdx], \
                          (kHeight + strideY - kIdy - 1) / strideY);
      XAI_TILE4D_SET_DIM3(subCoeffInfo[kernelIdx], \
                          XAI_TILE4D_GET_DIM4(coeffTile));
      XAI_TILE4D_SET_DIM4(subCoeffInfo[kernelIdx], \
                          XAI_TILE4D_GET_DIM3(coeffTile));
    }
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Implementation for getting the sub-kernel                  */
/*                related information.                                      */
/*               If getNumKernelsFlag is passed as 1, function returns the  */
/*               number of sub-kernels.                                     */
/*               If getNumKernelsFlag is passed as 0, function returns the  */
/*               tile dimension for the sub-kernels.                        */
/* Inputs      : Input Coeff Tile, stride along X & Y directions,           */
/*               getNumKernelsFlag                                          */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Array of Coeff Sub, numSubKernels.                         */
/* Assumptions : Coeff is in WHD format                                     */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvGetDim3D_WHD(const xai_pTile3D coeffTile,
                                   xai_pTile3D subCoeffInfo[],
                                   uint16_t *numSubKernels,
                                   const uint8_t strideX,
                                   const uint8_t strideY,
                                   const uint8_t getNumKernelsFlag)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    if (getNumKernelsFlag)
    {
      XAI_CHECK_POINTER(numSubKernels);
    }
    XAI_CHECK_ERROR((strideX > 0) && (strideY > 0),                                          \
                    XAI_ERR_BADARG, "strideX = %hhu, strideY = %hhu\nStride has to be >= 1", \
                    strideX, strideY);
  }
  if (getNumKernelsFlag)
  {
    *numSubKernels = strideX * strideY;
    return(XAI_ERROR_STATUS());
  }

  int32_t kIdx, kIdy;
  int32_t kernelIdx;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE3D(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(coeffTile, XAI_WHD);
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        XAI_CHECK_POINTER(subCoeffInfo[kernelIdx]);
      }
    }
  }

  const int32_t kWidth  = XAI_TILE3D_GET_DIM1(coeffTile);
  const int32_t kHeight = XAI_TILE3D_GET_DIM2(coeffTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_ERROR((strideX <= kWidth) && (strideY <= kHeight),                                                                                 \
                    XAI_ERR_BADARG,                                                                                                              \
                    "\nstrideX = %hhu, kWidth = %d and strideY = %hhu, kHeight = %d\nStride should be less than corresponding Kernel Dimension", \
                    strideX, kWidth, strideY, kHeight);
  }

  for (kIdy = 0; kIdy < strideY; kIdy++)
  {
    for (kIdx = 0; kIdx < strideX; kIdx++)
    {
      kernelIdx = kIdy * strideX + kIdx;

      XAI_TILE3D_SET_DIM1(subCoeffInfo[kernelIdx], \
                          (kWidth + strideX - kIdx - 1) / strideX);
      XAI_TILE3D_SET_DIM2(subCoeffInfo[kernelIdx], \
                          (kHeight + strideY - kIdy - 1) / strideY);
      XAI_TILE3D_SET_DIM3(subCoeffInfo[kernelIdx], \
                          XAI_TILE4D_GET_DIM3(coeffTile));
    }
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Implementation for coefficient reordering                  */
/*               The functions does the following:                          */
/*               - Convert from WHDN->WHND                                  */
/*               - Flips the coefficients across width and height which is  */
/*                 controlled by transposeCoeffsFlag.                       */
/*               - Breaks the kernel into sub-kernels.                      */
/* Inputs      : Input Coeff Tile, CNN convolution params structure,        */
/*               transposeCoeffsFlag                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Array of Coeff Sub & Super Tiles                           */
/* Assumptions : CoeffData is S8/U8                                         */
/*               Coeff is in WHDN format                                    */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvReOrder4D_I8_WHDN(const xai_pTile4D inTile,
                                        xai_pTile4D subCoeffs[],
                                        const xai_cnn_conv_params *param,
                                        const uint8_t transposeCoeffsFlag)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE4D_I8(inTile);
    XAI_CHECK_TILE4D_DATA_ORDER(inTile, XAI_WHDN);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_POINTER(subCoeffs);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) >= 1) &&                                                 \
                     (XAI_CNN_CONV_GET_STRIDEX(param) <= XAI_TILE4D_GET_DIM1(inTile))) &&                      \
                    ((XAI_CNN_CONV_GET_STRIDEY(param) >= 1) &&                                                 \
                     (XAI_CNN_CONV_GET_STRIDEY(param) <= XAI_TILE4D_GET_DIM2(inTile))), XAI_ERR_BADARG,        \
                    "StrideX = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Width) \
      \nStrideY = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Height)", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_TILE4D_GET_DIM1(inTile),                              \
                    XAI_CNN_CONV_GET_STRIDEY(param), XAI_TILE4D_GET_DIM2(inTile));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) == 1), \
                    XAI_ERR_BADARG, "\nDilation parameter is %hhu\nDilation parameter should be equal to 1", XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                          \
                    XAI_ERR_BADARG, "\nDilation along width is %hhu and dilation along height is %hhu are not same", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
  }

  int32_t kIdx, kIdy;
  int32_t kernelIdx;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    for (kIdy = 0; kIdy < XAI_CNN_CONV_GET_STRIDEY(param); kIdy++)
    {
      for (kIdx = 0; kIdx < XAI_CNN_CONV_GET_STRIDEX(param); kIdx++)
      {
        kernelIdx = kIdy * XAI_CNN_CONV_GET_STRIDEX(param) + kIdx;
        XAI_CHECK_TILE4D_I8(subCoeffs[kernelIdx]);
        XAI_CHECK_TILE4D_DATA_ORDER(subCoeffs[kernelIdx], XAI_WHDN);
      }
    }
  }
  int8_t *pInCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(inTile);

  const int32_t kWidth   = XAI_TILE4D_GET_DIM1(inTile); /* W */
  const int32_t kHeight  = XAI_TILE4D_GET_DIM2(inTile); /* H */
  const int32_t numInCh  = XAI_TILE4D_GET_DIM3(inTile); /* D */
  const int32_t numOutCh = XAI_TILE4D_GET_DIM4(inTile); /* N */

  const uint8_t strideX = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY = XAI_CNN_CONV_GET_STRIDEY(param);

  int32_t inCoeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(inTile);
  int32_t inCoeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(inTile);
  int32_t inCoeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(inTile);

  int32_t kx, ky, inCh, outCh, inIdx, outIdx = 0;
  int8_t *pSubCoeff;
  int32_t kxStart, kyStart;


  if (transposeCoeffsFlag)
  {
    /* Conversion from WHDN -> WHND,                       */
    /* transposing of kernels and formation of sub-kernels */
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        int8_t *pSubCoeff = \
          (int8_t *) XAI_TILE4D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = kHeight - 1 - ((kHeight + strideY - kIdy - 1) % strideY);
        kxStart = kWidth - 1 - ((kWidth + strideX - kIdx - 1) % strideX);

        for (inCh = 0; inCh < numInCh; inCh++)            /* D */
        {
          for (outCh = 0; outCh < numOutCh; outCh++)      /* N */
          {
            for (ky = kyStart; ky >= 0; ky -= strideY)    /* H */
            {
              for (kx = kxStart; kx >= 0; kx -= strideX)  /* W */
              {
                inIdx = outCh * inCoeffPitch3 + inCh * inCoeffPitch2 + \
                        ky * inCoeffPitch1 + kx;
                pSubCoeff[outIdx++] = pInCoeff[inIdx];
              }
            }
          }
        }
      }
    }
  }
  else
  {
    /* Conversion from WHDN -> WHND and formation of sub-kernels */
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        pSubCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = ((kHeight + strideY - kIdy - 1) % strideY);
        kxStart = ((kWidth + strideX - kIdx - 1) % strideX);

        for (inCh = 0; inCh < numInCh; inCh++)                 /* D */
        {
          for (outCh = 0; outCh < numOutCh; outCh++)           /* N */
          {
            for (ky = kyStart; ky < kHeight; ky += strideY)    /* H */
            {
              for (kx = kxStart; kx < kWidth; kx += strideX)   /* W */
              {
                inIdx = outCh * inCoeffPitch3 + inCh * inCoeffPitch2 + \
                        ky * inCoeffPitch1 + kx;
                pSubCoeff[outIdx++] = pInCoeff[inIdx];
              }
            }
          }
        }
      }
    }
  }

  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Implementation for coefficient reordering                  */
/*               The functions does the following:                          */
/*               - Flips the coefficients across width and height which is  */
/*                 controlled by transposeCoeffsFlag.                       */
/*               - Breaks the kernel into sub-kernels.                      */
/* Inputs      : Input Coeff Tile, CNN convolution params structure,        */
/*               transposeCoeffsFlag                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Array of Coeff Sub & Super Tiles                           */
/* Assumptions : CoeffData is S8/U8                                         */
/*               Coeff is in WHD format                                     */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvReOrder3D_I8_WHD(const xai_pTile3D inTile,
                                       xai_pTile3D subCoeffs[],
                                       const xai_cnn_depthwiseDilatedConv_params *param,
                                       const uint8_t transposeCoeffsFlag)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_I8(inTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_POINTER(subCoeffs);
    XAI_CHECK_ERROR(((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param) >= 1) &&                                          \
                     (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param) <= XAI_TILE3D_GET_DIM1(inTile))) &&               \
                    ((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param) >= 1) &&                                          \
                     (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param) <= XAI_TILE3D_GET_DIM2(inTile))), XAI_ERR_BADARG, \
                    "StrideX = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Width) \
            \nStrideY = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Height)",      \
                    XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param), XAI_TILE3D_GET_DIM1(inTile),                       \
                    XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param), XAI_TILE3D_GET_DIM2(inTile));
    XAI_CHECK_ERROR((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(param) == 1),                     \
                    XAI_ERR_BADARG, "\nDilation is %hhu\nDilation parameter should be equal to 1", \
                    XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONX(param) == XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONY(param), \
                    XAI_ERR_BADARG, "\nDilation along width is %hhu and dilation along height is %hhu are not same",            \
                    XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONX(param), XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONY(param));
  }

  int32_t kIdx, kIdy;
  int32_t kernelIdx;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    for (kIdy = 0; kIdy < XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param); kIdy++)
    {
      for (kIdx = 0; kIdx < XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param); kIdx++)
      {
        kernelIdx = kIdy * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param) + kIdx;
        XAI_CHECK_TILE3D_I8(subCoeffs[kernelIdx]);
        XAI_CHECK_TILE3D_DATA_ORDER(subCoeffs[kernelIdx], XAI_WHD);
      }
    }
  }
  int8_t *pInCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(inTile);

  const int32_t kWidth  = XAI_TILE4D_GET_DIM1(inTile);    /* W */
  const int32_t kHeight = XAI_TILE4D_GET_DIM2(inTile);    /* H */
  const int32_t numInCh = XAI_TILE4D_GET_DIM3(inTile);    /* D */


  const uint8_t strideX = XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param);
  const uint8_t strideY = XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param);

  int32_t inCoeffPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  int32_t inCoeffPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  int32_t kx, ky, inCh, inIdx, outIdx = 0;
  int8_t *pSubCoeff;
  int32_t kxStart, kyStart;


  if (transposeCoeffsFlag)
  {
    /* transposing of kernels and formation of sub-kernels */
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        int8_t *pSubCoeff = \
          (int8_t *) XAI_TILE4D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = kHeight - 1 - ((kHeight + strideY - kIdy - 1) % strideY);
        kxStart = kWidth - 1 - ((kWidth + strideX - kIdx - 1) % strideX);

        for (inCh = 0; inCh < numInCh; inCh++)            /* D */
        {
          for (ky = kyStart; ky >= 0; ky -= strideY)    /* H */
          {
            for (kx = kxStart; kx >= 0; kx -= strideX)  /* W */
            {
              inIdx = inCh * inCoeffPitch2 + \
                      ky * inCoeffPitch1 + kx;
              pSubCoeff[outIdx++] = pInCoeff[inIdx];
            }
          }
        }
      }
    }
  }
  else
  {
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        pSubCoeff = (int8_t *) XAI_TILE3D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = ((kHeight + strideY - kIdy - 1) % strideY);
        kxStart = ((kWidth + strideX - kIdx - 1) % strideX);

        for (inCh = 0; inCh < numInCh; inCh++)                 /* D */
        {
          for (ky = kyStart; ky < kHeight; ky += strideY)    /* H */
          {
            for (kx = kxStart; kx < kWidth; kx += strideX)   /* W */
            {
              inIdx = inCh * inCoeffPitch2 + \
                      ky * inCoeffPitch1 + kx;
              pSubCoeff[outIdx++] = pInCoeff[inIdx];
            }
          }
        }
      }
    }
  }

  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Implementation for extending the bias array in             */
/*               case of MOD deconvolution using superkernels.              */
/* Inputs      : Input Bias array,                                          */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Bias array                                          */
/****************************************************************************/
XAI_ERR_TYPE xaiBiasExtend_S32_MOD(const xai_pArray inBiasArray,
                                   xai_pArray outBiasArray)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_ARRAY_S32(inBiasArray);
    XAI_CHECK_ARRAY_S32(outBiasArray);
  }

  int32_t inWidth  = XAI_ARRAY_GET_WIDTH(inBiasArray);
  int32_t outWidth = XAI_ARRAY_GET_WIDTH(outBiasArray);
  int32_t strideX  = outWidth / inWidth;

  int32_t* pInBias  = (int32_t *) XAI_ARRAY_GET_DATA_PTR(inBiasArray);
  int32_t* pOutBias = (int32_t *) XAI_ARRAY_GET_DATA_PTR(outBiasArray);

  int32_t numX, inW;
  for (numX = 0; numX < strideX; numX++)
  {
    for (inW = 0; inW < inWidth; inW++)
    {
      pOutBias[inW + inWidth * numX] = pInBias[inW];
    }
  }
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************/
/* Description : Implementation for extending the outputscale array          */
/*               in case of MOD deconvolution using superkernels.            */
/* Inputs      : outputScale array,                                          */
/* Outputs     : XI Error Code                                               */
/* InOuts      : extended outputScale array                                  */
/*****************************************************************************/
XAI_ERR_TYPE xaiOutScaleExtend_U16_MOD(const xai_pArray outScaleArray,
                                       xai_pArray extendedOutScaleArray)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_ARRAY_U16(outScaleArray);
    XAI_CHECK_ARRAY_U16(extendedOutScaleArray);
  }

  int32_t inWidth  = XAI_ARRAY_GET_WIDTH(outScaleArray);
  int32_t outWidth = XAI_ARRAY_GET_WIDTH(extendedOutScaleArray);
  int32_t strideX  = outWidth / inWidth;

  uint16_t* pInScale  = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outScaleArray);
  uint16_t* pOutScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(extendedOutScaleArray);

  int32_t numX, inW;
  for (numX = 0; numX < strideX; numX++)
  {
    for (inW = 0; inW < inWidth; inW++)
    {
      pOutScale[inW + inWidth * numX] = pInScale[inW];
    }
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Implementation for getting the sub-kernel and              */
/*               super kernel related information.                          */
/*               If getNumKernelsFlag is passed as 1, function returns the  */
/*               number of sub-kernels and super kernels.                   */
/*               If getNumKernelsFlag is passed as 0, function returns the  */
/*               tile dimension for the sub-kernels and super kernels.      */
/* Inputs      : Input Coeff Tile, stride along X & Y directions,           */
/*               getNumKernelsFlag                                          */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Array of Coeff Sub & Super Tiles, numSubKernels and        */
/*               numSuperKernels                                            */
/* Assumptions : Coeff is in NDWH format                                    */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvGetDim4D_NDWH(const xai_pTile4D coeffTile,
                                    xai_pTile4D subCoeffInfo[],
                                    xai_pTile4D superCoeffInfo[],
                                    uint16_t *numSubKernels,
                                    uint16_t *numSuperKernels,
                                    const uint8_t strideX,
                                    const uint8_t strideY,
                                    const uint8_t getNumKernelsFlag)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    if (getNumKernelsFlag)
    {
      XAI_CHECK_POINTER(numSubKernels);
      XAI_CHECK_POINTER(numSuperKernels);
    }
    XAI_CHECK_ERROR((strideX > 0) && (strideY > 0),                                          \
                    XAI_ERR_BADARG, "strideX = %hhu, strideY = %hhu\nStride has to be >= 1", \
                    strideX, strideY);
  }
  if (getNumKernelsFlag)
  {
    *numSubKernels   = strideX * strideY;
    *numSuperKernels = strideY;
    return(XAI_ERROR_STATUS());
  }

  int32_t kIdx, kIdy;
  int32_t kernelIdx;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE4D(coeffTile);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_POINTER(subCoeffInfo);
    XAI_CHECK_POINTER(superCoeffInfo);
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        XAI_CHECK_POINTER(subCoeffInfo[kernelIdx]);
      }
      XAI_CHECK_POINTER(superCoeffInfo[kIdy]);
    }
  }

  const int32_t kWidth  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeight = XAI_TILE4D_GET_DIM4(coeffTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_ERROR((strideX <= kWidth) && (strideY <= kHeight), XAI_ERR_BADARG,     \
                    "StrideX = %hhu, value must be less than or equal to %d(kernel Width) \
            \nStrideY = %hhu, value must be ess than or equal to %d(kernel Height)", \
                    strideX, kWidth, strideY, kHeight);
  }

  for (kIdy = 0; kIdy < strideY; kIdy++)
  {
    for (kIdx = 0; kIdx < strideX; kIdx++)
    {
      kernelIdx = kIdy * strideX + kIdx;

      XAI_TILE4D_SET_DIM1(subCoeffInfo[kernelIdx], XAI_TILE4D_GET_DIM2(coeffTile));
      XAI_TILE4D_SET_DIM2(subCoeffInfo[kernelIdx], XAI_TILE4D_GET_DIM1(coeffTile));
      XAI_TILE4D_SET_DIM3(subCoeffInfo[kernelIdx], (kWidth + strideX - kIdx - 1) / strideX);
      XAI_TILE4D_SET_DIM4(subCoeffInfo[kernelIdx], (kHeight + strideY - kIdy - 1) / strideY);
    }
    XAI_TILE4D_SET_DIM1(superCoeffInfo[kIdy], XAI_TILE4D_GET_DIM1(subCoeffInfo[kIdy * strideX]) * strideX);
    XAI_TILE4D_SET_DIM2(superCoeffInfo[kIdy], XAI_TILE4D_GET_DIM2(subCoeffInfo[kIdy * strideX]));
    XAI_TILE4D_SET_DIM3(superCoeffInfo[kIdy], XAI_TILE4D_GET_DIM3(subCoeffInfo[kIdy * strideX]));
    XAI_TILE4D_SET_DIM4(superCoeffInfo[kIdy], XAI_TILE4D_GET_DIM4(subCoeffInfo[kIdy * strideX]));
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Implementation for getting the sub-kernel                  */
/*               related information.                                       */
/*               If getNumKernelsFlag is passed as 1, function returns the  */
/*               number of sub-kernels .                                    */
/*               If getNumKernelsFlag is passed as 0, function returns the  */
/*               tile dimension for the sub-kernels .                       */
/* Inputs      : Input Coeff Tile, stride along X & Y directions,           */
/*               getNumKernelsFlag                                          */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Array of Coeff Sub Tiles and numSubKernels                 */
/* Assumptions : Coeff is in DWH format                                     */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvGetDim3D_DWH(const xai_pTile3D coeffTile,
                                   xai_pTile3D subCoeffInfo[],
                                   uint16_t *numSubKernels,
                                   const uint8_t strideX,
                                   const uint8_t strideY,
                                   const uint8_t getNumKernelsFlag)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    if (getNumKernelsFlag)
    {
      XAI_CHECK_POINTER(numSubKernels);
    }
    XAI_CHECK_ERROR((strideX > 0) && (strideY > 0), XAI_ERR_BADARG,          \
                    "strideX = %hhu, strideY = %hhu\nStride has to be >= 1", \
                    strideX, strideY);
  }
  if (getNumKernelsFlag)
  {
    *numSubKernels = strideX * strideY;
    return(XAI_ERROR_STATUS());
  }

  int32_t kIdx, kIdy;
  int32_t kernelIdx;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE3D(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(coeffTile, XAI_DWH);
    XAI_CHECK_POINTER(subCoeffInfo);
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        XAI_CHECK_POINTER(subCoeffInfo[kernelIdx]);
      }
    }
  }

  const int32_t kWidth  = XAI_TILE3D_GET_DIM2(coeffTile);
  const int32_t kHeight = XAI_TILE3D_GET_DIM3(coeffTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_ERROR((strideX <= kWidth) && (strideY <= kHeight), XAI_ERR_BADARG, \
                    "StrideX = %hhu, value must be less than or equal to %d(kernel Width) \
       \nStrideY = %hhu, value must be ess than or equal to %d(kernel Height)",  \
                    strideX, kWidth, strideY, kHeight);
  }

  for (kIdy = 0; kIdy < strideY; kIdy++)
  {
    for (kIdx = 0; kIdx < strideX; kIdx++)
    {
      kernelIdx = kIdy * strideX + kIdx;

      XAI_TILE3D_SET_DIM1(subCoeffInfo[kernelIdx], XAI_TILE3D_GET_DIM1(coeffTile));
      XAI_TILE3D_SET_DIM2(subCoeffInfo[kernelIdx], (kWidth + strideX - kIdx - 1) / strideX);
      XAI_TILE3D_SET_DIM3(subCoeffInfo[kernelIdx], (kHeight + strideY - kIdy - 1) / strideY);
    }
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Implementation for coefficient reordering                  */
/*               The functions does the following:                          */
/*               - Convert from NDWH->DNWH                                  */
/*               - Flips the coefficients across width and height which is  */
/*                 controlled by transposeCoeffsFlag.                       */
/*               - Breaks the kernel into sub-kernels.                      */
/*               - Stacks sub-kernels to form super kernels.                */
/* Inputs      : Input Coeff Tile, CNN convolution params structure,        */
/*               transposeCoeffsFlag                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Array of Coeff Sub & Super Tiles                           */
/* Assumptions : CoeffData is S8/U8                                         */
/*               Coeff is in NDWH format                                    */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvReOrder4D_I8_NDWH(const xai_pTile4D inTile,
                                        xai_pTile4D subCoeffs[],
                                        xai_pTile4D superCoeffs[],
                                        const xai_cnn_conv_params *param,
                                        const uint8_t transposeCoeffsFlag)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE4D_I8(inTile);
    XAI_CHECK_TILE4D_DATA_ORDER(inTile, XAI_NDWH);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_POINTER(subCoeffs);
    XAI_CHECK_POINTER(superCoeffs);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) >= 1) &&                                                  \
                     (XAI_CNN_CONV_GET_STRIDEX(param) <= XAI_TILE4D_GET_DIM3(inTile))) &&                       \
                    ((XAI_CNN_CONV_GET_STRIDEY(param) >= 1) &&                                                  \
                     (XAI_CNN_CONV_GET_STRIDEY(param) <= XAI_TILE4D_GET_DIM4(inTile))), XAI_ERR_BADARG,         \
                    "StrideX = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Width) \
       \nStrideY = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Height)", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_TILE4D_GET_DIM3(inTile),                               \
                    XAI_CNN_CONV_GET_STRIDEY(param), XAI_TILE4D_GET_DIM4(inTile));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) == 1), \
                    XAI_ERR_BADARG, "\nDilation is %hhu\nDilation parameter should be equal to 1", XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                          \
                    XAI_ERR_BADARG, "\nDilation along width is %hhu and dilation along height is %hhu are not same", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
  }

  int32_t kIdx, kIdy;
  int32_t kernelIdx;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    for (kIdy = 0; kIdy < XAI_CNN_CONV_GET_STRIDEY(param); kIdy++)
    {
      for (kIdx = 0; kIdx < XAI_CNN_CONV_GET_STRIDEX(param); kIdx++)
      {
        kernelIdx = kIdy * XAI_CNN_CONV_GET_STRIDEX(param) + kIdx;
        XAI_CHECK_TILE4D_I8(subCoeffs[kernelIdx]);
        XAI_CHECK_TILE4D_DATA_ORDER(subCoeffs[kernelIdx], XAI_NDWH);
      }
      XAI_CHECK_TILE4D_I8(superCoeffs[kIdy]);
      XAI_CHECK_TILE4D_DATA_ORDER(superCoeffs[kIdy], XAI_NDWH);
    }
  }

  int8_t *pInCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(inTile);

  const int32_t numOutCh = XAI_TILE4D_GET_DIM1(inTile); /* N */
  const int32_t numInCh  = XAI_TILE4D_GET_DIM2(inTile); /* D */
  const int32_t kWidth   = XAI_TILE4D_GET_DIM3(inTile); /* W */
  const int32_t kHeight  = XAI_TILE4D_GET_DIM4(inTile); /* H */

  const uint8_t strideX = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY = XAI_CNN_CONV_GET_STRIDEY(param);

  int32_t inCoeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(inTile);
  int32_t inCoeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(inTile);
  int32_t inCoeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(inTile);

  int32_t kx, ky, inCh, outCh, inIdx, outIdx = 0;
  int8_t *pSuperCoeff;
  int8_t *pSubCoeff;
  int32_t subKPitch1, subKPitch2, subKPitch3;
  int32_t superKPitch1, superKPitch2;
  int32_t kW, kH, subkW;
  int32_t numInChSubCoeff;
  int32_t subKIdx;

  int32_t kxStart, kyStart;

  if (transposeCoeffsFlag)
  {
    /* Conversion from NDWH -> DNWH,                       */
    /* transposing of kernels and formation of sub-kernels */
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        int8_t *pSubCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = kHeight - 1 - ((kHeight + strideY - kIdy - 1) % strideY);

        for (ky = kyStart; ky >= 0; ky -= strideY)          /* H */
        {
          kxStart = kWidth - 1 - ((kWidth + strideX - kIdx - 1) % strideX);

          for (kx = kxStart; kx >= 0; kx -= strideX)        /* W */
          {
            for (outCh = 0; outCh < numOutCh; outCh++)      /* N */
            {
              for (inCh = 0; inCh < numInCh; inCh++)        /* D */
              {
                inIdx = ky * inCoeffPitch3 + kx * inCoeffPitch2 + \
                        inCh * inCoeffPitch1 + outCh;
                pSubCoeff[outIdx++] = pInCoeff[inIdx];
              }
              /* For stride alignment */
              outIdx += (outIdx % (2 * XCHAL_IVPN_SIMD_WIDTH)) ? ((2 * XCHAL_IVPN_SIMD_WIDTH) - (outIdx % (2 * XCHAL_IVPN_SIMD_WIDTH))) : 0;
            }
          }
        }
      }
    }
  }
  else
  {
    /* Conversion from NDWH -> DNWH and formation of sub-kernels */
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        int8_t *pSubCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = ((kHeight + strideY - kIdy - 1) % strideY);

        for (ky = kyStart; ky < kHeight; ky += strideY)          /* H */
        {
          kxStart = ((kWidth + strideX - kIdx - 1) % strideX);

          for (kx = kxStart; kx < kWidth; kx += strideX)         /* W */
          {
            for (outCh = 0; outCh < numOutCh; outCh++)           /* N */
            {
              for (inCh = 0; inCh < numInCh; inCh++)             /* D */
              {
                inIdx = ky * inCoeffPitch3 + kx * inCoeffPitch2 + \
                        inCh * inCoeffPitch1 + outCh;
                pSubCoeff[outIdx++] = pInCoeff[inIdx];
              }
              /* For stride alignment */
              outIdx += (outIdx % (2 * XCHAL_IVPN_SIMD_WIDTH)) ? ((2 * XCHAL_IVPN_SIMD_WIDTH) - (outIdx % (2 * XCHAL_IVPN_SIMD_WIDTH))) : 0;
            }
          }
        }
      }
    }
  }

  /* Form super-kernels by stacking sub-kernels */
  for (kernelIdx = 0; kernelIdx < strideY; kernelIdx++)
  {
    pSuperCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(superCoeffs[kernelIdx]);

    kW = XAI_TILE4D_GET_DIM3(superCoeffs[kernelIdx]);
    kH = XAI_TILE4D_GET_DIM4(superCoeffs[kernelIdx]);

    numInChSubCoeff = XAI_TILE4D_GET_DIM1(subCoeffs[kernelIdx * strideX]);
    superKPitch1    = XAI_TILE4D_GET_DIM1_PITCH(superCoeffs[kernelIdx]);
    superKPitch2    = XAI_TILE4D_GET_DIM2_PITCH(superCoeffs[kernelIdx]);

    for (subKIdx = 0; subKIdx < strideX; subKIdx++)
    {
      pSubCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(subCoeffs[kernelIdx * strideX + subKIdx]);

      subkW = XAI_TILE4D_GET_DIM3(subCoeffs[kernelIdx * strideX + subKIdx]);

      subKPitch1 = XAI_TILE4D_GET_DIM1_PITCH(subCoeffs[kernelIdx * strideX + subKIdx]);
      subKPitch2 = XAI_TILE4D_GET_DIM2_PITCH(subCoeffs[kernelIdx * strideX + subKIdx]);
      subKPitch3 = XAI_TILE4D_GET_DIM3_PITCH(subCoeffs[kernelIdx * strideX + subKIdx]);

      outIdx = numInChSubCoeff * subKIdx;

      for (ky = 0, kIdy = 0; ky < kH; ky++, kIdy++)          /* H */
      {
        for (kx = 0, kIdx = 0; kx < kW; kx++, kIdx++)        /* W */
        {
          /*In case of super kernels we have the first sub kernel width/height as the width/height of the superkernel     */
          /*In case the widths of the subkernel are not equal then we skip by differnce and start filling                 */
          /*Once the convolution is done the output junk data apprears at the end of the outtile.                         */
          /*In case of unequal heights this is handled using pointers in test app.                                        */
          if ((subkW < kW) && (kx == 0))
          {
            outIdx += superKPitch2;
            kIdx--;
            continue;
          }
          for (outCh = 0; outCh < numOutCh; outCh++)         /* N */
          {
            for (inCh = 0; inCh < numInChSubCoeff; inCh++)   /* D */
            {
              inIdx = kIdy * subKPitch3 + kIdx * subKPitch2 + \
                      outCh * subKPitch1 + inCh;
              pSuperCoeff[outIdx++] = pSubCoeff[inIdx];
            }
            outIdx += (superKPitch1 - numInChSubCoeff);
          }
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Implementation for coefficient reordering                  */
/*               The functions does the following:                          */
/*               - Flips the coefficients across width and height which is  */
/*                 controlled by transposeCoeffsFlag.                       */
/*               - Breaks the kernel into sub-kernels.                      */
/* Inputs      : Input Coeff Tile, CNN convolution params structure,        */
/*               transposeCoeffsFlag                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Array of Coeff Sub  Tiles                                  */
/* Assumptions : CoeffData is S8/U8                                         */
/*               Coeff is in DWH format                                     */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvReOrder3D_I8_DWH(const xai_pTile3D inTile,
                                       xai_pTile3D subCoeffs[],
                                       const xai_cnn_depthwiseDilatedConv_params *param,
                                       const uint8_t transposeCoeffsFlag)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_I8(inTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_POINTER(subCoeffs);
    XAI_CHECK_ERROR(((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param) >= 1) &&                                          \
                     (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param) <= XAI_TILE3D_GET_DIM2(inTile))) &&               \
                    ((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param) >= 1) &&                                          \
                     (XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param) <= XAI_TILE3D_GET_DIM3(inTile))), XAI_ERR_BADARG, \
                    "StrideX = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Width) \
       \nStrideY = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Height)",           \
                    XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param), XAI_TILE3D_GET_DIM2(inTile),                       \
                    XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param), XAI_TILE3D_GET_DIM3(inTile));
    XAI_CHECK_ERROR((XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(param) == 1),                     \
                    XAI_ERR_BADARG, "\nDilation is %hhu\nDilation parameter should be equal to 1", \
                    XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONX(param) == XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONY(param), \
                    XAI_ERR_BADARG, "\nDilation along width is %hhu and dilation along height is %hhu are not same",            \
                    XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONX(param), XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATIONY(param));
  }

  int32_t kIdx, kIdy;
  int32_t kernelIdx;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    for (kIdy = 0; kIdy < XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param); kIdy++)
    {
      for (kIdx = 0; kIdx < XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param); kIdx++)
      {
        kernelIdx = kIdy * XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param) + kIdx;
        XAI_CHECK_TILE3D_I8(subCoeffs[kernelIdx]);
        XAI_CHECK_TILE3D_DATA_ORDER(subCoeffs[kernelIdx], XAI_DWH);
      }
    }
  }

  int8_t *pInCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(inTile);


  const int32_t numInCh = XAI_TILE3D_GET_DIM1(inTile);    /* D */
  const int32_t kWidth  = XAI_TILE3D_GET_DIM2(inTile);    /* W */
  const int32_t kHeight = XAI_TILE3D_GET_DIM3(inTile);    /* H */

  const uint8_t strideX = XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(param);
  const uint8_t strideY = XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(param);

  int32_t inCoeffPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  int32_t inCoeffPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  int32_t kx, ky, inCh, inIdx, outIdx = 0;
  int32_t kxStart, kyStart;

  if (transposeCoeffsFlag)
  {
    /* transposing of kernels and formation of sub-kernels */
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        int8_t *pSubCoeff = (int8_t *) XAI_TILE3D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = kHeight - 1 - ((kHeight + strideY - kIdy - 1) % strideY);

        for (ky = kyStart; ky >= 0; ky -= strideY)          /* H */
        {
          kxStart = kWidth - 1 - ((kWidth + strideX - kIdx - 1) % strideX);

          for (kx = kxStart; kx >= 0; kx -= strideX)        /* W */
          {
            for (inCh = 0; inCh < numInCh; inCh++)          /* D */
            {
              inIdx               = ky * inCoeffPitch2 + kx * inCoeffPitch1 + inCh;
              pSubCoeff[outIdx++] = pInCoeff[inIdx];
            }
            /* For stride alignment */
            outIdx += (outIdx % (2 * XCHAL_IVPN_SIMD_WIDTH)) ? ((2 * XCHAL_IVPN_SIMD_WIDTH) - (outIdx % (2 * XCHAL_IVPN_SIMD_WIDTH))) : 0;
          }
        }
      }
    }
  }
  else
  {
    /* Formation of sub-kernels */
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        int8_t *pSubCoeff = (int8_t *) XAI_TILE3D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = ((kHeight + strideY - kIdy - 1) % strideY);

        for (ky = kyStart; ky < kHeight; ky += strideY)          /* H */
        {
          kxStart = ((kWidth + strideX - kIdx - 1) % strideX);

          for (kx = kxStart; kx < kWidth; kx += strideX)         /* W */
          {
            for (inCh = 0; inCh < numInCh; inCh++)             /* D */
            {
              inIdx               = ky * inCoeffPitch2 + kx * inCoeffPitch1 + inCh;
              pSubCoeff[outIdx++] = pInCoeff[inIdx];
            }
            /* For stride alignment */
            outIdx += (outIdx % (2 * XCHAL_IVPN_SIMD_WIDTH)) ? ((2 * XCHAL_IVPN_SIMD_WIDTH) - (outIdx % (2 * XCHAL_IVPN_SIMD_WIDTH))) : 0;
          }
        }
      }
    }
  }

  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Vision P6 implementation for interleaving the outputs      */
/*               generated by convolution functions using the sub-kernels   */
/* Inputs      : array of output tiles passed as input, CNN convolution     */
/*               params structure, output tile                              */
/* Outputs     : XI Error Code                                              */
/* InOuts      : output tile                                                */
/* Assumptions : Input Tile Data is S8/U8                                   */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvInterleave3D_I8_WHD(const xai_pTile3D inTile[],
                                          xai_pTile3D outTile,
                                          const xai_cnn_conv_params *convParams)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(inTile);
    XAI_CHECK_POINTER(convParams);
    XAI_CHECK_TILE3D_I8(outTile);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_WHD);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(outTile);
  }
  /* Getting parameters from the tile structures */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const uint8_t strideX = XAI_CNN_CONV_GET_STRIDEX(convParams);
  const uint8_t strideY = XAI_CNN_CONV_GET_STRIDEY(convParams);

  const int32_t outDataPitch1Offset = (outDataPitch1 * strideY);

  int8_t *pOutput = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int32_t ch, x, y, numX, numY, idx, remX;
  int8_t *pSubKernelOutput;
  int8_t *pOutput1;
  int8_t *pOutput2;
  int8_t *pInput1;
  int8_t *pInput2;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_ERROR(((strideX > 0) && (strideY > 0)), XAI_ERR_BADARG,        \
                    "strideX = %hhu, strideY = %hhu\nStride has to be >= 1", \
                    strideX, strideY);

    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outTile) >= strideX) &&                      \
                    (XAI_TILE3D_GET_DIM2(outTile) >= strideY), XAI_ERR_BADARG,        \
                    "\nOutTile width = %d, value must be greater than or equal to %hhu(strideX) \
       \nOutTile height = %d,  value must be greater than or equal to %hhu(strideY)", \
                    XAI_TILE3D_GET_DIM1(outTile), strideX, XAI_TILE3D_GET_DIM2(outTile), strideY);

    for (numY = 0; numY < strideY; numY++)
    {
      for (numX = 0; numX < strideX; numX++)
      {
        idx = numX + numY * strideX;
        XAI_CHECK_POINTER(inTile[idx]);
        XAI_CHECK_TILE3D_I8(inTile[idx]);
        XAI_CHECK_TILE3D_DATA_ORDER(inTile[idx], XAI_WHD);
        XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile[idx], outTile);
        XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inTile[idx]) == XAI_TILE3D_GET_DIM3(outTile), XAI_ERR_BADARG, \
                        "\nNumber of channels of each subkernel output = %d, final output = %d \
            \nNumber of channels of each subkernel output and final output should be the same",           \
                        XAI_TILE3D_GET_DIM3(inTile[idx]), XAI_TILE3D_GET_DIM3(outTile));
      }
    }
  }

  /* Scatter Index Calculations */
  /* Sequence - 0 1 2 3 4 ... 30 31 */
  xb_vecNx16U vecSelIdx1 = IVP_SEQNX16U();
  /* Sequence - 0 strideX 2*strideX 3*strideX 4*strideX .... 30*strideX 31*strideX*/
  xb_vecNx16U vecScatterOff1 = IVP_MULNX16UPACKL(vecSelIdx1, \
                                                 (uint16_t) strideX);

  xb_vecNx16U vecScatterOff2;
  /* Sequence - (32*strideX) (33*strideX) (34*strideX) ....(62*strideX) (63*strideX)*/
  vecScatterOff2 = IVP_ADDNX16(vecScatterOff1, (XCHAL_IVPN_SIMD_WIDTH * strideX));

  xb_vec2Nx8* restrict pdvecIn1;
  xb_vec2Nx8 dvecData1;
  valign vaInData1;
  vbool2N vecMsk;
  vboolN vecOffsetMsk1;
  vboolN vecOffsetMsk2;
  /* Sequence - 0 1 2 3 4 ... 62 63 */
  xb_vec2Nx8 vecCmp = IVP_SEQ2NX8U();
  /* Sequence - 0 1 2 3 4 ... 30 31 */
  xb_vecNx16U vecOffsetCmp = IVP_SEQNX16U();

  const int32_t vectorizationWidth = 2 * XCHAL_IVPN_SIMD_WIDTH;

  for (numY = 0; numY < strideY; numY++)
  {
    for (numX = 0; numX < strideX; numX++)
    {
      idx = numX + numY * strideX;
      int8_t *pInput             = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile[idx]);
      const int32_t inDataWidth  = XAI_TILE3D_GET_DIM1(inTile[idx]);
      const int32_t inDataHeight = XAI_TILE3D_GET_DIM2(inTile[idx]);
      const int32_t inChanNum    = XAI_TILE3D_GET_DIM3(inTile[idx]);
      const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile[idx]);
      const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile[idx]);
      pSubKernelOutput = (pOutput + numX + (numY * outDataPitch1));
      for (ch = 0; ch < inChanNum; ch++)
      {
        pOutput1 = (pSubKernelOutput + (ch * outDataPitch2));
        pInput1  = (pInput + (ch * inDataPitch2));
        for (x = 0; x <= (inDataWidth - vectorizationWidth); x += (vectorizationWidth))
        {
          pInput2  = (pInput1 + x);
          pOutput2 = (pOutput1 + (x * strideX));
          pdvecIn1 = (xb_vec2Nx8 *) pInput2;
          for (y = 0; y < inDataHeight; y++)
          {
            vaInData1 = IVP_LA2NX8_PP(pdvecIn1);
            IVP_LA2NX8_XP(dvecData1, vaInData1, pdvecIn1, inDataPitch1);
            IVP_SCATTER2NX8_L(dvecData1, pOutput2, vecScatterOff1);
            IVP_SCATTER2NX8_H(dvecData1, pOutput2, vecScatterOff2);
            pOutput2 += outDataPitch1Offset;
          }
        }
        /*To perform Interleaving for inputData widths that are less than the vectorization width*/
        if (inDataWidth - x)
        {
          pInput2  = (pInput1 + x);
          pOutput2 = ((pOutput1 + (x * strideX)));
          pdvecIn1 = (xb_vec2Nx8 *) pInput2;
          remX     = (inDataWidth - x);
          /*Creating Mask to scatter only the availble valid inputs that should be interleaved*/
          vecMsk = IVP_LT2NX8(vecCmp, remX);
          /*Creating Mask for the scatter operation to have only valid offsets based on the available inputs*/
          vecOffsetMsk1 = IVP_LTNX16(vecOffsetCmp, remX);
          vecOffsetMsk2 = IVP_LTNX16(vecOffsetCmp, (remX - XCHAL_IVPN_SIMD_WIDTH));
          for (y = 0; y < inDataHeight; y++)
          {
            vaInData1 = IVP_LA2NX8_PP(pdvecIn1);
            IVP_LA2NX8_XP(dvecData1, vaInData1, pdvecIn1, inDataPitch1);
            IVP_SCATTER2NX8T_L(dvecData1, pOutput2, IVP_MOVNX16T(vecScatterOff1, 0, vecOffsetMsk1), (vecMsk));
            IVP_SCATTER2NX8T_H(dvecData1, pOutput2, IVP_MOVNX16T(vecScatterOff2, 0, vecOffsetMsk2), (vecMsk));
            pOutput2 += outDataPitch1Offset;
          }
        }
      }
    }
  }

  IVP_SCATTERW();  /* Adding Memory Wait until all the scatter and store operations are completed */

  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Vision P6 implementation for interleaving the outputs      */
/*               generated by convolution functions using the sub-kernels   */
/* Inputs      : array of output tiles passed as input, CNN convolution     */
/*               params structure, output tile                              */
/* Outputs     : XI Error Code                                              */
/* InOuts      : output tile                                                */
/* Assumptions : Input Tile Data is S8/U8                                   */
/****************************************************************************/
XAI_ERR_TYPE xaiDepthwiseDeConvInterleave3D_I8_WHD(const xai_pTile3D inTile[],
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_depthwiseDilatedConv_params *convParams)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(inTile);
    XAI_CHECK_POINTER(convParams);
    XAI_CHECK_TILE3D_I8(outTile);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_WHD);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(outTile);
  }
  /* Getting parameters from the tile structures */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const uint8_t strideX = XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEX(convParams);
  const uint8_t strideY = XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDEY(convParams);

  const int32_t outDataPitch1Offset = (outDataPitch1 * strideY);

  int8_t *pOutput = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int32_t ch, x, y, numX, numY, idx, remX;
  int8_t *pSubKernelOutput;
  int8_t *pOutput1;
  int8_t *pOutput2;
  int8_t *pInput1;
  int8_t *pInput2;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_ERROR(((strideX > 0) && (strideY > 0)),
                    XAI_ERR_BADARG, "strideX = %hhu, strideY = %hhu\nStride has to be >= 1", \
                    strideX, strideY);

    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outTile) >= strideX) &&                      \
                    (XAI_TILE3D_GET_DIM2(outTile) >= strideY), XAI_ERR_BADARG,        \
                    "\nOutTile width = %d, value must be greater than or equal to %hhu(strideX) \
       \nOutTile height = %d,  value must be greater than or equal to %hhu(strideY)", \
                    XAI_TILE3D_GET_DIM1(outTile), strideX, XAI_TILE3D_GET_DIM2(outTile), strideY);

    for (numY = 0; numY < strideY; numY++)
    {
      for (numX = 0; numX < strideX; numX++)
      {
        idx = numX + numY * strideX;
        XAI_CHECK_POINTER(inTile[idx]);
        XAI_CHECK_TILE3D_I8(inTile[idx]);
        XAI_CHECK_TILE3D_DATA_ORDER(inTile[idx], XAI_WHD);
        XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile[idx], outTile);
        XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inTile[idx]) == XAI_TILE3D_GET_DIM3(outTile), XAI_ERR_BADARG, \
                        "\nNumber of channels of each subkernel output = %d, final output = %d \
           \nNumber of channels of each subkernel output and final output should be the same",            \
                        XAI_TILE3D_GET_DIM3(inTile[idx]), XAI_TILE3D_GET_DIM3(outTile));
      }
    }
  }

  /* Scatter Index Calculations */
  /* Sequence - 0 1 2 3 4 ... 30 31 */
  xb_vecNx16U vecSelIdx1 = IVP_SEQNX16U();
  /* Sequence - 0 strideX 2*strideX 3*strideX 4*strideX .... 30*strideX 31*strideX*/
  xb_vecNx16U vecScatterOff1 = IVP_MULNX16UPACKL(vecSelIdx1, \
                                                 (uint16_t) strideX);

  xb_vecNx16U vecScatterOff2;
  /* Sequence - (32*strideX) (33*strideX) (34*strideX) ....(62*strideX) (63*strideX)*/
  vecScatterOff2 = IVP_ADDNX16(vecScatterOff1, (XCHAL_IVPN_SIMD_WIDTH * strideX));

  xb_vec2Nx8* restrict pdvecIn1;
  xb_vec2Nx8 dvecData1;
  valign vaInData1;
  vbool2N vecMsk;
  vboolN vecOffsetMsk1;
  vboolN vecOffsetMsk2;
  /* Sequence - 0 1 2 3 4 ... 62 63 */
  xb_vec2Nx8 vecCmp = IVP_SEQ2NX8U();
  /* Sequence - 0 1 2 3 4 ... 30 31 */
  xb_vecNx16U vecOffsetCmp = IVP_SEQNX16U();

  const int32_t vectorizationWidth = 2 * XCHAL_IVPN_SIMD_WIDTH;

  for (numY = 0; numY < strideY; numY++)
  {
    for (numX = 0; numX < strideX; numX++)
    {
      idx = numX + numY * strideX;
      int8_t *pInput             = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile[idx]);
      const int32_t inDataWidth  = XAI_TILE3D_GET_DIM1(inTile[idx]);
      const int32_t inDataHeight = XAI_TILE3D_GET_DIM2(inTile[idx]);
      const int32_t inChanNum    = XAI_TILE3D_GET_DIM3(inTile[idx]);
      const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile[idx]);
      const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile[idx]);
      pSubKernelOutput = (pOutput + numX + (numY * outDataPitch1));
      for (ch = 0; ch < inChanNum; ch++)
      {
        pOutput1 = (pSubKernelOutput + (ch * outDataPitch2));
        pInput1  = (pInput + (ch * inDataPitch2));
        for (x = 0; x <= (inDataWidth - vectorizationWidth); x += (vectorizationWidth))
        {
          pInput2  = (pInput1 + x);
          pOutput2 = (pOutput1 + (x * strideX));
          pdvecIn1 = (xb_vec2Nx8 *) pInput2;
          for (y = 0; y < inDataHeight; y++)
          {
            vaInData1 = IVP_LA2NX8_PP(pdvecIn1);
            IVP_LA2NX8_XP(dvecData1, vaInData1, pdvecIn1, inDataPitch1);
            IVP_SCATTER2NX8_L(dvecData1, pOutput2, vecScatterOff1);
            IVP_SCATTER2NX8_H(dvecData1, pOutput2, vecScatterOff2);
            pOutput2 += outDataPitch1Offset;
          }
        }
        /*To perform Interleaving for inputData widths that are less than the vectorization width*/
        if (inDataWidth - x)
        {
          pInput2  = (pInput1 + x);
          pOutput2 = ((pOutput1 + (x * strideX)));
          pdvecIn1 = (xb_vec2Nx8 *) pInput2;
          remX     = (inDataWidth - x);
          /*Creating Mask to scatter only the availble valid inputs that should be interleaved*/
          vecMsk = IVP_LT2NX8(vecCmp, remX);
          /*Creating Mask for the scatter operation to have only valid offsets based on the available inputs*/
          vecOffsetMsk1 = IVP_LTNX16(vecOffsetCmp, remX);
          vecOffsetMsk2 = IVP_LTNX16(vecOffsetCmp, (remX - XCHAL_IVPN_SIMD_WIDTH));
          for (y = 0; y < inDataHeight; y++)
          {
            vaInData1 = IVP_LA2NX8_PP(pdvecIn1);
            IVP_LA2NX8_XP(dvecData1, vaInData1, pdvecIn1, inDataPitch1);
            IVP_SCATTER2NX8T_L(dvecData1, pOutput2, IVP_MOVNX16T(vecScatterOff1, 0, vecOffsetMsk1), (vecMsk));
            IVP_SCATTER2NX8T_H(dvecData1, pOutput2, IVP_MOVNX16T(vecScatterOff2, 0, vecOffsetMsk2), (vecMsk));
            pOutput2 += outDataPitch1Offset;
          }
        }
      }
    }
  }

  IVP_SCATTERW();  /* Adding Memory Wait until all the scatter and store operations are completed */

  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Vision P6 implementation for interleaving the outputs      */
/*               generated by convolution functions using the sub-kernels   */
/* Inputs      : array of output tiles passed as input, CNN convolution     */
/*               params structure, output tile                              */
/* Outputs     : XI Error Code                                              */
/* InOuts      : output tile                                                */
/* Assumptions : Input Tile Data is I16                                     */
/****************************************************************************/

XAI_ERR_TYPE xaiDeConvInterleave3D_I16_WHD(const xai_pTile3D inTile[],
                                           xai_pTile3D outTile,
                                           const xai_cnn_conv_params *convParams)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(inTile);
    XAI_CHECK_POINTER(convParams);
    XAI_CHECK_TILE3D_I16(outTile);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_WHD);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(outTile);
  }

  /* Getting parameters from the tile structures */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const uint8_t strideX             = XAI_CNN_CONV_GET_STRIDEX(convParams);
  const uint8_t strideY             = XAI_CNN_CONV_GET_STRIDEY(convParams);
  const int32_t outDataPitch1Offset = (outDataPitch1 * strideY);

  int16_t *pOutput = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int32_t ch, x, y, numX, numY, idx, remX;
  int16_t *pSubKernelOutput;
  int16_t *pOutput1;
  int16_t *pOutput2;
  int16_t *pInput1;
  int16_t *pInput2;


  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_ERROR(((strideX > 0) && (strideY > 0)),
                    XAI_ERR_BADARG, "strideX = %hhu, strideY = %hhu\nStride has to be >= 1", \
                    strideX, strideY);

    XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(outTile) >= strideX) &&                      \
                    (XAI_TILE3D_GET_DIM2(outTile) >= strideY), XAI_ERR_BADARG,        \
                    "\nOutTile width = %d, value must be greater than or equal to %hhu(strideX) \
       \nOutTile height = %d,  value must be greater than or equal to %hhu(strideY)", \
                    XAI_TILE3D_GET_DIM1(outTile), strideX, XAI_TILE3D_GET_DIM2(outTile), strideY);

    for (numY = 0; numY < strideY; numY++)
    {
      for (numX = 0; numX < strideX; numX++)
      {
        idx = numX + numY * strideX;
        XAI_CHECK_POINTER(inTile[idx]);
        XAI_CHECK_TILE3D_I16(inTile[idx]);
        XAI_CHECK_TILE3D_DATA_ORDER(inTile[idx], XAI_WHD);
        XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile[idx], outTile);
        XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM3(inTile[idx]) == XAI_TILE3D_GET_DIM3(outTile), XAI_ERR_BADARG, \
                        "\nNumber of channels of each subkernel output = %d, final output = %d \
           \nNumber of channels of each subkernel output and final output should be the same",            \
                        XAI_TILE3D_GET_DIM3(inTile[idx]), XAI_TILE3D_GET_DIM3(outTile));
      }
    }
  }

  /* Scatter Index Calculations */
  /* Sequence - 0 1 2 3 4 ... 30 31 */
  xb_vecNx16U vecSelIdx1 = IVP_SEQNX16U();
  /* Sequence - 0 strideX 2*strideX 3*strideX 4*strideX .... 30*strideX 31*strideX*/
  xb_vecNx16U vecScatterOff1 = IVP_MULNX16UPACKL(vecSelIdx1, \
                                                 (uint16_t) strideX * 2);

  xb_vecNx16* restrict pdvecIn1;
  xb_vecNx16 dvecData1;
  valign vaInData1;
  vboolN vecMsk;
  vboolN vecOffsetMsk1;
  /* Sequence - 0 1 2 3 4 ... 30 31 */
  xb_vecNx16U vecCmp       = IVP_SEQNX16U();
  xb_vecNx16U vecOffsetCmp = IVP_SEQNX16U();


  const int32_t vectorizationWidth = XCHAL_IVPN_SIMD_WIDTH;

  for (numY = 0; numY < strideY; numY++)
  {
    for (numX = 0; numX < strideX; numX++)
    {
      idx = numX + numY * strideX;
      int16_t *pInput            = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile[idx]);
      const int32_t inDataWidth  = XAI_TILE3D_GET_DIM1(inTile[idx]);
      const int32_t inDataHeight = XAI_TILE3D_GET_DIM2(inTile[idx]);
      const int32_t inChanNum    = XAI_TILE3D_GET_DIM3(inTile[idx]);
      const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile[idx]);
      const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile[idx]);
      pSubKernelOutput = (pOutput + numX + (numY * outDataPitch1));
      for (ch = 0; ch < inChanNum; ch++)
      {
        pOutput1 = (pSubKernelOutput + (ch * outDataPitch2));
        pInput1  = (pInput + (ch * inDataPitch2));
        for (x = 0; x <= (inDataWidth - vectorizationWidth); x += (vectorizationWidth))
        {
          pInput2  = (pInput1 + x);
          pOutput2 = (pOutput1 + (x * strideX));
          pdvecIn1 = (xb_vecNx16 *) pInput2;
          for (y = 0; y < inDataHeight; y++)
          {
            vaInData1 = IVP_LANX16_PP(pdvecIn1);
            IVP_LANX16_XP(dvecData1, vaInData1, pdvecIn1, (inDataPitch1 << 1));
            IVP_SCATTERNX16(dvecData1, pOutput2, vecScatterOff1);
            pOutput2 += outDataPitch1Offset;
          }
        }
        /*To perform Interleaving for inputData widths that are less than the vectorization width*/
        if (inDataWidth - x)
        {
          pInput2  = (pInput1 + x);
          pOutput2 = ((pOutput1 + (x * strideX)));
          pdvecIn1 = (xb_vecNx16 *) (pInput2);
          remX     = (inDataWidth - x);
          /*Creating Mask to scatter only the availble valid inputs that should be interleaved*/
          vecMsk = IVP_LTNX16(vecCmp, remX);
          /*Creating Mask for the scatter operation to have only valid offsets based on the available inputs*/
          vecOffsetMsk1 = IVP_LTNX16(vecOffsetCmp, remX);
          for (y = 0; y < inDataHeight; y++)
          {
            vaInData1 = IVP_LANX16_PP(pdvecIn1);
            IVP_LANX16_XP(dvecData1, vaInData1, pdvecIn1, (inDataPitch1 << 1));
            IVP_SCATTERNX16T(dvecData1, pOutput2, IVP_MOVNX16T(vecScatterOff1, 0, vecOffsetMsk1), (vecMsk));
            pOutput2 += outDataPitch1Offset;
          }
        }
      }
    }
  }

  IVP_SCATTERW();  /* Adding Memory Wait until all the scatter and store operations are completed */

  return(XAI_ERROR_STATUS());
}

/**********************xaiConvolvedBiasUpdate_S8S32*************************/
/* Description : Implementation of BiasUpdate calculation for             */
/*               It modifies the bias value by adding a fixup             */
/*               term to it. This function is called along with,          */
/*               Convolved3D_MOD functions which accepts U8 input tile    */
/*               and converts to S8 and also S8 coeff tile                */
/* Inputs      : Coeff Tile                                               */
/* InOuts      : biasArray                                                */
/* Assumptions : coeffData is S8 and biasData is S32                      */
/*               Coefficient tile is in NDWH format                       */
/**************************************************************************/
XAI_ERR_TYPE xaiConvolvedBiasUpdate_S8S32(const xai_pTile4D coeffTile,
                                          xai_pArray biasArray
                                          )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM1(coeffTile) <= XAI_ARRAY_GET_WIDTH(biasArray)), XAI_ERR_BADARG,                                        \
                    "\nNumber of Kernels = %d, Width of Bias Array = %d\nNumber of Kernels must be less than or equal to Width of Bias Array", \
                    XAI_TILE4D_GET_DIM1(coeffTile), XAI_ARRAY_GET_WIDTH(biasArray));
  }
#ifndef IVP_MULSUQA2N8XR8
  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pCoeff = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBias = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  /* Vector Pointers */
  xb_vec2Nx8*    restrict pdvecCoeff;
  xb_vecN_2x32v* restrict phvecBias = (xb_vecN_2x32v *) (pBias);
  xb_vecN_2x32v* phvecBiasIn        = phvecBias;
  xb_vecN_2x32v* phvecBiasOut       = phvecBias;
  valign vaInBias                   = IVP_LAN_2X32_PP(phvecBiasIn);
  valign vaOutBias                  = IVP_ZALIGN();

  /* Getting parameters from the tile structures */
  const int32_t outChanNum      = XAI_TILE4D_GET_DIM1(coeffTile);
  const int32_t inChanNum       = XAI_TILE4D_GET_DIM2(coeffTile);
  const uint16_t kWidthU        = XAI_TILE4D_GET_DIM3(coeffTile);
  const uint16_t kHeightU       = XAI_TILE4D_GET_DIM4(coeffTile);
  const int32_t coeffDataPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffDataPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffDataPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);
  int32_t accOverflowFlag       = 0;

  int32_t outCh, kx, ky, inCh;
  /*
     IF inputdata is S8
     convolutionS8 = summation(InputData * CoeffData)
     IF inputdata is U8
     convolutionU8 = summation((InputData - 128) * CoeffData) + summation(128 * CoeffData)
                = convolutionS8 + summation(128 * CoeffData)
                = convolutionS8 + 128 * summation( CoeffData)
                  128 * summation( CoeffData) is performed below
   */

  const int32_t vectorizationWidth = (XCHAL_IVPN_SIMD_WIDTH << 1);

  /* Iterate Over OutChannels */
  for (outCh = 0; outCh < outChanNum; outCh += vectorizationWidth)
  {
    /* Calculate remaining output channels */
    int32_t remOutCh = (outChanNum - outCh);

    /* Initialize Accumulator Vector */
    xb_vec2Nx24 daccSum = IVP_ZERO2NX24();

    /* Computes the sum of coeffs corresponding to the same outChannel */
    for (ky = 0; ky < kHeightU; ky++)
    {
      for (kx = 0; kx < kWidthU; kx++)
      {
        int32_t coeffIdx = outCh + kx * coeffDataPitch2 + ky * coeffDataPitch3;
        pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffIdx);

        for (inCh = 0; inCh < inChanNum - 3; inCh += 4)
        {
          xb_vec2Nx8 dvecCoeff1, dvecCoeff2, dvecCoeff3, dvecCoeff4;

          IVP_L2U2NX8_XP(dvecCoeff1, pdvecCoeff, coeffDataPitch1);
          IVP_L2U2NX8_XP(dvecCoeff2, pdvecCoeff, coeffDataPitch1);
          IVP_L2U2NX8_XP(dvecCoeff3, pdvecCoeff, coeffDataPitch1);
          IVP_L2U2NX8_XP(dvecCoeff4, pdvecCoeff, coeffDataPitch1);

          IVP_ADDWA2NX8(daccSum, dvecCoeff2, dvecCoeff1);
          IVP_ADDWA2NX8(daccSum, dvecCoeff4, dvecCoeff3);
        }
        for (; inCh < inChanNum - 1; inCh += 2)
        {
          xb_vec2Nx8 dvecCoeff1, dvecCoeff2;

          IVP_L2U2NX8_XP(dvecCoeff1, pdvecCoeff, coeffDataPitch1);
          IVP_L2U2NX8_XP(dvecCoeff2, pdvecCoeff, coeffDataPitch1);

          IVP_ADDWA2NX8(daccSum, dvecCoeff2, dvecCoeff1);
        }
        if (inCh < inChanNum)
        {
          xb_vec2Nx8 dvecCoeff;

          IVP_L2U2NX8_XP(dvecCoeff, pdvecCoeff, coeffDataPitch1);

          IVP_ADDWA2NX8(daccSum, (xb_vec2Nx8) 0, dvecCoeff);
        }
      }
    }

    /* Add Adjustment for Bias to Bias Vectors */
    xb_vecN_2x32v hvecBiasLL, hvecBiasLH, hvecBiasHL, hvecBiasHH;
    int32_t remBiasBytes = remOutCh * 4;

    /* Number of channels processed by N_2-way 32-bit vector */
    const int32_t numProcessCh = XCHAL_IVPN_SIMD_WIDTH >> 1;

    /* Convert Accumulated Double Accumulator Values to 4 Half Vectors */
    xb_vecN_2x32v hvecAccLL, hvecAccLH, hvecAccHL, hvecAccHH;
    hvecAccLL = IVP_CVT32S2NX24LL(daccSum); hvecAccLL = IVP_SLAN_2X32(hvecAccLL, 7);
    hvecAccLH = IVP_CVT32S2NX24LH(daccSum); hvecAccLH = IVP_SLAN_2X32(hvecAccLH, 7);
    hvecAccHL = IVP_CVT32S2NX24HL(daccSum); hvecAccHL = IVP_SLAN_2X32(hvecAccHL, 7);
    hvecAccHH = IVP_CVT32S2NX24HH(daccSum); hvecAccHH = IVP_SLAN_2X32(hvecAccHH, 7);

    hvecAccLL = IVP_MOVN_2X32T(hvecAccLL, (xb_vecN_2x32v) 0, \
                               IVP_LTN_2X32(IVP_SEQN_2X32(), (xb_vecN_2x32v) (remOutCh)));
    hvecAccLH = IVP_MOVN_2X32T(hvecAccLH, (xb_vecN_2x32v) 0, \
                               IVP_LTN_2X32(IVP_SEQN_2X32(), (xb_vecN_2x32v) (remOutCh - (numProcessCh))));
    hvecAccHL = IVP_MOVN_2X32T(hvecAccHL, (xb_vecN_2x32v) 0, \
                               IVP_LTN_2X32(IVP_SEQN_2X32(), (xb_vecN_2x32v) (remOutCh - (2 * numProcessCh))));
    hvecAccHH = IVP_MOVN_2X32T(hvecAccHH, (xb_vecN_2x32v) 0, \
                               IVP_LTN_2X32(IVP_SEQN_2X32(), (xb_vecN_2x32v) (remOutCh - (3 * numProcessCh))));

    IVP_LAVN_2X32_XP(hvecBiasLL, vaInBias, phvecBiasIn, remBiasBytes);
    IVP_LAVN_2X32_XP(hvecBiasLH, vaInBias, phvecBiasIn, remBiasBytes - (2 * XCHAL_IVPN_SIMD_WIDTH));
    IVP_LAVN_2X32_XP(hvecBiasHL, vaInBias, phvecBiasIn, remBiasBytes - (4 * XCHAL_IVPN_SIMD_WIDTH));
    IVP_LAVN_2X32_XP(hvecBiasHH, vaInBias, phvecBiasIn, remBiasBytes - (6 * XCHAL_IVPN_SIMD_WIDTH));

    /* Add Bias and its Adjustment */
    hvecBiasLL = IVP_ADDN_2X32(hvecBiasLL, hvecAccLL);
    hvecBiasLH = IVP_ADDN_2X32(hvecBiasLH, hvecAccLH);
    hvecBiasHL = IVP_ADDN_2X32(hvecBiasHL, hvecAccHL);
    hvecBiasHH = IVP_ADDN_2X32(hvecBiasHH, hvecAccHH);

    /* Check If Overflow is present and perform shifts as per requirement*/
    vboolN_2 hvbOverflow;

    /* hvecBiasLL */
    hvbOverflow      = IVP_ORBN_2(IVP_LTN_2X32(hvecBiasLL, S24_MIN), IVP_LTN_2X32(S24_MAX, hvecBiasLL));
    accOverflowFlag += (int32_t) IVP_RADDN_2X32T((xb_vecN_2x32v) 1, hvbOverflow);
    hvecBiasLL       = IVP_SLAN_2X32(hvecBiasLL, IVP_MOVN_2X32T((xb_vecN_2x32v) (8), (xb_vecN_2x32v) 0, hvbOverflow));
    hvecBiasLL       = IVP_SLAN_2X32(hvecBiasLL, IVP_MOVN_2X32T((xb_vecN_2x32v) (-8), (xb_vecN_2x32v) 0, hvbOverflow));

    /* hvecBiasLH */
    hvbOverflow      = IVP_ORBN_2(IVP_LTN_2X32(hvecBiasLH, S24_MIN), IVP_LTN_2X32(S24_MAX, hvecBiasLH));
    accOverflowFlag += (int32_t) IVP_RADDN_2X32T((xb_vecN_2x32v) 1, hvbOverflow);
    hvecBiasLH       = IVP_SLAN_2X32(hvecBiasLH, IVP_MOVN_2X32T((xb_vecN_2x32v) (8), (xb_vecN_2x32v) 0, hvbOverflow));
    hvecBiasLH       = IVP_SLAN_2X32(hvecBiasLH, IVP_MOVN_2X32T((xb_vecN_2x32v) (-8), (xb_vecN_2x32v) 0, hvbOverflow));

    /* hvecBiasHL */
    hvbOverflow      = IVP_ORBN_2(IVP_LTN_2X32(hvecBiasHL, S24_MIN), IVP_LTN_2X32(S24_MAX, hvecBiasHL));
    accOverflowFlag += (int32_t) IVP_RADDN_2X32T((xb_vecN_2x32v) 1, hvbOverflow);
    hvecBiasHL       = IVP_SLAN_2X32(hvecBiasHL, IVP_MOVN_2X32T((xb_vecN_2x32v) (8), (xb_vecN_2x32v) 0, hvbOverflow));
    hvecBiasHL       = IVP_SLAN_2X32(hvecBiasHL, IVP_MOVN_2X32T((xb_vecN_2x32v) (-8), (xb_vecN_2x32v) 0, hvbOverflow));

    /* hvecBiasHH */
    hvbOverflow      = IVP_ORBN_2(IVP_LTN_2X32(hvecBiasHH, S24_MIN), IVP_LTN_2X32(S24_MAX, hvecBiasHH));
    accOverflowFlag += (int32_t) IVP_RADDN_2X32T((xb_vecN_2x32v) 1, hvbOverflow);
    hvecBiasHH       = IVP_SLAN_2X32(hvecBiasHH, IVP_MOVN_2X32T((xb_vecN_2x32v) (8), (xb_vecN_2x32v) 0, hvbOverflow));
    hvecBiasHH       = IVP_SLAN_2X32(hvecBiasHH, IVP_MOVN_2X32T((xb_vecN_2x32v) (-8), (xb_vecN_2x32v) 0, hvbOverflow));

    /* Store Updated Bias */
    IVP_SAVN_2X32_XP(hvecBiasLL, vaOutBias, phvecBiasOut, remBiasBytes);
    IVP_SAVN_2X32_XP(hvecBiasLH, vaOutBias, phvecBiasOut, remBiasBytes - (2 * XCHAL_IVPN_SIMD_WIDTH));
    IVP_SAVN_2X32_XP(hvecBiasHL, vaOutBias, phvecBiasOut, remBiasBytes - (4 * XCHAL_IVPN_SIMD_WIDTH));
    IVP_SAVN_2X32_XP(hvecBiasHH, vaOutBias, phvecBiasOut, remBiasBytes - (6 * XCHAL_IVPN_SIMD_WIDTH));
  }

  IVP_SAPOSN_2X32_FP(vaOutBias, phvecBiasOut);

  if (accOverflowFlag)
  {
    return(XAI_ERR_OVERFLOW);
  }
#endif
  return(XAI_ERROR_STATUS());
}

/************************  xaiReOrder4DToIN32DWH_I16  ***********************/
/* Description : C-code implementation to reorder a tile from WHDN,        */
/*               DWHN or NDWH into IN32DWH format                          */
/* Inputs      : Coeff Tile  in WHDN or DWHN or NDWH format                */
/* Outputs     : Coeff Array in IN32DWH format                             */
/* Assumptions : The width and height of the coefficient tile are 1        */
/*               Input and Output tiles can be S16 / U16                   */
/***************************************************************************/
XAI_ERR_TYPE xaiReOrder4DToIN32DWH_I16(xai_pTile4D coeffTileIn, xai_pTile4D coeffTileOut)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE4D_I16(coeffTileIn);
    XAI_CHECK_TILE4D_I16(coeffTileOut);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_WHDN) ||                                                       \
                    (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_DWHN) || (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_NDWH), \
                    XAI_ERR_BADARG, "The Data Order of the input  is not supported by this function");
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTileOut, XAI_IN32DWH);
    XAI_CHECK_DIM_IN32DWH(coeffTileIn, coeffTileOut);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DATA_PTR(coeffTileIn) != XAI_ARRAY_GET_DATA_PTR(coeffTileOut)), XAI_ERR_INPLACE, "The input and output tile pointers overlap");
  }

  int32_t numInCh, numOutCh, minCh, coeffInPitch1, coeffInPitch3;

  if (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_WHDN)
  {
    numInCh       = XAI_TILE4D_GET_DIM3(coeffTileIn);
    numOutCh      = XAI_TILE4D_GET_DIM4(coeffTileIn);
    coeffInPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTileIn);
    coeffInPitch1 = 1;
  }
  else if (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_DWHN)
  {
    numInCh       = XAI_TILE4D_GET_DIM1(coeffTileIn);
    numOutCh      = XAI_TILE4D_GET_DIM4(coeffTileIn);
    coeffInPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTileIn);
    coeffInPitch1 = 1;
  }
  else /* If coeff tile NDWH */
  {
    numInCh       = XAI_TILE4D_GET_DIM2(coeffTileIn);
    numOutCh      = XAI_TILE4D_GET_DIM1(coeffTileIn);
    coeffInPitch3 = 1;
    coeffInPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTileIn);
  }

  int16_t *pCoeff    = (int16_t *) XAI_TILE4D_GET_DATA_PTR(coeffTileIn);
  int16_t *pCoeffOut = (int16_t *) XAI_ARRAY_GET_DATA_PTR(coeffTileOut);
  int32_t i, j, k;

  /* Reorder Coeff tile */
  /*
     The coefficient tile is reordered in the format IN64DWH:
     d0_0,....d0_31, d1_0,...d1_31, ....dN_0,...dN_31, d0_32,....d0_63, d1_32,...d1_63, ....dN_32,...dN_63,
     d0_64,....d0_95, d1_64,...d1_95, ....dN_64,...dN_95,...

     Here, d0, d1,....dN are input channels.
     where 'N' is the total input channels.
     d0_0 => 0_0 => inputChNumber_outputChNumber
   */

  for (i = 0; i < numOutCh; i += XCHAL_IVPN_SIMD_WIDTH)
  {
    for (j = 0; j < numInCh; j++)
    {
      minCh = (numOutCh - i) >= XCHAL_IVPN_SIMD_WIDTH ? XCHAL_IVPN_SIMD_WIDTH : (numOutCh - i);
      for (k = 0; k < minCh; k++)
      {
        int16_t val = *(pCoeff + (k + i) * coeffInPitch3 + j * coeffInPitch1);
        *(pCoeffOut + k + (j * XCHAL_IVPN_SIMD_WIDTH) + i * numInCh) = val;
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/*********************** xaiReOrder4DToIN64DWH_I8 ***************************/
/* Description : C-code implementation to reorder a tile from WHDN,        */
/*               DWHN or NDWH into IN64DWH format                          */
/* Inputs      : Coeff Tile  in WHDN or DWHN or NDWH format                */
/* Outputs     : Coeff Array in IN64DWH format                             */
/* Assumptions : The width and height of the coefficient tile are 1        */
/*               Input and Output tiles can be S16 / U16                   */
/***************************************************************************/
XAI_ERR_TYPE xaiReOrder4DToIN64DWH_I8(xai_pTile4D coeffTileIn, xai_pTile4D coeffTileOut)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE4D_I8(coeffTileIn);
    XAI_CHECK_TILE4D_I8(coeffTileOut);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_WHDN) ||                                                       \
                    (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_DWHN) || (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_NDWH), \
                    XAI_ERR_BADARG, "The Data Order of the input  is not supported by this function");
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTileOut, XAI_IN64DWH);
    XAI_CHECK_DIM_IN64DWH(coeffTileIn, coeffTileOut);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DATA_PTR(coeffTileIn) != XAI_ARRAY_GET_DATA_PTR(coeffTileOut)), XAI_ERR_INPLACE, "The input and output tile pointers overlap");
  }
  int32_t numInCh, numOutCh, minCh, coeffInPitch1, coeffInPitch3;

  if (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_WHDN)
  {
    numInCh       = XAI_TILE4D_GET_DIM3(coeffTileIn);
    numOutCh      = XAI_TILE4D_GET_DIM4(coeffTileIn);
    coeffInPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTileIn);
    coeffInPitch1 = 1;
  }
  else if (XAI_TILE4D_GET_DATA_ORDER(coeffTileIn) == XAI_DWHN)
  {
    numInCh       = XAI_TILE4D_GET_DIM1(coeffTileIn);
    numOutCh      = XAI_TILE4D_GET_DIM4(coeffTileIn);
    coeffInPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTileIn);
    coeffInPitch1 = 1;
  }
  else /* If coeff tile NDWH */
  {
    numInCh       = XAI_TILE4D_GET_DIM2(coeffTileIn);
    numOutCh      = XAI_TILE4D_GET_DIM1(coeffTileIn);
    coeffInPitch3 = 1;
    coeffInPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTileIn);
  }

  int8_t *pCoeff    = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTileIn);
  int8_t *pCoeffOut = (int8_t *) XAI_ARRAY_GET_DATA_PTR(coeffTileOut);
  int32_t i, j, k;

  /* Reorder Coeff tile */
  /*
     The coefficient tile is reordered in the format IN64DWH:
     d0_0,....d0_63, d1_0,...d1_63, ....dN_0,...dN_63, d0_64,....d0_127, d1_64,...d1_127, ....dN_64,...dN_127,
     d0_128,....d0_191, d1_128,...d1_191, ....dN_128,...dN_191,...

     Here, d0, d1,....dN are input channels.
     where 'N' is the total input channels.
     d0_0 => 0_0 => inputChNumber_outputChNumber
   */

  for (i = 0; i < numOutCh; i += 2 * XCHAL_IVPN_SIMD_WIDTH)
  {
    for (j = 0; j < numInCh; j++)
    {
      minCh = (numOutCh - i) >= (2 * XCHAL_IVPN_SIMD_WIDTH) ? (2 * XCHAL_IVPN_SIMD_WIDTH) : (numOutCh - i);
      for (k = 0; k < minCh; k++)
      {
        int8_t val = *(pCoeff + (k + i) * coeffInPitch3 + j * coeffInPitch1);
        *(pCoeffOut + k + (j * 2 * XCHAL_IVPN_SIMD_WIDTH) + i * numInCh) = val;
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

#if 0 //(XCHAL_HAVE_VISION_HP_VFPU == 1) // Disabled the F16 helper APIs which are not used anywhere

/****************************************************************************/
/* Description : Implementation for extending the bias array in             */
/*               case of MOD deconvolution using superkernels.              */
/* Inputs      : Input Bias array,                                          */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Bias array                                          */
/****************************************************************************/
XAI_ERR_TYPE xaiBiasExtend_F16_MOD(const xai_pArray inBiasArray,
                                   xai_pArray outBiasArray)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_ARRAY_F16(inBiasArray);
    XAI_CHECK_ARRAY_F16(outBiasArray);
  }

  int32_t inWidth  = XAI_ARRAY_GET_WIDTH(inBiasArray);
  int32_t outWidth = XAI_ARRAY_GET_WIDTH(outBiasArray);
  int32_t strideX  = outWidth / inWidth;

  xb_f16* pInBias  = (xb_f16 *) XAI_ARRAY_GET_DATA_PTR(inBiasArray);
  xb_f16* pOutBias = (xb_f16 *) XAI_ARRAY_GET_DATA_PTR(outBiasArray);

  int32_t numX, inW;
  for (numX = 0; numX < strideX; numX++)
  {
    for (inW = 0; inW < inWidth; inW++)
    {
      pOutBias[inW + inWidth * numX] = pInBias[inW];
    }
  }
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************/
/* Description : Implementation for extending the outputscale array          */
/*               in case of MOD deconvolution using superkernels.            */
/* Inputs      : outputScale array,                                          */
/* Outputs     : XI Error Code                                               */
/* InOuts      : extended outputScale array                                  */
/*****************************************************************************/
XAI_ERR_TYPE xaiOutScaleExtend_F16_MOD(const xai_pArray outScaleArray,
                                       xai_pArray extendedOutScaleArray)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_ARRAY_F16(outScaleArray);
    XAI_CHECK_ARRAY_F16(extendedOutScaleArray);
  }

  int32_t inWidth  = XAI_ARRAY_GET_WIDTH(outScaleArray);
  int32_t outWidth = XAI_ARRAY_GET_WIDTH(extendedOutScaleArray);
  int32_t strideX  = outWidth / inWidth;

  xb_f16* pInScale  = (xb_f16 *) XAI_ARRAY_GET_DATA_PTR(outScaleArray);
  xb_f16* pOutScale = (xb_f16 *) XAI_ARRAY_GET_DATA_PTR(extendedOutScaleArray);

  int32_t numX, inW;
  for (numX = 0; numX < strideX; numX++)
  {
    for (inW = 0; inW < inWidth; inW++)
    {
      pOutScale[inW + inWidth * numX] = pInScale[inW];
    }
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : Implementation for coefficient reordering                  */
/*               The functions does the following:                          */
/*               - Convert from NDWH->DNWH                                  */
/*               - Flips the coefficients across width and height which is  */
/*                 controlled by transposeCoeffsFlag.                       */
/*               - Breaks the kernel into sub-kernels.                      */
/*               - Stacks sub-kernels to form super kernels.                */
/* Inputs      : Input Coeff Tile, CNN convolution params structure,        */
/*               transposeCoeffsFlag                                        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Array of Coeff Sub & Super Tiles                           */
/* Assumptions : CoeffData is F16                                           */
/*               Coeff is in NDWH format                                    */
/****************************************************************************/
XAI_ERR_TYPE xaiDeConvReOrder4D_F16_NDWH(const xai_pTile4D inTile,
                                         xai_pTile4D subCoeffs[],
                                         xai_pTile4D superCoeffs[],
                                         const xai_cnn_conv_params *param,
                                         const uint8_t transposeCoeffsFlag)
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE4D_F16(inTile);
    XAI_CHECK_TILE4D_DATA_ORDER(inTile, XAI_NDWH);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_POINTER(subCoeffs);
    XAI_CHECK_POINTER(superCoeffs);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) >= 1) &&                                                  \
                     (XAI_CNN_CONV_GET_STRIDEX(param) <= XAI_TILE4D_GET_DIM3(inTile))) &&                       \
                    ((XAI_CNN_CONV_GET_STRIDEY(param) >= 1) &&                                                  \
                     (XAI_CNN_CONV_GET_STRIDEY(param) <= XAI_TILE4D_GET_DIM4(inTile))), XAI_ERR_BADARG,         \
                    "StrideX = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Width) \
       \nStrideY = %hhu, value must be greater than or equal to 1 and less than or equal to %d(inTile Height)", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_TILE4D_GET_DIM3(inTile),                               \
                    XAI_CNN_CONV_GET_STRIDEY(param), XAI_TILE4D_GET_DIM4(inTile));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) == 1), \
                    XAI_ERR_BADARG, "\nDilation is %hhu\nDilation parameter should be equal to 1", XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                          \
                    XAI_ERR_BADARG, "\nDilation along width is %hhu and dilation along height is %hhu are not same", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
  }

  int32_t kIdx, kIdy;
  int32_t kernelIdx;

  XAI_ERROR_CHECKS_CONTINUE()
  {
    for (kIdy = 0; kIdy < XAI_CNN_CONV_GET_STRIDEY(param); kIdy++)
    {
      for (kIdx = 0; kIdx < XAI_CNN_CONV_GET_STRIDEX(param); kIdx++)
      {
        kernelIdx = kIdy * XAI_CNN_CONV_GET_STRIDEX(param) + kIdx;
        XAI_CHECK_TILE4D_F16(subCoeffs[kernelIdx]);
        XAI_CHECK_TILE4D_DATA_ORDER(subCoeffs[kernelIdx], XAI_NDWH);
      }
      XAI_CHECK_TILE4D_F16(superCoeffs[kIdy]);
      XAI_CHECK_TILE4D_DATA_ORDER(superCoeffs[kIdy], XAI_NDWH);
    }
  }

  xb_f16 *pInCoeff = (xb_f16 *) XAI_TILE4D_GET_DATA_PTR(inTile);

  const int32_t numOutCh = XAI_TILE4D_GET_DIM1(inTile); /* N */
  const int32_t numInCh  = XAI_TILE4D_GET_DIM2(inTile); /* D */
  const int32_t kWidth   = XAI_TILE4D_GET_DIM3(inTile); /* W */
  const int32_t kHeight  = XAI_TILE4D_GET_DIM4(inTile); /* H */

  const uint8_t strideX = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY = XAI_CNN_CONV_GET_STRIDEY(param);

  int32_t inCoeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(inTile);
  int32_t inCoeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(inTile);
  int32_t inCoeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(inTile);

  int32_t kx, ky, inCh, outCh, inIdx, outIdx = 0;
  xb_f16 *pSuperCoeff;
  xb_f16 *pSubCoeff;
  int32_t subKPitch1, subKPitch2, subKPitch3;
  int32_t superKPitch1, superKPitch2;
  int32_t kW, kH, subkW;
  int32_t numInChSubCoeff;
  int32_t subKIdx;

  int32_t kxStart, kyStart;

  if (transposeCoeffsFlag)
  {
    /* Conversion from NDWH -> DNWH,                       */
    /* transposing of kernels and formation of sub-kernels */
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        xb_f16 *pSubCoeff = (xb_f16 *) XAI_TILE4D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = kHeight - 1 - ((kHeight + strideY - kIdy - 1) % strideY);

        for (ky = kyStart; ky >= 0; ky -= strideY)          /* H */
        {
          kxStart = kWidth - 1 - ((kWidth + strideX - kIdx - 1) % strideX);

          for (kx = kxStart; kx >= 0; kx -= strideX)        /* W */
          {
            for (outCh = 0; outCh < numOutCh; outCh++)      /* N */
            {
              for (inCh = 0; inCh < numInCh; inCh++)        /* D */
              {
                inIdx = ky * inCoeffPitch3 + kx * inCoeffPitch2 + \
                        inCh * inCoeffPitch1 + outCh;
                pSubCoeff[outIdx++] = pInCoeff[inIdx];
              }
              /* For stride alignment */
              outIdx += (outIdx % (XCHAL_IVPN_SIMD_WIDTH)) ? ((XCHAL_IVPN_SIMD_WIDTH) -(outIdx % (XCHAL_IVPN_SIMD_WIDTH))) : 0;
            }
          }
        }
      }
    }
  }
  else
  {
    /* Conversion from NDWH -> DNWH and formation of sub-kernels */
    for (kIdy = 0; kIdy < strideY; kIdy++)
    {
      for (kIdx = 0; kIdx < strideX; kIdx++)
      {
        kernelIdx = kIdy * strideX + kIdx;
        xb_f16 *pSubCoeff = (xb_f16 *) XAI_TILE4D_GET_DATA_PTR(subCoeffs[kernelIdx]);

        outIdx  = 0;
        kyStart = ((kHeight + strideY - kIdy - 1) % strideY);

        for (ky = kyStart; ky < kHeight; ky += strideY)          /* H */
        {
          kxStart = ((kWidth + strideX - kIdx - 1) % strideX);

          for (kx = kxStart; kx < kWidth; kx += strideX)         /* W */
          {
            for (outCh = 0; outCh < numOutCh; outCh++)           /* N */
            {
              for (inCh = 0; inCh < numInCh; inCh++)             /* D */
              {
                inIdx = ky * inCoeffPitch3 + kx * inCoeffPitch2 + \
                        inCh * inCoeffPitch1 + outCh;
                pSubCoeff[outIdx++] = pInCoeff[inIdx];
              }
              /* For stride alignment */
              outIdx += (outIdx % (XCHAL_IVPN_SIMD_WIDTH)) ? ((XCHAL_IVPN_SIMD_WIDTH) -(outIdx % (XCHAL_IVPN_SIMD_WIDTH))) : 0;
            }
          }
        }
      }
    }
  }

  /* Form super-kernels by stacking sub-kernels */
  for (kernelIdx = 0; kernelIdx < strideY; kernelIdx++)
  {
    pSuperCoeff = (xb_f16 *) XAI_TILE4D_GET_DATA_PTR(superCoeffs[kernelIdx]);

    kW = XAI_TILE4D_GET_DIM3(superCoeffs[kernelIdx]);
    kH = XAI_TILE4D_GET_DIM4(superCoeffs[kernelIdx]);

    numInChSubCoeff = XAI_TILE4D_GET_DIM1(subCoeffs[kernelIdx * strideX]);
    superKPitch1    = XAI_TILE4D_GET_DIM1_PITCH(superCoeffs[kernelIdx]);
    superKPitch2    = XAI_TILE4D_GET_DIM2_PITCH(superCoeffs[kernelIdx]);

    for (subKIdx = 0; subKIdx < strideX; subKIdx++)
    {
      pSubCoeff = (xb_f16 *) XAI_TILE4D_GET_DATA_PTR(subCoeffs[kernelIdx * strideX + subKIdx]);

      subkW = XAI_TILE4D_GET_DIM3(subCoeffs[kernelIdx * strideX + subKIdx]);

      subKPitch1 = XAI_TILE4D_GET_DIM1_PITCH(subCoeffs[kernelIdx * strideX + subKIdx]);
      subKPitch2 = XAI_TILE4D_GET_DIM2_PITCH(subCoeffs[kernelIdx * strideX + subKIdx]);
      subKPitch3 = XAI_TILE4D_GET_DIM3_PITCH(subCoeffs[kernelIdx * strideX + subKIdx]);

      outIdx = numInChSubCoeff * subKIdx;

      for (ky = 0, kIdy = 0; ky < kH; ky++, kIdy++)          /* H */
      {
        for (kx = 0, kIdx = 0; kx < kW; kx++, kIdx++)        /* W */
        {
          /*In case of super kernels we have the first sub kernel width/height as the width/height of the superkernel     */
          /*In case the widths of the subkernel are not equal then we skip by differnce and start filling                 */
          /*Once the convolution is done the output junk data apprears at the end of the outtile.                         */
          /*In case of unequal heights this is handled using pointers in test app.                                        */
          if ((subkW < kW) && (kx == 0))
          {
            outIdx += superKPitch2;
            kIdx--;
            continue;
          }
          for (outCh = 0; outCh < numOutCh; outCh++)         /* N */
          {
            for (inCh = 0; inCh < numInChSubCoeff; inCh++)   /* D */
            {
              inIdx = kIdy * subKPitch3 + kIdx * subKPitch2 + \
                      outCh * subKPitch1 + inCh;
              pSuperCoeff[outIdx++] = pSubCoeff[inIdx];
            }
            outIdx += (superKPitch1 - numInChSubCoeff);
          }
        }
      }
    }
  }
  return(XAI_ERROR_STATUS());
}
#endif //if (XCHAL_HAVE_VISION_HP_VFPU == 1)
#endif //if ((XCHAL_VISION_TYPE >= 6))

