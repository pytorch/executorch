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
#include <stdio.h>
#if ((XCHAL_VISION_TYPE >= 6))

/******************************************************************************
 * 3D convolution general version
 * Calls a specific convolution function based on parameters
 *****************************************************************************/
XAI_ERR_TYPE xaiConvolve3D(const xai_pTile3D inTile,
                           const xai_pTile4D coeffTile,
                           const xai_pArray biasArray,
                           xai_pTile3D outTile,
                           xai_cnn_conv_params *param)
{
  /* The arguments inTile, coeffTile and param are used by xaiGetConvolve3DVariant
   * helper function, to derive the appropriate convolution variant */
  if ((!inTile) || (!coeffTile) || (!param))
  {
    return(XAI_ERR_NULLARG);
  }

  /* Function Pointer */
  typedef XAI_ERR_TYPE (*fConvPtr)(const xai_pTile3D inTile,
                                   const xai_pTile4D coeffTile,
                                   const xai_pArray biasArray,
                                   xai_pTile3D outTile,
                                   xai_cnn_conv_params* param);

  /* Getting the function pointer of the convolution variant using xaiGetConvolve3DVariant function */
  fConvPtr xaiConvolve3D_opt = (fConvPtr) xaiGetConvolve3DVariant(inTile, coeffTile, biasArray, outTile, param);

  if (xaiConvolve3D_opt == NULL)
  {
    return(XAI_ERR_NO_VARIANT);
  }
  else
  {
    return(xaiConvolve3D_opt(inTile, coeffTile, biasArray, outTile, param));
  }
}

/**************************************************************************************
* 3D convolution helper function
* Returns the function pointer of a specific convolution variant based on parameters
**************************************************************************************/
XAI_ERR_TYPE *xaiGetConvolve3DVariant(const xai_pTile3D inTile,
                                      const xai_pTile4D coeffTile,
                                      const xai_pArray biasArray,
                                      xai_pTile3D outTile,
                                      xai_cnn_conv_params *param)
{
  if ((!inTile) || (!coeffTile) || (!param))
  {
    return(NULL);
  }

  xai_cnn_data_order coeffOrder = XAI_TILE4D_GET_DATA_ORDER(coeffTile);

  xai_cnn_data_order inOrder = XAI_TILE3D_GET_DATA_ORDER(inTile);
  int32_t kWidth, kHeight;
  uint8_t stride;

  if (coeffOrder == XAI_NDWH)
  {
    /* MOD variants */
    kWidth  = XAI_TILE4D_GET_DIM3(coeffTile);
    kHeight = XAI_TILE4D_GET_DIM4(coeffTile);

    if (inOrder == XAI_WHD)
    {
      if (kWidth == 1 && kHeight == 1)
      {
        return((XAI_ERR_TYPE *) &xaiConvolve3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH);
      }
      else if (kWidth == 3 && kHeight == 3)
      {
        return((XAI_ERR_TYPE *) &xaiConvolve3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH);
      }
      else if (kWidth == 5 && kHeight == 5)
      {
        return((XAI_ERR_TYPE *) &xaiConvolve3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH);
      }
      else if (kWidth == 7 && kHeight == 7)
      {
        return((XAI_ERR_TYPE *) &xaiConvolve3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH);
      }
      else
      {
        return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH);
      }
    }
    else if (inOrder == XAI_DWH)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (kWidth == 1 && kHeight == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_1x1_S8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 3 && kHeight == 3)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_3x3_S8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 5 && kHeight == 5)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_5x5_S8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 7 && kHeight == 7)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_7x7_S8S8IXCa2_MOD_DWH);
        }
        else
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxN_S8S8IXCa2_MOD_DWH);
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (kWidth == 1 && kHeight == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_1x1_U8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 3 && kHeight == 3)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_3x3_U8S8IXCa2_MOD_DWH);
        }
      }
    }
  }
  else if (coeffOrder == XAI_WHDN)
  {
    /* MOW variants */
    stride  = XAI_CNN_CONV_GET_STRIDE(param);
    kWidth  = XAI_TILE4D_GET_DIM1(coeffTile);
    kHeight = XAI_TILE4D_GET_DIM2(coeffTile);

    if (kWidth == 1 && kHeight == 1)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_1x1j1_S8S8IX_MOW_WHD);
        }
        else if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_1x1j2_S8S8IX_MOW_WHD);
        }
        else if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_1x1j4_S8S8IX_MOW_WHD);
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_1x1j1_U8S8IX_MOW_WHD);
        }
        if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_1x1j2_U8S8IX_MOW_WHD);
        }
        if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_1x1j4_U8S8IX_MOW_WHD);
        }
      }
    }
    else if (kWidth == 3 && kHeight == 3)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_3x3j1_S8S8IX_MOW_WHD);
        }
        else if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_3x3j2_S8S8IX_MOW_WHD);
        }
        else if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_3x3j4_S8S8IX_MOW_WHD);
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_3x3j1_U8S8IX_MOW_WHD);
        }
        if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_3x3j2_U8S8IX_MOW_WHD);
        }
        if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_3x3j4_U8S8IX_MOW_WHD);
        }
      }
    }
    else if (kWidth == 5 && kHeight == 5)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_5x5j1_S8S8IX_MOW_WHD);
        }
        else if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_5x5j2_S8S8IX_MOW_WHD);
        }
        else if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_5x5j4_S8S8IX_MOW_WHD);
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_5x5j1_U8S8IX_MOW_WHD);
        }
        if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_5x5j2_U8S8IX_MOW_WHD);
        }
        if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_5x5j4_U8S8IX_MOW_WHD);
        }
      }
    }
    else if (kWidth == 7 && kHeight == 7)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_7x7j1_S8S8IX_MOW_WHD);
        }
        else if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_7x7j2_S8S8IX_MOW_WHD);
        }
        else if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_7x7j4_S8S8IX_MOW_WHD);
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_7x7j1_U8S8IX_MOW_WHD);
        }
        if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_7x7j2_U8S8IX_MOW_WHD);
        }
        if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_7x7j4_U8S8IX_MOW_WHD);
        }
      }
    }
    else
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxNj1_S8S8IX_MOW_WHD);
        }
        else if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxNj2_S8S8IX_MOW_WHD);
        }
        else if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxNj4_S8S8IX_MOW_WHD);
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxNj1_U8S8IX_MOW_WHD);
        }
        if (stride == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxNj2_U8S8IX_MOW_WHD);
        }
        if (stride == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxNj4_U8S8IX_MOW_WHD);
        }
      }
    }
  }
  else if (coeffOrder == XAI_DWHN)
  {
    /* SO variants */
    if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
    {
      return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxN_S8S8IX_SO_DWH);
    }
    else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
    {
      return((XAI_ERR_TYPE *) &xaiConvolve3D_S_MxN_U8S8IX_SO_DWH);
    }
  }

  return(NULL);
}

/******************************************************************************
 * 3D convolution general version for dilation functions
 * Calls a specific dilated convolution function based on parameters
 *****************************************************************************/
XAI_ERR_TYPE xaiConvolved3D(const xai_pTile3D inTile,
                            const xai_pTile4D coeffTile,
                            const xai_pArray biasArray,
                            xai_pTile3D outTile,
                            const xai_cnn_conv_params *param)
{
  /* The arguments inTile, coeffTile and param are used by xaiGetConvolved3DVariant
   * helper function, to derive the appropriate convolution variant */
  if ((!inTile) || (!coeffTile) || (!param))
  {
    return(XAI_ERR_NULLARG);
  }

  /* Function Pointer */
  typedef XAI_ERR_TYPE (*fConvdPtr)(const xai_pTile3D inTile,
                                    const xai_pTile4D coeffTile,
                                    const xai_pArray biasArray,
                                    xai_pTile3D outTile,
                                    const xai_cnn_conv_params* param);

  /* Getting the function pointer of the convolution variant using xaiGetConvolved3DVariant function*/
  fConvdPtr xaiConvolve3D_opt =
    (fConvdPtr) xaiGetConvolved3DVariant(inTile, coeffTile, biasArray, outTile, param);

  if (xaiConvolve3D_opt == NULL)
  {
    return(XAI_ERR_NO_VARIANT);
  }
  else
  {
    return(xaiConvolve3D_opt(inTile, coeffTile, biasArray, outTile, param));
  }
}

/*********************************************************************************************
* 3D dilated convolution helper function
* Returns the function pointer of a specific dilated convolution variant based on parameters
*********************************************************************************************/
XAI_ERR_TYPE *xaiGetConvolved3DVariant(const xai_pTile3D inTile,
                                       const xai_pTile4D coeffTile,
                                       const xai_pArray biasArray,
                                       xai_pTile3D outTile,
                                       const xai_cnn_conv_params *param)
{
  if ((!inTile) || (!coeffTile) || (!param))
  {
    return(NULL);
  }

  uint8_t stride;
  uint8_t dilation;
  xai_cnn_data_order coeffOrder = XAI_TILE4D_GET_DATA_ORDER(coeffTile);


  int32_t kWidth, kHeight;
  xai_cnn_data_order inOrder = XAI_TILE3D_GET_DATA_ORDER(inTile);

  if (coeffOrder == XAI_NDWH)
  {
    /* MOD variants */
    kWidth  = XAI_TILE4D_GET_DIM3(coeffTile);
    kHeight = XAI_TILE4D_GET_DIM4(coeffTile);

    if (inOrder == XAI_WHD)
    {
      if (kWidth == 1 && kHeight == 1)
      {
        return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH);
      }
      else if (kWidth == 2 && kHeight == 2)
      {
        return((XAI_ERR_TYPE *) &xaiConvolved3D_S_2x2_S8S8IXCa2_MOD_WHD_DWH);
      }
      else if (kWidth == 3 && kHeight == 3)
      {
        return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH);
      }
      else if (kWidth == 4 && kHeight == 4)
      {
        return((XAI_ERR_TYPE *) &xaiConvolved3D_S_4x4_S8S8IXCa2_MOD_WHD_DWH);
      }
      else if (kWidth == 5 && kHeight == 5)
      {
        return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH);
      }
      else if (kWidth == 7 && kHeight == 7)
      {
        return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH);
      }
      else
      {
        return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH);
      }
    }
    else if (inOrder == XAI_DWH)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8) && XAI_TILE4D_CHECK_TYPE(coeffTile, XAI_S8))
      {
        if (XAI_CNN_CONV_GET_STRIDEX(param) != XAI_CNN_CONV_GET_STRIDEY(param))
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH);
        }
        else if (XAI_CNN_CONV_GET_STRIDE(param) != 1 && XAI_CNN_CONV_GET_STRIDE(param) != 2 \
                 && XAI_CNN_CONV_GET_STRIDE(param) != 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 1 && kHeight == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1_S8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 2 && kHeight == 2)
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_2x2_S8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 3 && kHeight == 3)
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3_S8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 4 && kHeight == 4)
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_4x4_S8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 5 && kHeight == 5)
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5_S8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 7 && kHeight == 7)
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7_S8S8IXCa2_MOD_DWH);
        }
        else
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH);
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8) && XAI_TILE4D_CHECK_TYPE(coeffTile, XAI_S8))
      {
        if (kWidth == 1 && kHeight == 1)
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1_U8S8IXCa2_MOD_DWH);
        }
        else if (kWidth == 3 && kHeight == 3)
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3_U8S8IXCa2_MOD_DWH);
        }
        else
        {
          return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxN_U8S8IXCa2_MOD_DWH);
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16) && XAI_TILE4D_CHECK_TYPE(coeffTile, XAI_S16))
      {
        return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxN_S16S16I16_MOD_DWH);
      }
    }
  }
  else if (coeffOrder == XAI_WHDN)
  {
    /* MOW variants */
    stride   = XAI_CNN_CONV_GET_STRIDE(param);
    dilation = XAI_CNN_CONV_GET_DILATION(param);
    kWidth   = XAI_TILE4D_GET_DIM1(coeffTile);
    kHeight  = XAI_TILE4D_GET_DIM2(coeffTile);

    if (kWidth == 1 && kHeight == 1)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1j1d1_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1j2d1_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1j4d1_S8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1j1d1_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1j2d1_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1j4d1_U8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj2d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj4d1_S16S16I16_MOW_WHD);
          }
        }
      }
#if 0  /* F16 disabled - no implementation available */
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_F16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_1x1j1d1_F16_MOW_WHD);
          }
        }
      }
#endif
    }
    else if (kWidth == 2 && kHeight == 2)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_2x2j1d1_S8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_2x2j1d1_U8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_S16S16I16_MOW_WHD);
          }
        }
        if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj2d1_S16S16I16_MOW_WHD);
          }
        }
        if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj4d1_S16S16I16_MOW_WHD);
          }
        }
      }
#if 0  /* F16 disabled - no implementation available */
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_F16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_2x2j1d1_F16_MOW_WHD);
          }
        }
      }
#endif
    }
    else if (kWidth == 3 && kHeight == 3)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j1d1_S8S8IX_MOW_WHD);
          }
          else if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j1d2_S8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j1d4_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j2d1_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j4d1_S8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j1d1_U8S8IX_MOW_WHD);
          }
          else if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j1d2_U8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j1d4_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j2d1_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j4d1_U8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj2d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj4d1_S16S16I16_MOW_WHD);
          }
        }
      }
#if 0  /* F16 disabled - no implementation available */
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_F16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j1d1_F16_MOW_WHD);
          }
        }
        if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_3x3j2d1_F16_MOW_WHD);
          }
        }
      }
#endif
    }
    else if (kWidth == 4 && kHeight == 4)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_4x4j1d1_S8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_4x4j1d1_U8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_S16S16I16_MOW_WHD);
          }
        }
        if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj2d1_S16S16I16_MOW_WHD);
          }
        }
        if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj4d1_S16S16I16_MOW_WHD);
          }
        }
      }
    }
    else if (kWidth == 5 && kHeight == 5)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j1d1_S8S8IX_MOW_WHD);
          }
          else if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j1d2_S8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j1d4_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j2d1_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j4d1_S8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j1d1_U8S8IX_MOW_WHD);
          }
          else if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j1d2_U8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j1d4_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j2d1_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_5x5j4d1_U8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj2d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj4d1_S16S16I16_MOW_WHD);
          }
        }
      }
    }
    else if (kWidth == 7 && kHeight == 7)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j1d1_S8S8IX_MOW_WHD);
          }
          else if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j1d2_S8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j1d4_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j2d1_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j4d1_S8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j1d1_U8S8IX_MOW_WHD);
          }
          else if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j1d2_U8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j1d4_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j2d1_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_7x7j4d1_U8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj2d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj4d1_S16S16I16_MOW_WHD);
          }
        }
      }
    }
    else
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_S8S8IX_MOW_WHD);
          }
          else if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d2_S8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d4_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj2d1_S8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj4d1_S8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_U8S8IX_MOW_WHD);
          }
          else if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d2_U8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d4_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj2d1_U8S8IX_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj4d1_U8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 2)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj2d1_S16S16I16_MOW_WHD);
          }
        }
        else if (stride == 4)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj4d1_S16S16I16_MOW_WHD);
          }
        }
      }
#if 0  /* F16 disabled - no implementation available */
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_F16))
      {
        if (stride == 1)
        {
          if (dilation == 1)
          {
            return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxNj1d1_F16_MOW_WHD);
          }
        }
      }
#endif
    }
  }
  else if (coeffOrder == XAI_DWHN)
  {
    /* SO variants */
    if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
    {
      return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxN_S8S8IX_SO_DWH);
    }
    else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
    {
      return((XAI_ERR_TYPE *) &xaiConvolved3D_S_MxN_U8S8IX_SO_DWH);
    }
  }

  return(NULL);
}

/******************************************************************************
 * Depthwise convolution general version
 * Calls a specific depthwise convolution function based in parameters
 *****************************************************************************/
XAI_ERR_TYPE xaiDepthwiseConvolve2D(const xai_pTile3D inTile,
                                    const xai_pTile3D coeffTile,
                                    const xai_pArray biasArray,
                                    xai_pTile3D outTile,
                                    const xai_cnn_conv_params *param)
{
  /* The arguments inTile, coeffTile and param are used by xaiGetDepthwiseConvolve2DVariant
   * helper function, to derive the appropriate convolution variant */
  if ((!inTile) || (!coeffTile) || (!param))
  {
    return(XAI_ERR_NULLARG);
  }

  /* Function Pointer */
  typedef XAI_ERR_TYPE (*fDepthwiseConvPtr)(const xai_pTile3D inTile,
                                            const xai_pTile3D coeffTile,
                                            const xai_pArray biasArray,
                                            xai_pTile3D outTile,
                                            const xai_cnn_conv_params* param);

  /* Getting the function pointer of the convolution variant using xaiGetDepthwiseConvolve2DVariant function */
  fDepthwiseConvPtr xaiDepthwiseConvolve2D_opt = (fDepthwiseConvPtr) xaiGetDepthwiseConvolve2DVariant(inTile,
                                                                                                      coeffTile,
                                                                                                      biasArray,
                                                                                                      outTile,
                                                                                                      param);

  if (xaiDepthwiseConvolve2D_opt == NULL)
  {
    return(XAI_ERR_NO_VARIANT);
  }
  else
  {
    return(xaiDepthwiseConvolve2D_opt(inTile, coeffTile, biasArray, outTile, param));
  }
}

/**************************************************************************************
* 2D depthwise convolution helper function
* Returns the function pointer of a specific depthwiseconvolution variant based
* on parameters
**************************************************************************************/
XAI_ERR_TYPE *xaiGetDepthwiseConvolve2DVariant(const xai_pTile3D inTile,
                                               const xai_pTile3D coeffTile,
                                               const xai_pArray biasArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_conv_params *param)
{
  if ((!inTile) || (!coeffTile) || (!outTile) || (!param))
  {
    return(NULL);
  }
  if (!(XAI_TILE3D_GET_DATA_ORDER(inTile) == XAI_TILE3D_GET_DATA_ORDER(outTile)))
  {
    return(NULL);
  }

  xai_cnn_data_order inOrder    = XAI_TILE3D_GET_DATA_ORDER(inTile);
  xai_cnn_data_order coeffOrder = XAI_TILE3D_GET_DATA_ORDER(coeffTile);
  int32_t kWidth, kHeight;

  if (coeffOrder == XAI_DWH)
  {
    /* MOD variants */
    kWidth  = XAI_TILE3D_GET_DIM2(coeffTile);
    kHeight = XAI_TILE3D_GET_DIM3(coeffTile);
    if (inOrder == XAI_DWH)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8) || XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (XAI_CNN_CONV_GET_STRIDEX(param) != XAI_CNN_CONV_GET_STRIDEY(param))
        {
          if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxN_S8S8IXCa2_MOD_DWH);
          }
          else
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxN_U8S8IXCa2_MOD_DWH);
          }
        }
        else if (XAI_CNN_CONV_GET_STRIDE(param) != 1 && XAI_CNN_CONV_GET_STRIDE(param) != 2 \
                 && XAI_CNN_CONV_GET_STRIDE(param) != 4)
        {
          if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxN_S8S8IXCa2_MOD_DWH);
          }
          else
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxN_U8S8IXCa2_MOD_DWH);
          }
        }
        else if (kWidth == 3 && kHeight == 3)
        {
          if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_3x3_S8S8IXCa2_MOD_DWH);
          }
          else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_3x3_U8S8IXCa2_MOD_DWH);
          }
        }
        else if (kWidth == 5 && kHeight == 5)
        {
          if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_5x5_S8S8IXCa2_MOD_DWH);
          }
          else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_5x5_U8S8IXCa2_MOD_DWH);
          }
        }
        else if (kWidth == 7 && kHeight == 7 && XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_7x7_S8S8IXCa2_MOD_DWH);
        }
        else
        {
          if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxN_S8S8IXCa2_MOD_DWH);
          }
          else
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxN_U8S8IXCa2_MOD_DWH);
          }
        }
      } /* if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8) || XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8)) */
      else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) || XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16))
      {
        return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxN_S16S16I16_MOD_DWH);
      } /* if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) || XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16))*/
    }   /* if(inOrder == XAI_DWH) */
  }     /* if(coeffOrder == XAI_DWH) */
  else if (coeffOrder == XAI_WHD)
  {
    /* MOW variants */
    uint8_t stride = XAI_CNN_CONV_GET_STRIDE(param);
    kWidth  = XAI_TILE3D_GET_DIM1(coeffTile);
    kHeight = XAI_TILE3D_GET_DIM2(coeffTile);
    if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8) || XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
    {
      if (kWidth == 3 && kHeight == 3)
      {
        if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
        {
          if (stride == 1)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_3x3j1_S8S8IX_MOW_WHD);
          }
          else if (stride == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_3x3j2_S8S8IX_MOW_WHD);
          }
          else if (stride == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_3x3j4_S8S8IX_MOW_WHD);
          }
        }
        else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
        {
          if (stride == 1)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_3x3j1_U8S8IX_MOW_WHD);
          }
          else if (stride == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_3x3j2_U8S8IX_MOW_WHD);
          }
          else if (stride == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_3x3j4_U8S8IX_MOW_WHD);
          }
        }
      }
      else if (kWidth == 5 && kHeight == 5)
      {
        if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
        {
          if (stride == 1)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_5x5j1_S8S8IX_MOW_WHD);
          }
          else if (stride == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_5x5j2_S8S8IX_MOW_WHD);
          }
          else if (stride == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_5x5j4_S8S8IX_MOW_WHD);
          }
        }
        else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
        {
          if (stride == 1)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_5x5j1_U8S8IX_MOW_WHD);
          }
          else if (stride == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_5x5j2_U8S8IX_MOW_WHD);
          }
          else if (stride == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_5x5j4_U8S8IX_MOW_WHD);
          }
        }
      }
      else if (kWidth == 7 && kHeight == 7)
      {
        if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
        {
          if (stride == 1)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_7x7j1_S8S8IX_MOW_WHD);
          }
          else if (stride == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_7x7j2_S8S8IX_MOW_WHD);
          }
          else if (stride == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_7x7j4_S8S8IX_MOW_WHD);
          }
        }
        else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
        {
          if (stride == 1)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_7x7j1_U8S8IX_MOW_WHD);
          }
          else if (stride == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_7x7j2_U8S8IX_MOW_WHD);
          }
          else if (stride == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_7x7j4_U8S8IX_MOW_WHD);
          }
        }
      }
      else
      {
        /* MOW Variants */
        if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
        {
          if (stride == 1)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxNj1_S8S8IX_MOW_WHD);
          }
          else if (stride == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxNj2_S8S8IX_MOW_WHD);
          }
          else if (stride == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxNj4_S8S8IX_MOW_WHD);
          }
        }
        else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
        {
          if (stride == 1)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxNj1_U8S8IX_MOW_WHD);
          }
          else if (stride == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxNj2_U8S8IX_MOW_WHD);
          }
          else if (stride == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxNj4_U8S8IX_MOW_WHD);
          }
        }
      }
/* #if XCHAL_VISION_QUAD_MAC_TYPE != 0 */
    } /* if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8) || XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8)) */
    else if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) || XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16))
    {
      if (stride == 1)
      {
        return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxNj1_S16S16I16_MOW_WHD);
      }
      else if (stride == 2)
      {
        return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxNj2_S16S16I16_MOW_WHD);
      }
      else if (stride == 4)
      {
        return((XAI_ERR_TYPE *) &xaiDepthwiseConvolve2D_S_MxNj4_S16S16I16_MOW_WHD);
      }
    } /* if (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) || XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16)) */
  }   /*  if(coeffOrder == XAI_WHD) */
  return(NULL);
}

/******************************************************************************
 * Depthwise dilated convolution general version
 * Calls a specific depthwise dilated convolution function based in parameters
 *****************************************************************************/
XAI_ERR_TYPE xaiDepthwiseConvolved2D(const xai_pTile3D inTile,
                                     const xai_pTile3D coeffTile,
                                     const xai_pArray biasArray,
                                     xai_pTile3D outTile,
                                     const xai_cnn_depthwiseDilatedConv_params *param)
{
  /* The arguments inTile, coeffTile and param are used by xaiGetDepthwiseConvolved2DVariant
   * helper function, to derive the appropriate convolution variant */
  if ((!inTile) || (!coeffTile) || (!param))
  {
    return(XAI_ERR_NULLARG);
  }

  /* Function Pointer */
  typedef XAI_ERR_TYPE (*fDepthwiseConvdPtr)(const xai_pTile3D inTile,
                                             const xai_pTile3D coeffTile,
                                             const xai_pArray biasArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_depthwiseDilatedConv_params* param);

  /* Getting the function pointer of the convolution variant using xaiGetDepthwiseConvolved2DVariant function */
  fDepthwiseConvdPtr xaiDepthwiseConvolved2D_opt = (fDepthwiseConvdPtr) xaiGetDepthwiseConvolved2DVariant(inTile, coeffTile, biasArray, outTile, param);

  if (xaiDepthwiseConvolved2D_opt == NULL)
  {
    return(XAI_ERR_NO_VARIANT);
  }
  else
  {
    return(xaiDepthwiseConvolved2D_opt(inTile, coeffTile, biasArray, outTile, param));
  }
}

/**************************************************************************************
* 2D depthwise convolution helper function
* Returns the function pointer of a specific depthwiseconvolution variant based
* on parameters
**************************************************************************************/
XAI_ERR_TYPE *xaiGetDepthwiseConvolved2DVariant(const xai_pTile3D inTile,
                                                const xai_pTile3D coeffTile,
                                                const xai_pArray biasArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_depthwiseDilatedConv_params *param)
{
  if ((!inTile) || (!coeffTile) || (!param))
  {
    return(NULL);
  }

  xai_cnn_data_order inOrder    = XAI_TILE3D_GET_DATA_ORDER(inTile);
  xai_cnn_data_order coeffOrder = XAI_TILE3D_GET_DATA_ORDER(coeffTile);
#if (XCHAL_HAVE_SUPERGATHER == 0)
  int32_t depthMultiplier = XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DEPTH_MULTIPLIER(param);
#endif
  uint8_t stride;
  uint8_t dilation;

  int32_t kWidth, kHeight;
  if (coeffOrder == XAI_DWH)
  {
    kWidth  = XAI_TILE3D_GET_DIM2(coeffTile);
    kHeight = XAI_TILE3D_GET_DIM3(coeffTile);
    /* MOD variants */
    if (inOrder == XAI_DWH)
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_MxN_U8S8IX_MOD_DWH);
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
#if (XCHAL_HAVE_SUPERGATHER == 0)
        if (kWidth == 3 && kHeight == 3 && depthMultiplier != 8)
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_3x3_S8S8IX_MOD_DWH);
        }
        else if (kWidth == 5 && kHeight == 5 && depthMultiplier != 8)
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_5x5_S8S8IX_MOD_DWH);
        }
        else if (kWidth == 7 && kHeight == 7 && depthMultiplier != 8)
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_7x7_S8S8IX_MOD_DWH);
        }
#else
        if (kWidth == 3 && kHeight == 3)
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_3x3_S8S8IX_MOD_DWH);
        }
        else if (kWidth == 5 && kHeight == 5)
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_5x5_S8S8IX_MOD_DWH);
        }
        else if (kWidth == 7 && kHeight == 7)
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_7x7_S8S8IX_MOD_DWH);
        }
#endif
        else
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_MxN_S8S8IX_MOD_DWH);
        }
      }
      else /* (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S16)) */
      {
        if (kWidth == 3 && kHeight == 3)
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_3x3_S16S16I16_MOD_DWH);
        }
        else if (kWidth == 5 && kHeight == 5)
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_5x5_S16S16I16_MOD_DWH);
        }
        else
        {
          return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_MxN_S16S16I16_MOD_DWH);
        }
      }
    }
  }
  /*else*/ if (coeffOrder == XAI_WHD)
  {
    /* MOW variants */

    stride   = XAI_CNN_DEPTHWISE_DILATED_CONV_GET_STRIDE(param);
    dilation = XAI_CNN_DEPTHWISE_DILATED_CONV_GET_DILATION(param);
//#endif
    /*if(kWidth == 3 && kHeight == 3)
       {
       if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
       {
        if(stride == 1)
        {
          if(dilation == 2)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_3x3j1d2_S8S8IX_MOW_WHD);
          }
          else if(dilation == 4)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_3x3j1d4_S8S8IX_MOW_WHD);
           }
        }
       }
       else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
       {
        if(stride == 1)
        {
          if(dilation == 2)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_3x3j1d2_U8S8IX_MOW_WHD);
          }
          else if(dilation == 4)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_3x3j1d4_U8S8IX_MOW_WHD);
           }
        }
       }
       }*/
    /*else if(kWidth == 5 && kHeight == 5)
       {
       if (xaiTile3DCheckType(inTile, XAI_S8))
       {
        if(stride == 1)
        {
          if(dilation == 2)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_5x5j1d2_S8S8IX_MOW_WHD);
          }
          else if(dilation == 4)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_5x5j1d4_S8S8IX_MOW_WHD);
          }
        }
       }
       else if (xaiTile3DCheckType(inTile, XAI_U8))
       {
        if(stride == 1)
        {
          if(dilation == 2)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_5x5j1d2_U8S8IX_MOW_WHD);
          }
          else if(dilation == 4)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_5x5j1d4_U8S8IX_MOW_WHD);
          }
        }
       }
       }
       else if(kWidth == 7 && kHeight == 7)
       {
       if (xaiTile3DCheckType(inTile, XAI_S8))
       {
        if(stride == 1)
        {
          if(dilation == 2)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_7x7j1d2_S8S8IX_MOW_WHD);
          }
          else if(dilation == 4)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_7x7j1d4_S8S8IX_MOW_WHD);
           }
        }
       }
       else if (xaiTile3DCheckType(inTile, XAI_U8))
       {
        if(stride == 1)
        {
          if(dilation == 2)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_7x7j1d2_U8S8IX_MOW_WHD);
          }
          else if(dilation == 4)
          {
            return ((XAI_ERR_TYPE *)&xaiDepthwiseConvolved2D_S_7x7j1d4_U8S8IX_MOW_WHD);
          }
        }
       }
       }*/
    /* else */
    {
      if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_S8))
      {
        if (stride == 1)
        {
          if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_MxNj1d2_S8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_MxNj1d4_S8S8IX_MOW_WHD);
          }
        }
      }
      else if (XAI_TILE3D_CHECK_TYPE(inTile, XAI_U8))
      {
        if (stride == 1)
        {
          if (dilation == 2)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_MxNj1d2_U8S8IX_MOW_WHD);
          }
          else if (dilation == 4)
          {
            return((XAI_ERR_TYPE *) &xaiDepthwiseConvolved2D_S_MxNj1d4_U8S8IX_MOW_WHD);
          }
        }
      }
//#endif
    }
  }
  return(NULL);
}
#endif //if ((XCHAL_VISION_TYPE >= 6))
