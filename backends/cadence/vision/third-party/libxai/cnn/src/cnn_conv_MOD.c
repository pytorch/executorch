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
#include "limits.h"

#if ((XCHAL_VISION_TYPE >= 6))

/******************************************************************************************
* MOD WHD variants
******************************************************************************************/


/*****************************************************************************
*  xaiConvolve3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 1x1xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

XAI_ERR_TYPE xaiConvolve3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 3x3xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/
XAI_ERR_TYPE xaiConvolve3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 5x5xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

XAI_ERR_TYPE xaiConvolve3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution .            */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 7x7xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

XAI_ERR_TYPE xaiConvolve3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk                                    */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

XAI_ERR_TYPE xaiConvolve3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
* MOD DWH variants
******************************************************************************************/



/*****************************************************************************
*  xaiConvolve3D_S_1x1_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 1x1xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

XAI_ERR_TYPE xaiConvolve3D_S_1x1_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_1x1_S8S8IXCa2_MOD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_1x1_U8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 1x1xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

XAI_ERR_TYPE xaiConvolve3D_S_1x1_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_1x1_U8S8IXCa2_MOD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_3x3_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 3x3xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/
XAI_ERR_TYPE xaiConvolve3D_S_3x3_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_3x3_S8S8IXCa2_MOD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_3x3_U8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 3x3xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/
XAI_ERR_TYPE xaiConvolve3D_S_3x3_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_3x3_U8S8IXCa2_MOD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_5x5_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 5x5xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

XAI_ERR_TYPE xaiConvolve3D_S_5x5_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_5x5_S8S8IXCa2_MOD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_7x7_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/*               Stride values = 1, 2 and 4 are supported.                  */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 7x7xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

XAI_ERR_TYPE xaiConvolve3D_S_7x7_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_7x7_S8S8IXCa2_MOD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolve3D_S_MxN_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

XAI_ERR_TYPE xaiConvolve3D_S_MxN_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params *param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH(inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************* end of MOD variants ***************************************/
/*******************************************************************************************/
#endif //if ((XCHAL_VISION_TYPE >= 6))
