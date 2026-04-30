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
#if ((XCHAL_VISION_TYPE >= 6))

#define MAKE_NAME_IMPL(name, MORPH_FNAME_SPECIFIER_IDT, suffix)  name ## _ ## MORPH_FNAME_SPECIFIER_IDT ## suffix

#if INPUT_DATA_TYPE == UNSIGNED8BIT

#define MAKE_NAME(name, suffix)  MAKE_NAME_IMPL(name, U8, suffix)
#define MORPH_IDT_CHECK              XAI_CHECK_TILE3D_U8
#define MORPH_IDT_SCALAR             uint8_t
#define MORPH_IDT_2Nx8               xb_vec2Nx8U
#define MORPH_OP_PRIME_2Nx8          IVP_LA2NX8U_PP
#define MORPH_OP_ALIGN_LOAD_2Nx8     IVP_LV2NX8U_XP
#define MORPH_OP_LOAD_2Nx8           IVP_LA2NX8U_XP
#define MORPH_OP_LOAD_2Nx8_IP        IVP_LA2NX8U_IP
#define MORPH_OP_LOAD_2Nx8_VARIABLE  IVP_LAV2NX8U_XP
#define MORPH_OP_MULA                IVP_MULUSA2N8XR16
#define MORPH_OP_MUL4TA              IVP_MULUS4TA2N8XR8
#define MORPH_OP_MULQA               IVP_MULUSQA2N8XR8
#define MORPH_OP_MULPA               IVP_MULUSPA2N8XR16

#elif INPUT_DATA_TYPE == SIGNED8BIT

#undef MAKE_NAME
#undef MORPH_IDT_CHECK
#undef MORPH_IDT_SCALAR
#undef MORPH_IDT_2Nx8
#undef MORPH_OP_PRIME_2Nx8
#undef MORPH_OP_ALIGN_LOAD_2Nx8
#undef MORPH_OP_LOAD_2Nx8_IP
#undef MORPH_OP_LOAD_2Nx8_VARIABLE
#undef MORPH_OP_LOAD_2Nx8
#undef MORPH_OP_MULA
#undef MORPH_OP_MUL4TA
#undef MORPH_OP_MULQA
#undef MORPH_OP_MULPA

#define MAKE_NAME(name, suffix)  MAKE_NAME_IMPL(name, S8, suffix)
#define MORPH_IDT_CHECK              XAI_CHECK_TILE3D_S8
#define MORPH_IDT_SCALAR             int8_t
#define MORPH_IDT_2Nx8               xb_vec2Nx8
#define MORPH_OP_PRIME_2Nx8          IVP_LA2NX8_PP
#define MORPH_OP_ALIGN_LOAD_2Nx8     IVP_LV2NX8_XP
#define MORPH_OP_LOAD_2Nx8           IVP_LA2NX8_XP
#define MORPH_OP_LOAD_2Nx8_IP        IVP_LA2NX8_IP
#define MORPH_OP_LOAD_2Nx8_VARIABLE  IVP_LAV2NX8_XP
#define MORPH_OP_MULA                IVP_MULA2N8XR16
#define MORPH_OP_MUL4TA              IVP_MUL4TA2N8XR8
#define MORPH_OP_MULQA               IVP_MULQA2N8XR8
#define MORPH_OP_MULPA               IVP_MULPA2N8XR16
#endif

/******************************************************************************************
* MOW Stride 1 varaints
******************************************************************************************/

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_1x1j1, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 1x1 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 1x1 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 1               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 1x1xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_1x1j1_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_1x1j1_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_1x1j1, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_1x1j1d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  MAKE_NAME(xaiConvolve3D_S_3x3j1, S8IX_MOW_WHD)
*  **************************************************************************/

/******************** xaiConvolve3D_S_3x3j1_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_3x3j1_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_3x3j1, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_3x3j1d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}



/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_5x5j1, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 5x5 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 5x5 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 1               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 5x5xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_5x5j1_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_5x5j1_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE  MAKE_NAME(xaiConvolve3D_S_5x5j1, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_5x5j1d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_7x7j1, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 7x7 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 7x7 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 1               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 7x7xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_7x7j1_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_7x7j1_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_7x7j1, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_7x7j1d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_MxNj1, S8IX_MOW_WHD)
*  ***************************************************************************************/
/******************************************************************************/
/* Description : P6 optimized generic implementation for MxN 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate MxN 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 1               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is MxNxDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_MxNj1_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_MxNj1_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_MxNj1, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_MxNj1d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
* MOW Stride 2 varaints
******************************************************************************************/


/*****************************************************************************
*  MAKE_NAME(xaiConvolve3D_S_1x1j2, S8IX_MOW_WHD)
*  **************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 1x1 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 1x1 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 2               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 1x1xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_1x1j2_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_1x1j2_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_1x1j2, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_1x1j2d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_3x3j2, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 3x3 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 3x3 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 2               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 3x3xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_3x3j2_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_3x3j2_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_3x3j2, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_3x3j2d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_5x5j2, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 5x5 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 5x5 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 2               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 5x5xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_5x5j2_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_5x5j2_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_5x5j2, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_5x5j2d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_7x7j2, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 7x7 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 7x7 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 2               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 7x7xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_7x7j2_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_7x7j2_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_7x7j2, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_7x7j2d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_MxNj2, S8IX_MOW_WHD)
*  ***************************************************************************************/
/******************************************************************************/
/* Description : P6 optimized generic implementation for MxN 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate MxN 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 2               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is MxNxDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_MxNj2_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_MxNj2_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_MxNj2, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_MxNj2d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
* MOW Stride 4 varaints
******************************************************************************************/

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_1x1j4, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 1x1 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 1x1 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 4               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 1x1xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_1x1j4_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_1x1j4_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_1x1j4, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_1x1j4d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_3x3j4, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 3x3 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 3x3 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 4               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 3x3xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_3x3j4_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_3x3j4_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_3x3j4, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_3x3j4d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_5x5j4, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 5x5 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 5x5 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 4               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 5x5xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_5x5j4_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_5x5j4_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_5x5j4, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_5x5j4d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_7x7j4, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for 7x7 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate 7x7 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 4               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is 7x7xDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_7x7j4_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_7x7j4_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_7x7j4, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_7x7j4d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}


/******************************************************************************************
*   MAKE_NAME(xaiConvolve3D_S_MxNj4, S8IX_MOW_WHD)
*  ***************************************************************************************/

/******************************************************************************/
/* Description : P6 optimized generic implementation for MxN 3D convolution.  */
/*               Based on MORPH pre-processor specifiers, code implementation */
/*               is generated during preprocessing stage. This method can be  */
/*               used to generate MxN 3D convolution function for U8 bit and  */
/*               S8 bit input data with input stride equal to 4               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,                */
/*               CNN convolution params structure                             */
/* Outputs     : XI Error Code                                                */
/* InOuts      : Output Tile                                                  */
/* Assumptions : CoeffData is S8                                              */
/*               biasArray is signed 32b, value not exceeding signed 24b      */
/*               OutData is S8 / U8 / S16                                     */
/*               Kernel Size is MxNxDxN                                       */
/*               Input and Output are in WHD format                           */
/*               Coeff is in WHDN format                                      */
/******************************************************************************/

/******************** xaiConvolve3D_S_MxNj4_S8S8IX_MOW_WHD *********************/
/******************** xaiConvolve3D_S_MxNj4_U8S8IX_MOW_WHD *********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_MxNj4, S8IX_MOW_WHD) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  xai_cnn_conv_params * param
  )
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_POINTER(param);
  }
  XAI_CNN_CONV_SET_DILATION_XY(param, 1, 1);

  return(MAKE_NAME(xaiConvolved3D_S_MxNj4d1, S8IX_MOW_WHD) (inTile, coeffTile, biasArray, outTile, param));

  return(XAI_ERROR_STATUS());
}

/********************************** end of MOW variants ************************************/
/*******************************************************************************************/
#endif //if ((XCHAL_VISION_TYPE >= 6))

