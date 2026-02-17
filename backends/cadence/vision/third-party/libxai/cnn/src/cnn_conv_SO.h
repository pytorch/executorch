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
#define MORPH_OP_MULPA               IVP_MULUSPA2NX8


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
#define MORPH_OP_MULPA               IVP_MULPA2NX8
#endif

/******************************************************************************************
* SO(Single output) variants
******************************************************************************************/

/***************************************************************************/
/*  xaiConvolve3D_S_MxN_S8_SO_DWH/xaiConvolve3D_S_MxN_U8_SO_DWH      */
/***************************************************************************/

/***********************************************************************/
/* Description : P6 Optimized implementation of 3D convolution in SO   */
/*               Vectorization Approach.                               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,         */
/*               CNN convolution params structure                      */
/* Outputs     : XI Error Code                                         */
/* InOuts      : Output Tile                                           */
/* Assumptions : InData is S8/U8                                       */
/*               CoeffData is S8                                       */
/*               OutData is S8 / U8 / S16                              */
/*               Kernel Size is close to that of Input Size.           */
/*               Input and Output is in DWH format.                    */
/*               Coeff is in DWHN format.                              */
/*               dim1Size of Input Tile is equal to dim1Pitch of Input */
/*               Tile.                                                 */
/***********************************************************************/

/******************* xaiConvolve3D_S_MxN_S8S8IX_SO_DWH ********************/
/******************* xaiConvolve3D_S_MxN_U8S8IX_SO_DWH ********************/

XAI_ERR_TYPE MAKE_NAME(xaiConvolve3D_S_MxN, S8IX_SO_DWH) (
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
  return(MAKE_NAME(xaiConvolved3D_S_MxN, S8IX_SO_DWH) (inTile, coeffTile, biasArray, outTile, param));
  return(XAI_ERROR_STATUS());
}

/****************************** end of SO variants *****************************************/
/*******************************************************************************************/
#endif //if ((XCHAL_VISION_TYPE >= 6))
