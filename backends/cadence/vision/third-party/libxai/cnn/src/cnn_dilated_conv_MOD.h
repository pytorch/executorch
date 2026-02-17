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
* MOD WHD DWH variants
******************************************************************************************/

/*****************************************************************************
*  xaiConvolvedVQ3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 1x1 MOD_WHD_DWH    */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 1x1 MOD_WHD_DWH 3D     */
/*               dilated convolution function and 1x1 MOD_WHD_DWH 3D VQ     */
/*               dilated convolution function                               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 1x1xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 1);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "Stride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                           \
                    XAI_ERR_BADARG, "\nStride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) == 1), \
                    XAI_ERR_BADARG, "\nDilation = %hhu\nDilation should be 1", XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                                             \
                    XAI_ERR_BADARG, "\nDilation along width = %hhu and height = %hhu\nDilation along width and height should be equal", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_CONSISTENCY_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }

#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    /* Max value of Gather Offset is (min(numInCh-1,7)*inDataPitch2 + stride*min(3,outWidth-1)) */
    if (numInCh > 1)
    {
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2_PITCH(inTile) <                                                                       \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) * XT_MIN(3, outW - 1)) / XT_MIN(numInCh - 1, 7)),            \
                      XAI_ERR_BADARG, "\ndim2Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                      XAI_TILE3D_GET_DIM2_PITCH(inTile),                                                                        \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) * XT_MIN(3, outW - 1)) / XT_MIN(numInCh - 1, 7)));
    }
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideU       = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  /* Pitch of Coefficient Data (NDWH) in dim1 (W = 1 and H = 1) */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);

  /* Pitches of Input Data (WHD) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecOut;

#if XCHAL_HAVE_SUPERGATHER == 0
  xb_vec2Nx8* pdvecCoeff1;
  xb_vec2Nx8* pdvecCoeff2;
  valign vIn;
  xb_vec2Nx8* pdvecIn1;
  xb_vec2Nx8* pdvecIn2;

  /*updating sel1 corresponding to 8 outCh and,4 width from input, hence
     for 8 input channel and 4 width elements from each load selection,
     sel1=0,64,0+strideU,64+strideU,0+2*strideU,64+2*strideU,0+3*strideU,64+3*strideU,0+4*strideU,64+4*strideU,...
     ...0+7*strideU,64+7*strideU*/
  xb_vec2Nx8U sel  = IVP_SEQ2NX8();
  xb_vecNx16U off  = IVP_MULNX16PACKL(IVP_ANDNX16(1, IVP_SEQNX16()), 64);
  xb_vec2Nx8U off1 = IVP_SEL2NX8I(IVP_MOV2NX8_FROMNX16(off), IVP_MOV2NX8_FROMNX16(off), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
  xb_vec2Nx8U sel1 = 0, sel2 = 0;
  sel2 = IVP_SEL2NX8UI(IVP_MUL2NX8(IVP_SEQ2NX8U(), strideU), IVP_MUL2NX8(IVP_SEQ2NX8U(), strideU), IVP_SELI_8B_INTERLEAVE_1_LO);
  sel2 = IVP_ADD2NX8U(sel2, off1);
  IVP_SEL2NX8UT(sel1, 0, sel2, IVP_SEQ2NX8U(), IVP_LT2NX8(sel, 16));

  xb_vec2Nx8 dvecIn  = 0, dvecIn1 = 0, dvecIn2 = 0, dvecIn3 = 0, dvecIn4 = 0;
  xb_vec2Nx8 dvecIn5 = 0, dvecIn6 = 0, dvecIn7 = 0, dvecIn8 = 0;

  /*implementation follows loading 8 input vectors corresponding to 8 inCh and ,first four elements
     along width */

  int32_t remainingInCh = numInCh - ((numInCh >> 3) << 3);

  uint8_t remCh1   = 0, remCh2 = 0, remCh3 = 0, remCh4 = 0, remCh5 = 0, remCh6 = 0;
  int32_t sumMask1 = 0, sumMask2 = 0;

  if (remainingInCh != 0) /* if numInCh is not a multiple of 8*/
  {
    /* Generating Coefficient mask such that coefficient load happens only for valid channel number*/
    /* Coefficient mask entries for channels greater than the remainingInCh are set to 0 */
    /* Generating Coefficient mask such that coefficient load happens only for valid channel number*/
    /* Coefficient mask entries for channels grea	ter than the remainingInCh are set to 0 */
    remCh1 = XT_SALT(1, remainingInCh);
    remCh2 = XT_SALT(2, remainingInCh);
    remCh3 = XT_SALT(3, remainingInCh);
    remCh4 = XT_SALT(4, remainingInCh);
    remCh5 = XT_SALT(5, remainingInCh);
    remCh6 = XT_SALT(6, remainingInCh);

    /*Generation of maskLut for handling cases when remainingInCh is not equal to 0   */
    /*eg. if remainingInCh is equal to 2 then sumMask1 is 00FFFFFF and sumMask2 is 0  */
    /*    if remainingInCh is equal to 3 then sumMask1 is FFFFFFFF and sumMask2 is 0  */
    /*    if remainingInCh is equal to 4 then sumMask1 is FFFFFFFF and sumMask2 is FF */
    const uint32_t maskLut[4] = { 0xff, 0xff00, 0xff0000, 0xff000000 };

    sumMask1 = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2 + maskLut[3] * remCh3;
    sumMask2 = maskLut[0] * remCh4 + maskLut[1] * remCh5 + maskLut[2] * remCh6;
  }

  /* Unrolling of 4 is done along output width and 8 along input channels */
  /**          Loop Starts            **/
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Along output channels*/
  {
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y++)   /* Along output height*/
    {
      for (x = 0; x < outW; x += 4)   /*Along output width*/
      {
        /* Input Data and Output Data Pointers */
        int8_t* pSrc = pInData + y * inDataPitch1 * strideU + x * strideU;
        int8_t* pOut = &pOutData[(y * outDataPitch2 + x * outDataPitch1) * bytesPerPixel];

        /*  For corner case handling  */
        int32_t remainingX = XT_MIN(4, outW - x);

        /* Loading bias and initializing sum with bias*/
        xb_vec2Nx24 dvecSum0, dvecSum1, dvecSum2, dvecSum3;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, dvecSum0, dvecSum1, dvecSum2, dvecSum3);

        /* Coefficient Pointer */
        pdvecCoeff1 = (xb_vec2Nx8 *) (&pCoeffData[outCh]);
        pdvecCoeff2 = (xb_vec2Nx8 *) (&pCoeffData[outCh] + coeffPitch1);
        pdvecIn1    = (xb_vec2Nx8 *) pSrc;
        pdvecIn2    = (xb_vec2Nx8 *) (pSrc + inDataPitch2);

        for (inCh = 0; inCh < (numInCh - 7); inCh += 8)
        {
          /*Loading input vector */
          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn1, vIn, pdvecIn1, 2 * inDataPitch2);

          vIn = IVP_LA2NX8_PP(pdvecIn2);
          IVP_LA2NX8_XP(dvecIn2, vIn, pdvecIn2, 2 * inDataPitch2);

          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn3, vIn, pdvecIn1, 2 * inDataPitch2);

          vIn = IVP_LA2NX8_PP(pdvecIn2);
          IVP_LA2NX8_XP(dvecIn4, vIn, pdvecIn2, 2 * inDataPitch2);

          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn5, vIn, pdvecIn1, 2 * inDataPitch2);

          vIn = IVP_LA2NX8_PP(pdvecIn2);
          IVP_LA2NX8_XP(dvecIn6, vIn, pdvecIn2, 2 * inDataPitch2);

          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn7, vIn, pdvecIn1, 2 * inDataPitch2);

          vIn = IVP_LA2NX8_PP(pdvecIn2);
          IVP_LA2NX8_XP(dvecIn8, vIn, pdvecIn2, 2 * inDataPitch2);

          /*dvecIn,dvecIn1 loaded with first 2 and next 2 elements of inChannels as x
             is unrolled 4 times loaded as first element of dvecIn1,first element of dvecIn2....first element of dvecIn4,
             second element of dvecIn1,second element of dvecIn2....second element of dvecIn4,
             third element of dvecIn1,third element of dvecIn2....third element of dvecIn4,
             fourth element of dvecIn1,fourth element of dvecIn2....fourth element of dvecIn4 for
             dvecIn, for dvecIn2 next four elements of input*/
          dvecIn  = IVP_SEL2NX8(dvecIn2, dvecIn1, sel1);
          dvecIn2 = IVP_SEL2NX8(dvecIn4, dvecIn3, sel1);
          dvecIn1 = IVP_SEL2NX8(dvecIn6, dvecIn5, sel1);
          dvecIn3 = IVP_SEL2NX8(dvecIn8, dvecIn7, sel1);
          dvecIn  = IVP_SEL2NX8I(dvecIn2, dvecIn, IVP_SELI_INTERLEAVE_1_LO);
          dvecIn1 = IVP_SEL2NX8I(dvecIn3, dvecIn1, IVP_SELI_INTERLEAVE_1_LO);

          /* 8 Coefficient Vector Loads */
          /* Load Coefficients to vector - coefficients already aligned  */
          xb_vec2Nx8 dvecCoeff0;
          IVP_LV2NX8_XP(dvecCoeff0, pdvecCoeff1, 2 * coeffPitch1);

          xb_vec2Nx8 dvecCoeff1;
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff2, 2 * coeffPitch1);

          xb_vec2Nx8 dvecCoeff2;
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, 2 * coeffPitch1);

          xb_vec2Nx8 dvecCoeff3;
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff2, 2 * coeffPitch1);

          xb_vec2Nx8 dvecCoeff4;
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff1, 2 * coeffPitch1);

          xb_vec2Nx8 dvecCoeff5;
          IVP_LV2NX8_XP(dvecCoeff5, pdvecCoeff2, 2 * coeffPitch1);

          xb_vec2Nx8 dvecCoeff6;
          IVP_LV2NX8_XP(dvecCoeff6, pdvecCoeff1, 2 * coeffPitch1);

          xb_vec2Nx8 dvecCoeff7;
          IVP_LV2NX8_XP(dvecCoeff7, pdvecCoeff2, 2 * coeffPitch1);


          /* Load 4 bytes(4 channels) of input data along the depth to int32_t scalar */
          xb_vecN_2x32v hvecIn  = IVP_MOVN_2X32_FROM2NX8(dvecIn);
          xb_vecN_2x32v hvecIn1 = IVP_MOVN_2X32_FROM2NX8(dvecIn1);

          int32_t scalarInData0 = IVP_EXTRN_2X32(hvecIn, 0);
          int32_t scalarInData1 = IVP_EXTRN_2X32(hvecIn1, 0);

          int32_t scalarInData2 = IVP_EXTRN_2X32(hvecIn, 1);
          int32_t scalarInData3 = IVP_EXTRN_2X32(hvecIn1, 1);

          int32_t scalarInData4 = IVP_EXTRN_2X32(hvecIn, 2);
          int32_t scalarInData5 = IVP_EXTRN_2X32(hvecIn1, 2);

          int32_t scalarInData6 = IVP_EXTRN_2X32(hvecIn, 3);
          int32_t scalarInData7 = IVP_EXTRN_2X32(hvecIn1, 3);

          /* Multiply and accumulate */
          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData0);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData2);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData4);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData6);

          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData1);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData3);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData5);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData7);
        } /* end of for(inCh = 0; inCh < numInCh; inCh+=8)*/

        if (inCh < numInCh)
        {
          /*Loading input vector */
          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn1, vIn, pdvecIn1, inDataPitch2 * remCh1);

          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn2, vIn, pdvecIn1, inDataPitch2 * remCh2);

          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn3, vIn, pdvecIn1, inDataPitch2 * remCh3);

          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn4, vIn, pdvecIn1, inDataPitch2 * remCh4);

          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn5, vIn, pdvecIn1, inDataPitch2 * remCh5);

          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn6, vIn, pdvecIn1, inDataPitch2 * remCh6);

          vIn = IVP_LA2NX8_PP(pdvecIn1);
          IVP_LA2NX8_XP(dvecIn7, vIn, pdvecIn1, inDataPitch2);

          dvecIn  = IVP_SEL2NX8(dvecIn2, dvecIn1, sel1);
          dvecIn2 = IVP_SEL2NX8(dvecIn4, dvecIn3, sel1);
          dvecIn1 = IVP_SEL2NX8(dvecIn6, dvecIn5, sel1);
          dvecIn3 = IVP_SEL2NX8(dvecIn8, dvecIn7, sel1);
          dvecIn  = IVP_SEL2NX8I(dvecIn2, dvecIn, IVP_SELI_INTERLEAVE_1_LO);
          dvecIn1 = IVP_SEL2NX8I(dvecIn3, dvecIn1, IVP_SELI_INTERLEAVE_1_LO);

          /* Load 4 bytes(4 channels) of input data along the depth to int32_t scalar */
          xb_vecN_2x32v hvecIn  = IVP_MOVN_2X32_FROM2NX8(dvecIn);
          xb_vecN_2x32v hvecIn1 = IVP_MOVN_2X32_FROM2NX8(dvecIn1);

          int32_t scalarInData0 = IVP_EXTRN_2X32(hvecIn, 0);
          int32_t scalarInData1 = IVP_EXTRN_2X32(hvecIn1, 0);

          int32_t scalarInData2 = IVP_EXTRN_2X32(hvecIn, 1);
          int32_t scalarInData3 = IVP_EXTRN_2X32(hvecIn1, 1);

          int32_t scalarInData4 = IVP_EXTRN_2X32(hvecIn, 2);
          int32_t scalarInData5 = IVP_EXTRN_2X32(hvecIn1, 2);

          int32_t scalarInData6 = IVP_EXTRN_2X32(hvecIn, 3);
          int32_t scalarInData7 = IVP_EXTRN_2X32(hvecIn1, 3);

          /* 8 Coefficient Vector Loads */
          /* Load Coefficients to vector - coefficients already aligned  */
          xb_vec2Nx8 dvecCoeff0;
          IVP_LV2NX8_XP(dvecCoeff0, pdvecCoeff1, coeffPitch1 * remCh1);

          xb_vec2Nx8 dvecCoeff1;
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff1, coeffPitch1 * remCh2);

          xb_vec2Nx8 dvecCoeff2;
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, coeffPitch1 * remCh3);

          xb_vec2Nx8 dvecCoeff3;
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff1, coeffPitch1 * remCh4);

          xb_vec2Nx8 dvecCoeff4;
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff1, coeffPitch1 * remCh5);

          xb_vec2Nx8 dvecCoeff5;
          IVP_LV2NX8_XP(dvecCoeff5, pdvecCoeff1, coeffPitch1 * remCh6);

          xb_vec2Nx8 dvecCoeff6;
          IVP_LV2NX8_XP(dvecCoeff6, pdvecCoeff1, coeffPitch1);

          /* Multiply and accumulate */
          /* Masking the scalarInData to avoid accumulation with unintended values*/
          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData0 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData2 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData4 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData6 & sumMask1);

          IVP_MULQA2N8XR8(dvecSum0, 0, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData1 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum1, 0, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData3 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum2, 0, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData5 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum3, 0, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData7 & sumMask2);
        } /* end of if (inCh < numInCh)*/

        /* Storing output vector to memory */
        xb_vec2Nx8 dvecOutData0L, dvecOutData1L, dvecOutData2L, dvecOutData3L;
        xb_vec2Nx8 dvecOutData0H, dvecOutData1H, dvecOutData2H, dvecOutData3H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData0L, dvecOutData0H, dvecSum0, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData1L, dvecOutData1H, dvecSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData2L, dvecOutData2H, dvecSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData3L, dvecOutData3H, dvecSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData0L, dvecOutData0H, dvecSum0, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData1L, dvecOutData1H, dvecSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData2L, dvecOutData2H, dvecSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData3L, dvecOutData3H, dvecSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        pdvecOut = (xb_vec2Nx8 *) &pOut[outCh * bytesPerPixel];
        valign vaOutData = IVP_ZALIGN();
        IVP_SAV2NX8_XP(dvecOutData0L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOutData0H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        pdvecOut = (xb_vec2Nx8 *) &pOut[(outCh + outDataPitch1) * bytesPerPixel * XT_SALT(0, remainingX - 1)];
        IVP_SAV2NX8_XP(dvecOutData1L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * XT_SALT(0, remainingX - 1));
        IVP_SAV2NX8_XP(dvecOutData1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * XT_SALT(0, remainingX - 1));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        pdvecOut = (xb_vec2Nx8 *) &pOut[(outCh + 2 * outDataPitch1) * bytesPerPixel * XT_SALT(0, remainingX - 2)];
        IVP_SAV2NX8_XP(dvecOutData2L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * XT_SALT(0, remainingX - 2));
        IVP_SAV2NX8_XP(dvecOutData2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * XT_SALT(0, remainingX - 2));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        pdvecOut = (xb_vec2Nx8 *) &pOut[(outCh + 3 * outDataPitch1) * bytesPerPixel * XT_SALT(0, remainingX - 3)];
        IVP_SAV2NX8_XP(dvecOutData3L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * XT_SALT(0, remainingX - 3));
        IVP_SAV2NX8_XP(dvecOutData3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * XT_SALT(0, remainingX - 3));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* end of for(x = 0; x < outW; x+=4)*/
    }   /* end of for(y = 0; y < outH; y++)*/
  }     /* end of for(outCh = 0; outCh < numOutCh; outCh+=2*XCHAL_IVPN_SIMD_WIDTH)*/

#else
#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  xb_vec2Nx8* restrict pdvecCoeff;

  /* This implementation uses gather operation to load 4 bytes of data each from 8 channels */

  /*****     Gather Offset Computation -  8channels, 4cols, 1row   *****/
  /*offset = pitch*[0 1 2 3 4 5 6 7 ... 0 1 2 3 4 5 6 7] +             */
  /*        stride*[0 0 0 0 0 0 0 0 ... 3 3 3 3 3 3 3 3]               */
  /* where [0 0 0 0 0 0 0 0 ... 3 3 3 3 3 3 3 3] =>> column indices    */
  /*       [0 1 2 3 4 5 6 7 ... 0 1 2 3 4 5 6 7] =>> channel indices   */
  xb_vecNx16U vecOffsets0 = IVP_MULNX16PACKL(IVP_ANDNX16(7, IVP_SEQNX16()), inDataPitch2);
  IVP_MULANX16PACKL(vecOffsets0, IVP_SRLINX16(IVP_SEQNX16(), 3), strideU);


  /*******           Gather Offset Computation and Coeff Mask           ********/
  /*******  for Corner Case : (InCh < numInCh) && (InCh > (numInCh -7)) ********/

  int32_t remainingInCh = numInCh - ((numInCh >> 3) << 3);

  xb_vecNx16U vecOffsets1 = (xb_vecNx16U) 0;
  uint8_t remCh1          = 0, remCh2 = 0, remCh3 = 0, remCh4 = 0, remCh5 = 0, remCh6 = 0;
  int32_t sumMask1        = 0, sumMask2 = 0;

  if (remainingInCh != 0) /* if numInCh is not a multiple of 8*/
  {
    /* Generating Coefficient mask such that coefficient load happens only for valid channel number*/
    /* Coefficient mask entries for channels greater than the remainingInCh are set to 0 */

    /* Finding the gather offset such that valid memory locations are accessed       */
    /* [0 1 2 3 4 5 6 7 ... 0 1 2 3 4 5 6 7] in offset calculation is modified such  */
    /* that columns greater than (remainingInCh-1) are set to (remainingInCh-1)      */
    xb_vecNx16 vecRemainingInChIdx = IVP_MINNX16(IVP_ANDNX16(7, IVP_SEQNX16()), remainingInCh - 1);
    vecOffsets1 = IVP_MULNX16PACKL(vecRemainingInChIdx, inDataPitch2);
    IVP_MULANX16PACKL(vecOffsets1, IVP_SRLINX16(IVP_SEQNX16(), 3), strideU);

    /* Generating Coefficient mask such that coefficient load happens only for valid channel number*/
    /* Coefficient mask entries for channels greater than the remainingInCh are set to 0 */
    remCh1 = XT_SALT(1, remainingInCh);
    remCh2 = XT_SALT(2, remainingInCh);
    remCh3 = XT_SALT(3, remainingInCh);
    remCh4 = XT_SALT(4, remainingInCh);
    remCh5 = XT_SALT(5, remainingInCh);
    remCh6 = XT_SALT(6, remainingInCh);

    /*Generation of maskLut for handling cases when remainingInCh is not equal to 0   */
    /*eg. if remainingInCh is equal to 2 then sumMask1 is 00FFFFFF and sumMask2 is 0  */
    /*    if remainingInCh is equal to 3 then sumMask1 is FFFFFFFF and sumMask2 is 0  */
    /*    if remainingInCh is equal to 4 then sumMask1 is FFFFFFFF and sumMask2 is FF */
    const uint32_t maskLut[4] = { 0xff, 0xff00, 0xff0000, 0xff000000 };

    sumMask1 = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2 + maskLut[3] * remCh3;
    sumMask2 = maskLut[0] * remCh4 + maskLut[1] * remCh5 + maskLut[2] * remCh6;
  }

  /* Unrolling of 4 is done along output width and 8 along input channels */
  /**          Loop Starts            **/
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Along output channels*/
  {
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y++)   /* Along output height*/
    {
      for (x = 0; x < outW; x += 4)   /*Along output width*/
      {
        xb_vecNx16U vecOffsets2;
        xb_vecNx16U vecOffsets3;
        /* Input Data and Output Data Pointers */
        int8_t* pSrc = pInData + y * inDataPitch1 * strideU + x * strideU;
        int8_t* pOut = &pOutData[(y * outDataPitch2 + x * outDataPitch1) * bytesPerPixel];

        /*  For corner case handling  */
        int32_t remainingX  = XT_MIN(4, outW - x);
        vboolN vbOffsetMask = IVP_LTRSN(8 * remainingX);   /*8 channels*/
        /* Assign valid address for predicated false lines */
        vecOffsets2 = IVP_MOVNX16UT(vecOffsets0, 0, vbOffsetMask);
        vecOffsets3 = IVP_MOVNX16UT(vecOffsets1, 0, vbOffsetMask);
        /* Loading bias and initializing sum with bias*/
        xb_vec2Nx24 dvecSum0, dvecSum1, dvecSum2, dvecSum3;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, dvecSum0, dvecSum1, dvecSum2, dvecSum3);

        /* Coefficient Pointer */
        pdvecCoeff = (xb_vec2Nx8 *) (&pCoeffData[outCh]);

        for (inCh = 0; inCh < (numInCh - 7); inCh += 8)
        {
          /* Gather Operation to load 8 channels of 1x4 block of input . dvecIn will contain data  */
          /* from 8 channels corresponding to same x and y value in consecutive positions.         */
          xb_gsr gatherReg  = IVP_GATHERANX8S(pSrc + inCh * inDataPitch2, vecOffsets2);
          xb_vec2Nx8 dvecIn = IVP_GATHERD2NX8_L(gatherReg);  /* LSB 8 bits of gatherReg contain the desired data*/

          /* 8 Coefficient Vector Loads */
          /* Load Coefficients to vector - coefficients already aligned  */
          xb_vec2Nx8 dvecCoeff0;
          IVP_LV2NX8_XP(dvecCoeff0, pdvecCoeff, coeffPitch1);

          xb_vec2Nx8 dvecCoeff1;
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);

          xb_vec2Nx8 dvecCoeff2;
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);

          xb_vec2Nx8 dvecCoeff3;
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

          xb_vec2Nx8 dvecCoeff4;
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

          xb_vec2Nx8 dvecCoeff5;
          IVP_LV2NX8_XP(dvecCoeff5, pdvecCoeff, coeffPitch1);

          xb_vec2Nx8 dvecCoeff6;
          IVP_LV2NX8_XP(dvecCoeff6, pdvecCoeff, coeffPitch1);

          xb_vec2Nx8 dvecCoeff7;
          IVP_LV2NX8_XP(dvecCoeff7, pdvecCoeff, coeffPitch1);


          /* Load 4 bytes(4 channels) of input data along the depth to int32_t scalar */
          int32_t scalarInData0 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 0);
          int32_t scalarInData1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 1);

          int32_t scalarInData2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 2);
          int32_t scalarInData3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 3);

          int32_t scalarInData4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 4);
          int32_t scalarInData5 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 5);

          int32_t scalarInData6 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 6);
          int32_t scalarInData7 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 7);

          /* Multiply and accumulate */
          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData0);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData2);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData4);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData6);

          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData1);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData3);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData5);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData7);
        } /* end of for(inCh = 0; inCh < numInCh; inCh+=8)*/

        if (inCh < numInCh)
        {
          /* Gather Operation to load remainingCh number of channels corresponding to 1x4 block */
          /* of input. The channels to be loaded are handled by vecOffsets1 */
          xb_gsr gatherReg  = IVP_GATHERANX8S(pSrc + inCh * inDataPitch2, vecOffsets3);
          xb_vec2Nx8 dvecIn = IVP_GATHERD2NX8_L(gatherReg); /* LSB 8 bits of gatherReg contain the desired data*/

          /* Load 4 bytes of input data along the depth to int32_t scalar */
          int32_t scalarInData0 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 0);
          int32_t scalarInData1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 1);

          int32_t scalarInData2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 2);
          int32_t scalarInData3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 3);

          int32_t scalarInData4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 4);
          int32_t scalarInData5 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 5);

          int32_t scalarInData6 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 6);
          int32_t scalarInData7 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 7);

          /* 8 Coefficient Vector Loads */
          /* Load Coefficients to vector - coefficients already aligned  */
          xb_vec2Nx8 dvecCoeff0;
          IVP_LV2NX8_XP(dvecCoeff0, pdvecCoeff, coeffPitch1 * remCh1);

          xb_vec2Nx8 dvecCoeff1;
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh2);

          xb_vec2Nx8 dvecCoeff2;
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh3);

          xb_vec2Nx8 dvecCoeff3;
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1 * remCh4);

          xb_vec2Nx8 dvecCoeff4;
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 * remCh5);

          xb_vec2Nx8 dvecCoeff5;
          IVP_LV2NX8_XP(dvecCoeff5, pdvecCoeff, coeffPitch1 * remCh6);
          xb_vec2Nx8 dvecCoeff6;
          IVP_LV2NX8_XP(dvecCoeff6, pdvecCoeff, coeffPitch1);

          /* Multiply and accumulate */
          /* Masking the scalarInData to avoid accumulation with unintended values*/
          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData0 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData2 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData4 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData6 & sumMask1);

          IVP_MULQA2N8XR8(dvecSum0, 0, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData1 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum1, 0, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData3 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum2, 0, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData5 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum3, 0, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData7 & sumMask2);
        } /* end of if (inCh < numInCh)*/

        /* Storing output vector to memory */
        xb_vec2Nx8 dvecOutData0L, dvecOutData1L, dvecOutData2L, dvecOutData3L;
        xb_vec2Nx8 dvecOutData0H, dvecOutData1H, dvecOutData2H, dvecOutData3H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData0L, dvecOutData0H, dvecSum0, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData1L, dvecOutData1H, dvecSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData2L, dvecOutData2H, dvecSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData3L, dvecOutData3H, dvecSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData0L, dvecOutData0H, dvecSum0, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData1L, dvecOutData1H, dvecSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData2L, dvecOutData2H, dvecSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData3L, dvecOutData3H, dvecSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        pdvecOut = (xb_vec2Nx8 *) &pOut[outCh * bytesPerPixel];
        valign vaOutData = IVP_ZALIGN();
        IVP_SAV2NX8_XP(dvecOutData0L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOutData0H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        pdvecOut = (xb_vec2Nx8 *) &pOut[(outCh + outDataPitch1) * bytesPerPixel * XT_SALT(0, remainingX - 1)];
        IVP_SAV2NX8_XP(dvecOutData1L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * XT_SALT(0, remainingX - 1));
        IVP_SAV2NX8_XP(dvecOutData1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * XT_SALT(0, remainingX - 1));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        pdvecOut = (xb_vec2Nx8 *) &pOut[(outCh + 2 * outDataPitch1) * bytesPerPixel * XT_SALT(0, remainingX - 2)];
        IVP_SAV2NX8_XP(dvecOutData2L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * XT_SALT(0, remainingX - 2));
        IVP_SAV2NX8_XP(dvecOutData2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * XT_SALT(0, remainingX - 2));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        pdvecOut = (xb_vec2Nx8 *) &pOut[(outCh + 3 * outDataPitch1) * bytesPerPixel * XT_SALT(0, remainingX - 3)];
        IVP_SAV2NX8_XP(dvecOutData3L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * XT_SALT(0, remainingX - 3));
        IVP_SAV2NX8_XP(dvecOutData3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * XT_SALT(0, remainingX - 3));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* end of for(x = 0; x < outW; x+=4)*/
    }   /* end of for(y = 0; y < outH; y++)*/
  }     /* end of for(outCh = 0; outCh < numOutCh; outCh+=2*XCHAL_IVPN_SIMD_WIDTH)*/
#endif

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_2x2_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 2x2 MOD_WHD_DWH    */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 2x2 MOD_WHD_DWH 3D     */
/*               dilated convolution function and 2x2 MOD_WHD_DWH 3D VQ     */
/*               dilated convolution function                               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 2x2xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_2x2_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_2x2_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 2);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "Stride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                           \
                    XAI_ERR_BADARG, "\nStride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) > 0),                                 \
                    XAI_ERR_BADARG, "\nDilation = %hhu, value should be greater than zero", \
                    XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                                             \
                    XAI_ERR_BADARG, "\nDilation along width = %hhu and height = %hhu\nDilation along width and height should be equal", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_EDGES_MOD_WHD(inTile, coeffTile, param);
    XAI_CHECK_CONSISTENCY_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    if (XAI_CNN_CONV_GET_DILATION(param) > 1)
    {
      XAI_CHECK_ERROR(XAI_CNN_CONV_GET_STRIDE(param) == 1,                                                                                  \
                      XAI_ERR_BADARG, "\nStride = %hhu, Dilation = %hhu\nWhen dilation parameter is more than 1 stride always has to be 1", \
                      XAI_CNN_CONV_GET_STRIDE(param), XAI_CNN_CONV_GET_DILATION(param));
    }
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }

#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    if (numInCh > 1)
    {
      /* Max value of Gather Offset is (stride* inDataPitch1)+ stride + (min(numInCh-1,3)*inDataPitch2 + dilation) */
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2_PITCH(inTile) <                                                                         \
                      ((USHRT_MAX - (XAI_CNN_CONV_GET_STRIDE(param) * XAI_TILE3D_GET_DIM1_PITCH(inTile) *                         \
                                     XT_MIN(1, outH - 1)) - XAI_CNN_CONV_GET_STRIDE(param)) - XAI_CNN_CONV_GET_DILATION(param)) / \
                      XT_MIN(numInCh - 1, 3),                                                                                     \
                      XAI_ERR_BADARG, "\ndim2Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d",   \
                      XAI_TILE3D_GET_DIM2_PITCH(inTile),                                                                          \
                      ((USHRT_MAX - (XAI_CNN_CONV_GET_STRIDE(param) * XAI_TILE3D_GET_DIM1_PITCH(inTile) *                         \
                                     XT_MIN(1, outH - 1)) - XAI_CNN_CONV_GET_STRIDE(param)) - XAI_CNN_CONV_GET_DILATION(param)) / \
                      XT_MIN(numInCh - 1, 3));
    }
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);
  const uint8_t dilation      = XAI_CNN_CONV_GET_DILATION(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);
  const int32_t kSizeU      = XAI_TILE4D_GET_DIM3(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  int32_t dilatedKWidthU      = dilation * (kSizeU - 1) + 1;
  int32_t dilatedKHeightU     = dilation * (kSizeU - 1) + 1;
  int32_t leftEdge, topEdge;

  if ((dilatedKWidthU % 2) != 0)
  {
    leftEdge = dilatedKWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedKWidthU / 2) : ((dilatedKWidthU / 2) - 1);
  }

  if ((dilatedKHeightU % 2) != 0)
  {
    topEdge = dilatedKHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedKHeightU / 2) : ((dilatedKHeightU / 2) - 1);
  }

  /* move to start of edge data including edges */
  pInData = &pInData[-(topEdge * inDataPitch1 + leftEdge)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y;
  valign vaOutData = IVP_ZALIGN();

  /* Only one Gather is used in the inner most loop in this
   * approach to get the Input Data for 4 Output Vectors.
   * In every Gather, 32 elements are read, where first 16
   * of them correspond to two vectors of Output along the width
   * and the other  16 of them correspond to two vectors of Output
   * along the height. To get the index values for the Gather,
   * the following calculations are made.
   */

  /* Sequence - 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 ... */
  xb_vecNx16 vecGather0123 = IVP_ANDNX16(IVP_SEQNX16(), 3);
  xb_vecNx16 vecSelIdx     = IVP_SEQNX16();
  /* To get the Select indexes as - 0 1 2 3 4...7 32 33 34 35 36.... */
  IVP_ADDNX16T(vecSelIdx, vecSelIdx, 24, IVP_NOTBN(IVP_LTRNI(8)));
  /* To get - 0 0 0 0 d*1 d*1 d*1 d*1 d*2 d*2 d*2 d*2 d*3 d*3 d*3 d*3... */
  xb_vecNx16U vecGatherOff = IVP_SRLINX16(IVP_SEQNX16(), 2);
  vecGatherOff = IVP_MULNX16UPACKL(vecGatherOff, (uint16_t) dilation);
  /* Sequence - 0 P2 2*P2 3*P2 d*1 P2+d*1 2*P2+d*1 3*P2+d*1 d*2 P2+d*2 2*P2+d*2 3*P2+d*2 .. */
  IVP_MULANX16PACKL(vecGatherOff, vecGather0123, inDataPitch2);
  vecGatherOff = IVP_SELNX16(IVP_ADDNX16(vecGatherOff, stride), \
                             vecGatherOff, vecSelIdx);
  vecSelIdx = IVP_SEQNX16();
  IVP_ADDNX16T(vecSelIdx, vecSelIdx, 16, IVP_NOTBN(IVP_LTRNI(16)));
  vecGatherOff = IVP_SELNX16(IVP_ADDNX16(vecGatherOff, stride * inDataPitch1), vecGatherOff, vecSelIdx);

  /*
     The generated sequence is:
   * 0               P2            2*P2           3*P2
   * d               P2+d          2*P2+d         3*P2+d
   * s               s+P2          s+2*P2         s+3*P2
   * s+d*1           s+P2+d        s+2*P2+d       s+3*P2+d
   * (s*P1)+0       (s*P1)+P2      (s*P1)+2*P2    (s*P1)+3*P2
   * (s*P1)+d       (s*P1)+P2+d   (s*P1)+2*P2+d   (s*P1)+3*P2+d
   * (s*P1)+s       (s*P1)+s+P2     (s*P1)+s+2*P2 (s*P1)+s+3*P2
   * (s*P1)+s+d     (s*P1)+s+P2+d (s*P1)+s+2*P2+d (s*P1)+s+3*P2+d
   */

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;


  int32_t remInCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remInCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint8_t remCh1 = XT_SALT(2, remInCh + 1);
  uint8_t remCh2 = XT_SALT(3, remInCh + 1);

  uint32_t sumMask = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* Unrolled by 2 along both Output Width and Output Height.
   * Also, unrolled along Input Channels by 4 and completely
   * along the Kernel Width. Gathers are used for loading Input Data.
   */
  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used to handle the corner case of OutHeight being odd */
      int32_t numY = XT_MIN(2, outH - y) - 1;
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff1;

        /* Variable used to handle the corner case of Output Width being odd */
        int32_t numX = XT_MIN(2, outW - x) - 1;

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) + (y * stride) * inDataPitch1;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean vectors to handle the corner cases of Out Width and Height being odd */
        vboolN vbXY = IVP_LTRSN((16 * numY) + 8 * (numX + 1));

        /* Initialise input data pointers */
        pData1 = pData;
        pData2 = pData + (dilation * inDataPitch1);

        /* Initialise co-efficient pointer */
        pdvecCoeff = (xb_vec2Nx8 *) (pCoeff);

        /* Assign gather offset considering corner cases of odd output height and width */
        vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, vbXY);

        for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels Loop */
        {
          /* Gather Input Data corresponding to ky=0 */
          xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff1);
          xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);

          /* Gather Input Data corresponding to ky=1 */
          xb_gsr gather2       = IVP_GATHERANX8S(pData2, vecGatherOff1);
          xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);

          /* kx = 0, ky =0 */
          /* Extracting scalar integers for QMULs */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 2);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 6);

          /* 4 Aligned Vector Loads of coefficients */
          xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, \
                                               coeffPitch2 - (3 * coeffPitch1));

          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

          /* kx = 1, ky = 0*/
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       1);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       3);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       5);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       7);

          /* 4 Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, -3 * coeffPitch1 - coeffPitch2 + coeffPitch3);

          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

          /* kx = 0, ky =1 */
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 2);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 4);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 6);

          /* 4 Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - (3 * coeffPitch1));

          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);


          /* kx = 1, ky = 1*/
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       1);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       3);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       5);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       7);

          /* 4 Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 - coeffPitch2 - coeffPitch3);

          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          pData1 += (4 * inDataPitch2);
          pData2 += (4 * inDataPitch2);
        } /* End Input Channels */

        /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
        if (remInCh)
        {
          vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);

          /* Gather Input Data */
          xb_vec2Nx8 dvecData1 = 0;

          /* Assign valid address for predicated false lines */
          vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbXY));
          xb_gsr gather1 = IVP_GATHERANX8S(pData1, vecGatherOff1);
          dvecData1 = IVP_GATHERD2NX8_L(gather1);

          /* Gather Input Data */
          xb_vec2Nx8 dvecData2 = 0;
          xb_gsr gather2       = IVP_GATHERANX8S(pData2, vecGatherOff1);
          dvecData2 = IVP_GATHERD2NX8_L(gather2);

          /* kx = 0, ky = 0 */
          /* Extracting scalar integers for QMULs */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 2);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 6);

          /* Aligned Vector Loads of coefficients */
          xb_vec2Nx8 dvecCoeff1;
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
          xb_vec2Nx8 dvecCoeff2;
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
          xb_vec2Nx8 dvecCoeff3;
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

          /* Masking the qmulScalar values to avoid accumulation with unintended values*/
          IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
          IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
          IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
          IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

          /* kx = 1, ky = 0 */
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       1);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       3);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       5);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       7);

          /* Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, (-(remCh1 + remCh2) * coeffPitch1) - coeffPitch2 + coeffPitch3);

          /* Masking the qmulScalar values to avoid accumulation with unintended values*/
          IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
          IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
          IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
          IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

          /* kx = 0, ky = 1 */
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 2);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 4);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 6);

          /* Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (remCh1 + remCh2) * coeffPitch1);

          /* Masking the qmulScalar values to avoid accumulation with unintended values*/
          IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
          IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
          IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
          IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

          /* kx = 1, ky = 1*/
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       1);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       3);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       5);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       7);

          /* Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, 0);

          /* Masking the qmulScalar values to avoid accumulation with unintended values*/
          IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
          IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
          IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
          IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
        } /* End Input Channels Corner case Handling */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * \
                       numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 3x3 MOD_WHD_DWH    */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 3x3 MOD_WHD_DWH 3D     */
/*               dilated convolution function and 3x3 MOD_WHD_DWH 3D VQ     */
/*               dilated convolution function                               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 3x3xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/


#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 3);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                         \
                    XAI_ERR_BADARG, "Stride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "\nStride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) > 0),                                 \
                    XAI_ERR_BADARG, "\nDilation = %hhu, value should be greater than zero", \
                    XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                                             \
                    XAI_ERR_BADARG, "\nDilation along width = %hhu and height = %hhu\nDilation along width and height should be equal", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_EDGE(inTile, 1 + (XAI_CNN_CONV_GET_DILATION(param) - 1));
    XAI_CHECK_CONSISTENCY_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));

    if (XAI_CNN_CONV_GET_DILATION(param) > 1)
    {
      XAI_CHECK_ERROR(XAI_CNN_CONV_GET_STRIDE(param) == 1,                                                                                  \
                      XAI_ERR_BADARG, "\nStride = %hhu, Dilation = %hhu\nWhen dilation parameter is more than 1 stride always has to be 1", \
                      XAI_CNN_CONV_GET_STRIDE(param), XAI_CNN_CONV_GET_DILATION(param));
    }
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }


#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    if (numInCh > 1)
    {
      /* Max value of Gather Offset is (min(numInCh-1,3)*inDataPitch2 + stride + 2 * dilation) */
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2_PITCH(inTile) <                                                                       \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) -                                                            \
                        2 * XAI_CNN_CONV_GET_DILATION(param)) / XT_MIN(numInCh - 1, 3)),                                        \
                      XAI_ERR_BADARG, "\ndim2Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                      XAI_TILE3D_GET_DIM2_PITCH(inTile),                                                                        \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) -                                                            \
                        2 * XAI_CNN_CONV_GET_DILATION(param)) / XT_MIN(numInCh - 1, 3)));
    }
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);
  const uint8_t dilation      = XAI_CNN_CONV_GET_DILATION(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);
  const int32_t kSizeU      = XAI_TILE4D_GET_DIM3(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t dilatedKSize = dilation * (kSizeU - 1) + 1;

  /* move to start of edge data only when input is already padded. */
  pInData = &pInData[-((dilatedKSize / 2) * inDataPitch1 + (dilatedKSize / 2))];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;
  valign vaOutData = IVP_ZALIGN();

  /* Only 2 Gathers are used in this approach to get the
   * Input Data for 4 Output Vectors. In each Gather,
   * 24 elements are read, where each 12 of them correspond
   * to one vector of Output along the width. To get the
   * index values for the Gather, the following calculations
   * are made.
   */

  /* Gather Index Calculations */
  /* Sequence - 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 ... */
  xb_vecNx16 vecGather0123 = IVP_ANDNX16(IVP_SEQNX16(), 3);
  xb_vecNx16 vecSelIdx     = IVP_SEQNX16();
  /* To get the Select indexes as - 0 1 2 3 4...11 32 33 34 35 36.... */
  IVP_ADDNX16T(vecSelIdx, vecSelIdx, (XCHAL_IVPN_SIMD_WIDTH - 12), IVP_NOTBN(IVP_LTRNI(12)));
  /* To get - 0 0 0 0 d*1 d*1 d*1 d*1 d*2 d*2 d*2 d*2 d*3 d*3 d*3 d*3... */
  xb_vecNx16U vecGatherOff = IVP_SRLINX16(IVP_SEQNX16(), 2);
  vecGatherOff = IVP_MULNX16UPACKL(vecGatherOff, (uint16_t) dilation);
  /* Sequence - 0 P2 2*P2 3*P2 d*1 P2+d*1 2*P2+d*1 3*P2+d*1 d*2 P2+d*2 2*P2+d*2 3*P2+d*2 .. */
  IVP_MULANX16PACKL(vecGatherOff, vecGather0123, inDataPitch2);
  vecGatherOff = IVP_SELNX16(IVP_ADDNX16(vecGatherOff, stride), \
                             vecGatherOff, vecSelIdx);
  /* Final Index Pattern is -
   * 0 P2 2*P2 3*P2 d*1 P2+d*1 2*P2+d*1 3*P2+d*1 d*2 P2+d*2 2*P2+d*2 3*P2+d*2
   * s s+P2 s+2*P2 s+3*P2 s+d*1 s+P2+d*1 s+2*P2+d*1 s+3*P2+d*1 s+2 s+P2+d*2 s+2*P2+d*2 s+3*P2+d*2*/

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;

  int32_t remInCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remInCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint8_t remCh1 = XT_SALT(2, remInCh + 1);
  uint8_t remCh2 = XT_SALT(3, remInCh + 1);

  uint32_t sumMask = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* Unrolled by 2 along both Output Width and Output Height.
   * Also, unrolled along Input Channels by 4 and completely
   * along the Kernel Width. Gathers are used for loading Input Data.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used to handle the corner case of OutHeight being odd */
      int32_t numY = XT_MIN(2, outH - y) - 1;
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff1;
        xb_vecNx16U vecGatherOff2;

        /* Variable used to handle the corner case of Output Width being odd */
        int32_t numX = XT_MIN(2, outW - x) - 1;

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) + (y * stride) * inDataPitch1;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean vectors to handle the corner cases of Out Width and Height being odd */
        vboolN vbX = IVP_LTRSN(12 * (numX + 1));
        vboolN vbY = IVP_LTRSN(12 * (numX + 1) * numY);

        for (ky = 0; ky < 3; ky++) /* Kernel Height Loop */
        {
          /* Pointer for Input Data Load */
          pData1 = pData + ky * dilation * inDataPitch1;
          pData2 = pData1 + (stride * inDataPitch1 * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);
          /* Assign valid address for predicated false lines */
          vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, vbX);
          vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, vbY);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels Loop */
          {
            /* Gather Input Data */
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff1);
            xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2       = IVP_GATHERANX8S(pData2, vecGatherOff2);
            xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);

            pData1 += 4 * inDataPitch2;
            pData2 += 4 * inDataPitch2;

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 3);

            /* 4 Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - \
                                                 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         4);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         5);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 - 2 * coeffPitch2);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */

          /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh  = numInCh - inCh;
            vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);

            /* Gather Input Data */
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            /* Assign valid address for predicated false lines */
            vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbX));
            vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbY));

            xb_gsr gather1 = IVP_GATHERANX8S(pData1, vecGatherOff1);
            dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2 = IVP_GATHERANX8S(pData2, vecGatherOff2);
            dvecData2 = IVP_GATHERD2NX8_L(gather2);

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 3);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh2 + remCh1)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values*/
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         4);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         5);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, 0);

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
          } /* End Input Channels Corner case Handling */
        }   /* End Kernel Height Loop */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * \
                       numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_4x4_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 2x2 MOD_WHD_DWH    */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 2x2 MOD_WHD_DWH 3D     */
/*               dilated convolution function and 2x2 MOD_WHD_DWH 3D VQ     */
/*               dilated convolution function                               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 4x4xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_4x4_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_4x4_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param)
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 4);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "Stride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                           \
                    XAI_ERR_BADARG, "\nStride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) > 0),                                 \
                    XAI_ERR_BADARG, "\nDilation = %hhu, value should be greater than zero", \
                    XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                                             \
                    XAI_ERR_BADARG, "\nDilation along width = %hhu and height = %hhu\nDilation along width and height should be equal", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_EDGES_MOD_WHD(inTile, coeffTile, param);
    XAI_CHECK_CONSISTENCY_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    if (XAI_CNN_CONV_GET_DILATION(param) > 1)
    {
      XAI_CHECK_ERROR(XAI_CNN_CONV_GET_STRIDE(param) == 1,                                                                                  \
                      XAI_ERR_BADARG, "\nStride = %hhu, Dilation = %hhu\nWhen dilation parameter is more than 1 stride always has to be 1", \
                      XAI_CNN_CONV_GET_STRIDE(param), XAI_CNN_CONV_GET_DILATION(param));
    }
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }

#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    if (numInCh > 1)
    {
      /* Max value of Gather Offset is (min(numInCh-1,3)*inDataPitch2 + stride + 3 * dilation) */
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2_PITCH(inTile) <                                                                       \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) - 3 * XAI_CNN_CONV_GET_DILATION(param)) /                    \
                       XT_MIN(numInCh - 1, 3)),                                                                                 \
                      XAI_ERR_BADARG, "\ndim2Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                      XAI_TILE3D_GET_DIM2_PITCH(inTile),                                                                        \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) - 3 * XAI_CNN_CONV_GET_DILATION(param)) /                    \
                       XT_MIN(numInCh - 1, 3)));
    }
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);
  const uint8_t dilation      = XAI_CNN_CONV_GET_DILATION(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);
  const int32_t kSizeU      = XAI_TILE4D_GET_DIM3(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t dilatedKSize = dilation * (kSizeU - 1) + 1;
  int32_t leftEdge, topEdge;

  if ((dilatedKSize % 2) != 0)
  {
    leftEdge = dilatedKSize / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedKSize / 2) : ((dilatedKSize / 2) - 1);
  }

  if ((dilatedKSize % 2) != 0)
  {
    topEdge = dilatedKSize / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedKSize / 2) : ((dilatedKSize / 2) - 1);
  }

  /* Move pointer to the start of the active data (including edge) */
  pInData = &pInData[-(topEdge * inDataPitch1 + leftEdge)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;
  valign vaOutData = IVP_ZALIGN();

  /* Only 2 Gathers are used in this approach to get the
   * Input Data for 4 Output Vectors. In each Gather,
   * 32 elements are read, where each 16 of them correspond
   * to one vector of Output along the width. To get the
   * index values for the Gather, the following calculations
   * are made.
   */

  /* Gather Index Calculations */
  /* Sequence - 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 ... */
  xb_vecNx16 vecGather0123 = IVP_ANDNX16(IVP_SEQNX16(), 3);
  xb_vecNx16 vecSelIdx     = IVP_SEQNX16();
  /* To get the Select indexes as - 0 1 2 3 4...11 32 33 34 35 36.... */
  IVP_ADDNX16T(vecSelIdx, vecSelIdx, 16, IVP_NOTBN(IVP_LTRNI(16)));
  /* To get - 0 0 0 0 d*1 d*1 d*1 d*1 d*2 d*2 d*2 d*2 d*3 d*3 d*3 d*3... */
  xb_vecNx16U vecGatherOff = IVP_SRLINX16(IVP_SEQNX16(), 2);
  vecGatherOff = IVP_MULNX16UPACKL(vecGatherOff, (uint16_t) dilation);
  /* Sequence - 0 P2 2*P2 3*P2 d*1 P2+d*1 2*P2+d*1 3*P2+d*1 d*2 P2+d*2 2*P2+d*2 3*P2+d*2 .. */
  IVP_MULANX16PACKL(vecGatherOff, vecGather0123, inDataPitch2);
  vecGatherOff = IVP_SELNX16(IVP_ADDNX16(vecGatherOff, stride), \
                             vecGatherOff, vecSelIdx);

  /* Final Index Pattern is -
   * First 16 elements
   * 0    P2      2*P2      3*P2
   * d*1  P2+d*1  2*P2+d*1  3*P2+d*1
   * d*2  P2+d*2  2*P2+d*2  3*P2+d*2
   * d*3  P2+d*3  2*P2+d*3  3*P2+d*3
   *
   * Last 16 elements
   * s      s+P2      s+2*P2      s+3*P2
   * s+d*1  s+P2+d*1  s+2*P2+d*1  s+3*P2+d*1
   * s+d*2  s+P2+d*2  s+2*P2+d*2  s+3*P2+d*2
   * s+d*3  s+P2+d*3  s+2*P2+d*3  s+3*P2+d*3
   */

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff1;
  xb_vec2Nx8* restrict pdvecCoeff2;
  xb_vec2Nx8* restrict pdvecCoeff3;
  xb_vec2Nx8* restrict pdvecCoeff4;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;


  int32_t remInCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remInCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint8_t remCh1 = XT_SALT(2, remInCh + 1);
  uint8_t remCh2 = XT_SALT(3, remInCh + 1);

  uint32_t sumMask = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* Unrolled by 2 along both Output Width and Output Height.
   * Also, unrolled along Input Channels by 4 and completely
   * along the Kernel Width. Gathers are used for loading Input Data.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used to handle the corner case of OutHeight being odd */
      int32_t numY = XT_MIN(2, outH - y) - 1;
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff1;
        xb_vecNx16U vecGatherOff2;

        /* Variable used to handle the corner case of Output Width being odd */
        int32_t numX = XT_MIN(2, outW - x) - 1;

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) + (y * stride) * inDataPitch1;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean vectors to handle the corner cases of Out Width and Height being odd */
        vboolN vbX = IVP_LTRSN(16 * (numX + 1));
        vboolN vbY = IVP_LTRSN(16 * (numX + 1) * numY);

        for (ky = 0; ky < 4; ky++) /* Kernel Height Loop */
        {
          /* Pointer for Input Data Load */
          pData1 = pData + ky * dilation * inDataPitch1;
          pData2 = pData1 + (stride * inDataPitch1 * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff1 = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);
          pdvecCoeff2 = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3 + coeffPitch2);
          pdvecCoeff3 = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3 + 2 * coeffPitch2);
          pdvecCoeff4 = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3 + 3 * coeffPitch2);
          /* Assign valid address for predicated false lines */
          vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, vbX);
          vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, vbY);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels Loop */
          {
            /* Gather Input Data */
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff1);
            xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2       = IVP_GATHERANX8S(pData2, vecGatherOff2);
            xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);

            pData1 += 4 * inDataPitch2;
            pData2 += 4 * inDataPitch2;

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 4);

            /* 4 Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff1, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff1, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff1, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         5);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff2, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff2, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff2, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff2, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         6);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff3, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff3, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff3, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff3, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 4 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         3);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         7);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         3);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         7);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff4, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff4, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff4, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff4, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */

          /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh  = numInCh - inCh;
            vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);

            /* Gather Input Data */
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            /* Assign valid address for predicated false lines */
            vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbX));
            vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbY));

            xb_gsr gather1 = IVP_GATHERANX8S(pData1, vecGatherOff1);
            dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2 = IVP_GATHERANX8S(pData2, vecGatherOff2);
            dvecData2 = IVP_GATHERD2NX8_L(gather2);

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 4);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff1, coeffPitch1 * remCh1);
            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, coeffPitch1 * remCh2);
            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff1, 0);

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         5);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff2, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff2, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff2, 0);

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         6);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff3, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff3, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff3, 0);

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 4 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         3);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         7);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         3);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         7);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff4, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff4, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff4, 0);

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
          } /* End Input Channels Corner case Handling */
        }   /* End Kernel Height Loop */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * \
                       numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 5x5 MOD_WHD_DWH    */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 5x5 MOD_WHD_DWH 3D     */
/*               dilated convolution function and 5x5 MOD_WHD_DWH 3D VQ     */
/*               dilated convolution function                               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 5x5xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 5);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                         \
                    XAI_ERR_BADARG, "Stride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "\nStride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) > 0),                                 \
                    XAI_ERR_BADARG, "\nDilation = %hhu, value should be greater than zero", \
                    XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                                             \
                    XAI_ERR_BADARG, "\nDilation along width = %hhu and height = %hhu\nDilation along width and height should be equal", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_EDGE(inTile, 2 + 2 * (XAI_CNN_CONV_GET_DILATION(param) - 1));
    XAI_CHECK_CONSISTENCY_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    if (XAI_CNN_CONV_GET_DILATION(param) > 1)
    {
      XAI_CHECK_ERROR(XAI_CNN_CONV_GET_STRIDE(param) == 1,                                                                                  \
                      XAI_ERR_BADARG, "\nStride = %hhu, Dilation = %hhu\nWhen dilation parameter is more than 1 stride always has to be 1", \
                      XAI_CNN_CONV_GET_STRIDE(param), XAI_CNN_CONV_GET_DILATION(param));
    }
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }

#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    if (numInCh > 1)
    {
      /* Max value of Gather Offset is (min(numInCh-1,3)*inDataPitch2 + stride + 4 * dilation) */
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2_PITCH(inTile) <                                                                       \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) - 4 * XAI_CNN_CONV_GET_DILATION(param)) /                    \
                       XT_MIN(numInCh - 1, 3)),                                                                                 \
                      XAI_ERR_BADARG, "\ndim2Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                      XAI_TILE3D_GET_DIM2_PITCH(inTile),                                                                        \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) - 4 * XAI_CNN_CONV_GET_DILATION(param)) /                    \
                       XT_MIN(numInCh - 1, 3)));
    }
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);
  const uint8_t dilation      = XAI_CNN_CONV_GET_DILATION(param);
  const int32_t kSizeU        = XAI_TILE4D_GET_DIM3(coeffTile);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t dilatedKSize = dilation * (kSizeU - 1) + 1;

  /* move to start of edge data only when input is already padded. */
  pInData = &pInData[-((dilatedKSize / 2) * inDataPitch1 + (dilatedKSize / 2))];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;


  /* 4 Gathers are being used to Load Input Data. Many common elements
   * will be loaded in separate Gathers, especially in the case of
   * stride 1 and 2. To take the advantage of having common offsets in
   * a single Gather, 2 Gather Patterns are generated as given below.
   * For example, in the Gather Patterns generated below,
   * if stride is 1 and dilation equal to 1, then 8 offsets are common and if stride is 2, 4 offsets
   * are common in each Gather.
   */

  /* Gather Index Calculations */
  xb_vecNx16 vecGather = IVP_SRLINX16(IVP_SEQNX16(), 2);
  vecGather = IVP_MULNX16UPACKL(vecGather, (uint16_t) dilation);
  IVP_MULANX16PACKL(vecGather, inDataPitch2, IVP_ANDNX16(IVP_SEQNX16(), 3));
  xb_vecNx16 vecGather1 = IVP_ADDNX16(vecGather, stride);

  xb_vecNx16 vecSelIdx1 = IVP_SEQNX16();
  IVP_ADDNX16T(vecSelIdx1, vecSelIdx1, (XCHAL_IVPN_SIMD_WIDTH - 12), IVP_NOTBN(IVP_LTRNI(12)));
  xb_vecNx16U vecGatherOff1 = IVP_SELNX16(vecGather1, vecGather, vecSelIdx1);
  xb_vecNx16 vecSelIdx2     = IVP_ADDNX16(IVP_SEQNX16(), 12);
  IVP_ADDNX16T(vecSelIdx2, vecSelIdx2, (XCHAL_IVPN_SIMD_WIDTH - 12), IVP_NOTBN(IVP_LTRNI(8)));
  xb_vecNx16U vecGatherOff2 = IVP_SELNX16(vecGather1, vecGather, vecSelIdx2);
  /* Index Pattern of vecGatherOff1 is -
   * 0 P2 2*P2 3*P2 d*1 P2+d*1 2*P2+d*1 3*P2+d*1 d*2 P2+d*2 2*P2+d*2 3*P2+d*2
   * s s+P2 s+2*P2 s+3*P2 s+d*1 s+d*1+P2 s+d*1+2*P2 s+d*1+3*P2 */

  /* Index Pattern of vecGatherOff2 is -
   * d*3 P2+d*3 2*P2+d*3 3*P2+d*3 d*4 P2+d*4 2*P2+d*4 3*P2+d*4 s+d*2 s+d*2+P2 s+d*2+2*P2 s+d*2+3*P2
   * s+d*3 s+d*3+P2 s+d*3+2*P2 s+d*3+3*P2 s+d*4 s+d*4+P2 s+d*4+2*P2 s+d*4+3*P2 */

  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;

  int32_t remInCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remInCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint32_t sumMask = maskLut[0] + maskLut[1] * XT_SALT(2, remInCh + 1) + maskLut[2] * XT_SALT(3, remInCh + 1);


#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* 4 Gathers used for Input Data Load. Unrolled along */
  /* Output Width and Height by 2. Also, unrolled along */
  /* Input Channels by 4 and Kernel Width.              */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Out Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used for corner case handling of Out Height odd */
      int32_t numY = XT_MIN(2, outH - y) - 1;
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff00;
        xb_vecNx16U vecGatherOff01;
        xb_vecNx16U vecGatherOff10;
        xb_vecNx16U vecGatherOff11;

        /* Variable used for corner case handling of Out Width odd */
        int32_t numX = XT_MIN(2, outW - x) - 1;

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) + (y * stride) * inDataPitch1;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean Vectors for Predicate Gather with corner cases  */
        /* handled for Out Width and Height being odd numbers      */
        vboolN vb1 = IVP_ORBN(IVP_LTRNI(12), IVP_LTRSN(20 * numX));
        vboolN vb2 = IVP_ORBN(IVP_LTRNI(8), IVP_LTRSN(20 * numX));
        vboolN vb3 = IVP_ANDBN(vb1, IVP_LTRSN(20 * numY));
        vboolN vb4 = IVP_ANDBN(vb2, IVP_LTRSN(20 * numY));

        for (ky = 0; ky < 5; ky++) /* Kernel Height */
        {
          /* Pointer for Input Data Load */
          pData1 = pData + ky * dilation * inDataPitch1;
          pData2 = pData1 + (stride * inDataPitch1 * numY);
          /* Assign valid address for predicated false lines */
          vecGatherOff00 = IVP_MOVNX16UT(vecGatherOff1, 0, vb1);
          vecGatherOff01 = IVP_MOVNX16UT(vecGatherOff2, 0, vb2);
          vecGatherOff10 = IVP_MOVNX16UT(vecGatherOff1, 0, vb3);
          vecGatherOff11 = IVP_MOVNX16UT(vecGatherOff2, 0, vb4);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            /* Gather Load of Input Data */
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff00);
            xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2       = IVP_GATHERANX8S(pData1, vecGatherOff01);
            xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);
            xb_gsr gather3       = IVP_GATHERANX8S(pData2, vecGatherOff10);
            xb_vec2Nx8 dvecData3 = IVP_GATHERD2NX8_L(gather3);
            xb_gsr gather4       = IVP_GATHERANX8S(pData2, vecGatherOff11);
            xb_vec2Nx8 dvecData4 = IVP_GATHERD2NX8_L(gather4);

            pData1 += 4 * inDataPitch2;
            pData2 += 4 * inDataPitch2;

            /* kx = 1 */
            /* Extracting scalars for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 3);

            /* 4 Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * \
                                                 coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 2 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         4);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 3 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         2);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 4 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         0);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         3);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         0);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         3);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 5 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         4);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 - 4 * coeffPitch2);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */

          /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
          if (remInCh)
          {
            vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);
            /* Assign valid address for predicated false lines */
            vecGatherOff00 = IVP_MOVNX16UT(vecGatherOff1, 0, IVP_ANDBN(vbRemInCh, vb1));
            vecGatherOff01 = IVP_MOVNX16UT(vecGatherOff2, 0, IVP_ANDBN(vbRemInCh, vb2));
            vecGatherOff10 = IVP_MOVNX16UT(vecGatherOff1, 0, IVP_ANDBN(vbRemInCh, vb3));
            vecGatherOff11 = IVP_MOVNX16UT(vecGatherOff2, 0, IVP_ANDBN(vbRemInCh, vb4));

            /* Gather Input Data */
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            xb_vec2Nx8 dvecData3 = 0;
            xb_vec2Nx8 dvecData4 = 0;
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff00);
            dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2 = IVP_GATHERANX8S(pData1, vecGatherOff01);
            dvecData2 = IVP_GATHERD2NX8_L(gather2);
            xb_gsr gather3 = IVP_GATHERANX8S(pData2, vecGatherOff10);
            dvecData3 = IVP_GATHERD2NX8_L(gather3);
            xb_gsr gather4 = IVP_GATHERANX8S(pData2, vecGatherOff11);
            dvecData4 = IVP_GATHERD2NX8_L(gather4);

            /* kx = 1 */
            /* Extracting scalars for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 3);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * XT_SALT(2, remInCh + 1));
            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * XT_SALT(3, \
                                                                        remInCh + 1));
            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - coeffPitch1 * (XT_SALT(2, remInCh + 1) + XT_SALT(3, remInCh + 1)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 2 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         4);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * XT_SALT(2, remInCh + 1));
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * XT_SALT(3, remInCh + 1));
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - coeffPitch1 * (XT_SALT(2, remInCh + 1) + XT_SALT(3, remInCh + 1)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 3 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         2);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * XT_SALT(2, remInCh + 1));
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * XT_SALT(3, remInCh + 1));
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - coeffPitch1 * (XT_SALT(2, remInCh + 1) + XT_SALT(3, remInCh + 1)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 4 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         0);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         3);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         0);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         3);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * XT_SALT(2, remInCh + 1));
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * XT_SALT(3, remInCh + 1));
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (XT_SALT(2, remInCh + 1) + XT_SALT(3, remInCh + 1))));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 5 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         4);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * XT_SALT(2, remInCh + 1));
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * XT_SALT(3, remInCh + 1));
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, 0);

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
          } /* End Input Channels corner case handling */
        }   /* End Kernel Height */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * \
                       numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 7x7 MOD_WHD_DWH    */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 7x7 MOD_WHD_DWH 3D     */
/*               dilated convolution function and 7x7 MOD_WHD_DWH 3D VQ     */
/*               dilated convolution function                               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 7x7xDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 7);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                         \
                    XAI_ERR_BADARG, "Stride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "\nStride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) == 1) ||                                                          \
                    ((XAI_CNN_CONV_GET_DILATION(param) >= 1) &&                                                         \
                     (XAI_CNN_CONV_GET_STRIDE(param) == 1)), XAI_ERR_BADARG,                                            \
                    "\nDilation = %hhu\nDilation should be 1. It can be greater than 1 only when stride is equal to 1", \
                    XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                                             \
                    XAI_ERR_BADARG, "\nDilation along width = %hhu and height = %hhu\nDilation along width and height should be equal", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_EDGE(inTile, 3 + 3 * (XAI_CNN_CONV_GET_DILATION(param) - 1));
    XAI_CHECK_CONSISTENCY_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }

#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    if (numInCh > 1)
    {
      /* Max value of Gather Offset is (min(numInCh-1,3)*inDataPitch2 + stride + 6*dilation) */
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2_PITCH(inTile) <                                                                             \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) - 6 * XAI_CNN_CONV_GET_DILATION(param)) / XT_MIN(numInCh - 1, 3)), \
                      XAI_ERR_BADARG, "dim2Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d",         \
                      XAI_TILE3D_GET_DIM2_PITCH(inTile),                                                                              \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) - 6 * XAI_CNN_CONV_GET_DILATION(param)) / XT_MIN(numInCh - 1, 3)));
    }
  }

  /* Kernel Size (NDWH) */
  const int32_t kSizeU = XAI_TILE4D_GET_DIM3(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);
  const uint8_t dilation      = XAI_CNN_CONV_GET_DILATION(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Effective Kernel size = dilation(KernelSize - 1) + 1                */
  /* Effective kernel size is used for calculating the min required edge */
  int32_t dilatedKSizeU = dilation * (kSizeU - 1) + 1;

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((dilatedKSizeU / 2) * inDataPitch1 + (dilatedKSizeU / 2))];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;

  /* 4 Gathers are being used to Load Input Data. Many common elements
   * will be loaded in separate Gathers, especially in the case of
   * stride 1 and 2. To take the advantage of having common offsets in
   * a single Gather, 2 Gather Patterns are generated as given below.
   * For example, in the Gather Patterns generated below,
   * if stride is 1 and dilation = 1, then 12 offsets are common and
   * if stride is 2 and dilation = 1, 8 offsets are common in each Gather.
   */
  /* Gather Index Calculations */
  xb_vecNx16 vecGather = IVP_MULNX16PACKL(dilation, IVP_SRLINX16(IVP_SEQNX16(), 2));
  IVP_MULANX16PACKL(vecGather, inDataPitch2, IVP_ANDNX16(IVP_SEQNX16(), 3));
  xb_vecNx16 vecGather1 = IVP_ADDNX16(vecGather, stride);

  xb_vecNx16 vecSelIdx1 = IVP_SEQNX16();
  IVP_ADDNX16T(vecSelIdx1, vecSelIdx1, (XCHAL_IVPN_SIMD_WIDTH - 16), IVP_NOTBN(IVP_LTRNI(16)));
  xb_vecNx16U vecGatherOff1 = IVP_SELNX16(vecGather1, vecGather, vecSelIdx1);
  xb_vecNx16 vecSelIdx2     = IVP_ADDNX16(IVP_SEQNX16(), 16);
  IVP_ADDNX16T(vecSelIdx2, vecSelIdx2, (XCHAL_IVPN_SIMD_WIDTH - 16), IVP_NOTBN(IVP_LTRNI(12)));
  xb_vecNx16U vecGatherOff2 = IVP_SELNX16(vecGather1, vecGather, vecSelIdx2);

  /* Index Pattern of vecGatherOff1 is -
   * 0 P2 2*P2 3*P2 1*d P2+1*d 2*P2+1*d 3*P2+1*d 2*d P2+2*d 2*P2+2*d 3*P2+2*d 3*d P2+3*d 2*P2+3*d 3*P2+3*d
   * s s+P2 s+2*P2 s+3*P2 s+1*d s+1*d+P2 s+1*d+2*P2 s+1*d+3*P2 s+2*d s+2*d+P2 s+2*d+2*P2 s+2*d+3*P2 */

  /* Index Pattern of vecGatherOff2 is -
   * 4*d P2+4*d 2*P2+4*d 3*P2+4*d 5*d P2+5*d 2*P2+5*d 3*P2+5*d 6*d P2+6*d 2*P2+6*d 3*P2+6*d s+3*d s+3*d+P2 s+3*d+2*P2 s+3*d+3*P2
   * s+4*d s+4*d+P2 s+4*d+2*P2 s+4*d+3*P2 s+5*d s+5*d+P2 s+5*d+2*P2 s+5*d+3*P2 s+6*d s+6*d+P2 s+6*d+2*P2 s+6*d+3*P2   */

  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;

  int32_t remInCh = numInCh & 3;

  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint8_t remCh1 = XT_SALT(2, remInCh + 1);
  uint8_t remCh2 = XT_SALT(3, remInCh + 1);

  uint32_t sumMask = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* 4 Gathers are used for Input Data Load corresponding to 4         */
  /* Output Vectors. Loop unrolled along Output Width and Height by 2. */
  /* Also unrolled along Input Channels by 4 and Kernel Width.         */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used for corner case handling of Out Height odd */
      int32_t numY = XT_MIN(1, outH - y - 1);
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff00;
        xb_vecNx16U vecGatherOff01;
        xb_vecNx16U vecGatherOff10;
        xb_vecNx16U vecGatherOff11;
        /* Variable used for corner case handling of Out Width odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) + (y * stride) * inDataPitch1;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);


        /* Boolean Vectors for Predicate Gather with corner cases  */
        /* handled for Out Width and Height being odd numbers      */
        vboolN vb1 = IVP_ORBN(IVP_LTRNI(16), IVP_LTRSN(28 * numX));
        vboolN vb2 = IVP_ORBN(IVP_LTRNI(12), IVP_LTRSN(28 * numX));
        vboolN vb3 = IVP_ANDBN(vb1, IVP_LTRSN(28 * numY));
        vboolN vb4 = IVP_ANDBN(vb2, IVP_LTRSN(28 * numY));

        for (ky = 0; ky < 7; ky++) /* Kernel Height */
        {
          /* Pointer for Input Data Load */
          pData1 = pData + ky * inDataPitch1 * dilation;
          pData2 = pData1 + (stride * inDataPitch1 * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);
          /* Assign valid address for predicated false lines */
          vecGatherOff00 = IVP_MOVNX16UT(vecGatherOff1, 0, vb1);
          vecGatherOff01 = IVP_MOVNX16UT(vecGatherOff2, 0, vb2);
          vecGatherOff10 = IVP_MOVNX16UT(vecGatherOff1, 0, vb3);
          vecGatherOff11 = IVP_MOVNX16UT(vecGatherOff2, 0, vb4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Number of Input Channels */
          {
            /* Gathers for Input Loads */
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff00);
            xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2       = IVP_GATHERANX8S(pData1, vecGatherOff01);
            xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);
            xb_gsr gather3       = IVP_GATHERANX8S(pData2, vecGatherOff10);
            xb_vec2Nx8 dvecData3 = IVP_GATHERD2NX8_L(gather3);
            xb_gsr gather4       = IVP_GATHERANX8S(pData2, vecGatherOff11);
            xb_vec2Nx8 dvecData4 = IVP_GATHERD2NX8_L(gather4);

            pData1 += 4 * inDataPitch2;
            pData2 += 4 * inDataPitch2;

            /* kx = 1 */
            /* Extracting Scalars for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 4);

            /* 4 Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 2 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 5);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 3 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 6);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 4 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 3);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 3);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 3);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 5 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 0);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 4);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 6 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 5);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 7 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 6);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 - 6 * coeffPitch2);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */

          /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
          if (remInCh)
          {
            vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);
            /* Assign valid address for predicated false lines */
            vecGatherOff00 = IVP_MOVNX16UT(vecGatherOff1, 0, IVP_ANDBN(vbRemInCh, vb1));
            vecGatherOff01 = IVP_MOVNX16UT(vecGatherOff2, 0, IVP_ANDBN(vbRemInCh, vb2));
            vecGatherOff10 = IVP_MOVNX16UT(vecGatherOff1, 0, IVP_ANDBN(vbRemInCh, vb3));
            vecGatherOff11 = IVP_MOVNX16UT(vecGatherOff2, 0, IVP_ANDBN(vbRemInCh, vb4));

            /* Gather Input Data */
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            xb_vec2Nx8 dvecData3 = 0;
            xb_vec2Nx8 dvecData4 = 0;
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff00);
            dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2 = IVP_GATHERANX8S(pData1, vecGatherOff01);
            dvecData2 = IVP_GATHERD2NX8_L(gather2);
            xb_gsr gather3 = IVP_GATHERANX8S(pData2, vecGatherOff10);
            dvecData3 = IVP_GATHERD2NX8_L(gather3);
            xb_gsr gather4 = IVP_GATHERANX8S(pData2, vecGatherOff11);
            dvecData4 = IVP_GATHERD2NX8_L(gather4);


            /* kx = 1 */
            /* Extracting scalars for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 4);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 2 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 5);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 3 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 6);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 4 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 3);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 3);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 3);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 5 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 0);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 4);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 6 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 5);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 7 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 6);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, 0);

            /* Masking the qmulScalar values to avoid accumulation with unintended values */
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
          } /* End Input Channels corner case handling */
        }   /* End Kernel Height */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_WHD_DWH    */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_WHD_DWH 3D     */
/*               dilated convolution function and MxN MOD_WHD_DWH 3D VQ     */
/*               dilated convolution function                               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structur        */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxN                                     */
/*               Input is in WHD and Output is in DWH format                */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                         \
                    XAI_ERR_BADARG, "Stride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 16) &&                                                                         \
                    (XAI_TILE4D_GET_DIM4(coeffTile) <= 16),                                                                           \
                    XAI_ERR_KSIZE, "\nKernel height = %d and width = %d\nKernel width and height should be less than or equal to 16", \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_WHD(inTile, coeffTile, param);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "\nStride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) == 1) ||                                                           \
                    ((XAI_CNN_CONV_GET_DILATIONX(param) >= 1) &&                                                          \
                     (XAI_CNN_CONV_GET_STRIDE(param) == 1)), XAI_ERR_BADARG,                                              \
                    "\nDilationX = %hhu\nDilationX should be 1. It can be greater than 1 only when stride is equal to 1", \
                    XAI_CNN_CONV_GET_DILATIONX(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONY(param) == 1) ||                                                           \
                    ((XAI_CNN_CONV_GET_DILATIONY(param) >= 1) &&                                                          \
                     (XAI_CNN_CONV_GET_STRIDE(param) == 1)), XAI_ERR_BADARG,                                              \
                    "\nDilationY = %hhu\nDilationY should be 1. It can be greater than 1 only when stride is equal to 1", \
                    XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_WHD);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_CONSISTENCY_MOD_WHD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }

#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Getting parameters from the tile structures */
  const int32_t outW      = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH      = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh   = XAI_TILE3D_GET_DIM3(inTile);
  const int32_t numOutCh  = XAI_TILE3D_GET_DIM1(outTile);
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    if (numInCh > 1)
    {
      /* Max value of Gather Offset is (min(numInCh-1,7)*inDataPitch + stride*min(3,outWidth-1)) */
      XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM2_PITCH(inTile) <                                                                       \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) * XT_MIN(3, outW - 1)) / XT_MIN(numInCh - 1, 7)),            \
                      XAI_ERR_BADARG, "\ndim2Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                      XAI_TILE3D_GET_DIM2_PITCH(inTile),                                                                        \
                      ((USHRT_MAX - XAI_CNN_CONV_GET_STRIDE(param) * XT_MIN(3, outW - 1)) / XT_MIN(numInCh - 1, 7)));
    }
  }

  /* Kernel Size (NDWH) */
  const int32_t kWidthU   = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU  = XAI_TILE4D_GET_DIM4(coeffTile);
  int32_t dilatedkWidthU  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedkHeightU = dilationY * (kHeightU - 1) + 1;

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideU       = XAI_CNN_CONV_GET_STRIDE(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;
  if ((dilatedkWidthU % 2) != 0)
  {
    leftEdge = dilatedkWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedkWidthU / 2) : ((dilatedkWidthU / 2) - 1);
  }

  if ((dilatedkHeightU % 2) != 0)
  {
    topEdge = dilatedkHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedkHeightU / 2) : ((dilatedkHeightU / 2) - 1);
  }

  /* Move pointer to the start of the active data (including edge) */
  pInData = &pInData[-(topEdge * inDataPitch1 + leftEdge)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y;
  int32_t k;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecOut;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* The loop across kernel width and kernel height can be combined. In this  */
  /* case the address offsets for input and coefficient need to be derived    */
  /* from vector registers. These vector registers are initialized as follows */

  xb_vecN_2x32v hvecCoeffAddrOffInit = IVP_PACKVRNRN_2X64W(IVP_MULN_2X16X32_0 \
                                                             (IVP_MOVNX16_FROMN_2X32(IVP_SEQN_2X32()), coeffPitch2), 0);

  xb_vecN_2x32v hvecInAddrOffInit = IVP_PACKVRNRN_2X64W(IVP_MULHN_2X16X32_1 \
                                                          ((xb_vecNx16) dilationX, IVP_SEQN_2X32()), 16);

  /* This implementation uses one gather operation to load 4 bytes of data each from 8 channels */

  /*****    Gather Offset Computation (used inside InCh for-loop)    *****/
  /*               InCh for-loop is executed when inCh>8                 */
  /*                                                                     */
  /* offset = pitch*[0 1 2 3 4 5 6 7 ... 0 1 2 3 4 5 6 7] +              */
  /*         stride*[0 0 0 0 0 0 0 0 ... 3 3 3 3 3 3 3 3]                */
  /*  where [0 0 0 0 0 0 0 0 ... 3 3 3 3 3 3 3 3] =>> column indices     */
  /*        [0 1 2 3 4 5 6 7 ... 0 1 2 3 4 5 6 7] =>> channel indices    */
  xb_vecNx16U vecOffsets0 = IVP_ADDNX16(IVP_MULNX16PACKL(IVP_ANDNX16(7, IVP_SEQNX16()), inDataPitch2), \
                                        IVP_MULNX16PACKL(IVP_SRLINX16(IVP_SEQNX16(), 3), strideU));

  /*******  Gather Offset Computation and Coeff Mask (outside InCh for-loop)   ********/

  /* ((numInCh>>3)<<3) = largest multiple of 8 less numInCh-8 */
  /* Loop across inCh is executed only when numInCh > 8       */
  int32_t remainingInCh = (numInCh - ((numInCh >> 3) << 3));
  remainingInCh = remainingInCh != 0 ? remainingInCh : 8;

  /* Generating Coefficient mask such that coefficient load happens only for valid channel number*/
  /* Coefficient mask entries for channels greater than the remainingInCh are set to 0 */
  uint8_t remCh1 = XT_SALT(1, remainingInCh);
  uint8_t remCh2 = XT_SALT(2, remainingInCh);
  uint8_t remCh3 = XT_SALT(3, remainingInCh);
  uint8_t remCh4 = XT_SALT(4, remainingInCh);
  uint8_t remCh5 = XT_SALT(5, remainingInCh);
  uint8_t remCh6 = XT_SALT(6, remainingInCh);
  uint8_t remCh7 = XT_SALT(7, remainingInCh);

  /*Generation of maskLut for handling cases when remainingInCh is not equal to 0   */
  /*eg. if remainingInCh is equal to 2 then sumMask1 is 00FFFFFF and sumMask2 is 0  */
  /*    if remainingInCh is equal to 3 then sumMask1 is FFFFFFFF and sumMask2 is 0  */
  /*    if remainingInCh is equal to 4 then sumMask1 is FFFFFFFF and sumMask2 is FF */
  const uint32_t maskLut[4] = { 0xff, 0xff00, 0xff0000, 0xff000000 };

  int32_t sumMask1 = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2 + maskLut[3] * remCh3;
  int32_t sumMask2 = maskLut[0] * remCh4 + maskLut[1] * remCh5 + maskLut[2] * remCh6 + maskLut[3] * remCh7;

  /* Finding the gather offset such that valid memory locations are accessed       */
  /* [0 1 2 3 4 5 6 7 ... 0 1 2 3 4 5 6 7] in offset calculation is modified such  */
  /* that columns greater than (remainingInCh-1) are set to (remainingInCh-1)      */
  xb_vecNx16 vecRemainingInChIdx = IVP_MINNX16(IVP_ANDNX16(7, IVP_SEQNX16()), remainingInCh - 1);
  xb_vecNx16U vecOffsets1        = IVP_ADDNX16(IVP_MULNX16PACKL(vecRemainingInChIdx, inDataPitch2), \
                                               IVP_MULNX16PACKL(IVP_SRLINX16(IVP_SEQNX16(), 3), strideU));

  /**  Output width is unrolled by 4 and Input Channels is unrolled by 8 **/

  /********* Loop Starts ************/
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Along output channels*/
  {
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y++)   /* Along output height*/
    {
      xb_vecNx16U vecOffsets2;
      xb_vecNx16U vecOffsets3;
      for (x = 0; x < outW; x += 4)   /*Along output width*/
      {
        /*  For corner case handling  */
        int32_t remainingX  = XT_MIN(4, outW - x);
        vboolN vbOffsetMask = IVP_LTRSN(8 * remainingX);     /* 8 channels*/
        /* Assign valid address for predicated false lines */
        vecOffsets2 = IVP_MOVNX16UT(vecOffsets0, 0, vbOffsetMask);
        vecOffsets3 = IVP_MOVNX16UT(vecOffsets1, 0, vbOffsetMask);

        /*  Output pointer */
        int8_t* pOut = &pOutData[(y * outDataPitch2 + x * outDataPitch1) * bytesPerPixel];

        /* Loading bias and initializing sum with bias*/
        xb_vec2Nx24 dvecSum0 = 0, dvecSum1 = 0, dvecSum2 = 0, dvecSum3 = 0;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, dvecSum0, dvecSum1, dvecSum2, dvecSum3);

        /* Input Data and Coeff Data Pointers */
        int8_t *pSrc1  = pInData + x * strideU + y * strideU * inDataPitch1;
        int8_t *pCoeff = pCoeffData + outCh;

        xb_vecN_2x32v hvecInAddrOff    = hvecInAddrOffInit;
        xb_vecN_2x32v hvecCoeffAddrOff = hvecCoeffAddrOffInit;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t index, inAddrOff, coeffAddrOff;

        for (k = 0; k < kHeightU * kWidthU; k++) /* Kernel Height * Kernel Width */
        {
          /* Condition checks performed to get the Input and Coefficient        */
          /* Pointer Offsets after combining the Kernel Width and Height Loops  */
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch1 * dilationY, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3, vbN_2);
          index = IVP_EXTRN_2X32(hvecLaneIdx, 0);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff    = IVP_EXTRVRN_2X32(hvecInAddrOff, 4 * index);
          coeffAddrOff = IVP_EXTRVRN_2X32(hvecCoeffAddrOff, 4 * index);
          hvecLaneIdx  = IVP_ADDN_2X32(hvecLaneIdx, 1);

          /* Pointers for Input Data Loads */
          int8_t *pSrc = (pSrc1 + inAddrOff);

          /* Pointer for Coefficient Load */
#ifdef IS_VISION_130
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);
          xb_vec2Nx8* pdvecCoeff1 = (xb_vec2Nx8 *) (pCoeff + 4 * coeffPitch1 + coeffAddrOff);

          for (inCh = 0; inCh < (numInCh - 8); inCh += 8)
          {
            /* Gather Operation to load 8 channels of 1x4 block of input . dvecIn will contain data  */
            /* from 8 channels corresponding to same x and y value in consecutive positions.         */
            xb_gsr gatherReg  = IVP_GATHERANX8S(pSrc + inCh * inDataPitch2, vecOffsets2);
            xb_vec2Nx8 dvecIn = IVP_GATHERD2NX8_L(gatherReg);  /* LSB 8 bits of gatherReg  contain the desired data*/

            /* 8 Coefficient Vector Loads */
            /* Load Coefficients to vector - coefficients already aligned  */
            xb_vec2Nx8 dvecCoeff0;
            IVP_L2U2NX8_XP(dvecCoeff0, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff1;
            IVP_L2U2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff2;
            IVP_L2U2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff3;
            IVP_L2U2NX8_XP(dvecCoeff3, pdvecCoeff, 5 * coeffPitch1);

            xb_vec2Nx8 dvecCoeff4;
            IVP_L2U2NX8_XP(dvecCoeff4, pdvecCoeff1, coeffPitch1);

            xb_vec2Nx8 dvecCoeff5;
            IVP_L2U2NX8_XP(dvecCoeff5, pdvecCoeff1, coeffPitch1);

            xb_vec2Nx8 dvecCoeff6;
            IVP_L2U2NX8_XP(dvecCoeff6, pdvecCoeff1, coeffPitch1);

            xb_vec2Nx8 dvecCoeff7;
            IVP_L2U2NX8_XP(dvecCoeff7, pdvecCoeff1, 5 * coeffPitch1);

            /* Load 4 bytes of input data along the depth to int32_t scalar */
            int32_t scalarInData0 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 0);
            int32_t scalarInData1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 1);

            int32_t scalarInData2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 2);
            int32_t scalarInData3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 3);

            int32_t scalarInData4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 4);
            int32_t scalarInData5 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 5);

            int32_t scalarInData6 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 6);
            int32_t scalarInData7 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 7);

            /* Multiply and accumulate */
            IVP_MULQA2N8XR8(dvecSum0, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData0);
            IVP_MULQA2N8XR8(dvecSum1, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData2);
            IVP_MULQA2N8XR8(dvecSum2, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData4);
            IVP_MULQA2N8XR8(dvecSum3, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData6);

            IVP_MULQA2N8XR8(dvecSum0, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData1);
            IVP_MULQA2N8XR8(dvecSum1, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData3);
            IVP_MULQA2N8XR8(dvecSum2, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData5);
            IVP_MULQA2N8XR8(dvecSum3, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData7);
          }  /* end of for(inCh = 0; inCh < (numInCh-8); inCh+=8)*/

          /*Gather Operation to load remainingCh number of channels corresponding to 1x4 block   */
          /*of input. The channels to be loaded are handled by vecOffsets1 */
          xb_gsr gatherReg  = IVP_GATHERANX8S(pSrc + inCh * inDataPitch2, vecOffsets3);
          xb_vec2Nx8 dvecIn = IVP_GATHERD2NX8_L(gatherReg); /* LSB 8 bits of gatherReg contain the desired data*/

          /* Load 4 bytes of input data along the depth to int32_t scalar */
          int32_t scalarInData0 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 0);
          int32_t scalarInData1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 1);

          int32_t scalarInData2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 2);
          int32_t scalarInData3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 3);

          int32_t scalarInData4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 4);
          int32_t scalarInData5 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 5);

          int32_t scalarInData6 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 6);
          int32_t scalarInData7 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 7);

          /* 8 Coefficient Vector Loads */
          /* Load Coefficients to vector - coefficients already aligned  */
          xb_vec2Nx8 dvecCoeff0;
          IVP_LV2NX8_XP(dvecCoeff0, pdvecCoeff, coeffPitch1 * remCh1);

          xb_vec2Nx8 dvecCoeff1;
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh2);

          xb_vec2Nx8 dvecCoeff2;
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh3);

          xb_vec2Nx8 dvecCoeff3;
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1 * remCh4);

          xb_vec2Nx8 dvecCoeff4;
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 * remCh5);

          xb_vec2Nx8 dvecCoeff5;
          IVP_LV2NX8_XP(dvecCoeff5, pdvecCoeff, coeffPitch1 * remCh6);

          xb_vec2Nx8 dvecCoeff6;
          IVP_LV2NX8_XP(dvecCoeff6, pdvecCoeff, coeffPitch1 * remCh7);

          xb_vec2Nx8 dvecCoeff7;
          IVP_LV2NX8_XP(dvecCoeff7, pdvecCoeff, coeffPitch1);

          /* Multiply and accumulate */
          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData0 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData2 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData4 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData6 & sumMask1);

          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData1 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData3 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData5 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData7 & sumMask2);
        } /* end of for (k = 0; k < kHeightU * kWidthU; k++)*/

#else
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          for (inCh = 0; inCh < (numInCh - 8); inCh += 8)
          {
            /* Gather Operation to load 8 channels of 1x4 block of input . dvecIn will contain data  */
            /* from 8 channels corresponding to same x and y value in consecutive positions.         */
            xb_gsr gatherReg  = IVP_GATHERANX8S(pSrc + inCh * inDataPitch2, vecOffsets2);
            xb_vec2Nx8 dvecIn = IVP_GATHERD2NX8_L(gatherReg);    /* LSB 8 bits of gatherReg  contain the desired data*/

            /* 8 Coefficient Vector Loads */
            /* Load Coefficients to vector - coefficients already aligned  */
            xb_vec2Nx8 dvecCoeff0;
            IVP_LV2NX8_XP(dvecCoeff0, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff4;
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff5;
            IVP_LV2NX8_XP(dvecCoeff5, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff6;
            IVP_LV2NX8_XP(dvecCoeff6, pdvecCoeff, coeffPitch1);

            xb_vec2Nx8 dvecCoeff7;
            IVP_LV2NX8_XP(dvecCoeff7, pdvecCoeff, coeffPitch1);

            /* Load 4 bytes of input data along the depth to int32_t scalar */
            int32_t scalarInData0 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 0);
            int32_t scalarInData1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 1);

            int32_t scalarInData2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 2);
            int32_t scalarInData3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 3);

            int32_t scalarInData4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 4);
            int32_t scalarInData5 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 5);

            int32_t scalarInData6 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 6);
            int32_t scalarInData7 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                     (IVP_MOVNX16_FROM2NX8(dvecIn)), 7);

            /* Multiply and accumulate */
            IVP_MULQA2N8XR8(dvecSum0, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData0);
            IVP_MULQA2N8XR8(dvecSum1, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData2);
            IVP_MULQA2N8XR8(dvecSum2, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData4);
            IVP_MULQA2N8XR8(dvecSum3, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData6);

            IVP_MULQA2N8XR8(dvecSum0, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData1);
            IVP_MULQA2N8XR8(dvecSum1, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData3);
            IVP_MULQA2N8XR8(dvecSum2, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData5);
            IVP_MULQA2N8XR8(dvecSum3, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData7);
          }  /* end of for(inCh = 0; inCh < (numInCh-8); inCh+=8)*/

          /*Gather Operation to load remainingCh number of channels corresponding to 1x4 block   */
          /*of input. The channels to be loaded are handled by vecOffsets1 */
          xb_gsr gatherReg  = IVP_GATHERANX8S(pSrc + inCh * inDataPitch2, vecOffsets3);
          xb_vec2Nx8 dvecIn = IVP_GATHERD2NX8_L(gatherReg); /* LSB 8 bits of gatherReg contain the desired data*/

          /* Load 4 bytes of input data along the depth to int32_t scalar */
          int32_t scalarInData0 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 0);
          int32_t scalarInData1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 1);

          int32_t scalarInData2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 2);
          int32_t scalarInData3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 3);

          int32_t scalarInData4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 4);
          int32_t scalarInData5 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 5);

          int32_t scalarInData6 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 6);
          int32_t scalarInData7 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecIn)), 7);

          /* 8 Coefficient Vector Loads */
          /* Load Coefficients to vector - coefficients already aligned  */
          xb_vec2Nx8 dvecCoeff0;
          IVP_LV2NX8_XP(dvecCoeff0, pdvecCoeff, coeffPitch1 * remCh1);

          xb_vec2Nx8 dvecCoeff1;
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh2);

          xb_vec2Nx8 dvecCoeff2;
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh3);

          xb_vec2Nx8 dvecCoeff3;
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1 * remCh4);

          xb_vec2Nx8 dvecCoeff4;
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 * remCh5);

          xb_vec2Nx8 dvecCoeff5;
          IVP_LV2NX8_XP(dvecCoeff5, pdvecCoeff, coeffPitch1 * remCh6);

          xb_vec2Nx8 dvecCoeff6;
          IVP_LV2NX8_XP(dvecCoeff6, pdvecCoeff, coeffPitch1 * remCh7);

          xb_vec2Nx8 dvecCoeff7;
          IVP_LV2NX8_XP(dvecCoeff7, pdvecCoeff, coeffPitch1);

          /* Multiply and accumulate */
          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData0 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData2 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData4 & sumMask1);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff3, dvecCoeff2, dvecCoeff1, dvecCoeff0, scalarInData6 & sumMask1);

          IVP_MULQA2N8XR8(dvecSum0, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData1 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum1, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData3 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum2, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData5 & sumMask2);
          IVP_MULQA2N8XR8(dvecSum3, dvecCoeff7, dvecCoeff6, dvecCoeff5, dvecCoeff4, scalarInData7 & sumMask2);
        } /* end of for (k = 0; k < kHeightU * kWidthU; k++)*/
#endif

        /* Storing output vector to memory */
        xb_vec2Nx8 dvecOutData0L = 0, dvecOutData1L = 0, dvecOutData2L = 0, dvecOutData3L = 0;
        xb_vec2Nx8 dvecOutData0H = 0, dvecOutData1H = 0, dvecOutData2H = 0, dvecOutData3H = 0;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData0L, dvecOutData0H, dvecSum0, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData1L, dvecOutData1H, dvecSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData2L, dvecOutData2H, dvecSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData3L, dvecOutData3H, dvecSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData0L, dvecOutData0H, dvecSum0, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData1L, dvecOutData1H, dvecSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData2L, dvecOutData2H, dvecSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData3L, dvecOutData3H, dvecSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        pdvecOut = (xb_vec2Nx8 *) &pOut[outCh * bytesPerPixel];
        valign vaOutData = IVP_ZALIGN();
        IVP_SAV2NX8_XP(dvecOutData0L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOutData0H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        pdvecOut = (xb_vec2Nx8 *) &pOut[(outCh + outDataPitch1) * bytesPerPixel * XT_SALT(0, remainingX - 1)];
        IVP_SAV2NX8_XP(dvecOutData1L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * XT_SALT(0, remainingX - 1));
        IVP_SAV2NX8_XP(dvecOutData1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * XT_SALT(0, remainingX - 1));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        pdvecOut = (xb_vec2Nx8 *) &pOut[(outCh + 2 * outDataPitch1) * bytesPerPixel * XT_SALT(0, remainingX - 2)];
        IVP_SAV2NX8_XP(dvecOutData2L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * XT_SALT(0, remainingX - 2));
        IVP_SAV2NX8_XP(dvecOutData2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * XT_SALT(0, remainingX - 2));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        pdvecOut = (xb_vec2Nx8 *) &pOut[(outCh + 3 * outDataPitch1) * bytesPerPixel * XT_SALT(0, remainingX - 3)];
        IVP_SAV2NX8_XP(dvecOutData3L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * XT_SALT(0, remainingX - 3));
        IVP_SAV2NX8_XP(dvecOutData3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * XT_SALT(0, remainingX - 3));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* end of for(x = 0; x < outW; x+=4)*/
    }   /* end of for(y = 0; y < outH; y++)*/
  }     /* end of for(outCh = 0; outCh < numOutCh; outCh+=2*XCHAL_IVPN_SIMD_WIDTH)*/
  return(XAI_ERROR_STATUS());
}

/******************************************************************************************
* MOD DWH variants
******************************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution for handling */
/*               cases where kwidth * numInch is a multiple of 4            */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;

  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];


  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();
  int32_t numIter  = kWidthU * numInCh;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
#ifdef __XCC__
#pragma loop_count min=1
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);

#ifdef __XCC__
#pragma loop_count min=1
#endif
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

#ifdef __XCC__
#pragma loop_count min=1
#endif
          for (k = 0; k < numIter; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t numIter = kWidthU * numInCh;

  int32_t leftEdge, topEdge;
  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

          for (k = 0; k < numIter - 3; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);


            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }   /* End Input Channels */

          /* Corner case handling as numIter is not a multiple of 4 */
          {
            int32_t remInCh = numIter - k;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, remInCh);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);
            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);


            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/******************************************************************************************
* MOD DWH variants
******************************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution for handling */
/*               cases where kwidth * numInch is a multiple of 4            */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/
#ifdef IVP_MULSUQA2N8XR8
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;

  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];


  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();
  int32_t numIter  = kWidthU * numInCh;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
#ifdef __XCC__
#pragma loop_count min=1
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);

#ifdef __XCC__
#pragma loop_count min=1
#endif
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8U_PP(pdvecData4);

#ifdef __XCC__
#pragma loop_count min=1
#endif
          for (k = 0; k < numIter; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecData3; IVP_LAV2NX8U_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecData4; IVP_LAV2NX8U_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            xb_vec2Nx8U dvecS1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar1)));
            xb_vec2Nx8U dvecS2 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar2)));
            xb_vec2Nx8U dvecS3 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar3)));
            xb_vec2Nx8U dvecS4 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar4)));

            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 0), dvecCoeff1, IVP_REP2NX8U(dvecS1, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 2), dvecCoeff3, IVP_REP2NX8U(dvecS1, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 0), dvecCoeff1, IVP_REP2NX8U(dvecS2, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 2), dvecCoeff3, IVP_REP2NX8U(dvecS2, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 0), dvecCoeff1, IVP_REP2NX8U(dvecS3, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 2), dvecCoeff3, IVP_REP2NX8U(dvecS3, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 0), dvecCoeff1, IVP_REP2NX8U(dvecS4, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 2), dvecCoeff3, IVP_REP2NX8U(dvecS4, 3), dvecCoeff4);
#endif
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}
#endif

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution              */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/
#ifdef IVP_MULSUQA2N8XR8
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t numIter = kWidthU * numInCh;

  int32_t leftEdge, topEdge;
  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8U_PP(pdvecData4);

          for (k = 0; k < numIter - 3; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecData3; IVP_LAV2NX8U_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecData4; IVP_LAV2NX8U_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);


#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            xb_vec2Nx8U dvecS1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar1)));
            xb_vec2Nx8U dvecS2 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar2)));
            xb_vec2Nx8U dvecS3 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar3)));
            xb_vec2Nx8U dvecS4 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar4)));

            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 0), dvecCoeff1, IVP_REP2NX8U(dvecS1, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 2), dvecCoeff3, IVP_REP2NX8U(dvecS1, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 0), dvecCoeff1, IVP_REP2NX8U(dvecS2, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 2), dvecCoeff3, IVP_REP2NX8U(dvecS2, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 0), dvecCoeff1, IVP_REP2NX8U(dvecS3, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 2), dvecCoeff3, IVP_REP2NX8U(dvecS3, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 0), dvecCoeff1, IVP_REP2NX8U(dvecS4, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 2), dvecCoeff3, IVP_REP2NX8U(dvecS4, 3), dvecCoeff4);
#endif
          }   /* End Input Channels */

          /* Corner case handling as numIter is not a multiple of 4 */
          {
            int32_t remInCh = numIter - k;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8U dvecData3; IVP_LAV2NX8U_XP(dvecData3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8U dvecData4; IVP_LAV2NX8U_XP(dvecData4, vaData4, pdvecData4, remInCh);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData4)), 0);
            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);


#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            xb_vec2Nx8U dvecS1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar1)));
            xb_vec2Nx8U dvecS2 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar2)));
            xb_vec2Nx8U dvecS3 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar3)));
            xb_vec2Nx8U dvecS4 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar4)));

            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 0), dvecCoeff1, IVP_REP2NX8U(dvecS1, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 2), dvecCoeff3, IVP_REP2NX8U(dvecS1, 3), 0);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 0), dvecCoeff1, IVP_REP2NX8U(dvecS2, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 2), dvecCoeff3, IVP_REP2NX8U(dvecS2, 3), 0);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 0), dvecCoeff1, IVP_REP2NX8U(dvecS3, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 2), dvecCoeff3, IVP_REP2NX8U(dvecS3, 3), 0);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 0), dvecCoeff1, IVP_REP2NX8U(dvecS4, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 2), dvecCoeff3, IVP_REP2NX8U(dvecS4, 3), 0);
#endif
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * numX + outDataPitch2 * numY) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}
#endif

/*****************************************************************************
*  convolvedVQ3D_S_1x1_S8S8IXCa2_MOD_DWH_x4
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution for handling */
/*               cases where kwidth * numInch is a multiple of 4            */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 1x1xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_1x1_S8S8IXCa2_MOD_DWH_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_1x1_S8S8IXCa2_MOD_DWH_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /* Unrolled by 2 along both Output Width and Height.
   * Inner loop unrolled by 4 along the Input number of Channels.
   * Input Number of Channels less than 4 handled in a
   * separate loop.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
#ifdef __XCC__
#pragma loop_count min=1
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Corner case Handling if height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);
#ifdef __XCC__
#pragma loop_count min=1
#endif
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Corner case Handling if width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Pointer for Coefficient Load */
        int8_t *pCoeff = pCoeffData + outCh;
        pdvecCoeff = (xb_vec2Nx8 *) pCoeff;

        /* Input Data Pointers */
        int8_t *pData = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        pdvecData1 = (xb_vec2Nx8 *) pData;
        pdvecData2 = (xb_vec2Nx8 *) (pData + stride * inDataPitch1 * numX);
        pdvecData3 = (xb_vec2Nx8 *) (pData + stride * inDataPitch2 * numY);
        pdvecData4 = (xb_vec2Nx8 *) (pData + stride * (inDataPitch1 + inDataPitch2) * numX * numY);

        valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
        valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
        valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
        valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (inCh = 0; inCh < numInCh; inCh += 4) /* Input Channels */
        {
          /* Aligning variable vector load of pixels */
          xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
          xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
          xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);
          xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, 4);

          /* Extracting first 4 bytes of vector into address register */
          /* Scalar integers to be used for QMUL                      */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

          /* 4 Aligned Vector Loads of coefficients */
          xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

          /* Quad Muls */
          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
        } /* End Input Channels */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        int8_t *pOut = pOutData + (outCh + x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        valign vaOutData = IVP_ZALIGN();
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pOut     = pOutData + (outCh + (x + 1) * outDataPitch1 + y * outDataPitch2) * bytesPerPixel * numX;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pOut     = pOutData + (outCh + x * outDataPitch1 + (y + 1) * outDataPitch2) * bytesPerPixel * numY;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pOut     = pOutData + (outCh + (x + 1) * outDataPitch1 + (y + 1) * outDataPitch2) * bytesPerPixel * numX * numY;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/*****************************************************************************
*  convolvedVQ3D_S_1x1_U8S8IXCa2_MOD_DWH_x4
*  **************************************************************************/
/****************************************************************************/
/* Description : P6 optimized implementation of 3D convolution for handling */
/*               cases where kwidth * numInch is a multiple of 4            */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 1x1xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_1x1_U8S8IXCa2_MOD_DWH_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_1x1_U8S8IXCa2_MOD_DWH_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, k, x, y;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /* Vector data registers */
  xb_vec2Nx8U dvecInData1, dvecInData2, dvecInData3, dvecInData4;
  valign vaIn1, vaIn2, vaIn3, vaIn4;

  /* Unrolled by 2 along both Output Width and Height.
   * Inner loop unrolled by 4 along the Input number of Channels.
   * Input Number of Channels less than 4 handled in a
   * separate loop.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
#ifdef __XCC__
#pragma loop_count min=1
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Corner case Handling if height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);
#ifdef __XCC__
#pragma loop_count min=1
#endif
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Corner case Handling if width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Pointer for Coefficient Load */
        int8_t *pCoeff = pCoeffData + outCh;
        pdvecCoeff = (xb_vec2Nx8 *) pCoeff;

        /* Input Data Pointers */
        uint8_t *pData = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        pdvecData1 = (xb_vec2Nx8U *) pData;
        pdvecData2 = (xb_vec2Nx8U *) (pData + stride * inDataPitch1 * numX);
        pdvecData3 = (xb_vec2Nx8U *) (pData + stride * inDataPitch2 * numY);
        pdvecData4 = (xb_vec2Nx8U *) (pData + stride * (inDataPitch1 + inDataPitch2) * numX * numY);

        vaIn1 = IVP_LA2NX8U_PP(pdvecData1);
        vaIn2 = IVP_LA2NX8U_PP(pdvecData2);
        vaIn3 = IVP_LA2NX8U_PP(pdvecData3);
        vaIn4 = IVP_LA2NX8U_PP(pdvecData4);

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (k = 0; k < numInCh; k += 4)   /* (Input Channels * kWidth) loops combined */
        {
          /* Load 4 bytes of input data */
          IVP_LAV2NX8U_XP(dvecInData1, vaIn1, pdvecData1, 4);
          IVP_LAV2NX8U_XP(dvecInData2, vaIn2, pdvecData2, 4);
          IVP_LAV2NX8U_XP(dvecInData3, vaIn3, pdvecData3, 4);
          IVP_LAV2NX8U_XP(dvecInData4, vaIn4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
          /* Extracting first 4 bytes of vector into address register */
          /* Scalar integers to be used for QMUL                      */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
          xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
          xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
          xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
          xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
          xb_vecNx16 vecData1, vecData2;
          xb_vecNx16 vecData3, vecData4;
          xb_vecNx16 vecData5, vecData6;
          xb_vecNx16 vecData7, vecData8;
          xb_vecNx16 vecTemp1, vecTemp2;

          /* Custom select pattern for DSELs */
          int16_t sel1       = ((XCHAL_IVPN_SIMD_WIDTH << 8));
          xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
          int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
          xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);

          /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
          IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
          IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
          IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
          IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

          /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
          /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
          dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
          dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
          dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
          dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

          /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif
          /* 4 Aligned Vector Loads of coefficients */
          xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
          IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
          /* Multiply unsigned x signed and accumulate to 24-bits */
          IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, dvecCoeff4);
          IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, dvecCoeff4);
          IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, dvecCoeff4);
          IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, dvecCoeff4);
#endif
        } /* End Corner case handling */


        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        int8_t *pOut = pOutData + (outCh + x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        valign vaOutData = IVP_ZALIGN();
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pOut     = pOutData + (outCh + (x + 1) * outDataPitch1 + y * outDataPitch2) * bytesPerPixel * numX;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pOut     = pOutData + (outCh + x * outDataPitch1 + (y + 1) * outDataPitch2) * bytesPerPixel * numY;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pOut     = pOutData + (outCh + (x + 1) * outDataPitch1 + (y + 1) * outDataPitch2) * bytesPerPixel * numX * numY;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_1x1_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 1x1 MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 1x1 MOD_DWH 3D         */
/*               dilated convolution function and 1x1 MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               stride equal to 1                                          */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 1x1xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/
#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_1x1_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 1);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "Stride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                           \
                    XAI_ERR_BADARG, "\nStride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  if (numInCh % 4 == 0)
  {
#ifdef DILATED_VQ_CONV
    convolvedVQ3D_S_1x1_S8S8IXCa2_MOD_DWH_x4(inTile, coeffTile, biasArray, outputScaleArray, outTile, param);
#else
    convolved3D_S_1x1_S8S8IXCa2_MOD_DWH_x4(inTile, coeffTile, biasArray, outTile, param);
#endif
    return(XAI_ERROR_STATUS());
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }

  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /* Unrolled by 2 along both Output Width and Height.
   * Inner loop unrolled by 4 along the Input number of Channels.
   * Input Number of Channels less than 4 handled in a
   * separate loop.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Corner case Handling if height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Corner case Handling if width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Pointer for Coefficient Load */
        int8_t *pCoeff = pCoeffData + outCh;
        pdvecCoeff = (xb_vec2Nx8 *) pCoeff;

        /* Input Data Pointers */
        int8_t *pData = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        pdvecData1 = (xb_vec2Nx8 *) pData;
        pdvecData2 = (xb_vec2Nx8 *) (pData + stride * inDataPitch1 * numX);
        pdvecData3 = (xb_vec2Nx8 *) (pData + stride * inDataPitch2 * numY);
        pdvecData4 = (xb_vec2Nx8 *) (pData + stride * (inDataPitch1 + inDataPitch2) * numX * numY);

        valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
        valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
        valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
        valign vaData4 = IVP_LA2NX8_PP(pdvecData4);
        for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
        {
          /* Aligning variable vector load of pixels */
          xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
          xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
          xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);
          xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, 4);

          /* Extracting first 4 bytes of vector into address register */
          /* Scalar integers to be used for QMUL                      */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

          /* 4 Aligned Vector Loads of coefficients */
          xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

          /* Quad Muls */
          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
        } /* End Input Channels */

        /* Corner Case Handling as No. of Input Channels not multiple of 4 */
        {
          int32_t remInCh = numInCh - inCh;

          /* Aligning variable vector load of pixels */
          xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, remInCh);
          xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, remInCh);
          xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, remInCh);
          xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, remInCh);

          /* Extracting first 4 bytes of vector into address register */
          /* Scalar integers to be used for QMUL                      */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

          /* For conditional coefficient loads */
          int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
          int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

          /* Coefficient Loads */
          xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
          xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
          xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

          IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
        } /* End Corner case handling */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        int8_t *pOut = pOutData + (outCh + x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        valign vaOutData = IVP_ZALIGN();
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pOut     = pOutData + (outCh + (x + 1) * outDataPitch1 + y * outDataPitch2) * bytesPerPixel * numX;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pOut     = pOutData + (outCh + x * outDataPitch1 + (y + 1) * outDataPitch2) * bytesPerPixel * numY;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pOut     = pOutData + (outCh + (x + 1) * outDataPitch1 + (y + 1) * outDataPitch2) * bytesPerPixel * numX * numY;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_1x1_U8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 1x1 MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 1x1 MOD_DWH 3D         */
/*               dilated convolution function and 1x1 MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               stride equal to 1                                          */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 1x1xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/
#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_1x1_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 1);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "Stride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                           \
                    XAI_ERR_BADARG, "\nStride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  if (numInCh % 4 == 0)
  {
#ifdef DILATED_VQ_CONV
    convolvedVQ3D_S_1x1_U8S8IXCa2_MOD_DWH_x4(inTile, coeffTile, biasArray, outputScaleArray, outTile, param);
#else
    convolved3D_S_1x1_U8S8IXCa2_MOD_DWH_x4(inTile, coeffTile, biasArray, outTile, param);
#endif
    return(XAI_ERROR_STATUS());
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }

  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, k, x, y;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /* Vector data registers */
  xb_vec2Nx8U dvecInData1, dvecInData2, dvecInData3, dvecInData4;
  valign vaIn1, vaIn2, vaIn3, vaIn4;

  /* Unrolled by 2 along both Output Width and Height.
   * Inner loop unrolled by 4 along the Input number of Channels.
   * Input Number of Channels less than 4 handled in a
   * separate loop.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Corner case Handling if height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Corner case Handling if width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Pointer for Coefficient Load */
        int8_t *pCoeff = pCoeffData + outCh;
        pdvecCoeff = (xb_vec2Nx8 *) pCoeff;

        /* Input Data Pointers */
        uint8_t *pData = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        pdvecData1 = (xb_vec2Nx8U *) pData;
        pdvecData2 = (xb_vec2Nx8U *) (pData + stride * inDataPitch1 * numX);
        pdvecData3 = (xb_vec2Nx8U *) (pData + stride * inDataPitch2 * numY);
        pdvecData4 = (xb_vec2Nx8U *) (pData + stride * (inDataPitch1 + inDataPitch2) * numX * numY);

        vaIn1 = IVP_LA2NX8U_PP(pdvecData1);
        vaIn2 = IVP_LA2NX8U_PP(pdvecData2);
        vaIn3 = IVP_LA2NX8U_PP(pdvecData3);
        vaIn4 = IVP_LA2NX8U_PP(pdvecData4);

        for (k = 0; k < numInCh - 3; k += 4)   /* Input Channels  */
        {
          /* Aligning variable vector load of pixels */
          IVP_LAV2NX8U_XP(dvecInData1, vaIn1, pdvecData1, 4);
          IVP_LAV2NX8U_XP(dvecInData2, vaIn2, pdvecData2, 4);
          IVP_LAV2NX8U_XP(dvecInData3, vaIn3, pdvecData3, 4);
          IVP_LAV2NX8U_XP(dvecInData4, vaIn4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
          /* Extracting first 4 bytes of vector into address register */
          /* Scalar integers to be used for QMUL                      */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
          xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
          xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
          xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
          xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
          xb_vecNx16 vecData1, vecData2;
          xb_vecNx16 vecData3, vecData4;
          xb_vecNx16 vecData5, vecData6;
          xb_vecNx16 vecData7, vecData8;
          xb_vecNx16 vecTemp1, vecTemp2;

          /* Custom select pattern for DSELs */
          int16_t sel1       = ((XCHAL_IVPN_SIMD_WIDTH << 8));
          xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
          int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
          xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);

          /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
          IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
          IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
          IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
          IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

          /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
          /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
          dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
          dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
          dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
          dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

          /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif
          /* 4 Aligned Vector Loads of coefficients */
          xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
          xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
          IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
          /* Multiply unsigned x signed and accumulate to 24-bits */
          IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, dvecCoeff4);
          IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, dvecCoeff4);
          IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, dvecCoeff4);
          IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, dvecCoeff4);
#endif
        } /* End Input Channels */

        /* Corner Case Handling as No. of Input Channels not multiple of 4 */
        {
          int32_t remInCh = numInCh - k;

          /* Aligning variable vector load of pixels */
          IVP_LAV2NX8U_XP(dvecInData1, vaIn1, pdvecData1, remInCh);
          IVP_LAV2NX8U_XP(dvecInData2, vaIn2, pdvecData2, remInCh);
          IVP_LAV2NX8U_XP(dvecInData3, vaIn3, pdvecData3, remInCh);
          IVP_LAV2NX8U_XP(dvecInData4, vaIn4, pdvecData4, remInCh);

#ifdef IVP_MULSUQA2N8XR8
          /* Extracting first 4 bytes of vector into address register */
          /* Scalar integers to be used for QMUL                      */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
          xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
          xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
          xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
          xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
          xb_vecNx16 vecData1, vecData2;
          xb_vecNx16 vecData3, vecData4;
          xb_vecNx16 vecData5, vecData6;
          xb_vecNx16 vecData7, vecData8;
          xb_vecNx16 vecTemp1, vecTemp2;

          /* Custom select pattern for DSELs */
          int16_t sel1       = ((XCHAL_IVPN_SIMD_WIDTH << 8));
          xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
          int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
          xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);

          /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
          IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
          IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
          IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
          IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

          /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
          /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
          dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
          dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
          dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
          dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
          dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

          /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
          IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
          dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif
          /* For conditional coefficient loads */
          int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
          int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

          /* Coefficient Loads */
          xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
          xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
          xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
          IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
          /* Multiply unsigned x signed and accumulate to 24-bits */
          IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
          IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, 0);
          IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, 0);
          IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, 0);
          IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, 0);
#endif
        } /* End Corner case handling */


        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        int8_t *pOut = pOutData + (outCh + x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        valign vaOutData = IVP_ZALIGN();
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pOut     = pOutData + (outCh + (x + 1) * outDataPitch1 + y * outDataPitch2) * bytesPerPixel * numX;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pOut     = pOutData + (outCh + x * outDataPitch1 + (y + 1) * outDataPitch2) * bytesPerPixel * numY;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pOut     = pOutData + (outCh + (x + 1) * outDataPitch1 + (y + 1) * outDataPitch2) * bytesPerPixel * numX * numY;
        pdvecOut = (xb_vec2Nx8 *) pOut;
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */

  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_2x2_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 2x2 MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 2x2 MOD_DWH 3D         */
/*               dilated convolution function and 2x2 MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               stride equal to 1                                          */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 2x2xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_2x2_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_2x2_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param)
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 2);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "Stride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                           \
                    XAI_ERR_BADARG, "\nStride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile),                                                                                          \
                    XAI_ERR_DATASIZE, "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Calling further optimized function if dim1Size == dim1Pitch */
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && dilationX == 1 && dilationY == 1)
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                                outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                              outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                             coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                           coeffTile, biasArray, outTile, param);
#endif
    }
    return(XAI_ERROR_STATUS());
  }

  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    /* Max value of Gather Offset is ((stride*min(1, outW-1) + dilation) * inDataPitch1 +
     * min(3, numInCh - 1) + ((stride*min(1, outH-1) * inDataPitch2)) */
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1_PITCH(inTile) <                                                                       \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1) - XAI_CNN_CONV_GET_STRIDE(param) * XT_MIN(1, outH - 1) *             \
                      XAI_TILE3D_GET_DIM2_PITCH(inTile)) / (XAI_CNN_CONV_GET_STRIDE(param) *                                  \
                                                            XT_MIN(1, outW - 1) + XAI_CNN_CONV_GET_DILATION(param))),         \
                    XAI_ERR_BADARG, "\ndim1Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                    XAI_TILE3D_GET_DIM1_PITCH(inTile),                                                                        \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1) - XAI_CNN_CONV_GET_STRIDE(param) * XT_MIN(1, outH - 1) *             \
                      XAI_TILE3D_GET_DIM2_PITCH(inTile)) / (XAI_CNN_CONV_GET_STRIDE(param) *                                  \
                                                            XT_MIN(1, outW - 1) + XAI_CNN_CONV_GET_DILATIONX(param))));
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  int32_t dilatedKWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedKHeight = dilationY * (kHeightU - 1) + 1;

  const uint8_t leftEdgeFlag = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag  = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  int32_t leftEdge, topEdge;

  if ((dilatedKWidth % 2) != 0)
  {
    leftEdge = dilatedKWidth / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedKWidth / 2) : ((dilatedKWidth / 2) - 1);
  }

  if ((dilatedKHeight % 2) != 0)
  {
    topEdge = dilatedKHeight / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedKHeight / 2) : ((dilatedKHeight / 2) - 1);
  }

  /* Move pointer to the start of the active data (including edge) */
  pInData = &pInData[-(topEdge * inDataPitch2 + leftEdge * inDataPitch1)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y;
  valign vaOutData = IVP_ZALIGN();

  /* Only 1 Gather is used in this approach to get the
   * Input Data for 4 Output Vectors. In every Gather,
   * 32 elements are read, where first 16 of them correspond
   * to two vectors of Output along the width and the other
   * 16 of them correspond to two vectors of Output along the height.
   * To get the index values for the Gather, the following
   * calculations are made.
   */

  /* Gather Index Calculations */
  /* Sequence - 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 ... */
  xb_vecNx16U vecGatherOff = IVP_ANDNX16(IVP_SEQNX16(), 3);
  xb_vecNx16 vecSelIdx     = IVP_SEQNX16();
  /* To get the Select indexes as - 0 1 2 3 4 5 6 7 32 33 34 35 36.... */
  IVP_ADDNX16T(vecSelIdx, vecSelIdx, 24, IVP_NOTBN(IVP_LTRNI(8)));
  /* To get - 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 ... */
  xb_vecNx16 vecSeqDiv4 = IVP_SRLINX16(IVP_SEQNX16(), 2);
  /* Sequence - 0 1 2 3  d*P1 d*P1+1 d*P1+2 d*P1+3 */
  IVP_MULANX16PACKL(vecGatherOff, vecSeqDiv4, dilationX * inDataPitch1);
  vecGatherOff = IVP_SELNX16(IVP_ADDNX16(vecGatherOff, stride * inDataPitch1), \
                             vecGatherOff, vecSelIdx);

  xb_vecNx16 vecSelIdx2 = IVP_SEQNX16();
  /* To get the Select indexes as - 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 32 33 34 35 36.... */
  IVP_ADDNX16T(vecSelIdx2, vecSelIdx2, 16, IVP_NOTBN(IVP_LTRNI(16)));

  vecGatherOff = IVP_SELNX16(IVP_ADDNX16(vecGatherOff, stride * inDataPitch2), \
                             vecGatherOff, vecSelIdx2);

  /* Final Index Pattern is -
   *
   * First 8 elements :
   *    0       1       2       3
   * d*P1  d*P1+1  d*P1+2  d*P1+3
   *
   * Second 8 elements :
   *     s*P1      s*P1+1      s*P1+2      s*P1+3
   * (s+d)*P1  (s+d)*P1+1  (s+d)*P1+2  (s+d)*P1+3
   *
   * Third 8 elements :
   *    0+(s*P2)       1+(s*P2)       2+(s*P2)       3+(s*P2)
   * d*P1+(s*P2)  d*P1+1+(s*P2)  d*P1+2+(s*P2)  d*P1+3+(s*P2)
   *
   * Last 8 elements :
   *     s*P1+(s*P2)      s*P1+1+(s*P2)      s*P1+2+(s*P2)      s*P1+3+(s*P2)
   * (s+d)*P1+(s*P2)  (s+d)*P1+1+(s*P2)  (s+d)*P1+2+(s*P2)  (s+d)*P1+3+(s*P2)
   *
   */

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff1;
  xb_vec2Nx8* restrict pdvecCoeff2;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;

  int32_t remCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint8_t remCh1 = XT_SALT(2, remCh + 1);
  uint8_t remCh2 = XT_SALT(3, remCh + 1);

  uint32_t sumMask = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* Unrolled by 2 along both Output Width and Output Height.
   * Also, unrolled along Input Channels by 4 and completely
   * along the Kernel Width. Gathers are used for loading Input Data.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used to handle the corner case of OutHeight being odd */
      int32_t numY = XT_MIN(2, outH - y) - 1;

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff1;

        /* Variable used to handle the corner case of Output Width being odd */
        int32_t numX = XT_MIN(2, outW - x) - 1;

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean vectors to handle the corner cases of Out Width and Height being odd */
        vboolN vbXY = IVP_LTRSN((16 * numY) + 8 * (numX + 1));

        /* Pointer for Coefficient Load */
        pdvecCoeff1 = (xb_vec2Nx8 *) (pCoeff);
        pdvecCoeff2 = (xb_vec2Nx8 *) (pCoeff + coeffPitch3);

        /* Assign valid address for predicated false lines */
        vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, vbXY);

        /* Pointer for Input Data Load corresponding to ky = 0 */
        pData1 = pData;

        /* Pointer for Input Data Load corresponding to ky = 1 */
        pData2 = pData1 + (dilationY * inDataPitch2);

        for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels Loop */
        {
          /* Gather Input Data correspoinding to ky = 0 */
          xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff1);
          xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);

          /* Gather Input Data corresponding to ky = 1 */
          xb_gsr gather2       = IVP_GATHERANX8S(pData2, vecGatherOff1);
          xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);

          /* ky = 0, kx = 0 */
          /* Extracting scalar integers for QMULs */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 2);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 6);

          /* 4 Aligned Vector Loads of coefficients */
          xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff1, coeffPitch1);
          xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, coeffPitch1);
          xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff1, coeffPitch1);
          xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff1, coeffPitch2 - \
                                               3 * coeffPitch1);

          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

          /* ky = 0, kx = 1 */
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       1);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       3);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       5);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       7);

          /* 4 Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff1, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff1, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff1, coeffPitch1 - coeffPitch2);

          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

          /* ky = 1, kx = 0 */
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 2);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 4);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 6);

          /* 4 Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff2, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff2, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff2, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff2, coeffPitch2 - 3 * coeffPitch1);

          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

          /* ky = 1, kx = 1 */
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       1);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       3);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       5);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       7);

          /* 4 Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff2, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff2, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff2, coeffPitch1);
          IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff2, coeffPitch1 - coeffPitch2);

          IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
          IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
          IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

          pData1 += 4;
          pData2 += 4;
        } /* End Input Channels */

        /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
        if (inCh < numInCh)
        {
          int32_t remInCh  = numInCh - inCh;
          vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);

          /* Gather Input Data */
          xb_vec2Nx8 dvecData1 = 0;
          xb_vec2Nx8 dvecData2 = 0;

          /* Pointer for Input Data Load corresponding to ky = 0 */
          pData1 = pData + inCh;

          /* Pointer for Input Data Load corresponding to ky = 1 */
          pData2 = pData1 + (dilationY * inDataPitch2);

          /* Assign valid address for predicated false lines */
          vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbXY));

          /* Gather Input Data corresponding to ky = 0*/
          xb_gsr gather1 = IVP_GATHERANX8S(pData1, vecGatherOff1);
          dvecData1 = IVP_GATHERD2NX8_L(gather1);

          /* Gather Input Data corresponding to ky = 1 */
          xb_gsr gather2 = IVP_GATHERANX8S(pData2, vecGatherOff1);
          dvecData2 = IVP_GATHERD2NX8_L(gather2);

          /* ky = 0, kx = 0 */
          /* Extracting scalar integers for QMULs */
          int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
          int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 2);
          int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
          int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                 (IVP_MOVNX16_FROM2NX8(dvecData1)), 6);

          /* Aligned Vector Loads of coefficients */
          xb_vec2Nx8 dvecCoeff1;
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff1, coeffPitch1 * remCh1);
          xb_vec2Nx8 dvecCoeff2;
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, coeffPitch1 * remCh2);
          xb_vec2Nx8 dvecCoeff3;
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff1, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

          IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
          IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
          IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
          IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

          /* ky = 0, kx = 1 */
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       1);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       3);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       5);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                       7);

          /* Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff1, coeffPitch1 * remCh1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, coeffPitch1 * remCh2);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff1, 0);

          IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
          IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
          IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
          IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

          /* ky = 1, kx = 0 */
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 2);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 4);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                         (IVP_MOVNX16_FROM2NX8(dvecData2)), 6);

          /* Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff2, coeffPitch1 * remCh1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff2, coeffPitch1 * remCh2);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff2, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));


          IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
          IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
          IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
          IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

          /* ky = 1, kx = 1 */
          /* Extracting scalar integers for QMULs */
          qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       1);
          qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       3);
          qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       5);
          qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                       7);

          /* Aligned Vector Loads of coefficients */
          IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff2, coeffPitch1 * remCh1);
          IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff2, coeffPitch1 * remCh2);
          IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff2, 0);

          IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
          IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
          IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
          IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
        } /* End Input Channels Corner case Handling */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_3x3_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 3x3 MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 3x3 MOD_DWH 3D         */
/*               dilated convolution function and 3x3 MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               stride equal to 1                                          */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 3x3xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_3x3_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 3);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                         \
                    XAI_ERR_BADARG, "Stride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "\nStride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_EDGE2(inTile, 1 + 1 * (XAI_CNN_CONV_GET_DILATIONX(param) - 1), 1 + 1 * (XAI_CNN_CONV_GET_DILATIONY(param) - 1));
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Calling further optimized function if dim1Size == dim1Pitch */
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && dilationX == 1 && dilationY == 1)
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                                outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                              outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                             coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                           coeffTile, biasArray, outTile, param);
#endif
    }
    return(XAI_ERROR_STATUS());
  }

  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    /* Max value of Gather Offset is ((stride*min(1, outW-1) + 2 * dilationX) * inDataPitch1 +
     * min(3, numInCh - 1)) */
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1_PITCH(inTile) <                                                                       \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) / (XAI_CNN_CONV_GET_STRIDE(param) *                                 \
                                                             XT_MIN(1, outW - 1) + 2 * XAI_CNN_CONV_GET_DILATION(param))),    \
                    XAI_ERR_BADARG, "\ndim1Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                    XAI_TILE3D_GET_DIM1_PITCH(inTile),                                                                        \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) / (XAI_CNN_CONV_GET_STRIDE(param) *                                 \
                                                             XT_MIN(1, outW - 1) + 2 * XAI_CNN_CONV_GET_DILATION(param))));
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  int32_t dilatedKWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedKHeight = dilationY * (kHeightU - 1) + 1;

  /* move to start of edge data only when input is already padded. */
  pInData = &pInData[-(int32_t) ((dilatedKHeight / 2) * inDataPitch2 + (dilatedKWidth / 2) * inDataPitch1)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;
  valign vaOutData = IVP_ZALIGN();

  /* Only 2 Gathers are used in this approach to get the
   * Input Data for 4 Output Vectors. In each Gather,
   * 24 elements are read, where each 12 of them correspond
   * to one vector of Output along the width. To get the
   * index values for the Gather, the following calculations
   * are made.
   */

  /* Gather Index Calculations */
  /* Sequence - 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 ... */
  xb_vecNx16U vecGatherOff = IVP_ANDNX16(IVP_SEQNX16(), 3);
  xb_vecNx16 vecSelIdx     = IVP_SEQNX16();
  /* To get the Select indexes as - 0 1 2 3 4...11 32 33 34 35 36.... */
  IVP_ADDNX16T(vecSelIdx, vecSelIdx, 20, IVP_NOTBN(IVP_LTRNI(12)));
  /* To get - 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 ... */
  xb_vecNx16 vecSeqDiv4 = IVP_SRLINX16(IVP_SEQNX16(), 2);
  /* Sequence - 0 1 2 3  d*P1 d*P1+1 d*P1+2 d*P1+3 2.d*P1 2.d*P1+1 2.d*P1+2 2.d*P1+3 ... */
  IVP_MULANX16PACKL(vecGatherOff, vecSeqDiv4, dilationX * inDataPitch1);
  vecGatherOff = IVP_SELNX16(IVP_ADDNX16(vecGatherOff, stride * inDataPitch1), \
                             vecGatherOff, vecSelIdx);
  /* Final Index Pattern is -
   * 0 1 2 3 d*P1 d*p1+1 d*P1+2 d*P1+3 d*2*P1 d*2*P1+1 d*2*P1+2 d*2*P1+3
   * s*P1 s*P1+1 s*P1+2 s*P1+3 (s+1*d)*P1 (s+1*d)*P1+1 (s+1*d)*P1+2 (s+1*d)*P1+3
   * (s+2*d)*P1 (s+2*d)*P1+1 (s+2*d)*P1+2 (s+2*d)*P1+3 */

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;

  int32_t remCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remInCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint8_t remCh1 = XT_SALT(2, remCh + 1);
  uint8_t remCh2 = XT_SALT(3, remCh + 1);

  uint32_t sumMask = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* Unrolled by 2 along both Output Width and Output Height.
   * Also, unrolled along Input Channels by 4 and completely
   * along the Kernel Width. Gathers are used for loading Input Data.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used to handle the corner case of OutHeight being odd */
      int32_t numY = XT_MIN(2, outH - y) - 1;

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff1;
        xb_vecNx16U vecGatherOff2;

        /* Variable used to handle the corner case of Output Width being odd */
        int32_t numX = XT_MIN(2, outW - x) - 1;

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean vectors to handle the corner cases of Out Width and Height being odd */
        vboolN vbX = IVP_LTRSN(12 * (numX + 1));
        vboolN vbY = IVP_LTRSN(12 * (numX + 1) * numY);

        for (ky = 0; ky < 3; ky++) /* Kernel Height Loop */
        {
          /* Pointer for Input Data Load */
          pData1 = pData + ky * dilationY * inDataPitch2;
          pData2 = pData1 + (stride * inDataPitch2 * numY);
          /* Assign valid address for predicated false lines */
          vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, vbX);
          vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, vbY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);
          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels Loop */
          {
            /* Gather Input Data */
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff1);
            xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2       = IVP_GATHERANX8S(pData2, vecGatherOff2);
            xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);


            pData1 += 4;
            pData2 += 4;

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 3);

            /* 4 Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * \
                                                 coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         4);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         5);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 - 2 * coeffPitch2);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */

          /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh  = numInCh - inCh;
            vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);

            /* Gather Input Data */
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            /* Assign valid address for predicated false lines */
            vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbX));
            vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbY));

            xb_gsr gather1 = IVP_GATHERANX8S(pData1, vecGatherOff1);
            dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2 = IVP_GATHERANX8S(pData2, vecGatherOff2);
            dvecData2 = IVP_GATHERD2NX8_L(gather2);

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 3);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));


            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         4);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - coeffPitch1 * (remCh1 + remCh2));

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         5);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, 0);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
          } /* End Input Channels Corner case Handling */
        }   /* End Kernel Height Loop */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * \
                       numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 *
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_3x3_U8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 3x3 MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 3x3 MOD_DWH 3D         */
/*               dilated convolution function and 3x3 MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               stride equal to 1                                          */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 3x3xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_3x3_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 3);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                         \
                    XAI_ERR_BADARG, "Stride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "\nStride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_EDGE2(inTile, 1 + 1 * (XAI_CNN_CONV_GET_DILATIONX(param) - 1), 1 + 1 * (XAI_CNN_CONV_GET_DILATIONY(param) - 1));
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

#ifdef IVP_MULSUQA2N8XR8
  /* Calling further optimized function if dim1Size == dim1Pitch */
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && dilationX == 1 && dilationY == 1)
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                                outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                              outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                             coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                           coeffTile, biasArray, outTile, param);
#endif
    }
    return(XAI_ERROR_STATUS());
  }
#endif //#ifdef IVP_MULSUQA2N8XR8
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    /* Max value of Gather Offset is ((stride*min(1, outW-1) + 2 * dilationX) * inDataPitch1 +
     * min(3, numInCh - 1)) */
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1_PITCH(inTile) <                                                                       \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) / (XAI_CNN_CONV_GET_STRIDE(param) *                                 \
                                                             XT_MIN(1, outW - 1) + 2 * XAI_CNN_CONV_GET_DILATION(param))),    \
                    XAI_ERR_BADARG, "\ndim1Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                    XAI_TILE3D_GET_DIM1_PITCH(inTile),                                                                        \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) / (XAI_CNN_CONV_GET_STRIDE(param) *                                 \
                                                             XT_MIN(1, outW - 1) + 2 * XAI_CNN_CONV_GET_DILATION(param))));
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  int32_t dilatedKWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedKHeight = dilationY * (kHeightU - 1) + 1;

  /* move to start of edge data only when input is already padded. */
  pInData = &pInData[-(int32_t) ((dilatedKHeight / 2) * inDataPitch2 + (dilatedKWidth / 2) * inDataPitch1)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;
  valign vaOutData = IVP_ZALIGN();

  /* Only 2 Gathers are used in this approach to get the
   * Input Data for 4 Output Vectors. In each Gather,
   * 24 elements are read, where each 12 of them correspond
   * to one vector of Output along the width. To get the
   * index values for the Gather, the following calculations
   * are made.
   */

  /* Gather Index Calculations */
  /* Sequence - 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 ... */
  xb_vecNx16U vecGatherOff = IVP_ANDNX16(IVP_SEQNX16(), 3);
  xb_vecNx16 vecSelIdx     = IVP_SEQNX16();
  /* To get the Select indexes as - 0 1 2 3 4...11 32 33 34 35 36.... */
  IVP_ADDNX16T(vecSelIdx, vecSelIdx, 20, IVP_NOTBN(IVP_LTRNI(12)));
  /* To get - 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 ... */
  xb_vecNx16 vecSeqDiv4 = IVP_SRLINX16(IVP_SEQNX16(), 2);
  /* Sequence - 0 1 2 3  d*P1 d*P1+1 d*P1+2 d*P1+3 2.d*P1 2.d*P1+1 2.d*P1+2 2.d*P1+3 ... */
  IVP_MULANX16PACKL(vecGatherOff, vecSeqDiv4, dilationX * inDataPitch1);
  vecGatherOff = IVP_SELNX16(IVP_ADDNX16(vecGatherOff, stride * inDataPitch1), \
                             vecGatherOff, vecSelIdx);
  /* Final Index Pattern is -
   * 0 1 2 3 d*P1 d*p1+1 d*P1+2 d*P1+3 d*2*P1 d*2*P1+1 d*2*P1+2 d*2*P1+3
   * s*P1 s*P1+1 s*P1+2 s*P1+3 (s+1*d)*P1 (s+1*d)*P1+1 (s+1*d)*P1+2 (s+1*d)*P1+3
   * (s+2*d)*P1 (s+2*d)*P1+1 (s+2*d)*P1+2 (s+2*d)*P1+3 */

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecOut;
  uint8_t*     restrict pData1;
  uint8_t*     restrict pData2;

  int32_t remCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remInCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };
  uint8_t remCh1            = XT_SALT(2, remCh + 1);
  uint8_t remCh2            = XT_SALT(3, remCh + 1);
  uint32_t sumMask          = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;


#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* Unrolled by 2 along both Output Width and Output Height.
   * Also, unrolled along Input Channels by 4 and completely
   * along the Kernel Width. Gathers are used for loading Input Data.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used to handle the corner case of OutHeight being odd */
      int32_t numY = XT_MIN(2, outH - y) - 1;

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff1;
        xb_vecNx16U vecGatherOff2;

        /* Variable used to handle the corner case of Output Width being odd */
        int32_t numX = XT_MIN(2, outW - x) - 1;

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        uint8_t *pData = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean vectors to handle the corner cases of Out Width and Height being odd */
        vboolN vbX = IVP_LTRSN(12 * (numX + 1));
        vboolN vbY = IVP_LTRSN(12 * (numX + 1) * numY);

        for (ky = 0; ky < 3; ky++) /* Kernel Height Loop */
        {
          /* Pointer for Input Data Load */
          pData1 = pData + ky * dilationY * inDataPitch2;
          pData2 = pData1 + (stride * inDataPitch2 * numY);
          /* Assign valid address for predicated false lines */
          vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, vbX);
          vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, vbY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);
          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels Loop */
          {
            /* Gather Input Data */
            xb_gsr gather1        = IVP_GATHERANX8U(pData1, vecGatherOff1);
            xb_vec2Nx8U dvecData1 = IVP_GATHERD2NX8U_L(gather1);
            xb_gsr gather2        = IVP_GATHERANX8U(pData2, vecGatherOff2);
            xb_vec2Nx8U dvecData2 = IVP_GATHERD2NX8U_L(gather2);


            pData1 += 4;
            pData2 += 4;

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 3);

            /* 4 Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * \
                                                 coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            xb_vec2Nx8U dvecS1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar1)));
            xb_vec2Nx8U dvecS2 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar2)));
            xb_vec2Nx8U dvecS3 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar3)));
            xb_vec2Nx8U dvecS4 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar4)));

            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 0), dvecCoeff1, IVP_REP2NX8U(dvecS1, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 2), dvecCoeff3, IVP_REP2NX8U(dvecS1, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 0), dvecCoeff1, IVP_REP2NX8U(dvecS2, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 2), dvecCoeff3, IVP_REP2NX8U(dvecS2, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 0), dvecCoeff1, IVP_REP2NX8U(dvecS3, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 2), dvecCoeff3, IVP_REP2NX8U(dvecS3, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 0), dvecCoeff1, IVP_REP2NX8U(dvecS4, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 2), dvecCoeff3, IVP_REP2NX8U(dvecS4, 3), dvecCoeff4);
#endif

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData2)), \
                                         4);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            dvecS1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar1)));
            dvecS2 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar2)));
            dvecS3 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar3)));
            dvecS4 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar4)));

            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 0), dvecCoeff1, IVP_REP2NX8U(dvecS1, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 2), dvecCoeff3, IVP_REP2NX8U(dvecS1, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 0), dvecCoeff1, IVP_REP2NX8U(dvecS2, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 2), dvecCoeff3, IVP_REP2NX8U(dvecS2, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 0), dvecCoeff1, IVP_REP2NX8U(dvecS3, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 2), dvecCoeff3, IVP_REP2NX8U(dvecS3, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 0), dvecCoeff1, IVP_REP2NX8U(dvecS4, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 2), dvecCoeff3, IVP_REP2NX8U(dvecS4, 3), dvecCoeff4);
#endif

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData2)), \
                                         5);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 - 2 * coeffPitch2);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            dvecS1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar1)));
            dvecS2 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar2)));
            dvecS3 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar3)));
            dvecS4 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar4)));

            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 0), dvecCoeff1, IVP_REP2NX8U(dvecS1, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 2), dvecCoeff3, IVP_REP2NX8U(dvecS1, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 0), dvecCoeff1, IVP_REP2NX8U(dvecS2, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 2), dvecCoeff3, IVP_REP2NX8U(dvecS2, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 0), dvecCoeff1, IVP_REP2NX8U(dvecS3, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 2), dvecCoeff3, IVP_REP2NX8U(dvecS3, 3), dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 0), dvecCoeff1, IVP_REP2NX8U(dvecS4, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 2), dvecCoeff3, IVP_REP2NX8U(dvecS4, 3), dvecCoeff4);
#endif
          } /* End Input Channels */

          /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh  = numInCh - inCh;
            vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);

            /* Gather Input Data */
            xb_vec2Nx8U dvecData1 = 0;
            xb_vec2Nx8U dvecData2 = 0;
            /* Assign valid address for predicated false lines */
            vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbX));
            vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbY));

            xb_gsr gather1 = IVP_GATHERANX8U(pData1, vecGatherOff1);
            dvecData1 = IVP_GATHERD2NX8U_L(gather1);
            xb_gsr gather2 = IVP_GATHERANX8U(pData2, vecGatherOff2);
            dvecData2 = IVP_GATHERD2NX8U_L(gather2);

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 3);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
#else
            xb_vec2Nx8U dvecS1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar1 & sumMask)));
            xb_vec2Nx8U dvecS2 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar2 & sumMask)));
            xb_vec2Nx8U dvecS3 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar3 & sumMask)));
            xb_vec2Nx8U dvecS4 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar4 & sumMask)));

            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 0), dvecCoeff1, IVP_REP2NX8U(dvecS1, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 2), dvecCoeff3, IVP_REP2NX8U(dvecS1, 3), 0);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 0), dvecCoeff1, IVP_REP2NX8U(dvecS2, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 2), dvecCoeff3, IVP_REP2NX8U(dvecS2, 3), 0);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 0), dvecCoeff1, IVP_REP2NX8U(dvecS3, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 2), dvecCoeff3, IVP_REP2NX8U(dvecS3, 3), 0);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 0), dvecCoeff1, IVP_REP2NX8U(dvecS4, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 2), dvecCoeff3, IVP_REP2NX8U(dvecS4, 3), 0);
#endif

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData2)), \
                                         4);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - coeffPitch1 * (remCh1 + remCh2));

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
#else
            dvecS1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar1 & sumMask)));
            dvecS2 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar2 & sumMask)));
            dvecS3 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar3 & sumMask)));
            dvecS4 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar4 & sumMask)));

            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 0), dvecCoeff1, IVP_REP2NX8U(dvecS1, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 2), dvecCoeff3, IVP_REP2NX8U(dvecS1, 3), 0);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 0), dvecCoeff1, IVP_REP2NX8U(dvecS2, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 2), dvecCoeff3, IVP_REP2NX8U(dvecS2, 3), 0);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 0), dvecCoeff1, IVP_REP2NX8U(dvecS3, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 2), dvecCoeff3, IVP_REP2NX8U(dvecS3, 3), 0);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 0), dvecCoeff1, IVP_REP2NX8U(dvecS4, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 2), dvecCoeff3, IVP_REP2NX8U(dvecS4, 3), 0);
#endif

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8U(dvecData2)), \
                                         5);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, 0);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
#else
            dvecS1 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar1 & sumMask)));
            dvecS2 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar2 & sumMask)));
            dvecS3 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar3 & sumMask)));
            dvecS4 = IVP_MOV2NX8U_FROMNX16(IVP_MOVNX16_FROMN_2X32(IVP_MOVVA32(qmulScalar4 & sumMask)));

            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 0), dvecCoeff1, IVP_REP2NX8U(dvecS1, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, IVP_REP2NX8U(dvecS1, 2), dvecCoeff3, IVP_REP2NX8U(dvecS1, 3), 0);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 0), dvecCoeff1, IVP_REP2NX8U(dvecS2, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, IVP_REP2NX8U(dvecS2, 2), dvecCoeff3, IVP_REP2NX8U(dvecS2, 3), 0);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 0), dvecCoeff1, IVP_REP2NX8U(dvecS3, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, IVP_REP2NX8U(dvecS3, 2), dvecCoeff3, IVP_REP2NX8U(dvecS3, 3), 0);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 0), dvecCoeff1, IVP_REP2NX8U(dvecS4, 1), dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, IVP_REP2NX8U(dvecS4, 2), dvecCoeff3, IVP_REP2NX8U(dvecS4, 3), 0);
#endif
          } /* End Input Channels Corner case Handling */
        }   /* End Kernel Height Loop */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * \
                       numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 *
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_4x4_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 4x4 MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 4x4 MOD_DWH 3D         */
/*               dilated convolution function and 4x4 MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               stride equal to 1                                          */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 4x4xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_4x4_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_4x4_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 4);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "Stride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                           \
                    XAI_ERR_BADARG, "\nStride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));

    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Calling further optimized function if dim1Size == dim1Pitch */
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && dilationX == 1 && dilationY == 1)
  {
#ifdef DILATED_VQ_CONV
    convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile,
                                                              coeffTile,
                                                              biasArray,
                                                              outputScaleArray,
                                                              outTile,
                                                              param);
#else
    convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile,
                                                            coeffTile,
                                                            biasArray,
                                                            outTile,
                                                            param);
#endif
    return(XAI_ERROR_STATUS());
  }

  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);

    /* Max value of Gather Offset is ((stride*min(outW-1, 1) + 3 * dilationX) * inDataPitch1 +
     * min(3, numInCh - 1)) */
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1_PITCH(inTile) <                                                                                    \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) / (XAI_CNN_CONV_GET_STRIDE(param) *                                              \
                                                             XT_MIN(1, outW - 1) + 3 * XAI_CNN_CONV_GET_DILATION(param))), XAI_ERR_BADARG, \
                    "\ndim1Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d",                              \
                    XAI_TILE3D_GET_DIM1_PITCH(inTile),                                                                                     \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) / (XAI_CNN_CONV_GET_STRIDE(param) *                                              \
                                                             XT_MIN(1, outW - 1) + 3 * XAI_CNN_CONV_GET_DILATION(param))));
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);

  const uint8_t outShiftU  = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride     = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  int32_t dilatedKWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedKHeight = dilationY * (kHeightU - 1) + 1;

  const uint8_t leftEdgeFlag = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag  = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  int32_t leftEdge, topEdge;

  if ((dilatedKWidth % 2) != 0)
  {
    leftEdge = dilatedKWidth / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedKWidth / 2) : ((dilatedKWidth / 2) - 1);
  }

  if ((dilatedKHeight % 2) != 0)
  {
    topEdge = dilatedKHeight / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedKHeight / 2) : ((dilatedKHeight / 2) - 1);
  }

  /* Move pointer to the start of the active data (including edge) */
  pInData = &pInData[-(topEdge * inDataPitch2 + leftEdge * inDataPitch1)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;
  valign vaOutData = IVP_ZALIGN();

  /* Only 2 Gathers are used in this approach to get the
   * Input Data for 4 Output Vectors. In each Gather,
   * 32 elements are read, where each 16 of them correspond
   * to one vector of Output along the width. To get the
   * index values for the Gather, the following calculations
   * are made.
   */

  /* Gather Index Calculations */
  /* Sequence - 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 ... */
  xb_vecNx16U vecGatherOff = IVP_ANDNX16(IVP_SEQNX16(), 3);
  xb_vecNx16 vecSelIdx     = IVP_SEQNX16();
  /* To get the Select indexes as - 0 1 2 3 4...11 12 13 14 15 32 33 34 35 36.... */
  IVP_ADDNX16T(vecSelIdx, vecSelIdx, 16, IVP_NOTBN(IVP_LTRNI(16)));
  /* To get - 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 ... */
  xb_vecNx16 vecSeqDiv4 = IVP_SRLINX16(IVP_SEQNX16(), 2);
  /* Sequence - 0 1 2 3  d*P1 d*P1+1 d*P1+2 d*P1+3 2.d*P1 2.d*P1+1 2.d*P1+2 2.d*P1+3
     3.d*P1 3.d*P1+1 3.d*P1+2 3.d*P1+3 ... */
  IVP_MULANX16PACKL(vecGatherOff, vecSeqDiv4, dilationX * inDataPitch1);
  vecGatherOff = IVP_SELNX16(IVP_ADDNX16(vecGatherOff, stride * inDataPitch1), \
                             vecGatherOff, vecSelIdx);

  /* Final Index Pattern is :
   *
   * First 16 elements :
   *      0         1         2         3
   * d*1*P1  d*1*P1+1  d*1*P1+2  d*1*P1+3
   * d*2*P1  d*2*P1+1  d*2*P1+2  d*2*P1+3
   * d*3*P1  d*3*P1+1  d*3*P1+2  d*3*P1+3
   *
   * Last 16 elements :
   *       s*P1        s*P1+1        s*P1+2        s*P1+3
   * (s+1*d)*P1  (s+1*d)*P1+1  (s+1*d)*P1+2  (s+1*d)*P1+3
   * (s+2*d)*P1  (s+2*d)*P1+1  (s+2*d)*P1+2  (s+2*d)*P1+3
   * (s+3*d)*P1  (s+3*d)*P1+1  (s+3*d)*P1+2  (s+3*d)*P1+3
   *
   */

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff1;
  xb_vec2Nx8* restrict pdvecCoeff2;
  xb_vec2Nx8* restrict pdvecCoeff3;
  xb_vec2Nx8* restrict pdvecCoeff4;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;

  int32_t remCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remInCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint8_t remCh1 = XT_SALT(2, remCh + 1);
  uint8_t remCh2 = XT_SALT(3, remCh + 1);

  uint32_t sumMask = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* Unrolled by 2 along both Output Width and Output Height.
   * Also, unrolled along Input Channels by 4 and completely
   * along the Kernel Width. Gathers are used for loading Input Data.
   */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used to handle the corner case of OutHeight being odd */
      int32_t numY = XT_MIN(2, outH - y) - 1;

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff1;
        xb_vecNx16U vecGatherOff2;

        /* Variable used to handle the corner case of Output Width being odd */
        int32_t numX = XT_MIN(2, outW - x) - 1;

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean vectors to handle the corner cases of Out Width and Height being odd */
        vboolN vbX = IVP_LTRSN(16 * (numX + 1));
        vboolN vbY = IVP_LTRSN(16 * (numX + 1) * numY);

        for (ky = 0; ky < 4; ky++) /* Kernel Height Loop */
        {
          /* Pointer for Input Data Load */
          pData1 = pData + ky * dilationY * inDataPitch2;
          pData2 = pData1 + (stride * inDataPitch2 * numY);
          /* Assign valid address for predicated false lines */
          vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, vbX);
          vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, vbY);

          /* Pointer for Coefficient Load */
          pdvecCoeff1 = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);
          pdvecCoeff2 = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3 + coeffPitch2);
          pdvecCoeff3 = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3 + 2 * coeffPitch2);
          pdvecCoeff4 = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3 + 3 * coeffPitch2);
          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels Loop */
          {
            /* Gather Input Data */
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff1);
            xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2       = IVP_GATHERANX8S(pData2, vecGatherOff2);
            xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);


            pData1 += 4;
            pData2 += 4;

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 4);

            /* 4 Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff1, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff1, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff1, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         5);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff2, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff2, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff2, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff2, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         6);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff3, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff3, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff3, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff3, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 4 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         3);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         7);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         3);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         7);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff4, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff4, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff4, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff4, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */

          /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh  = numInCh - inCh;
            vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);

            /* Gather Input Data */
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            /* Assign valid address for predicated false lines */
            vecGatherOff1 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbX));
            vecGatherOff2 = IVP_MOVNX16UT(vecGatherOff, 0, IVP_ANDBN(vbRemInCh, vbY));

            xb_gsr gather1 = IVP_GATHERANX8S(pData1, vecGatherOff1);
            dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2 = IVP_GATHERANX8S(pData2, vecGatherOff2);
            dvecData2 = IVP_GATHERD2NX8_L(gather2);

            /* kx = 1 */
            /* Extracting scalar integers for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 4);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff1, coeffPitch1 * remCh1);
            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff1, coeffPitch1 * remCh2);
            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff1, 0);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 2 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         5);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff2, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff2, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff2, 0);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 3 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         6);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff3, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff3, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff3, 0);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 4 */
            /* Extracting scalar integers for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         3);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         7);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         3);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         7);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff4, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff4, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff4, 0);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
          } /* End Input Channels Corner case Handling */
        }   /* End Kernel Height Loop */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * \
                       numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 *
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_5x5_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 5x5 MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 5x5 MOD_DWH 3D         */
/*               dilated convolution function and 5x5 MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               stride equal to 1                                          */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 5x5xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_5x5_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 5);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                         \
                    XAI_ERR_BADARG, "Stride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "\nStride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_EDGE2(inTile, 2 + 2 * (XAI_CNN_CONV_GET_DILATIONX(param) - 1), 2 + 2 * (XAI_CNN_CONV_GET_DILATIONY(param) - 1));
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }

#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif


  /* Calling further optimized function if dim1Size == dim1Pitch */
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && \
      (XAI_CNN_CONV_GET_DILATIONX(param) == 1) && (XAI_CNN_CONV_GET_DILATIONY(param) == 1))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                                outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                              outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                             coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                           coeffTile, biasArray, outTile, param);
#endif
    }
    return(XAI_ERROR_STATUS());
  }

  /* Getting parameters from the tile structures */
  const int32_t outW      = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH      = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh   = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh  = XAI_TILE3D_GET_DIM1(outTile);
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    /* Max value of Gather Offset is ((stride*min(outW-1, 1) + 4 * dilationX) * inDataPitch1 +
     * min(3, numInCh - 1)) */
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1_PITCH(inTile) <                                                                     \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) / (XAI_CNN_CONV_GET_STRIDE(param) *                               \
                                                             XT_MIN(1, outW - 1) + 4 * XAI_CNN_CONV_GET_DILATION(param))),  \
                    XAI_ERR_BADARG, "dim1Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                    XAI_TILE3D_GET_DIM1_PITCH(inTile),                                                                      \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) / (XAI_CNN_CONV_GET_STRIDE(param) *                               \
                                                             XT_MIN(1, outW - 1) + 4 * XAI_CNN_CONV_GET_DILATION(param))));
  }

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  int32_t dilatedKWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedKHeight = dilationY * (kHeightU - 1) + 1;

  /* move to start of edge data only when input is already padded. */
  pInData = &pInData[-(int32_t) ((dilatedKHeight / 2) * inDataPitch2 + (dilatedKWidth / 2) * inDataPitch1)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;


  /* 4 Gathers are being used to Load Input Data. Many common elements
   * will be loaded in separate Gathers, especially in the case of
   * stride 1 and 2. To take the advantage of having common offsets in
   * a single Gather, 2 Gather Patterns are generated as given below.
   * For example, in the Gather Patterns generated below,
   * if stride is 1 and dilation equal to 1, then 8 offsets are common and if stride is 2, 4 offsets
   * are common in each Gather.
   */
  /* Gather Index Calculations */
  xb_vecNx16 vecGather = IVP_ANDNX16(IVP_SEQNX16(), 3);
  IVP_MULANX16PACKL(vecGather, inDataPitch1 * dilationX, IVP_SRLINX16(IVP_SEQNX16(), 2));
  xb_vecNx16 vecGather1 = IVP_ADDNX16(vecGather, stride * inDataPitch1);

  xb_vecNx16 vecSelIdx1 = IVP_SEQNX16();
  IVP_ADDNX16T(vecSelIdx1, vecSelIdx1, 20, IVP_NOTBN(IVP_LTRNI(12)));
  xb_vecNx16U vecGatherOff1 = IVP_SELNX16(vecGather1, vecGather, vecSelIdx1);
  xb_vecNx16 vecSelIdx2     = IVP_ADDNX16(IVP_SEQNX16(), 12);
  IVP_ADDNX16T(vecSelIdx2, vecSelIdx2, 20, IVP_NOTBN(IVP_LTRNI(8)));
  xb_vecNx16U vecGatherOff2 = IVP_SELNX16(vecGather1, vecGather, vecSelIdx2);
  /* Index Pattern of vecGatherOff1 is -
   * 0 1 2 3 d*P1 d*P1+1 d*P1+2 d*P1+3 d*2*P1 d*2*P1+1 d*2*P1+2 d*2*P1+3
   * s*P1 s*P1+1 s*P1+2 s*dP1+3 (s+1*d)*P1 (s+1*d)*P1+1 (s+1*d)*P1+2 (s+1*d)*P1+3 */

  /* Index Pattern of vecGatherOff2 is -
   * d*3*P1 d*3*P1+1 d*3*P1+2 d*3*P1+3 d*4*P1 d*4*P1+1 d*4*P1+2 d*4*P1+3
   * (s+2*d)*P1 (s+2*d)*P1+1 (s+2*d)*P1+2 (s+2*d)*P1+3
   * (s+3*d)*P1 (s+3*d)*P1+1 (s+3*d)*P1+2 (s+3*d)*P1+3
   * (s+4*d)*P1 (s+4*d)*P1+1 (s+4*d)*P1+2 (s+4*d)*P1+3 */

  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;

  int32_t remCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remInCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint8_t remCh1 = XT_SALT(2, remCh + 1);
  uint8_t remCh2 = XT_SALT(3, remCh + 1);

  uint32_t sumMask = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* 4 Gathers used for Input Data Load. Unrolled along */
  /* Output Width and Height by 2. Also, unrolled along */
  /* Input Channels by 4 and Kernel Width.              */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Out Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used for corner case handling of Out Height odd */
      int32_t numY = XT_MIN(2, outH - y) - 1;
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff00;
        xb_vecNx16U vecGatherOff01;
        xb_vecNx16U vecGatherOff10;
        xb_vecNx16U vecGatherOff11;
        /* Variable used for corner case handling of Out Width odd */
        int32_t numX = XT_MIN(2, outW - x) - 1;

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean Vectors for Predicate Gather with corner cases  */
        /* handled for Out Width and Height being odd numbers      */
        vboolN vb1 = IVP_ORBN(IVP_LTRNI(12), IVP_LTRSN(20 * numX));
        vboolN vb2 = IVP_ORBN(IVP_LTRNI(8), IVP_LTRSN(20 * numX));
        vboolN vb3 = IVP_ANDBN(vb1, IVP_LTRSN(20 * numY));
        vboolN vb4 = IVP_ANDBN(vb2, IVP_LTRSN(20 * numY));

        for (ky = 0; ky < 5; ky++) /* Kernel Height */
        {
          /* Pointer for Input Data Load */
          pData1 = pData + ky * dilationY * inDataPitch2;
          pData2 = pData1 + (stride * inDataPitch2 * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);
          /* Assign valid address for predicated false lines */
          vecGatherOff00 = IVP_MOVNX16UT(vecGatherOff1, 0, vb1);
          vecGatherOff01 = IVP_MOVNX16UT(vecGatherOff2, 0, vb2);
          vecGatherOff10 = IVP_MOVNX16UT(vecGatherOff1, 0, vb3);
          vecGatherOff11 = IVP_MOVNX16UT(vecGatherOff2, 0, vb4);


          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            /* Gather Load of Input Data */
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff00);
            xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2       = IVP_GATHERANX8S(pData1, vecGatherOff01);
            xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);
            xb_gsr gather3       = IVP_GATHERANX8S(pData2, vecGatherOff10);
            xb_vec2Nx8 dvecData3 = IVP_GATHERD2NX8_L(gather3);
            xb_gsr gather4       = IVP_GATHERANX8S(pData2, vecGatherOff11);
            xb_vec2Nx8 dvecData4 = IVP_GATHERD2NX8_L(gather4);


            pData1 += 4;
            pData2 += 4;

            /* kx = 1 */
            /* Extracting scalars for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 3);

            /* 4 Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * \
                                                 coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 2 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         4);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 3 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         2);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 4 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         0);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         3);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         0);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         3);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 5 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         4);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 - 4 * coeffPitch2);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */

          /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh  = numInCh - inCh;
            vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);
            /* Assign valid address for predicated false lines */
            vecGatherOff00 = IVP_MOVNX16UT(vecGatherOff1, 0, IVP_ANDBN(vbRemInCh, vb1));
            vecGatherOff01 = IVP_MOVNX16UT(vecGatherOff2, 0, IVP_ANDBN(vbRemInCh, vb2));
            vecGatherOff10 = IVP_MOVNX16UT(vecGatherOff1, 0, IVP_ANDBN(vbRemInCh, vb3));
            vecGatherOff11 = IVP_MOVNX16UT(vecGatherOff2, 0, IVP_ANDBN(vbRemInCh, vb4));

            /* Gather Input Data */
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            xb_vec2Nx8 dvecData3 = 0;
            xb_vec2Nx8 dvecData4 = 0;
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff00);
            dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2 = IVP_GATHERANX8S(pData1, vecGatherOff01);
            dvecData2 = IVP_GATHERD2NX8_L(gather2);
            xb_gsr gather3 = IVP_GATHERANX8S(pData2, vecGatherOff10);
            dvecData3 = IVP_GATHERD2NX8_L(gather3);
            xb_gsr gather4 = IVP_GATHERANX8S(pData2, vecGatherOff11);
            dvecData4 = IVP_GATHERD2NX8_L(gather4);


            /* kx = 1 */
            /* Extracting scalars for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 3);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 2 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         4);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 3 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), \
                                         2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         2);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), \
                                         2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         2);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 4 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         0);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         3);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         0);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         3);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - coeffPitch1 * (remCh1 + remCh2));

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 5 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), \
                                         4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), \
                                         4);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, 0);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
          } /* End Input Channels corner case handling */
        }   /* End Kernel Height */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * \
                       numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_7x7_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for 7x7 MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate 7x7 MOD_DWH 3D         */
/*               dilated convolution function and 7x7 MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported.                  */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is 7x7xDxN                                     */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_7x7_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_KERNEL_SIZE(coeffTile, 7);
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDEX(param) == XAI_CNN_CONV_GET_STRIDEY(param)),                                         \
                    XAI_ERR_BADARG, "Stride along width = %hhu and height = %hhu\nStride along width and height should be equal", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_STRIDE(param) == 1) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 2) ||               \
                    (XAI_CNN_CONV_GET_STRIDE(param) == 4), XAI_ERR_BADARG, \
                    "\nStride = %hhu, value should be 1, 2 or 4", XAI_CNN_CONV_GET_STRIDE(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_EDGE(inTile, 3 + 3 * (XAI_CNN_CONV_GET_DILATION(param) - 1));
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }

#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Calling further optimized function if dim1Size == dim1Pitch */
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && \
      (XAI_CNN_CONV_GET_DILATIONX(param) == 1) && (XAI_CNN_CONV_GET_DILATIONY(param) == 1))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                                outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                              outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                             coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                           coeffTile, biasArray, outTile, param);
#endif
    }
    return(XAI_ERROR_STATUS());
  }

  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  XAI_ERROR_CHECKS_CONTINUE()
  {
    XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(inTile);
    /* Max value of Gather Offset is ((stride*min(1,outW-1) + 6*dilationX) * inDataPitch1 + min(3,numInCh-1)) */
    XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1_PITCH(inTile) <                                                                       \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) /                                                                   \
                     (XAI_CNN_CONV_GET_STRIDE(param) * XT_MIN(1, outW - 1) + 6 * XAI_CNN_CONV_GET_DILATION(param))),          \
                    XAI_ERR_BADARG, "\ndim1Pitch value of inTile = %d, should be less than Gather Offset(16-bit limit) - %d", \
                    XAI_TILE3D_GET_DIM1_PITCH(inTile),                                                                        \
                    ((USHRT_MAX - XT_MIN(3, numInCh - 1)) /                                                                   \
                     (XAI_CNN_CONV_GET_STRIDE(param) * XT_MIN(1, outW - 1) + 6 * XAI_CNN_CONV_GET_DILATION(param))));
  }

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t stride        = XAI_CNN_CONV_GET_STRIDE(param);
  const uint8_t dilationX     = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  /* Effective Kernel size = dilation(KernelSize - 1) + 1                */
  /* Effective kernel size is used for calculating the min required edge */
  int32_t dilatedKWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedKHeight = dilationY * (kHeightU - 1) + 1;
  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((dilatedKWidth / 2) * inDataPitch1 + (dilatedKHeight / 2) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, ky;

  /* 4 Gathers are being used to Load Input Data. Many common elements
   * will be loaded in separate Gathers, especially in the case of
   * stride 1 and 2. To take the advantage of having common offsets in
   * a single Gather, 2 Gather Patterns are generated as given below.
   * For example, in the Gather Patterns generated below,
   * if stride is 1 and dilation = 1, then 12 offsets are common and  8 offsets
   * if stride is 2 and dilation = 1, are common in each Gather.
   */
  /* Gather Index Calculations */
  xb_vecNx16 vecGather = IVP_ANDNX16(IVP_SEQNX16(), 3);
  IVP_MULANX16PACKL(vecGather, inDataPitch1 * dilationX, IVP_SRLINX16(IVP_SEQNX16(), 2));
  xb_vecNx16 vecGather1 = IVP_ADDNX16(vecGather, stride * inDataPitch1);

  xb_vecNx16 vecSelIdx1 = IVP_SEQNX16();
  IVP_ADDNX16T(vecSelIdx1, vecSelIdx1, 16, IVP_NOTBN(IVP_LTRNI(16)));
  xb_vecNx16U vecGatherOff1 = IVP_SELNX16(vecGather1, vecGather, vecSelIdx1);
  xb_vecNx16 vecSelIdx2     = IVP_ADDNX16(IVP_SEQNX16(), 16);
  IVP_ADDNX16T(vecSelIdx2, vecSelIdx2, 16, IVP_NOTBN(IVP_LTRNI(12)));
  xb_vecNx16U vecGatherOff2 = IVP_SELNX16(vecGather1, vecGather, vecSelIdx2);
  /* Index Pattern of vecGatherOff1 is -
   * 0 1 2 3 P1*d P1*d+1 P1*d+2 P1*d+3 2*P1*d 2*P1*d+1 2*P1*d+2 2*P1*d+3 3*P1*d 3*P1*d+1 3*P1*d+2 3*P1*d+3
   * s*P1 s*P1+1 s*P1+2 s*P1+3 (s+1*d)*P1 (s+1*d)*P1+1 (s+1*d)*P1+2 (s+1*d)*P1+3
   * (s+2*d)*P1  (s+2*d)*P1+1  (s+2*d)*P1+2  (s+2*d)*P1+3 */

  /* Index Pattern of vecGatherOff2 is -
   * 4*P1*d 4*P1*d+1 4*P1*d+2 4*P1*d+3 5*P1*d 5*P1*d+1 5*P1*d+2 5*P1*d+3 6*P1*d 6*P1*d+1 6*P1*d+2 6*P1*d+3
   * (s+3*d)*P1 (s+3*d)*P1+1 (s+3*d)*P1+2 (s+3*d)*P1+3 (s+4*d)*P1 (s+4*d)*P1*d+1 (s+4*d)*P1+2 (s+4*d)*P1+3
   * (s+5*d)*P1 (s+5*d)*P1+1 (s+5*d)*P1+2 (s+5*d)*P1+3 (s+6*d)*P1 (s+6*d)*P1+1 (s+6*d)*P1+2 (s+6*d)*P1+3 */

  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecOut;
  int8_t*     restrict pData1;
  int8_t*     restrict pData2;

  int32_t remCh = numInCh & 3;

  /*Generation of maskLut for handling cases when remCh is not equal to 0   */
  /*eg. if remInCh is equal to 1 then sumMask is 0000FFFF  */
  /*    if remInCh is equal to 2 then sumMask is 00FFFFFF  */
  const uint32_t maskLut[3] = { 0xff, 0xff00, 0xff0000 };

  uint8_t remCh1 = XT_SALT(2, remCh + 1);
  uint8_t remCh2 = XT_SALT(3, remCh + 1);

  uint32_t sumMask = maskLut[0] + maskLut[1] * remCh1 + maskLut[2] * remCh2;

#ifdef __XCC__
  XT_MEMW(); /* Adding Memory Wait as Gather and Normal Load/Stores are not synchronized */
#endif

  /* 4 Gathers are used for Input Data Load corresponding to 4         */
  /* Output Vectors. Loop unrolled along Output Width and Height by 2. */
  /* Also unrolled along Input Channels by 4 and Kernel Width.         */

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable used for corner case handling of Out Height odd */
      int32_t numY = XT_MIN(1, outH - y - 1);
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        xb_vecNx16U vecGatherOff00;
        xb_vecNx16U vecGatherOff01;
        xb_vecNx16U vecGatherOff10;
        xb_vecNx16U vecGatherOff11;
        /* Variable used for corner case handling of Out Width odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output, Input and Coefficient Data Pointers */
        int8_t *pOut   = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int8_t *pData  = pInData + (x * stride) * inDataPitch1 + (y * stride) * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Boolean Vectors for Predicate Gather with corner cases  */
        /* handled for Out Width and Height being odd numbers      */
        vboolN vb1 = IVP_ORBN(IVP_LTRNI(16), IVP_LTRSN(28 * numX));
        vboolN vb2 = IVP_ORBN(IVP_LTRNI(12), IVP_LTRSN(28 * numX));
        vboolN vb3 = IVP_ANDBN(vb1, IVP_LTRSN(28 * numY));
        vboolN vb4 = IVP_ANDBN(vb2, IVP_LTRSN(28 * numY));

        for (ky = 0; ky < 7; ky++) /* Kernel Height */
        {
          /* Pointer for Input Data Load */
          pData1 = pData + ky * inDataPitch2 * dilationY;
          pData2 = pData1 + (stride * inDataPitch2 * numY);
          /* Assign valid address for predicated false lines */
          vecGatherOff00 = IVP_MOVNX16UT(vecGatherOff1, 0, vb1);
          vecGatherOff01 = IVP_MOVNX16UT(vecGatherOff2, 0, vb2);
          vecGatherOff10 = IVP_MOVNX16UT(vecGatherOff1, 0, vb3);
          vecGatherOff11 = IVP_MOVNX16UT(vecGatherOff2, 0, vb4);
          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Number of Input Channels */
          {
            /* Gathers for Input Loads */
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff00);
            xb_vec2Nx8 dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2       = IVP_GATHERANX8S(pData1, vecGatherOff01);
            xb_vec2Nx8 dvecData2 = IVP_GATHERD2NX8_L(gather2);
            xb_gsr gather3       = IVP_GATHERANX8S(pData2, vecGatherOff10);
            xb_vec2Nx8 dvecData3 = IVP_GATHERD2NX8_L(gather3);
            xb_gsr gather4       = IVP_GATHERANX8S(pData2, vecGatherOff11);
            xb_vec2Nx8 dvecData4 = IVP_GATHERD2NX8_L(gather4);

            pData1 += 4;
            pData2 += 4;

            /* kx = 1 */
            /* Extracting Scalars for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 4);

            /* 4 Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 2 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 5);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 3 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 6);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 4 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 3);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 3);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 3);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 5 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 0);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 4);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 6 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 5);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch2 - 3 * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);

            /* kx = 7 */
            /* Extracting Scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 6);

            /* 4 Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1 - 6 * coeffPitch2);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */

          /* Handling Corner cases of Number of Input Channels not being multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh  = numInCh - inCh;
            vboolN vbRemInCh = IVP_LTNX16(IVP_ANDNX16(IVP_SEQNX16(), 3), remInCh);
            /* Assign valid address for predicated false lines */
            vecGatherOff00 = IVP_MOVNX16UT(vecGatherOff1, 0, IVP_ANDBN(vbRemInCh, vb1));
            vecGatherOff01 = IVP_MOVNX16UT(vecGatherOff2, 0, IVP_ANDBN(vbRemInCh, vb2));
            vecGatherOff10 = IVP_MOVNX16UT(vecGatherOff1, 0, IVP_ANDBN(vbRemInCh, vb3));
            vecGatherOff11 = IVP_MOVNX16UT(vecGatherOff2, 0, IVP_ANDBN(vbRemInCh, vb4));

            /* Gather Input Data */
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            xb_vec2Nx8 dvecData3 = 0;
            xb_vec2Nx8 dvecData4 = 0;
            xb_gsr gather1       = IVP_GATHERANX8S(pData1, vecGatherOff00);
            dvecData1 = IVP_GATHERD2NX8_L(gather1);
            xb_gsr gather2 = IVP_GATHERANX8S(pData1, vecGatherOff01);
            dvecData2 = IVP_GATHERD2NX8_L(gather2);
            xb_gsr gather3 = IVP_GATHERANX8S(pData2, vecGatherOff10);
            dvecData3 = IVP_GATHERD2NX8_L(gather3);
            xb_gsr gather4 = IVP_GATHERANX8S(pData2, vecGatherOff11);
            dvecData4 = IVP_GATHERD2NX8_L(gather4);


            /* kx = 1 */
            /* Extracting scalars for QMULs */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 4);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 4);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1;
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            xb_vec2Nx8 dvecCoeff2;
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            xb_vec2Nx8 dvecCoeff3;
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - coeffPitch1 * (remCh1 + remCh2));

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 2 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 5);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh1 + remCh2)));

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 3 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 6);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - coeffPitch1 * (remCh1 + remCh2));

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 4 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData1)), 3);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 3);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData3)), 3);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 3);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (coeffPitch1 * (remCh2 + remCh1)));

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 5 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 4);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 0);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 4);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (remCh1 + remCh2) * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 6 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 1);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 5);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 1);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 5);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch2 - (remCh1 + remCh2) * coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);

            /* kx = 7 */
            /* Extracting scalars for QMULs */
            qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 2);
            qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData2)), 6);
            qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 2);
            qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16(IVP_MOVNX16_FROM2NX8(dvecData4)), 6);

            /* Aligned Vector Loads of coefficients */
            IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * remCh1);
            IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * remCh2);
            IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, 0);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1 & sumMask);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2 & sumMask);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3 & sumMask);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4 & sumMask);
          } /* End Input Channels corner case handling */
        }   /* End Kernel Height */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D         */
/*               dilated convolution function and MxN MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/
#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE, \
                    "\nKernel height = %d, width = %d\nKernel width and height should be less than or equal to 64",  \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) > 0) && (XAI_CNN_CONV_GET_STRIDEY(param) > 0)) &&                                      \
                    ((XAI_CNN_CONV_GET_STRIDEX(param) <= 64) && (XAI_CNN_CONV_GET_STRIDEY(param) <= 64)), XAI_ERR_BADARG,                    \
                    "\nStrideX = %hhu, StrideY = %hhu\nStride along width and height should be greater than 0 and less than or equal to 64", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Calling further optimized function if dim1Size == dim1Pitch */
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && \
      (XAI_CNN_CONV_GET_DILATIONX(param) == 1) && (XAI_CNN_CONV_GET_DILATIONY(param) == 1))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                                outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                              outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                             coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                           coeffTile, biasArray, outTile, param);
#endif
    }
    return(XAI_ERROR_STATUS());
  }

  /* Getting parameters from the tile structures */
  const int32_t outW      = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH      = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh   = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh  = XAI_TILE3D_GET_DIM1(outTile);
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU   = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU  = XAI_TILE4D_GET_DIM4(coeffTile);
  int32_t dilatedkWidthU  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedkHeightU = dilationY * (kHeightU - 1) + 1;

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;
  if ((dilatedkWidthU % 2) != 0)
  {
    leftEdge = dilatedkWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedkWidthU / 2) : ((dilatedkWidthU / 2) - 1);
  }

  if ((dilatedkHeightU % 2) != 0)
  {
    topEdge = dilatedkHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedkHeightU / 2) : ((dilatedkHeightU / 2) - 1);
  }


  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, k;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);
      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

        xb_vecN_2x32v hvecInAddrOff    = 0;
        xb_vecN_2x32v hvecCoeffAddrOff = 0;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t inAddrOff, coeffAddrOff;

        for (k = 0; k < kHeightU * kWidthU; k++) /* Kernel Height * Kernel Width */
        {
          /* Condition checks performed to get the Input and Coefficient        */
          /* Pointer Offsets after combining the Kernel Width and Height Loops  */
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch2 * dilationY - kWidthU * inDataPitch1 * dilationX, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3 - kWidthU * coeffPitch2, vbN_2);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff        = IVP_EXTRN_2X32(hvecInAddrOff, 0);
          coeffAddrOff     = IVP_EXTRN_2X32(hvecCoeffAddrOff, 0);
          hvecLaneIdx      = IVP_ADDN_2X32(hvecLaneIdx, 1);
          hvecCoeffAddrOff = IVP_ADDN_2X32(hvecCoeffAddrOff, coeffPitch2);
          hvecInAddrOff    = IVP_ADDN_2X32(hvecInAddrOff, inDataPitch1 * dilationX);

          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + inAddrOff);
          pdvecData2 = (xb_vec2Nx8 *) (pData + inAddrOff + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8 *) (pData + inAddrOff + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8 *) (pData + inAddrOff + strideX * inDataPitch1 * numX + strideY * inDataPitch2 * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */

          /* Corner Case Handling if number of input channels not multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;
            vaData1 = IVP_LA2NX8_PP(pdvecData1);

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, remInCh);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Coefficient Loads */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Corner case handling */
        }   /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : further optimized function if dim1Size == dim1Pitch        */
/*               of 3D convolution for handling                             */
/*               cases where kwidth * numInch is a multiple of 4            */
/****************************************************************************/
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
  xb_vecNx16U* restrict pOutScaleData;
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;

  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];


  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();
  int32_t numIter  = kWidthU * numInCh;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (y = 0; y < outH; y++)
  {
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
      for (x = 0; x < (outW - 3); x += 4) /* Image Width */
      {                                   /* walk across the columns */
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + strideX * inDataPitch1);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + 2 * strideX * inDataPitch1);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + 3 * strideX * inDataPitch1);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

#ifdef __XCC__
#pragma loop_count min=1
#endif
          for (k = 0; k < numIter; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 3 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
      if (x < outW)
      {
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + 2 * strideX * inDataPitch1 * enable3rdWidth);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);

#ifdef __XCC__
#pragma loop_count min=1
#endif
          for (k = 0; k < numIter; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable2ndWidth);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable2ndWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable3rdWidth);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable3rdWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      }
    }     /* End Output Channels */
  }
}

/****************************************************************************/
/* Description : further optimized function if dim1Size == dim1Pitch        */
/*               of 3D convolution                                          */
/****************************************************************************/
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
  xb_vecNx16U* restrict pOutScaleData;
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t numIter = kWidthU * numInCh;

  int32_t leftEdge, topEdge;
  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (y = 0; y < outH; y++)
  {
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
      for (x = 0; x < outW - 3; x += 4) /* Image Width */
      {                                 /* walk across the columns */
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + strideX * inDataPitch1);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + 2 * strideX * inDataPitch1);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + 3 * strideX * inDataPitch1);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

          for (k = 0; k < numIter - 3; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);


            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }   /* End Input Channels */

          /* Corner case handling as numIter is not a multiple of 4 */
          {
            int32_t remInCh = numIter - k;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, remInCh);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);
            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);


            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 3 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
      if (x < outW)
      {
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 + 2 * strideX * inDataPitch1 * enable3rdWidth);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);

          for (k = 0; k < numIter - 3; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);


            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          }   /* End Input Channels */

          /* Corner case handling as numIter is not a multiple of 4 */
          {
            int32_t remInCh = numIter - k;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, remInCh);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);


            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable2ndWidth);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable2ndWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable3rdWidth);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable3rdWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      }
    }     /* End Output Channels */
  }
}

/***************xaiConvolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH***********/
/***************xaiConvolve3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH**************/
/* Description : P6 optimized implementation for MxN MOD_DWH 3D convolution.*/
/*               with loop across outTile as outermost loop. For H=1 , The  */
/*               outermost loop will be executed only once                  */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/
#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE, \
                    "\nKernel height = %d, width = %d\nKernel width and height should be less than or equal to 64",  \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) > 0) && (XAI_CNN_CONV_GET_STRIDEY(param) > 0)) &&                                      \
                    ((XAI_CNN_CONV_GET_STRIDEX(param) <= 64) && (XAI_CNN_CONV_GET_STRIDEY(param) <= 64)), XAI_ERR_BADARG,                    \
                    "\nStrideX = %hhu, StrideY = %hhu\nStride along width and height should be greater than 0 and less than or equal to 64", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Calling further optimized function if dim1Size == dim1Pitch */
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && \
      (XAI_CNN_CONV_GET_DILATIONX(param) == 1 && XAI_CNN_CONV_GET_DILATIONY(param) == 1))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                                          outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, \
                                                                        outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(inTile, \
                                                                       coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(inTile, \
                                                                     coeffTile, biasArray, outTile, param);
#endif
    }
    return(XAI_ERROR_STATUS());
  }

  /* Getting parameters from the tile structures */
  const int32_t outW      = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH      = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh   = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh  = XAI_TILE3D_GET_DIM1(outTile);
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);
  int32_t dilatedkWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedkHeight = dilationY * (kHeightU - 1) + 1;

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
  xb_vecNx16U* restrict pOutScaleData;
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;
  if ((dilatedkWidth % 2) != 0)
  {
    leftEdge = dilatedkWidth / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedkWidth / 2) : ((dilatedkWidth / 2) - 1);
  }

  if ((dilatedkHeight % 2) != 0)
  {
    topEdge = dilatedkHeight / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedkHeight / 2) : ((dilatedkHeight / 2) - 1);
  }


  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, k, x, y;
  int32_t inCh;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /* Loops Start */
  for (y = 0; y < outH; y++)
  {
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;

#ifdef DILATED_VQ_CONV
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif

      for (x = 0; x < outW - 3; x += 4) /* Image Width */
      {
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

        xb_vecN_2x32v hvecInAddrOff    = 0;
        xb_vecN_2x32v hvecCoeffAddrOff = 0;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t inAddrOff              = 0, coeffAddrOff = 0;

        for (k = 0; k < kWidthU * kHeightU; k++) /* Kernel Height * Kernel Width */
        {
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch2 * dilationY - kWidthU * inDataPitch1 * dilationX, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3 - kWidthU * coeffPitch2, vbN_2);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff        = IVP_EXTRN_2X32(hvecInAddrOff, 0);
          coeffAddrOff     = IVP_EXTRN_2X32(hvecCoeffAddrOff, 0);
          hvecLaneIdx      = IVP_ADDN_2X32(hvecLaneIdx, 1);
          hvecCoeffAddrOff = IVP_ADDN_2X32(hvecCoeffAddrOff, coeffPitch2);
          hvecInAddrOff    = IVP_ADDN_2X32(hvecInAddrOff, inDataPitch1 * dilationX);

          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + inAddrOff);
          pdvecData2 = (xb_vec2Nx8 *) (pData + inAddrOff + strideX * inDataPitch1);
          pdvecData3 = (xb_vec2Nx8 *) (pData + inAddrOff + strideX * inDataPitch1 * 2);
          pdvecData4 = (xb_vec2Nx8 *) (pData + inAddrOff + strideX * inDataPitch1 * 3);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          } /* End Input Channels */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;
            vaData1 = IVP_LA2NX8_PP(pdvecData1);

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8 dvecData4; IVP_LAV2NX8_XP(dvecData4, vaData4, pdvecData4, remInCh);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData4)), 0);

            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Coefficient Loads */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }
        }
        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 3 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      }
      if (x < outW)
      {
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

        xb_vecN_2x32v hvecInAddrOff    = 0;
        xb_vecN_2x32v hvecCoeffAddrOff = 0;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t inAddrOff              = 0, coeffAddrOff = 0;

        for (k = 0; k < kWidthU * kHeightU; k++) /* Kernel Height * Kernel Width */
        {
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch2 * dilationY - kWidthU * inDataPitch1 * dilationX, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3 - kWidthU * coeffPitch2, vbN_2);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff        = IVP_EXTRN_2X32(hvecInAddrOff, 0);
          coeffAddrOff     = IVP_EXTRN_2X32(hvecCoeffAddrOff, 0);
          hvecLaneIdx      = IVP_ADDN_2X32(hvecLaneIdx, 1);
          hvecCoeffAddrOff = IVP_ADDN_2X32(hvecCoeffAddrOff, coeffPitch2);
          hvecInAddrOff    = IVP_ADDN_2X32(hvecInAddrOff, inDataPitch1 * dilationX);

          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + inAddrOff);
          pdvecData2 = (xb_vec2Nx8 *) (pData + inAddrOff + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8 *) (pData + inAddrOff + strideX * inDataPitch1 * 2 * enable3rdWidth);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          } /* End Input Channels */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;
            vaData1 = IVP_LA2NX8_PP(pdvecData1);

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8 dvecData1; IVP_LAV2NX8_XP(dvecData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8 dvecData2; IVP_LAV2NX8_XP(dvecData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8 dvecData3; IVP_LAV2NX8_XP(dvecData3, vaData3, pdvecData3, remInCh);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8(dvecData3)), 0);

            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Coefficient Loads */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          }
        }
        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable2ndWidth);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable3rdWidth);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D         */
/*               dilated convolution function and MxN MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               inChannels is a multiple of 2                              */
/*               Active data pointer is aligned to 2-bytes                  */
/****************************************************************************/
#ifndef IVP_MULSUQA2N8XR8
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_U8S8IXCa2_depth2X_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_U8S8IXCa2_depth2X_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t numIter = kWidthU * numInCh;

  int32_t leftEdge, topEdge;
  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  uint16_t* restrict pData1;
  uint16_t* restrict pData2;
  uint16_t* restrict pData3;
  uint16_t* restrict pData4;
  xb_vec2Nx8* restrict pdvecOut;

  xb_vecNx16 vecData1, vecData2, vecData3, vecData4;
  xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
  xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
  xb_vecNx16 vecTemp1, vecTemp2;

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = ((int8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pData1 = (uint16_t *) (pData + ky * inDataPitch2);
          pData2 = (uint16_t *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * numX);
          pData3 = (uint16_t *) (pData + ky * inDataPitch2 + strideY * inDataPitch2 * numY);
          pData4 = (uint16_t *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * numX + strideY * inDataPitch2 * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

#ifdef __XCC__
#pragma loop_count min=1
#endif
          for (k = 0; k < numIter; k += 2) /* (Input Channels * kWidth) loops combined */
          {
            /* Load 2 bytes of input data */
            IVP_LSRNX16U_XP(vecData1, pData1, 2);
            IVP_LSRNX16U_XP(vecData2, pData2, 2);
            IVP_LSRNX16U_XP(vecData3, pData3, 2);
            IVP_LSRNX16U_XP(vecData4, pData4, 2);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData1, vecData1, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData1 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData2 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData2, vecData2, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData3 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData4 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData3, vecData3, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData5 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData6 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData4, vecData4, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData7 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData8 = IVP_MOV2NX8U_FROMNX16(vecTemp2);

            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}
#endif

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D         */
/*               dilated convolution function and MxN MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               Implementation also supports dilation > 1 for stride = 1   */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/****************************************************************************/
/* Although this routine supports IVP_MULSUQA2N8XR8, it has been intentionally disabled because we are not using it for the core that supports IVP_MULSUQA2N8XR8.
   We will be using convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth_x4 and convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth.
   These routines are faster than convolvedVQ3D_S_MxNdX_U8S8IXCa2_MOD_DWH */
#ifndef IVP_MULSUQA2N8XR8
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxNdX_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxNdX_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW      = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH      = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh   = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh  = XAI_TILE3D_GET_DIM1(outTile);
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);
  int32_t dilatedkWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedkHeight = dilationY * (kHeightU - 1) + 1;

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;
  if ((dilatedkWidth % 2) != 0)
  {
    leftEdge = dilatedkWidth / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedkWidth / 2) : ((dilatedkWidth / 2) - 1);
  }

  if ((dilatedkHeight % 2) != 0)
  {
    topEdge = dilatedkHeight / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedkHeight / 2) : ((dilatedkHeight / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, k;
  valign vaOutData = IVP_ZALIGN();

  /* Vector data pointers */
  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

#ifndef IVP_MULSUQA2N8XR8
  /* Vector data registers */
  xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
  xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
  xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
  xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
  xb_vecNx16 vecData1, vecData2;
  xb_vecNx16 vecData3, vecData4;
  xb_vecNx16 vecData5, vecData6;
  xb_vecNx16 vecData7, vecData8;
  xb_vecNx16 vecTemp1, vecTemp2;

  /* Custom select pattern for DSELs */
  int16_t sel1       = (XCHAL_IVPN_SIMD_WIDTH << 8);
  xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
  int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
  xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);
#endif

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * inDataPitch1 + y * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

        xb_vecN_2x32v hvecInAddrOff    = 0;
        xb_vecN_2x32v hvecCoeffAddrOff = 0;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t inAddrOff, coeffAddrOff;

        for (k = 0; k < kHeightU * kWidthU; k++) /* Kernel Height * Kernel Width */
        {
          /* Condition checks performed to get the Input and Coefficient        */
          /* Pointer Offsets after combining the Kernel Width and Height Loops  */
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch2 * dilationY - kWidthU * inDataPitch1 * dilationX, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3 - kWidthU * coeffPitch2, vbN_2);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff        = IVP_EXTRN_2X32(hvecInAddrOff, 0);
          coeffAddrOff     = IVP_EXTRN_2X32(hvecCoeffAddrOff, 0);
          hvecLaneIdx      = IVP_ADDN_2X32(hvecLaneIdx, 1);
          hvecCoeffAddrOff = IVP_ADDN_2X32(hvecCoeffAddrOff, coeffPitch2);
          hvecInAddrOff    = IVP_ADDN_2X32(hvecInAddrOff, inDataPitch1 * dilationX);

          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + inAddrOff);
          pdvecData2 = (xb_vec2Nx8U *) (pData + inAddrOff + inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8U *) (pData + inAddrOff + inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8U *) (pData + inAddrOff + inDataPitch1 * numX + inDataPitch2 * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Priming input loads */
          valign vaIn1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaIn2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaIn3 = IVP_LA2NX8U_PP(pdvecData3);
          valign vaIn4 = IVP_LA2NX8U_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            xb_vec2Nx8U dvecInData1, dvecInData2, dvecInData3, dvecInData4;
            /* Aligning variable vector load of pixels */
            IVP_LAV2NX8U_XP(dvecInData1, vaIn1, pdvecData1, 4);
            IVP_LAV2NX8U_XP(dvecInData2, vaIn2, pdvecData2, 4);
            IVP_LAV2NX8U_XP(dvecInData3, vaIn3, pdvecData3, 4);
            IVP_LAV2NX8U_XP(dvecInData4, vaIn4, pdvecData4, 4);
#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, dvecCoeff4);
#endif
          }   /* End Input Channels */
          /* Corner Case Handling if number of input channels not multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;
            vaIn1 = IVP_LA2NX8U_PP(pdvecData1);
            xb_vec2Nx8U dvecInData1, dvecInData2, dvecInData3, dvecInData4;
            /* Aligning variable vector load of pixels */
            IVP_LAV2NX8U_XP(dvecInData1, vaIn1, pdvecData1, remInCh);
            IVP_LAV2NX8U_XP(dvecInData2, vaIn2, pdvecData2, remInCh);
            IVP_LAV2NX8U_XP(dvecInData3, vaIn3, pdvecData3, remInCh);
            IVP_LAV2NX8U_XP(dvecInData4, vaIn4, pdvecData4, remInCh);

            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif
            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSA2NX8(daccSum1, dvecData9, dvecCoeff3);
            IVP_MULUSA2NX8(daccSum2, dvecData11, dvecCoeff3);
            IVP_MULUSA2NX8(daccSum3, dvecData13, dvecCoeff3);
            IVP_MULUSA2NX8(daccSum4, dvecData15, dvecCoeff3);
#endif
          }    /* End Corner case handling */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}
#endif

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D         */
/*               dilated convolution function and MxN MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               No edges along dimension 1 of inTile                       */
/****************************************************************************/

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_MxN_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE, \
                    "\nKernel height = %d, width = %d\nKernel width and height should be less than or equal to 64",  \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) > 0) && (XAI_CNN_CONV_GET_STRIDEY(param) > 0)) &&                                      \
                    ((XAI_CNN_CONV_GET_STRIDEX(param) <= 64) && (XAI_CNN_CONV_GET_STRIDEY(param) <= 64)), XAI_ERR_BADARG,                    \
                    "\nStrideX = %hhu, StrideY = %hhu\nStride along width and height should be greater than 0 and less than or equal to 64", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_PITCH(inTile) == XAI_TILE3D_GET_DIM1(inTile)) \
                      && XAI_CNN_CONV_GET_DILATION(param) == 1) || XAI_CNN_CONV_GET_DILATION(param) > 1),
                    XAI_ERR_BADARG, "Edges along input channels is not supported if dilation = 1.");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

#ifdef IVP_MULSUQA2N8XR8 // only for Vision_130
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && \
      (XAI_CNN_CONV_GET_DILATIONX(param) == 1 && XAI_CNN_CONV_GET_DILATIONY(param) == 1))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth(inTile, coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth(inTile, coeffTile, biasArray, outTile, param);
#endif
    }
    return(XAI_ERROR_STATUS());
  }
#else // Vision_P6
  if (XAI_CNN_CONV_GET_DILATIONX(param) > 1 && XAI_CNN_CONV_GET_DILATIONY(param) > 1)
  {
#ifdef DILATED_VQ_CONV
    convolvedVQ3D_S_MxNdX_U8S8IXCa2_MOD_DWH(inTile, \
                                            coeffTile, biasArray, outputScaleArray, outTile, param);
#else
    convolved3D_S_MxNdX_U8S8IXCa2_MOD_DWH(inTile, \
                                          coeffTile, biasArray, outTile, param);
#endif
    return(XAI_ERROR_STATUS());
  }
  if ((XAI_CNN_CONV_GET_DILATIONX(param) == 1 && XAI_CNN_CONV_GET_DILATIONY(param) == 1) && \
      (XAI_TILE3D_GET_DIM1(inTile) % 2) == 0                                                \
      && ((XAI_PTR_TO_ADDR(XAI_TILE3D_GET_DATA_PTR(inTile)) & (2 - 1)) == 0))
  {
#ifdef DILATED_VQ_CONV
    convolvedVQ3D_S_MxN_U8S8IXCa2_depth2X_MOD_DWH(inTile, \
                                                  coeffTile, biasArray, outputScaleArray, outTile, param);
#else
    convolved3D_S_MxN_U8S8IXCa2_depth2X_MOD_DWH(inTile, \
                                                coeffTile, biasArray, outTile, param);
#endif
    return(XAI_ERROR_STATUS());
  }
#endif

  /* Getting parameters from the tile structures */
  const int32_t outW      = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH      = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh   = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh  = XAI_TILE3D_GET_DIM1(outTile);
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);
  int32_t dilatedkWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedkHeight = dilationY * (kHeightU - 1) + 1;

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  //int32_t numIter = kWidthU * numInCh;

  int32_t leftEdge, topEdge;
  if ((dilatedkWidth % 2) != 0)
  {
    leftEdge = dilatedkWidth / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedkWidth / 2) : ((dilatedkWidth / 2) - 1);
  }

  if ((dilatedkHeight % 2) != 0)
  {
    topEdge = dilatedkHeight / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedkHeight / 2) : ((dilatedkHeight / 2) - 1);
  }


  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, k;
  valign vaOutData = IVP_ZALIGN();

  /* Vector data pointers */
  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /* Vector data registers */
  xb_vec2Nx8U dvecInData1, dvecInData2, dvecInData3, dvecInData4;

  valign vaIn1, vaIn2, vaIn3, vaIn4;

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
  {                                                                     /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
    xb_vecNx16U outScaleDataEven, outScaleDataOdd;
    /*Load output scale values*/
    VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
    for (y = 0; y < outH; y += 2) /* Image Height */
    {                             /* walk down the rows */
      /* Variable to handle corner case when height is odd */
      int32_t numY = XT_MIN(1, outH - y - 1);

      for (x = 0; x < outW; x += 2) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t numX = XT_MIN(1, outW - x - 1);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

        xb_vecN_2x32v hvecInAddrOff    = 0;
        xb_vecN_2x32v hvecCoeffAddrOff = 0;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t inAddrOff, coeffAddrOff;

        for (k = 0; k < kHeightU * kWidthU; k++) /* Kernel Height * Kernel Width */
        {
          /* Condition checks performed to get the Input and Coefficient        */
          /* Pointer Offsets after combining the Kernel Width and Height Loops  */
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch2 * dilationY - kWidthU * inDataPitch1 * dilationX, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3 - kWidthU * coeffPitch2, vbN_2);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff        = IVP_EXTRN_2X32(hvecInAddrOff, 0);
          coeffAddrOff     = IVP_EXTRN_2X32(hvecCoeffAddrOff, 0);
          hvecLaneIdx      = IVP_ADDN_2X32(hvecLaneIdx, 1);
          hvecCoeffAddrOff = IVP_ADDN_2X32(hvecCoeffAddrOff, coeffPitch2);
          hvecInAddrOff    = IVP_ADDN_2X32(hvecInAddrOff, inDataPitch1 * dilationX);

          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + inAddrOff);
          pdvecData2 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8U *) (pData + inAddrOff + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1 * numX + strideY * inDataPitch2 * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Priming input loads */
          vaIn1 = IVP_LA2NX8U_PP(pdvecData1);
          vaIn2 = IVP_LA2NX8U_PP(pdvecData2);
          vaIn3 = IVP_LA2NX8U_PP(pdvecData3);
          vaIn4 = IVP_LA2NX8U_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            /* Aligning variable vector load of pixels */


            /* Load 4 bytes of input data */
            IVP_LAV2NX8U_XP(dvecInData1, vaIn1, pdvecData1, 4);
            IVP_LAV2NX8U_XP(dvecInData2, vaIn2, pdvecData2, 4);
            IVP_LAV2NX8U_XP(dvecInData3, vaIn3, pdvecData3, 4);
            IVP_LAV2NX8U_XP(dvecInData4, vaIn4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
            xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
            xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
            xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
            xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
            xb_vecNx16 vecData1, vecData2;
            xb_vecNx16 vecData3, vecData4;
            xb_vecNx16 vecData5, vecData6;
            xb_vecNx16 vecData7, vecData8;
            xb_vecNx16 vecTemp1, vecTemp2;

            /* Custom select pattern for DSELs */
            int16_t sel1       = ((XCHAL_IVPN_SIMD_WIDTH << 8));
            xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
            int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
            xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);

            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif
            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, dvecCoeff4);
#endif
          } /* End Input Channels */

          /* Corner Case Handling if number of input channels not multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;
            vaIn1 = IVP_LA2NX8U_PP(pdvecData1);

            /* Load 4 bytes of input data */
            IVP_LAV2NX8U_XP(dvecInData1, vaIn1, pdvecData1, remInCh);
            IVP_LAV2NX8U_XP(dvecInData2, vaIn2, pdvecData2, remInCh);
            IVP_LAV2NX8U_XP(dvecInData3, vaIn3, pdvecData3, remInCh);
            IVP_LAV2NX8U_XP(dvecInData4, vaIn4, pdvecData4, remInCh);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
            xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
            xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
            xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
            xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
            xb_vecNx16 vecData1, vecData2;
            xb_vecNx16 vecData3, vecData4;
            xb_vecNx16 vecData5, vecData6;
            xb_vecNx16 vecData7, vecData8;
            xb_vecNx16 vecTemp1, vecTemp2;

            /* Custom select pattern for DSELs */
            int16_t sel1       = ((XCHAL_IVPN_SIMD_WIDTH << 8));
            xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
            int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
            xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);

            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif

            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Coefficient Loads */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, 0);
            IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, 0);
            IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, 0);
            IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, 0);
#endif
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel * numX);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * bytesPerPixel * numY);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numY);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 + outDataPitch2) * bytesPerPixel * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * numX * numY);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX * numY);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : P6 optimized implementation for noUnrollH MxN MOD_DWH      */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D         */
/*               dilated convolution function and MxN MOD_DWH 3D VQ         */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 15.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               No edges along dimension 1 of inTile                       */
/****************************************************************************/
/* Although this routine supports IVP_MULSUQA2N8XR8, it has been intentionally disabled because we are not using it for the core that supports IVP_MULSUQA2N8XR8.
   We will be using convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth_x4 and convolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH_contiguous_depth.
   These routines are faster than convolvedVQ3D_S_MxNdX_U8S8IXCa2_MOD_DWH */
#ifndef IVP_MULSUQA2N8XR8
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxNdX_U8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxNdX_U8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW      = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH      = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh   = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh  = XAI_TILE3D_GET_DIM1(outTile);
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);
  int32_t dilatedkWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedkHeight = dilationY * (kHeightU - 1) + 1;

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
  xb_vecNx16U* restrict pOutScaleData;
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;
  if ((dilatedkWidth % 2) != 0)
  {
    leftEdge = dilatedkWidth / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedkWidth / 2) : ((dilatedkWidth / 2) - 1);
  }

  if ((dilatedkHeight % 2) != 0)
  {
    topEdge = dilatedkHeight / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedkHeight / 2) : ((dilatedkHeight / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, k;
  valign vaOutData = IVP_ZALIGN();

  /* Vector data pointers */
  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

#ifndef IVP_MULSUQA2N8XR8
  /* Vector data registers */
  xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
  xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
  xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
  xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
  xb_vecNx16 vecData1, vecData2;
  xb_vecNx16 vecData3, vecData4;
  xb_vecNx16 vecData5, vecData6;
  xb_vecNx16 vecData7, vecData8;
  xb_vecNx16 vecTemp1, vecTemp2;

  /* Custom select pattern for DSELs */
  int16_t sel1       = (XCHAL_IVPN_SIMD_WIDTH << 8);
  xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
  int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
  xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);
#endif

  /* Loops Start */
  for (y = 0; y < outH; y++) /* Image Height */
  {
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
      for (x = 0; x < outW; x += 4) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner cases */
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        int32_t enable4thWidth = XT_SALT(3, outW - x);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * inDataPitch1 + y * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;



        xb_vecN_2x32v hvecInAddrOff    = 0;
        xb_vecN_2x32v hvecCoeffAddrOff = 0;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t inAddrOff, coeffAddrOff;

        for (k = 0; k < kHeightU * kWidthU; k++) /* Kernel Height * Kernel Width */
        {
          /* Condition checks performed to get the Input and Coefficient        */
          /* Pointer Offsets after combining the Kernel Width and Height Loops  */
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch2 * dilationY - kWidthU * inDataPitch1 * dilationX, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3 - kWidthU * coeffPitch2, vbN_2);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff        = IVP_EXTRN_2X32(hvecInAddrOff, 0);
          coeffAddrOff     = IVP_EXTRN_2X32(hvecCoeffAddrOff, 0);
          hvecLaneIdx      = IVP_ADDN_2X32(hvecLaneIdx, 1);
          hvecCoeffAddrOff = IVP_ADDN_2X32(hvecCoeffAddrOff, coeffPitch2);
          hvecInAddrOff    = IVP_ADDN_2X32(hvecInAddrOff, inDataPitch1 * dilationX);

          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + inAddrOff);
          pdvecData2 = (xb_vec2Nx8U *) (pData + inAddrOff + inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8U *) (pData + inAddrOff + inDataPitch1 * 2 * enable3rdWidth);
          pdvecData4 = (xb_vec2Nx8U *) (pData + inAddrOff + inDataPitch1 * 3 * enable4thWidth);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Priming input loads */
          valign vaIn1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaIn2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaIn3 = IVP_LA2NX8U_PP(pdvecData3);
          valign vaIn4 = IVP_LA2NX8U_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            xb_vec2Nx8U dvecInData1, dvecInData2, dvecInData3, dvecInData4;
            /* Aligning variable vector load of pixels */
            IVP_LAV2NX8U_XP(dvecInData1, vaIn1, pdvecData1, 4);
            IVP_LAV2NX8U_XP(dvecInData2, vaIn2, pdvecData2, 4);
            IVP_LAV2NX8U_XP(dvecInData3, vaIn3, pdvecData3, 4);
            IVP_LAV2NX8U_XP(dvecInData4, vaIn4, pdvecData4, 4);
#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, dvecCoeff4);
#endif
          }   /* End Input Channels */
          /* Corner Case Handling if number of input channels not multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;
            vaIn1 = IVP_LA2NX8U_PP(pdvecData1);
            xb_vec2Nx8U dvecInData1, dvecInData2, dvecInData3, dvecInData4;
            /* Aligning variable vector load of pixels */
            IVP_LAV2NX8U_XP(dvecInData1, vaIn1, pdvecData1, remInCh);
            IVP_LAV2NX8U_XP(dvecInData2, vaIn2, pdvecData2, remInCh);
            IVP_LAV2NX8U_XP(dvecInData3, vaIn3, pdvecData3, remInCh);
            IVP_LAV2NX8U_XP(dvecInData4, vaIn4, pdvecData4, remInCh);

            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif
            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSA2NX8(daccSum1, dvecData9, dvecCoeff3);
            IVP_MULUSA2NX8(daccSum2, dvecData11, dvecCoeff3);
            IVP_MULUSA2NX8(daccSum3, dvecData13, dvecCoeff3);
            IVP_MULUSA2NX8(daccSum4, dvecData15, dvecCoeff3);
#endif
          }    /* End Corner case handling */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * enable2ndWidth) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable2ndWidth);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable2ndWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * 2 * enable3rdWidth) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable3rdWidth);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable3rdWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * 3 * enable4thWidth) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * enable4thWidth);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable4thWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}
#endif

#ifndef IVP_MULSUQA2N8XR8
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_depth2X_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_U8S8IXCa2_noUnrollH_depth2X_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
  xb_vecNx16U* restrict pOutScaleData;
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t numIter = kWidthU * numInCh;

  int32_t leftEdge, topEdge;
  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  uint16_t* restrict pData1;
  uint16_t* restrict pData2;
  uint16_t* restrict pData3;
  uint16_t* restrict pData4;
  xb_vec2Nx8* restrict pdvecOut;

  xb_vecNx16 vecData1, vecData2, vecData3, vecData4;
  xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
  xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
  xb_vecNx16 vecTemp1, vecTemp2;

  /* Loops Start */
  for (y = 0; y < outH; y++) /* Image Height */
  {
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
      for (x = 0; x < outW; x += 4) /* Image Width */
      {                             /* walk across the columns */
        /* Variables to handle corner cases */
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        int32_t enable4thWidth = XT_SALT(3, outW - x);

        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = ((int8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pData1 = (uint16_t *) (pData + ky * inDataPitch2);
          pData2 = (uint16_t *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * enable2ndWidth);
          pData3 = (uint16_t *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * 2 * enable3rdWidth);
          pData4 = (uint16_t *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * 3 * enable4thWidth);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

#ifdef __XCC__
#pragma loop_count min=1
#endif
          for (k = 0; k < numIter; k += 2) /* (Input Channels * kWidth) loops combined */
          {
            /* Load 2 bytes of input data */
            IVP_LSRNX16U_XP(vecData1, pData1, 2);
            IVP_LSRNX16U_XP(vecData2, pData2, 2);
            IVP_LSRNX16U_XP(vecData3, pData3, 2);
            IVP_LSRNX16U_XP(vecData4, pData4, 2);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData1, vecData1, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData1 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData2 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData2, vecData2, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData3 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData4 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData3, vecData3, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData5 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData6 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData4, vecData4, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData7 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData8 = IVP_MOV2NX8U_FROMNX16(vecTemp2);

            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * enable2ndWidth) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable2ndWidth);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable2ndWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * 2 * enable3rdWidth) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable3rdWidth);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable3rdWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1 * 3 * enable4thWidth) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh * enable4thWidth);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable4thWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}
#endif

/****************************************************************************/
/* Description : further optimized function if dim1Size == dim1Pitch        */
/*               of 3D convolution for handling                             */
/*               cases where kwidth * numInch is a multiple of 4            */
/****************************************************************************/
#ifdef IVP_MULSUQA2N8XR8
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;

  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];


  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();
  int32_t numIter  = kWidthU * numInCh;

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (y = 0; y < outH; y++)
  {
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif

      for (x = 0; x < (outW - 3); x += 4) /* Image Width */
      {                                   /* walk across the columns */
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + strideX * inDataPitch1);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + 2 * strideX * inDataPitch1);
          pdvecData4 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + 3 * strideX * inDataPitch1);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8U_PP(pdvecData4);

#ifdef __XCC__
#pragma loop_count min=1
#endif
          for (k = 0; k < numIter; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecData3; IVP_LAV2NX8U_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecData4; IVP_LAV2NX8U_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 3 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
      if (x < outW)
      {
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + 2 * strideX * inDataPitch1 * enable3rdWidth);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);

#ifdef __XCC__
#pragma loop_count min=1
#endif
          for (k = 0; k < numIter; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecData3; IVP_LAV2NX8U_XP(dvecData3, vaData3, pdvecData3, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData3)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable2ndWidth);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable2ndWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable3rdWidth);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable3rdWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      }
    }     /* End Output Channels */
  }
}
#endif

/****************************************************************************/
/* Description : further optimized function if dim1Size == dim1Pitch        */
/*               of 3D convolution                                          */
/****************************************************************************/
#ifdef IVP_MULSUQA2N8XR8
#ifdef DILATED_VQ_CONV
static _XAI_INLINE_ void convolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void convolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t numIter = kWidthU * numInCh;

  int32_t leftEdge, topEdge;
  if ((kWidthU % 2) != 0)
  {
    leftEdge = kWidthU / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (kWidthU / 2) : ((kWidthU / 2) - 1);
  }

  if ((kHeightU % 2) != 0)
  {
    topEdge = kHeightU / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (kHeightU / 2) : ((kHeightU / 2) - 1);
  }

  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t outCh, x, y, ky, k;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (y = 0; y < outH; y++)
  {
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
      for (x = 0; x < outW - 3; x += 4) /* Image Width */
      {                                 /* walk across the columns */
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + strideX * inDataPitch1);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + 2 * strideX * inDataPitch1);
          pdvecData4 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + 3 * strideX * inDataPitch1);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8U_PP(pdvecData4);

          for (k = 0; k < numIter - 3; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecData3; IVP_LAV2NX8U_XP(dvecData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecData4; IVP_LAV2NX8U_XP(dvecData4, vaData4, pdvecData4, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData4)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);


            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }   /* End Input Channels */

          /* Corner case handling as numIter is not a multiple of 4 */
          {
            int32_t remInCh = numIter - k;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8U dvecData3; IVP_LAV2NX8U_XP(dvecData3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8U dvecData4; IVP_LAV2NX8U_XP(dvecData4, vaData4, pdvecData4, remInCh);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData4)), 0);
            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);


            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 3 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * \
                       remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End image width */
      if (x < outW)
      {
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 + 2 * strideX * inDataPitch1 * enable3rdWidth);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);

          for (k = 0; k < numIter - 3; k += 4) /* (Input Channels * kWidth) loops combined */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecData3; IVP_LAV2NX8U_XP(dvecData3, vaData3, pdvecData3, 4);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData3)), 0);

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);


            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          }   /* End Input Channels */

          /* Corner case handling as numIter is not a multiple of 4 */
          {
            int32_t remInCh = numIter - k;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8U dvecData3; IVP_LAV2NX8U_XP(dvecData3, vaData3, pdvecData3, remInCh);

            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecData3)), 0);
            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable2ndWidth);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable2ndWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable3rdWidth);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable3rdWidth);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      }
    }     /* End Output Channels */
  }
}
#endif

#ifdef DILATED_VQ_CONV
XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiConvolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_U8(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE, \
                    "\nKernel height = %d, width = %d\nKernel width and height should be less than or equal to 64",  \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) > 0) && (XAI_CNN_CONV_GET_STRIDEY(param) > 0)) &&                                      \
                    ((XAI_CNN_CONV_GET_STRIDEX(param) <= 64) && (XAI_CNN_CONV_GET_STRIDEY(param) <= 64)), XAI_ERR_BADARG,                    \
                    "\nStrideX = %hhu, StrideY = %hhu\nStride along width and height should be greater than 0 and less than or equal to 64", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) > 0 && XAI_CNN_CONV_GET_DILATIONY(param) > 0), \
                    XAI_ERR_BADARG, "dilation parameter has to be >= 1");
    XAI_CHECK_ERROR((((XAI_TILE3D_GET_DIM1_PITCH(inTile) == XAI_TILE3D_GET_DIM1(inTile)) \
                      && XAI_CNN_CONV_GET_DILATION(param) == 1) || XAI_CNN_CONV_GET_DILATION(param) > 1),
                    XAI_ERR_BADARG, "Edges along input channels is not supported if dilation = 1.");
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif
  }
#ifndef DILATED_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

#ifdef IVP_MULSUQA2N8XR8 // only for Vision_130
  if (XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile) && \
      (XAI_CNN_CONV_GET_DILATIONX(param) == 1) && (XAI_CNN_CONV_GET_DILATIONY(param) == 1))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(inTile, coeffTile, biasArray, outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV
      convolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(inTile, coeffTile, biasArray, outputScaleArray, outTile, param);
#else
      convolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(inTile, coeffTile, biasArray, outTile, param);
#endif
    }
    return(XAI_ERROR_STATUS());
  }
#else // Vision_P6
  if (XAI_CNN_CONV_GET_DILATIONX(param) > 1 && XAI_CNN_CONV_GET_DILATIONY(param) > 1)
  {
#ifdef DILATED_VQ_CONV
    convolvedVQ3D_S_MxNdX_U8S8IXCa2_noUnrollH_MOD_DWH(inTile, \
                                                      coeffTile, biasArray, outputScaleArray, outTile, param);
#else
    convolved3D_S_MxNdX_U8S8IXCa2_noUnrollH_MOD_DWH(inTile, \
                                                    coeffTile, biasArray, outTile, param);
#endif
    return(XAI_ERROR_STATUS());
  }
  /* If number of input channels is a multiple of 2 &
     the active data pointer is aligned to 2-bytes,
     call a more optimal variant */
  if ((XAI_CNN_CONV_GET_DILATIONX(param) == 1 && XAI_CNN_CONV_GET_DILATIONY(param) == 1) && \
      (XAI_TILE3D_GET_DIM1(inTile) % 2) == 0                                                \
      && ((XAI_PTR_TO_ADDR(XAI_TILE3D_GET_DATA_PTR(inTile)) & (2 - 1)) == 0))
  {
#ifdef DILATED_VQ_CONV
    convolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_depth2X_MOD_DWH(inTile, \
                                                            coeffTile, biasArray, outputScaleArray, outTile, param);
#else
    convolved3D_S_MxN_U8S8IXCa2_noUnrollH_depth2X_MOD_DWH(inTile, \
                                                          coeffTile, biasArray, outTile, param);
#endif
    return(XAI_ERROR_STATUS());
  }
#endif

  /* Getting parameters from the tile structures */
  const int32_t outW      = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH      = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh   = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh  = XAI_TILE3D_GET_DIM1(outTile);
  const uint8_t dilationX = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY = XAI_CNN_CONV_GET_DILATIONY(param);

  /* Kernel Size (NDWH) */
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM4(coeffTile);
  int32_t dilatedKWidth  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedKHeight = dilationY * (kHeightU - 1) + 1;

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);
#ifdef DILATED_VQ_CONV
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
  xb_vecNx16U* restrict pOutScaleData;
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  /* Pitches of Coefficient Data (NDWH) in dim1, dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t leftEdge, topEdge;
  if ((dilatedKWidth % 2) != 0)
  {
    leftEdge = dilatedKWidth / 2;
  }
  else
  {
    leftEdge = leftEdgeFlag ? (dilatedKWidth / 2) : ((dilatedKWidth / 2) - 1);
  }

  if ((dilatedKHeight % 2) != 0)
  {
    topEdge = dilatedKHeight / 2;
  }
  else
  {
    topEdge = topEdgeFlag ? (dilatedKHeight / 2) : ((dilatedKHeight / 2) - 1);
  }


  /* Move pointer to the start of the data (including edge) */
  pInData = &pInData[-((leftEdge) * inDataPitch1 + (topEdge) * inDataPitch2)];

  /* Setting the limits for output data according to ReLu Flag and outTileType */
  int32_t minLim, maxLim;
  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? \
             SHRT_MIN : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MIN : 0);
    maxLim = XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX \
             : (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S8) ? SCHAR_MAX : UCHAR_MAX);
  }
  const int8_t typeFlag       = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16)) ? 1 : 0;
  const uint8_t bytesPerPixel = XAI_TILE3D_GET_ELEMENT_SIZE(outTile);

  /* Variable Declarations */
  int32_t inCh, outCh, x, y, k;
  valign vaOutData = IVP_ZALIGN();

  /* Vector data pointers */
  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;

  /* Loops Start */
  for (y = 0; y < outH; y++)
  {
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH) /* Output Channels */
    {                                                                     /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = (numOutCh - outCh);
#ifdef DILATED_VQ_CONV
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif

      for (x = 0; x < outW - 3; x += 4) /* Image Width */
      {
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

        xb_vecN_2x32v hvecInAddrOff    = 0;
        xb_vecN_2x32v hvecCoeffAddrOff = 0;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t inAddrOff              = 0, coeffAddrOff = 0;

        for (k = 0; k < kWidthU * kHeightU; k++) /* Kernel Height * Kernel Width */
        {
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch2 * dilationY - kWidthU * inDataPitch1 * dilationX, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3 - kWidthU * coeffPitch2, vbN_2);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff        = IVP_EXTRN_2X32(hvecInAddrOff, 0);
          coeffAddrOff     = IVP_EXTRN_2X32(hvecCoeffAddrOff, 0);
          hvecLaneIdx      = IVP_ADDN_2X32(hvecLaneIdx, 1);
          hvecCoeffAddrOff = IVP_ADDN_2X32(hvecCoeffAddrOff, coeffPitch2);
          hvecInAddrOff    = IVP_ADDN_2X32(hvecInAddrOff, inDataPitch1 * dilationX);

          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + inAddrOff);
          pdvecData2 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1);
          pdvecData3 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1 * 2);
          pdvecData4 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1 * 3);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8U_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecInData1; IVP_LAV2NX8U_XP(dvecInData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecInData2; IVP_LAV2NX8U_XP(dvecInData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecInData3; IVP_LAV2NX8U_XP(dvecInData3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecInData4; IVP_LAV2NX8U_XP(dvecInData4, vaData4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
            xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
            xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
            xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
            xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
            xb_vecNx16 vecData1, vecData2;
            xb_vecNx16 vecData3, vecData4;
            xb_vecNx16 vecData5, vecData6;
            xb_vecNx16 vecData7, vecData8;
            xb_vecNx16 vecTemp1, vecTemp2;

            /* Custom select pattern for DSELs */
            int16_t sel1       = ((XCHAL_IVPN_SIMD_WIDTH << 8));
            xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
            int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
            xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);

            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif
            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, dvecCoeff4);
#endif
          }   /* End Input Channels */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;
            vaData1 = IVP_LA2NX8U_PP(pdvecData1);

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecInData1; IVP_LAV2NX8U_XP(dvecInData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8U dvecInData2; IVP_LAV2NX8U_XP(dvecInData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8U dvecInData3; IVP_LAV2NX8U_XP(dvecInData3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8U dvecInData4; IVP_LAV2NX8U_XP(dvecInData4, vaData4, pdvecData4, remInCh);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData4)), 0);
#else
            xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
            xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
            xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
            xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
            xb_vecNx16 vecData1, vecData2;
            xb_vecNx16 vecData3, vecData4;
            xb_vecNx16 vecData5, vecData6;
            xb_vecNx16 vecData7, vecData8;
            xb_vecNx16 vecTemp1, vecTemp2;

            /* Custom select pattern for DSELs */
            int16_t sel1       = ((XCHAL_IVPN_SIMD_WIDTH << 8));
            xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
            int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
            xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);

            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif
            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Coefficient Loads */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULSUQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, 0);
            IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, 0);
            IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, 0);
            IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, 0);
#endif
          }
        }
        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut4L, dvecOut4H, daccSum4, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut4 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 3 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      }
      if (x < outW)
      {
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        /* Output Data pointer */
        int8_t *pOut = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
        ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = ((uint8_t *) pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2);
        int8_t *pCoeff = pCoeffData + outCh;

        xb_vecN_2x32v hvecInAddrOff    = 0;
        xb_vecN_2x32v hvecCoeffAddrOff = 0;
        xb_vecN_2x32v hvecLaneIdx      = 0;
        int32_t inAddrOff              = 0, coeffAddrOff = 0;

        for (k = 0; k < kWidthU * kHeightU; k++) /* Kernel Height * Kernel Width */
        {
          vboolN_2 vbN_2 = IVP_EQN_2X32(hvecLaneIdx, kWidthU);
          /* hvecLaneIdx will be reset to zero after every kWidth */
          hvecLaneIdx = IVP_MOVN_2X32T(0, hvecLaneIdx, vbN_2);
          /* InPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecInAddrOff, hvecInAddrOff, inDataPitch2 * dilationY - kWidthU * inDataPitch1 * dilationX, vbN_2);
          /* CoeffPitch added after every kWidth */
          IVP_ADDN_2X32T(hvecCoeffAddrOff, hvecCoeffAddrOff, coeffPitch3 - kWidthU * coeffPitch2, vbN_2);
          /* Extracting Input and Coefficient address offsets */
          inAddrOff        = IVP_EXTRN_2X32(hvecInAddrOff, 0);
          coeffAddrOff     = IVP_EXTRN_2X32(hvecCoeffAddrOff, 0);
          hvecLaneIdx      = IVP_ADDN_2X32(hvecLaneIdx, 1);
          hvecCoeffAddrOff = IVP_ADDN_2X32(hvecCoeffAddrOff, coeffPitch2);
          hvecInAddrOff    = IVP_ADDN_2X32(hvecInAddrOff, inDataPitch1 * dilationX);

          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + inAddrOff);
          pdvecData2 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1 * 2 * enable3rdWidth);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecInData1; IVP_LAV2NX8U_XP(dvecInData1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecInData2; IVP_LAV2NX8U_XP(dvecInData2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecInData3; IVP_LAV2NX8U_XP(dvecInData3, vaData3, pdvecData3, 4);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);

#else
            xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
            xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
            xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
            xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
            xb_vecNx16 vecData1, vecData2;
            xb_vecNx16 vecData3, vecData4;
            xb_vecNx16 vecData5, vecData6;
            xb_vecNx16 vecData7, vecData8;
            xb_vecNx16 vecTemp1, vecTemp2;
            xb_vec2Nx8U dvecInData4 = 0;

            /* Custom select pattern for DSELs */
            int16_t sel1       = ((XCHAL_IVPN_SIMD_WIDTH << 8));
            xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
            int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
            xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);

            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif

            /* Aligned Vector Loads of coefficients */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);
            xb_vec2Nx8 dvecCoeff4; IVP_LV2NX8_XP(dvecCoeff4, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, dvecCoeff4);
            IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, dvecCoeff4);
#endif
          } /* End Input Channels */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;
            vaData1 = IVP_LA2NX8U_PP(pdvecData1);

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecInData1; IVP_LAV2NX8U_XP(dvecInData1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8U dvecInData2; IVP_LAV2NX8U_XP(dvecInData2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8U dvecInData3; IVP_LAV2NX8U_XP(dvecInData3, vaData3, pdvecData3, remInCh);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInData3)), 0);
#else
            xb_vec2Nx8U dvecData1, dvecData2, dvecData3, dvecData4;
            xb_vec2Nx8U dvecData5, dvecData6, dvecData7, dvecData8;
            xb_vec2Nx8U dvecData9, dvecData10, dvecData11, dvecData12;
            xb_vec2Nx8U dvecData13, dvecData14, dvecData15, dvecData16;
            xb_vecNx16 vecData1, vecData2;
            xb_vecNx16 vecData3, vecData4;
            xb_vecNx16 vecData5, vecData6;
            xb_vecNx16 vecData7, vecData8;
            xb_vecNx16 vecTemp1, vecTemp2;
            xb_vec2Nx8U dvecInData4 = 0;

            /* Custom select pattern for DSELs */
            int16_t sel1       = ((XCHAL_IVPN_SIMD_WIDTH << 8));
            xb_vec2Nx8 vecSel1 = IVP_MOV2NX8_FROMNX16(sel1);
            int16_t sel2       = (((XCHAL_IVPN_SIMD_WIDTH + 1) << 8) | 1);
            xb_vec2Nx8 vecSel2 = IVP_MOV2NX8_FROMNX16(sel2);

            /* Broadcast a0, a1, a2, a3.... | b0, b1, b2, b3.... using DSELs into a0, a1, a0, a1.... | b0, b1, b0, b1.... */
            IVP_DSELNX16(vecData2, vecData1, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel1);
            IVP_DSELNX16(vecData4, vecData3, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel1);
            IVP_DSELNX16(vecData6, vecData5, IVP_MOVNX16_FROM2NX8U(dvecInData2), IVP_MOVNX16_FROM2NX8U(dvecInData1), vecSel2);
            IVP_DSELNX16(vecData8, vecData7, IVP_MOVNX16_FROM2NX8U(dvecInData4), IVP_MOVNX16_FROM2NX8U(dvecInData3), vecSel2);

            /* Splitting 8 DSELI operations into 4 DSELIs and 8 SELIs for balancing loop schedule */
            /* Separate a0, a1, a0, a1 using SELIs into a0, a0, a0... */
            dvecData1 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData2 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData1), IVP_MOV2NX8U_FROMNX16(vecData1), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData3 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData4 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData2), IVP_MOV2NX8U_FROMNX16(vecData2), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData5 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData6 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData3), IVP_MOV2NX8U_FROMNX16(vecData3), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);
            dvecData7 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);
            dvecData8 = IVP_SEL2NX8I(IVP_MOV2NX8U_FROMNX16(vecData4), IVP_MOV2NX8U_FROMNX16(vecData4), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_1);

            /* De-interleave a b a b a b... and move to a a a a... and b b b b... */
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData5, vecData5, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData9 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData10 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData6, vecData6, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData11 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData12 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData7, vecData7, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData13 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData14 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
            IVP_DSELNX16I(vecTemp2, vecTemp1, vecData8, vecData8, IVP_DSELI_8B_DEINTERLEAVE_1);
            dvecData15 = IVP_MOV2NX8U_FROMNX16(vecTemp1); dvecData16 = IVP_MOV2NX8U_FROMNX16(vecTemp2);
#endif

            /* For conditional coefficient loads */
            int32_t enable2 = XT_SALT(1, remInCh); /* Will be 1 if remInCh > 1 */
            int32_t enable3 = XT_SALT(2, remInCh); /* Will be 1 if remInCh > 2 */

            /* Coefficient Loads */
            xb_vec2Nx8 dvecCoeff1; IVP_LV2NX8_XP(dvecCoeff1, pdvecCoeff, coeffPitch1 * enable2);
            xb_vec2Nx8 dvecCoeff2; IVP_LV2NX8_XP(dvecCoeff2, pdvecCoeff, coeffPitch1 * enable3);
            xb_vec2Nx8 dvecCoeff3; IVP_LV2NX8_XP(dvecCoeff3, pdvecCoeff, coeffPitch1);

#ifdef IVP_MULSUQA2N8XR8
            IVP_MULSUQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULSUQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULSUQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
#else
            /* Multiply unsigned x signed and accumulate to 24-bits */
            IVP_MULUSPA2NX8(daccSum1, dvecData1, dvecCoeff1, dvecData2, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum2, dvecData3, dvecCoeff1, dvecData4, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum3, dvecData5, dvecCoeff1, dvecData6, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum4, dvecData7, dvecCoeff1, dvecData8, dvecCoeff2);
            IVP_MULUSPA2NX8(daccSum1, dvecData9, dvecCoeff3, dvecData10, 0);
            IVP_MULUSPA2NX8(daccSum2, dvecData11, dvecCoeff3, dvecData12, 0);
            IVP_MULUSPA2NX8(daccSum3, dvecData13, dvecCoeff3, dvecData14, 0);
            IVP_MULUSPA2NX8(daccSum4, dvecData15, dvecCoeff3, dvecData16, 0);
#endif
          }
        }
        /* Pack, Output Scale, Output Shift and clamping */
        xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L;
        xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H;
#ifdef DILATED_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1L, dvecOut1H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut2L, dvecOut2H, daccSum2, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut3L, dvecOut3H, daccSum3, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Store the output dvecOut1 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
        IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut2 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable2ndWidth);
        IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

        /* Store the output dvecOut3 along the output depth */
        pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + 2 * outDataPitch1) * bytesPerPixel);
        IVP_SAV2NX8_XP(dvecOut3L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable3rdWidth);
        IVP_SAV2NX8_XP(dvecOut3H, vaOutData, pdvecOut, typeFlag * 2 * \
                       (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      }
    }
  }
  return(XAI_ERROR_STATUS());
}

/******************************* end of VQ MOD variants ***************************************/
/**********************************************************************************************/
#endif /*#if ((XCHAL_VISION_TYPE >= 6))*/
