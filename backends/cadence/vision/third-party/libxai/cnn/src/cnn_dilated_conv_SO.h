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
/* convolved3D_S_MxN_S8S8IXCa2_SO_DWH_INPUTNOEDGE                      */
/* convolved3D_S_MxN_U8S8IXCa2_SO_DWH_INPUTNOEDGE                      */
/***********************************************************************/
/* Description : P6 Optimized implementation of 3D convolution in SO   */
/*               for cases where                                       */
/*               . there are no edges along depth for input tile       */
/*                 and coeff tile                                      */
/*               . dilation = 1                                        */
/*               . dim2pitch of coeff tile is a multiple of 64         */
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
#ifdef DILATED_SO_VQ_CONV
static _XAI_INLINE_ void MAKE_NAME(convolvedVQ3D_S_MxN, S8IXCa2_SO_DWH_INPUTNOEDGE) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params * param
  )
#else
static _XAI_INLINE_ void MAKE_NAME(convolved3D_S_MxN, S8IXCa2_SO_DWH_INPUTNOEDGE) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params * param
  )
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM2(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM3(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
#ifdef DILATED_SO_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  const uint8_t outShiftU    = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu   = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t leftEdgeFlag = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag  = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t dilation     = XAI_CNN_CONV_GET_DILATION(param);
  const uint8_t strideX      = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY      = XAI_CNN_CONV_GET_STRIDEY(param);

  /* Data Pointers of input, output, coefficient and bias data */
  MORPH_IDT_SCALAR *pInData = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData          = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData        = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData        = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  /* Pitches of Coefficient Data (DWHN) in dim2 and dim3 */
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t dilatedKWidthU  = dilation * (kWidthU - 1) + 1;
  int32_t dilatedKHeightU = dilation * (kHeightU - 1) + 1;
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

  int32_t outCh, k, x, y, ky;

  MORPH_IDT_2Nx8* restrict pdvecData;
  MORPH_IDT_2Nx8* restrict pdvecData1;
  MORPH_IDT_2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecCoeff1;
  xb_vec2Nx8* restrict pdvecCoeff2;
  xb_vec2Nx8* restrict pdvecCoeff3;
  xb_vec2Nx8* restrict pdvecCoeff4;
  xb_vec2Nx8* restrict pdvecOut;
  xb_vecN_2x32v* restrict phvecBias;

  valign vaOutData = IVP_ZALIGN();
  if (numOutCh * outW * outH == 1 && kHeightU * kWidthU == 1 && (numInCh & (4 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
  {
#ifdef DILATED_SO_VQ_CONV
    const uint16_t outScale = ((int16_t *) pOutScaleData)[0];
#endif

    /* Initialize Accumulator */
    xb_vec2Nx24 daccSum1 = 0;

    /* Input, Output and Coefficient Pointers */
    int8_t *pOut           = pOutData;
    MORPH_IDT_SCALAR * pIn = pInData;
    int8_t *pCoeff1        = pCoeffData;

    pdvecData   = (MORPH_IDT_2Nx8 *) (pIn);
    pdvecCoeff1 = (xb_vec2Nx8 *) (pCoeff1);

    /* Priming Load for Input Data */
    valign vaData = MORPH_OP_PRIME_2Nx8(pdvecData);

    /* Multiplying and Accumulating 4 * XCHAL_IVPN_SIMD_WIDTH bytes at a time using PMULs */
    for (k = 0; k < numInCh; k += 4 * XCHAL_IVPN_SIMD_WIDTH)
    {
      /* Input Data Load */
      MORPH_IDT_2Nx8 dvecData1; MORPH_OP_LOAD_2Nx8_IP(dvecData1, vaData, pdvecData);
      MORPH_IDT_2Nx8 dvecData2; MORPH_OP_LOAD_2Nx8_IP(dvecData2, vaData, pdvecData);

      /* Coefficient Data Load */
      xb_vec2Nx8 dvecCoeff11; IVP_LV2NX8_IP(dvecCoeff11, pdvecCoeff1, 2 * XCHAL_IVPN_SIMD_WIDTH);
      xb_vec2Nx8 dvecCoeff12; IVP_LV2NX8_IP(dvecCoeff12, pdvecCoeff1, 2 * XCHAL_IVPN_SIMD_WIDTH);

      /* Pair Multiply and Accumulates */
      MORPH_OP_MULPA(daccSum1, dvecData2, dvecCoeff12, dvecData1, dvecCoeff11);
    }

    /* Reduction Addition and Bias Addition */
    xb_vecN_2x32v hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum1), \
                                               IVP_CVT32S2NX24HL(daccSum1));
    xb_vecN_2x32v hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum1), \
                                               IVP_CVT32S2NX24LL(daccSum1));
    int32_t sum1 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));


    sum1 += pBiasData[0];
    xb_vecN_2x32v hvecOut = (xb_vecN_2x32v) sum1;

    /* Truncate to 24-bit values */
    daccSum1 = IVP_CVT24UNX32L(hvecOut, hvecOut);

    xb_vecNx16 outData = IVP_PACKVR2NX24_0(daccSum1, packShiftAccU);
    xb_vecNx48 m_wvec  = IVP_MULUSNX16((xb_vecNx16U) outScale, outData);
    outData = IVP_PACKVRNX48(m_wvec, outShiftU);
    outData = IVP_MAXNX16(IVP_MINNX16(outData, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);

    /* Save the output values */
    pdvecOut = (xb_vec2Nx8 *) (pOut);
    IVP_SAV2NX8_XP(IVP_MOV2NX8_FROMNX16(outData), vaOutData, pdvecOut, bytesPerPixel);
    IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
  }
  else
  {
    /* Output Channels Loop is unrolled by 4 */
    for (outCh = 0; outCh < numOutCh - 3; outCh += 4) /* Output Channels Loop */
    {
#ifdef DILATED_SO_VQ_CONV
      xb_vecNx16U outScaleData, outScaleDataEven, outScaleDataOdd;
      valign vascale;
      //Load output scale values
      vascale = IVP_LANX16U_PP(pOutScaleData);
      IVP_LAVNX16_XP(outScaleData, vascale, pOutScaleData, 8);
      outScaleDataEven = IVP_SELNX16UI(outScaleData,
                                       outScaleData,
                                       IVP_SELI_16B_EXTRACT_1_OF_2_OFF_0);
      outScaleDataOdd = IVP_SELNX16UI(outScaleData,
                                      outScaleData,
                                      IVP_SELI_16B_EXTRACT_1_OF_2_OFF_1);
#endif
      for (y = 0; y < outH; y++) /* Output Height Loop */
      {
        for (x = 0; x < outW; x++) /* Output Width Loop */
        {
          /* Initialize Accumulator */
          xb_vec2Nx24 daccSum1 = 0;
          xb_vec2Nx24 daccSum2 = 0;
          xb_vec2Nx24 daccSum3 = 0;
          xb_vec2Nx24 daccSum4 = 0;

          int8_t *pOut = pOutData + (outCh + x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

          /* Input and Coefficient Pointers */
          MORPH_IDT_SCALAR * pIn = (pInData + x * strideX * inDataPitch1 + (y * strideY) * inDataPitch2);
          int8_t *pCoeff1        = (pCoeffData + outCh * coeffPitch3);
          int8_t *pCoeff2        = (pCoeffData + (outCh + 1) * coeffPitch3);
          int8_t *pCoeff3        = (pCoeffData + (outCh + 2) * coeffPitch3);
          int8_t *pCoeff4        = (pCoeffData + (outCh + 3) * coeffPitch3);

          for (ky = 0; ky < kHeightU; ky++) /* Kernel Height Loop */
          {
            pdvecData1  = (MORPH_IDT_2Nx8 *) (pIn);
            pdvecData2  = (MORPH_IDT_2Nx8 *) (pIn + 2 * XCHAL_IVPN_SIMD_WIDTH);
            pdvecCoeff1 = (xb_vec2Nx8 *) (pCoeff1);
            pdvecCoeff2 = (xb_vec2Nx8 *) (pCoeff2);
            pdvecCoeff3 = (xb_vec2Nx8 *) (pCoeff3);
            pdvecCoeff4 = (xb_vec2Nx8 *) (pCoeff4);


            /* Multiplying and Accumulating 4 * XCHAL_IVPN_SIMD_WIDTH bytes at a time using PMULs */
            for (k = 0; k < kWidthU * numInCh - 4 * XCHAL_IVPN_SIMD_WIDTH; k += 4 * XCHAL_IVPN_SIMD_WIDTH)
            {
              /* Input Data Load */
              valign vaData1 = MORPH_OP_PRIME_2Nx8(pdvecData1);
              valign vaData2 = MORPH_OP_PRIME_2Nx8(pdvecData2);
              MORPH_IDT_2Nx8 dvecData1; MORPH_OP_LOAD_2Nx8(dvecData1, vaData1, pdvecData1, 4 * XCHAL_IVPN_SIMD_WIDTH);
              MORPH_IDT_2Nx8 dvecData2; MORPH_OP_LOAD_2Nx8(dvecData2, vaData2, pdvecData2, 4 * XCHAL_IVPN_SIMD_WIDTH);

              /* Coefficient Data Load */
              xb_vec2Nx8 dvecCoeff11; IVP_LV2NX8_IP(dvecCoeff11, pdvecCoeff1, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff12; IVP_LV2NX8_IP(dvecCoeff12, pdvecCoeff1, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff21; IVP_LV2NX8_IP(dvecCoeff21, pdvecCoeff2, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff22; IVP_LV2NX8_IP(dvecCoeff22, pdvecCoeff2, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff31; IVP_LV2NX8_IP(dvecCoeff31, pdvecCoeff3, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff32; IVP_LV2NX8_IP(dvecCoeff32, pdvecCoeff3, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff41; IVP_LV2NX8_IP(dvecCoeff41, pdvecCoeff4, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff42; IVP_LV2NX8_IP(dvecCoeff42, pdvecCoeff4, 2 * XCHAL_IVPN_SIMD_WIDTH);

              /* Pair Multiply and Accumulates */
              MORPH_OP_MULPA(daccSum1, dvecData2, dvecCoeff12, dvecData1, dvecCoeff11);
              MORPH_OP_MULPA(daccSum2, dvecData2, dvecCoeff22, dvecData1, dvecCoeff21);
              MORPH_OP_MULPA(daccSum3, dvecData2, dvecCoeff32, dvecData1, dvecCoeff31);
              MORPH_OP_MULPA(daccSum4, dvecData2, dvecCoeff42, dvecData1, dvecCoeff41);
            }
            /* Corner case handling if numInCh  is not a multiple of 4 * XCHAL_IVPN_SIMD_WIDTH */

            int32_t remK = kWidthU * numInCh - k;
            /* remLoad is set to 1 if kWidthU * numInCh - k is greater than 64*/
            int32_t remLoad = XT_SALT(2 * XCHAL_IVPN_SIMD_WIDTH, kWidthU * numInCh - k);

            /* Input Data Load */
            valign vaData1 = MORPH_OP_PRIME_2Nx8(pdvecData1);
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData1, pdvecData1, remK);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData1, pdvecData1, remK - 2 * XCHAL_IVPN_SIMD_WIDTH);

            /* Coefficient Data Load */
            xb_vec2Nx8 dvecCoeff11; IVP_LV2NX8_XP(dvecCoeff11, pdvecCoeff1, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff12; IVP_LV2NX8_XP(dvecCoeff12, pdvecCoeff1, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff21; IVP_LV2NX8_XP(dvecCoeff21, pdvecCoeff2, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff22; IVP_LV2NX8_XP(dvecCoeff22, pdvecCoeff2, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff31; IVP_LV2NX8_XP(dvecCoeff31, pdvecCoeff3, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff32; IVP_LV2NX8_XP(dvecCoeff32, pdvecCoeff3, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff41; IVP_LV2NX8_XP(dvecCoeff41, pdvecCoeff4, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff42; IVP_LV2NX8_XP(dvecCoeff42, pdvecCoeff4, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);

            /* Pair Multiply and Accumulates */
            MORPH_OP_MULPA(daccSum1, dvecData2, dvecCoeff12, dvecData1, dvecCoeff11);
            MORPH_OP_MULPA(daccSum2, dvecData2, dvecCoeff22, dvecData1, dvecCoeff21);
            MORPH_OP_MULPA(daccSum3, dvecData2, dvecCoeff32, dvecData1, dvecCoeff31);
            MORPH_OP_MULPA(daccSum4, dvecData2, dvecCoeff42, dvecData1, dvecCoeff41);

            /* Update Pointer*/
            pIn     += inDataPitch2;
            pCoeff1 += coeffPitch2;
            pCoeff2 += coeffPitch2;
            pCoeff3 += coeffPitch2;
            pCoeff4 += coeffPitch2;
          } /* End Kernel Height Loop */

          /* Reduction Addition and Bias Addition */
          xb_vecN_2x32v hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum1), \
                                                     IVP_CVT32S2NX24HL(daccSum1));
          xb_vecN_2x32v hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum1), \
                                                     IVP_CVT32S2NX24LL(daccSum1));
          int32_t sum1 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));

          /* Reduction Addition and Bias Addition */
          hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum2), IVP_CVT32S2NX24HL(daccSum2));
          hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum2), IVP_CVT32S2NX24LL(daccSum2));
          int32_t sum2 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));

          /* Reduction Addition and Bias Addition */
          hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum3), IVP_CVT32S2NX24HL(daccSum3));
          hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum3), IVP_CVT32S2NX24LL(daccSum3));
          int32_t sum3 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));

          /* Reduction Addition and Bias Addition */
          hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum4), IVP_CVT32S2NX24HL(daccSum4));
          hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum4), IVP_CVT32S2NX24LL(daccSum4));
          int32_t sum4 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));

          /* Moving all the scalar sums to a 32-bit vector */
          xb_vecN_2x32v hvecOut = (xb_vecN_2x32v) sum4;
          hvecOut = IVP_MOVN_2X32T((xb_vecN_2x32v) sum3, hvecOut, IVP_LTRN_2I(3));
          hvecOut = IVP_MOVN_2X32T((xb_vecN_2x32v) sum2, hvecOut, IVP_LTRN_2I(2));
          hvecOut = IVP_MOVN_2X32T((xb_vecN_2x32v) sum1, hvecOut, IVP_LTRN_2I(1));

          /* Load bias values corresponding to two outChannels */
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          valign vaBias = IVP_LAN_2X32_PP(phvecBias);
          xb_vecN_2x32v hvecBias;  IVP_LAVN_2X32_XP(hvecBias, vaBias, phvecBias, 16);
          hvecOut = IVP_ADDN_2X32(hvecOut, hvecBias);

          /* Truncate to 24-bit values */
          daccSum1 = IVP_CVT24UNX32L(hvecOut, hvecOut);

          /* Pack, Scale, Shift and Clamp the accumulator output */
          xb_vec2Nx8 dvecOutData0L, dvecOutData0H;
#ifdef DILATED_SO_VQ_CONV
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData0L, dvecOutData0H, daccSum1, packShiftAccU, \
                                           outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
          PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData0L, dvecOutData0H, daccSum1, packShiftAccU, \
                                        outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
          /* Save the output values */
          pdvecOut = (xb_vec2Nx8 *) (pOut);
          IVP_SAV2NX8_XP(dvecOutData0L, vaOutData, pdvecOut, 4 * bytesPerPixel);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        } /* End Output Width Loop */
      }   /* End Output Height Loop */
    }     /* End Output Channels Loop */

    /* Corner case handling if Number of Output Channels is not a multiple of 4 */
    if (outCh < numOutCh)
    {
#ifdef DILATED_SO_VQ_CONV
      xb_vecNx16U outScaleData, outScaleDataEven, outScaleDataOdd;
      valign vascale;
      //Load output scale values
      vascale = IVP_LANX16U_PP(pOutScaleData);
      IVP_LAVNX16_XP(outScaleData, vascale, pOutScaleData, 6);
      outScaleDataEven = IVP_SELNX16UI(outScaleData,
                                       outScaleData,
                                       IVP_SELI_16B_EXTRACT_1_OF_2_OFF_0);
      outScaleDataOdd = IVP_SELNX16UI(outScaleData,
                                      outScaleData,
                                      IVP_SELI_16B_EXTRACT_1_OF_2_OFF_1);
#endif

      int32_t remOutCh = numOutCh - outCh;
      for (y = 0; y < outH; y++)
      {
        for (x = 0; x < outW; x++)
        {
          /* Initialize Accumulator */
          xb_vec2Nx24 daccSum1 = 0;
          xb_vec2Nx24 daccSum2 = 0;
          xb_vec2Nx24 daccSum3 = 0;

          /* Input, Output and Coefficient Pointers */
          int8_t *pOut           = pOutData + (outCh + x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
          MORPH_IDT_SCALAR * pIn = (pInData + x * strideX * inDataPitch1 + \
                                    (y * strideY) * inDataPitch2);
          int8_t *pCoeff1 = (pCoeffData + outCh * coeffPitch3);
          int8_t *pCoeff2 = (pCoeffData + (outCh + XT_MIN(1, remOutCh - 1)) * coeffPitch3);
          int8_t *pCoeff3 = (pCoeffData + (outCh + XT_MIN(2, remOutCh - 1)) * coeffPitch3);

          for (ky = 0; ky < kHeightU; ky++) /* Kernel Height Loop */
          {
            pdvecData   = (MORPH_IDT_2Nx8 *) (pIn);
            pdvecCoeff1 = (xb_vec2Nx8 *) (pCoeff1);
            pdvecCoeff2 = (xb_vec2Nx8 *) (pCoeff2);
            pdvecCoeff3 = (xb_vec2Nx8 *) (pCoeff3);

            /* Priming Load for Input Data */
            valign vaData = MORPH_OP_PRIME_2Nx8(pdvecData);

            /* Multiplying and Accumulating 128 bytes at a time using PMULs */
            for (k = 0; k < kWidthU * numInCh - 4 * XCHAL_IVPN_SIMD_WIDTH; k += 4 * XCHAL_IVPN_SIMD_WIDTH)
            {
              /* Input Data Load */
              MORPH_IDT_2Nx8 dvecData1; MORPH_OP_LOAD_2Nx8_IP(dvecData1, vaData, pdvecData);
              MORPH_IDT_2Nx8 dvecData2; MORPH_OP_LOAD_2Nx8_IP(dvecData2, vaData, pdvecData);

              /* Coefficient Data Load */
              xb_vec2Nx8 dvecCoeff11; IVP_LV2NX8_IP(dvecCoeff11, pdvecCoeff1, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff12; IVP_LV2NX8_IP(dvecCoeff12, pdvecCoeff1, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff21; IVP_LV2NX8_IP(dvecCoeff21, pdvecCoeff2, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff22; IVP_LV2NX8_IP(dvecCoeff22, pdvecCoeff2, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff31; IVP_LV2NX8_IP(dvecCoeff31, pdvecCoeff3, 2 * XCHAL_IVPN_SIMD_WIDTH);
              xb_vec2Nx8 dvecCoeff32; IVP_LV2NX8_IP(dvecCoeff32, pdvecCoeff3, 2 * XCHAL_IVPN_SIMD_WIDTH);

              /* Pair Multiply and Accumulates */
              MORPH_OP_MULPA(daccSum1, dvecData2, dvecCoeff12, dvecData1, dvecCoeff11);
              MORPH_OP_MULPA(daccSum2, dvecData2, dvecCoeff22, dvecData1, dvecCoeff21);
              MORPH_OP_MULPA(daccSum3, dvecData2, dvecCoeff32, dvecData1, dvecCoeff31);
            }
            int32_t remK = kWidthU * numInCh - k;
            /* remLoad is set to 1 if kWidthU * numInCh - k is greater than 64*/
            int32_t remLoad = XT_SALT(2 * XCHAL_IVPN_SIMD_WIDTH, kWidthU * numInCh - k);

            /* Input Data Load */
            xb_vec2Nx8U dvecData1; IVP_LAV2NX8U_XP(dvecData1, vaData, pdvecData, remK);
            xb_vec2Nx8U dvecData2; IVP_LAV2NX8U_XP(dvecData2, vaData, pdvecData, remK - 2 * XCHAL_IVPN_SIMD_WIDTH);

            /* Coefficient Data Load */
            xb_vec2Nx8 dvecCoeff11; IVP_LV2NX8_XP(dvecCoeff11, pdvecCoeff1, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff12; IVP_LV2NX8_XP(dvecCoeff12, pdvecCoeff1, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff21; IVP_LV2NX8_XP(dvecCoeff21, pdvecCoeff2, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff22; IVP_LV2NX8_XP(dvecCoeff22, pdvecCoeff2, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff31; IVP_LV2NX8_XP(dvecCoeff31, pdvecCoeff3, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);
            xb_vec2Nx8 dvecCoeff32; IVP_LV2NX8_XP(dvecCoeff32, pdvecCoeff3, remLoad * 2 * XCHAL_IVPN_SIMD_WIDTH);

            /* Pair Multiply and Accumulates */
            MORPH_OP_MULPA(daccSum1, dvecData2, dvecCoeff12, dvecData1, dvecCoeff11);
            MORPH_OP_MULPA(daccSum2, dvecData2, dvecCoeff22, dvecData1, dvecCoeff21);
            MORPH_OP_MULPA(daccSum3, dvecData2, dvecCoeff32, dvecData1, dvecCoeff31);

            /* Update Pointer*/
            pIn     += inDataPitch2;
            pCoeff1 += coeffPitch2;
            pCoeff2 += coeffPitch2;
            pCoeff3 += coeffPitch2;
          } /* End Kernel Height Loop */
            /* Reduction Addition and Bias Addition */
          xb_vecN_2x32v hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum1), \
                                                     IVP_CVT32S2NX24HL(daccSum1));
          xb_vecN_2x32v hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum1), \
                                                     IVP_CVT32S2NX24LL(daccSum1));
          int32_t sum1 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));

          /* Reduction Addition */
          hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum2), IVP_CVT32S2NX24HL(daccSum2));
          hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum2), IVP_CVT32S2NX24LL(daccSum2));
          int32_t sum2 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));

          /* Reduction Addition */
          hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum3), IVP_CVT32S2NX24HL(daccSum3));
          hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum3), IVP_CVT32S2NX24LL(daccSum3));
          int32_t sum3 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));

          /* Moving all the scalar sums to a 32-bit vector */
          xb_vecN_2x32v hvecOut = (xb_vecN_2x32v) sum3;
          hvecOut = IVP_MOVN_2X32T((xb_vecN_2x32v) sum2, hvecOut, IVP_LTRN_2I(2));
          hvecOut = IVP_MOVN_2X32T((xb_vecN_2x32v) sum1, hvecOut, IVP_LTRN_2I(1));

          /* Load bias values corresponding to two outChannels */
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          valign vaBias = IVP_LAN_2X32_PP(phvecBias);
          xb_vecN_2x32v hvecBias;  IVP_LAVN_2X32_XP(hvecBias, vaBias, phvecBias, 4 * remOutCh);

          /* Add bias to the accumulated value*/
          hvecOut = IVP_ADDN_2X32(hvecOut, hvecBias);

          /* Truncate to 24-bit values */
          daccSum1 = IVP_CVT24UNX32L(hvecOut, hvecOut);

          /* Pack, Scale, Shift and Clamp the accumulator output */
          xb_vec2Nx8 dvecOutData0L, dvecOutData0H;
#ifdef DILATED_SO_VQ_CONV
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData0L, dvecOutData0H, daccSum1, packShiftAccU, \
                                           outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
          PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData0L, dvecOutData0H, daccSum1, packShiftAccU, \
                                        outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
          /* Save the output values */
          pdvecOut = (xb_vec2Nx8 *) (pOut);
          IVP_SAV2NX8_XP(dvecOutData0L, vaOutData, pdvecOut, remOutCh * bytesPerPixel);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        } /* End Output Width Loop */
      }   /* End Output Height Loop */
    }     /* End of if (outCh < numOutCh) */
  }       /*End else*/
}


/***************************************************************************/
/*  xaiConvolved(VQ)3D_S_MxN_S8_SO_DWH/xaiConvolve(VQ)3D_S_MxN_U8_SO_DWH     */
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
/***********************************************************************/

/***************** xaiConvolvedVQ3D_S_MxN_S8S8IX_SO_DWH *****************/
/***************** xaiConvolvedVQ3D_S_MxN_U8S8IX_SO_DWH *****************/
/****************** xaiConvolved3D_S_MxN_S8S8IX_SO_DWH ******************/
/****************** xaiConvolved3D_S_MxN_U8S8IX_SO_DWH ******************/

#ifdef DILATED_SO_VQ_CONV
XAI_ERR_TYPE MAKE_NAME(xaiConvolvedVQ3D_S_MxN, S8IX_SO_DWH) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params * param
  )
#else
XAI_ERR_TYPE MAKE_NAME(xaiConvolved3D_S_MxN, S8IX_SO_DWH) (
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D outTile,
  const xai_cnn_conv_params * param
  )
#endif
{
  /* Error Checks */
  XAI_ERROR_CHECKS()
  {
    MORPH_IDT_CHECK(inTile);
    XAI_CHECK_CONV_OUTPUT_TILE3D(outTile);
    XAI_CHECK_TILE4D_S8(coeffTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_ARRAY_S32(biasArray);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
    XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) > 0) && (XAI_CNN_CONV_GET_STRIDEY(param) > 0)) &&                                      \
                    ((XAI_CNN_CONV_GET_STRIDEX(param) <= 64) && (XAI_CNN_CONV_GET_STRIDEY(param) <= 64)), XAI_ERR_BADARG,                    \
                    "\nStrideX = %hhu, StrideY = %hhu\nStride along width and height should be greater than 0 and less than or equal to 64", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) == 1) ||                                                            \
                    ((XAI_CNN_CONV_GET_DILATIONX(param) >= 1) &&                                                           \
                     (XAI_CNN_CONV_GET_STRIDEX(param) == 1)), XAI_ERR_BADARG,                                              \
                    "\nDilationX = %hhu\nDilationX should be 1. It can be greater than 1 only when strideX is equal to 1", \
                    XAI_CNN_CONV_GET_DILATIONX(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONY(param) == 1) ||                                                            \
                    ((XAI_CNN_CONV_GET_DILATIONY(param) >= 1) &&                                                           \
                     (XAI_CNN_CONV_GET_STRIDEY(param) == 1)), XAI_ERR_BADARG,                                              \
                    "\nDilationY = %hhu\nDilationY should be 1. It can be greater than 1 only when strideY is equal to 1", \
                    XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_EDGES_SO(inTile, coeffTile, param);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_DWHN);
    XAI_CHECK_CONSISTENCY_SO_DWH(inTile, coeffTile, biasArray, outTile, param);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_SO_VQ_CONV
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM4(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM4(coeffTile));
#endif
  }
#ifndef DILATED_SO_VQ_CONV
  if (XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0)
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* If
   * 1) there are no edges along depth (dim1) for input and coeff and dilation = 1
   * 2) the coeff pointer is aligned to (XCHAL_IVPN_SIMD_WIDTH << 1) and dim2pitch is a multiple of (XCHAL_IVPN_SIMD_WIDTH << 1)
   * Call MAKE_NAME(convolved3D_S_MxN, S8IXCa2_SO_DWH_INPUTNOEDGE)
   */
  if ((XAI_TILE3D_GET_DIM1_PITCH(inTile) == XAI_TILE3D_GET_DIM1(inTile)) &&
      (XAI_TILE4D_GET_DIM1_PITCH(coeffTile) == XAI_TILE4D_GET_DIM1(coeffTile)) && \
      (XAI_CNN_CONV_GET_DILATIONX(param) == 1) && (XAI_CNN_CONV_GET_DILATIONY(param) == 1))
  {
    if ((XAI_TILE4D_IS_PTR_ALIGNED_2NX8(coeffTile) && \
         (XAI_TILE4D_GET_DIM2_PITCH(coeffTile) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0))
    {
#ifdef DILATED_SO_VQ_CONV
      MAKE_NAME(convolvedVQ3D_S_MxN, S8IXCa2_SO_DWH_INPUTNOEDGE) (inTile,
                                                                  coeffTile,
                                                                  biasArray,
                                                                  outputScaleArray,
                                                                  outTile,
                                                                  param);
#else
      MAKE_NAME(convolved3D_S_MxN, S8IXCa2_SO_DWH_INPUTNOEDGE) (inTile,
                                                                coeffTile,
                                                                biasArray,
                                                                outTile,
                                                                param);
#endif
      return(XAI_ERROR_STATUS());
    }
  }

  /* Getting parameters from the tile structures */
  const int32_t outW     = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outH     = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t numInCh  = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t kWidthU  = XAI_TILE4D_GET_DIM2(coeffTile);
  const int32_t kHeightU = XAI_TILE4D_GET_DIM3(coeffTile);

  /* CNN convolution parameters */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
#ifdef DILATED_SO_VQ_CONV
  xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif
  const uint8_t outShiftU    = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu   = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t leftEdgeFlag = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag  = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t dilationX    = XAI_CNN_CONV_GET_DILATIONX(param);
  const uint8_t dilationY    = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t strideX      = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY      = XAI_CNN_CONV_GET_STRIDEY(param);

  /* Data Pointers of input, output, coefficient and bias data */
  MORPH_IDT_SCALAR *pInData = (MORPH_IDT_SCALAR *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData          = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData        = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData        = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  /* Pitches of Coefficient Data (DWHN) in dim2 and dim3 */
  const int32_t coeffPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);

  /* Pitches of Input Data (DWH) in dim1 and dim2 */
  const int32_t inDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(inTile);

  /* Pitch of Output Data (DWH) in dim1 and dim2 */
  const int32_t outDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(outTile);

  int32_t dilatedKWidthU  = dilationX * (kWidthU - 1) + 1;
  int32_t dilatedKHeightU = dilationY * (kHeightU - 1) + 1;
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

  int32_t outCh, inCh, x, y, ky, kx;

  MORPH_IDT_2Nx8* restrict pdvecData;
  xb_vec2Nx8* restrict pdvecCoeff1;
  xb_vec2Nx8* restrict pdvecCoeff2;
  xb_vec2Nx8* restrict pdvecOut;

  valign vaOutData = IVP_ZALIGN();

  /* Output Channels Loop is unrolled by 2 */
  for (outCh = 0; outCh < numOutCh - 1; outCh += 2) /* Output Channels Loop */
  {
#ifdef DILATED_SO_VQ_CONV
    xb_vecNx16U outScaleData, outScaleDataEven, outScaleDataOdd;
    valign vascale;
    //Load output scale values
    vascale = IVP_LANX16U_PP(pOutScaleData);
    IVP_LAVNX16_XP(outScaleData, vascale, pOutScaleData, 4);
    outScaleDataEven = IVP_SELNX16UI(outScaleData,
                                     outScaleData,
                                     IVP_SELI_16B_EXTRACT_1_OF_2_OFF_0);
    outScaleDataOdd = IVP_SELNX16UI(outScaleData,
                                    outScaleData,
                                    IVP_SELI_16B_EXTRACT_1_OF_2_OFF_1);
#endif

    for (y = 0; y < outH; y++) /* Output Height Loop */
    {
      for (x = 0; x < outW; x++) /* Output Width Loop */
      {
        /* Initialize Accumulator */
        xb_vec2Nx24 daccSum1 = 0;
        xb_vec2Nx24 daccSum2 = 0;

        int8_t *pOut = pOutData + (outCh + x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height Loop */
        {
          /* Input and Coefficient Pointers */
          MORPH_IDT_SCALAR * pIn = (pInData + x * strideX * inDataPitch1 + \
                                    (y * strideY + ky * dilationY) * inDataPitch2);
          int8_t *pCoeff1 = (pCoeffData + outCh * coeffPitch3 + ky * coeffPitch2);
          int8_t *pCoeff2 = (pCoeffData + (outCh + 1) * coeffPitch3 + ky * coeffPitch2);

          for (kx = 0; kx < kWidthU; kx++) /* Kernel Width Loop */
          {
            pdvecData   = (MORPH_IDT_2Nx8 *) (pIn);
            pdvecCoeff1 = (xb_vec2Nx8 *) (pCoeff1);
            pdvecCoeff2 = (xb_vec2Nx8 *) (pCoeff2);

            /* Priming Loads for Input and Coefficient Data */
            valign vaData   = MORPH_OP_PRIME_2Nx8(pdvecData);
            valign vaCoeff1 = IVP_LA2NX8_PP(pdvecCoeff1);
            valign vaCoeff2 = IVP_LA2NX8_PP(pdvecCoeff2);

            /* Multiplying and Accumulating 4 * XCHAL_IVPN_SIMD_WIDTH bytes at a time using PMULs */
            for (inCh = 0; inCh < numInCh - 4 * XCHAL_IVPN_SIMD_WIDTH; inCh += 4 * XCHAL_IVPN_SIMD_WIDTH)
            {
              /* Input Data Load */
              MORPH_IDT_2Nx8 dvecData1; MORPH_OP_LOAD_2Nx8_IP(dvecData1, vaData, pdvecData);
              MORPH_IDT_2Nx8 dvecData2; MORPH_OP_LOAD_2Nx8_IP(dvecData2, vaData, pdvecData);

              /* Coefficient Data Load */
              xb_vec2Nx8 dvecCoeff11; IVP_LA2NX8_IP(dvecCoeff11, vaCoeff1, pdvecCoeff1);
              xb_vec2Nx8 dvecCoeff12; IVP_LA2NX8_IP(dvecCoeff12, vaCoeff1, pdvecCoeff1);
              xb_vec2Nx8 dvecCoeff21; IVP_LA2NX8_IP(dvecCoeff21, vaCoeff2, pdvecCoeff2);
              xb_vec2Nx8 dvecCoeff22; IVP_LA2NX8_IP(dvecCoeff22, vaCoeff2, pdvecCoeff2);

              /* Pair Multiply and Accumulates */
              MORPH_OP_MULPA(daccSum1, dvecData2, dvecCoeff12, dvecData1, dvecCoeff11);
              MORPH_OP_MULPA(daccSum2, dvecData2, dvecCoeff22, dvecData1, dvecCoeff21);
            }
            /* Corner case handling if numInCh  is not a multiple of 4 * XCHAL_IVPN_SIMD_WIDTH */
            int32_t remLength = numInCh - inCh;

            /* Input Data Load */
            MORPH_IDT_2Nx8 dvecData1;
            MORPH_OP_LOAD_2Nx8_VARIABLE(dvecData1, vaData, pdvecData, remLength);
            MORPH_IDT_2Nx8 dvecData2;
            MORPH_OP_LOAD_2Nx8_VARIABLE(dvecData2, vaData, pdvecData, \
                                        remLength - 2 * XCHAL_IVPN_SIMD_WIDTH);

            /* Coefficient Data Load */
            xb_vec2Nx8 dvecCoeff11, dvecCoeff12, dvecCoeff21, dvecCoeff22;
            IVP_LAV2NX8_XP(dvecCoeff11, vaCoeff1, pdvecCoeff1, remLength);
            IVP_LAV2NX8_XP(dvecCoeff12, vaCoeff1, pdvecCoeff1, remLength - 2 * XCHAL_IVPN_SIMD_WIDTH);
            IVP_LAV2NX8_XP(dvecCoeff21, vaCoeff2, pdvecCoeff2, remLength);
            IVP_LAV2NX8_XP(dvecCoeff22, vaCoeff2, pdvecCoeff2, remLength - 2 * XCHAL_IVPN_SIMD_WIDTH);

            MORPH_OP_MULPA(daccSum1, dvecData2, dvecCoeff12, dvecData1, dvecCoeff11);
            MORPH_OP_MULPA(daccSum2, dvecData2, dvecCoeff22, dvecData1, dvecCoeff21);

            pIn     += dilationX * inDataPitch1;
            pCoeff1 += coeffPitch1;
            pCoeff2 += coeffPitch1;
          } /* End Kernel Width Loop */
        }   /* End Kernel Height Loop */

        /* Reduction Addition and Bias Addition */
        xb_vecN_2x32v hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum1), \
                                                   IVP_CVT32S2NX24HL(daccSum1));
        xb_vecN_2x32v hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum1), \
                                                   IVP_CVT32S2NX24LL(daccSum1));
        int32_t sum1 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));
        sum1 += pBiasData[outCh];

        /* Reduction Addition and Bias Addition */
        hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum2), IVP_CVT32S2NX24HL(daccSum2));
        hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum2), IVP_CVT32S2NX24LL(daccSum2));
        int32_t sum2 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));
        sum2 += pBiasData[outCh + 1];

        /* Moving all the scalar sums to a 32-bit vector */
        xb_vecN_2x32v hvecOut = 0;
        hvecOut = IVP_MOVN_2X32T((xb_vecN_2x32v) sum2, hvecOut, IVP_LTRN_2I(2));
        hvecOut = IVP_MOVN_2X32T((xb_vecN_2x32v) sum1, hvecOut, IVP_LTRN_2I(1));

        /* Truncate to 24-bit values */
        daccSum1 = IVP_CVT24UNX32L(hvecOut, hvecOut);

        /* Pack, Scale, Shift and Clamp the accumulator output */
        xb_vec2Nx8 dvecOutData0L, dvecOutData0H;
#ifdef DILATED_SO_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData0L, dvecOutData0H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData0L, dvecOutData0H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Save the output values */
        pdvecOut = (xb_vec2Nx8 *) (pOut);
        IVP_SAV2NX8_XP(dvecOutData0L, vaOutData, pdvecOut, 2 * bytesPerPixel);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End Output Width Loop */
    }   /* End Output Height Loop */
  }     /* End Output Channels Loop */

  /* Corner case handling if Number of Output Channels is odd */
  if (outCh < numOutCh)
  {
#ifdef DILATED_SO_VQ_CONV
    xb_vecNx16U outScaleData, outScaleDataEven, outScaleDataOdd;
    valign vascale;
    //Load output scale values
    vascale = IVP_LANX16U_PP(pOutScaleData);
    IVP_LAVNX16_XP(outScaleData, vascale, pOutScaleData, 2);
    outScaleDataEven = IVP_SELNX16UI(outScaleData,
                                     outScaleData,
                                     IVP_SELI_16B_EXTRACT_1_OF_2_OFF_0);
    outScaleDataOdd = IVP_SELNX16UI(outScaleData,
                                    outScaleData,
                                    IVP_SELI_16B_EXTRACT_1_OF_2_OFF_1);
#endif
    for (y = 0; y < outH; y++)
    {
      for (x = 0; x < outW; x++)
      {
        /* Initialize Accumulator */
        xb_vec2Nx24 daccSum1 = 0;

        int8_t *pOut = pOutData + (outCh + x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;

        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height Loop */
        {
          /* Input and Coefficient Pointers */
          MORPH_IDT_SCALAR * pIn = (pInData + x * strideX * inDataPitch1 + \
                                    (y * strideY + ky * dilationY) * inDataPitch2);
          int8_t *pCoeff1 = (pCoeffData + outCh * coeffPitch3 + ky * coeffPitch2);

          for (kx = 0; kx < kWidthU; kx++) /* Kernel Width Loop */
          {
            pdvecData   = (MORPH_IDT_2Nx8 *) (pIn);
            pdvecCoeff1 = (xb_vec2Nx8 *) (pCoeff1);

            /* Priming Loads for Input and Coefficient Data */
            valign vaData   = MORPH_OP_PRIME_2Nx8(pdvecData);
            valign vaCoeff1 = IVP_LA2NX8_PP(pdvecCoeff1);

            /* Multiplying and Accumulating 4 * XCHAL_IVPN_SIMD_WIDTH bytes at a time using PMULs */
            for (inCh = 0; inCh < numInCh - 4 * XCHAL_IVPN_SIMD_WIDTH; inCh += 4 * XCHAL_IVPN_SIMD_WIDTH)
            {
              /* Input Data Load */
              MORPH_IDT_2Nx8 dvecData1; MORPH_OP_LOAD_2Nx8_IP(dvecData1, vaData, pdvecData);
              MORPH_IDT_2Nx8 dvecData2; MORPH_OP_LOAD_2Nx8_IP(dvecData2, vaData, pdvecData);

              /* Coefficient Data Load */
              xb_vec2Nx8 dvecCoeff11; IVP_LA2NX8_IP(dvecCoeff11, vaCoeff1, pdvecCoeff1);
              xb_vec2Nx8 dvecCoeff12; IVP_LA2NX8_IP(dvecCoeff12, vaCoeff1, pdvecCoeff1);

              /* Pair Multiply and Accumulates */
              MORPH_OP_MULPA(daccSum1, dvecData2, dvecCoeff12, dvecData1, dvecCoeff11);
            }
            /* Corner case handling if numInCh is not a multiple of 4 * XCHAL_IVPN_SIMD_WIDTH */
            int32_t remLength = numInCh - inCh;

            /* Input Data Load */
            MORPH_IDT_2Nx8 dvecData1;
            MORPH_OP_LOAD_2Nx8_VARIABLE(dvecData1, vaData, pdvecData, remLength);
            MORPH_IDT_2Nx8 dvecData2;
            MORPH_OP_LOAD_2Nx8_VARIABLE(dvecData2, vaData, pdvecData, \
                                        remLength - 2 * XCHAL_IVPN_SIMD_WIDTH);

            /* Coefficient Data Load */
            xb_vec2Nx8 dvecCoeff11, dvecCoeff12;
            IVP_LAV2NX8_XP(dvecCoeff11, vaCoeff1, pdvecCoeff1, remLength);
            IVP_LAV2NX8_XP(dvecCoeff12, vaCoeff1, pdvecCoeff1, remLength - 2 * XCHAL_IVPN_SIMD_WIDTH);

            MORPH_OP_MULPA(daccSum1, dvecData2, dvecCoeff12, dvecData1, dvecCoeff11);

            pIn     += dilationX * inDataPitch1;
            pCoeff1 += coeffPitch1;
          } /* End Kernel Width Loop */
        }   /* End Kernel Height Loop */
            /* Reduction Addition and Bias Addition */
        xb_vecN_2x32v hvecSumUpper = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum1), \
                                                   IVP_CVT32S2NX24HL(daccSum1));
        xb_vecN_2x32v hvecSumLower = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum1), \
                                                   IVP_CVT32S2NX24LL(daccSum1));
        int32_t sum1 = IVP_RADDN_2X32(IVP_ADDN_2X32(hvecSumUpper, hvecSumLower));
        sum1 += pBiasData[outCh];

        /* Moving all the scalar sums to a 32-bit vector */
        xb_vecN_2x32v hvecOut = (xb_vecN_2x32v) sum1;

        /* Truncate to 24-bit values */
        daccSum1 = IVP_CVT24UNX32L(hvecOut, hvecOut);

        /* Pack, Scale, Shift and Clamp the accumulator output */
        xb_vec2Nx8 dvecOutData0L, dvecOutData0H;
#ifdef DILATED_SO_VQ_CONV
        PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOutData0L, dvecOutData0H, daccSum1, packShiftAccU, \
                                         outScaleDataEven, outScaleDataOdd, outShiftU, minLim, maxLim, typeFlag);
#else
        PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOutData0L, dvecOutData0H, daccSum1, packShiftAccU, \
                                      outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
        /* Save the output values */
        pdvecOut = (xb_vec2Nx8 *) (pOut);
        IVP_SAV2NX8_XP(dvecOutData0L, vaOutData, pdvecOut, bytesPerPixel);
        IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
      } /* End Output Width Loop */
    }   /* End Output Height Loop */
  }     /* End of if (outCh < numOutCh) */

  return(XAI_ERROR_STATUS());
}


/****************************** end of SO variants *****************************************/
/*******************************************************************************************/
#endif //if ((XCHAL_VISION_TYPE >= 6))
