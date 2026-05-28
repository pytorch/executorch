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


/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

  int32_t numIter = kWidthU * numInCh;

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
  xb_vecN_2x32v* restrict phvecAcc;

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
#ifdef DILATED_VQ_CONV_PARTIAL
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
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;


        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

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

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * numX * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
          IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut3 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * numY * bytesPerPixel);
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1) * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch2) * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * numX + accDataPitch2 * numY));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

  int32_t numIter = kWidthU * numInCh;

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
  xb_vecN_2x32v* restrict phvecAcc;

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
#ifdef DILATED_VQ_CONV_PARTIAL
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
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;


        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

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
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1;
            xb_vec2Nx8 dvecData2;
            xb_vec2Nx8 dvecData3;
            xb_vec2Nx8 dvecData4;

            dvecData1 = IVP_SUB2NX8U(dvecInp1, 128);
            dvecData2 = IVP_SUB2NX8U(dvecInp2, 128);
            dvecData3 = IVP_SUB2NX8U(dvecInp3, 128);
            dvecData4 = IVP_SUB2NX8U(dvecInp4, 128);

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
            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * numX * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
          IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut3 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * numY * bytesPerPixel);
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1) * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch2) * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * numX + accDataPitch2 * numY));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_S8S8IXCa2_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

  int32_t numIter = kWidthU * numInCh;

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
  xb_vecN_2x32v* restrict phvecAcc;

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
#ifdef DILATED_VQ_CONV_PARTIAL
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
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

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
          if (k < numIter)
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
          }   /* End if( k < numIter)*/
        } /* End Kernel Height * Width */

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * numX * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
          IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut3 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * numY * bytesPerPixel);
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1) * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch2) * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * numX + accDataPitch2 * numY));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_U8S8IXCa2_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_U8S8IXCa2_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

  int32_t numIter = kWidthU * numInCh;

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
  xb_vecN_2x32v* restrict phvecAcc;

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
#ifdef DILATED_VQ_CONV_PARTIAL
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
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

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
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1;
            xb_vec2Nx8 dvecData2;
            xb_vec2Nx8 dvecData3;
            xb_vec2Nx8 dvecData4;

            dvecData1 = IVP_SUB2NX8U(dvecInp1, 128);
            dvecData2 = IVP_SUB2NX8U(dvecInp2, 128);
            dvecData3 = IVP_SUB2NX8U(dvecInp3, 128);
            dvecData4 = IVP_SUB2NX8U(dvecInp4, 128);

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
            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          }   /* End Input Channels */
          /* Corner case handling as numIter is not a multiple of 4 */
          if (k < numIter)
          {
            int32_t remInCh = numIter - k;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, remInCh);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            xb_vec2Nx8 dvecData3 = 0;
            xb_vec2Nx8 dvecData4 = 0;

            IVP_SUB2NX8UT(dvecData1, dvecInp1, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData2, dvecInp2, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData3, dvecInp3, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData4, dvecInp4, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));

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
#endif
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
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          }   /* End if( k < numIter)*/
        } /* End Kernel Height * Width */

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * numX * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
          IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut3 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * numY * bytesPerPixel);
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1) * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch2) * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * numX + accDataPitch2 * numY));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/
#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

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
  xb_vecN_2x32v* restrict phvecAcc;

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV_PARTIAL
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
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

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
          pdvecData4 = (xb_vec2Nx8 *) (pData + inAddrOff + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
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

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * numX * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
          IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut3 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * numY * bytesPerPixel);
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1) * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch2) * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * numX + accDataPitch2 * numY));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile4D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      const xai_pArray outputScaleArray,
                                                                      xai_pTile3D accTile,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_conv_params *param
                                                                      )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxN_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    xai_pTile3D accTile,
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
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

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
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;
  xb_vecN_2x32v* restrict phvecAcc;

  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV_PARTIAL
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
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
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
          pdvecData4 = (xb_vec2Nx8U *) (pData + inAddrOff + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8U_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1;
            xb_vec2Nx8 dvecData2;
            xb_vec2Nx8 dvecData3;
            xb_vec2Nx8 dvecData4;

            dvecData1 = IVP_SUB2NX8U(dvecInp1, 128);
            dvecData2 = IVP_SUB2NX8U(dvecInp2, 128);
            dvecData3 = IVP_SUB2NX8U(dvecInp3, 128);
            dvecData4 = IVP_SUB2NX8U(dvecInp4, 128);

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
            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          } /* End Input Channels */

          /* Corner Case Handling if number of input channels not multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, remInCh);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            xb_vec2Nx8 dvecData3 = 0;
            xb_vec2Nx8 dvecData4 = 0;

            IVP_SUB2NX8UT(dvecData1, dvecInp1, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData2, dvecInp2, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData3, dvecInp3, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData4, dvecInp4, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));

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
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          } /* End Corner case handling */
        }   /* End Kernel Height * Width */

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * numX * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
          IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut3 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * numY * bytesPerPixel);
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1) * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch2) * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh * numY);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * numX + accDataPitch2 * numY));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_MOD_DWH_QM32_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_S8S8IXCa2_MOD_DWH_QM32_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

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
  int32_t outCh, x, y, ky, k, j;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;
  xb_vecN_2x32v* restrict phvecAcc;

  xb_vecN_2x32v hvecSum1LL, hvecSum1LH, hvecSum1HL, hvecSum1HH;
  xb_vecN_2x32v hvecSum2LL, hvecSum2LH, hvecSum2HL, hvecSum2HH;
  xb_vecN_2x32v hvecSum3LL, hvecSum3LH, hvecSum3HL, hvecSum3HH;
  xb_vecN_2x32v hvecSum4LL, hvecSum4LH, hvecSum4HL, hvecSum4HH;

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
#ifdef DILATED_VQ_CONV_PARTIAL
    xb_vecNx16U outScaleDataL, outScaleDataH;
    /*Load output scale values*/
    valign vaScale = IVP_LANX16U_PP(pOutScaleData);
    IVP_LAVNX16_XP(outScaleDataL, vaScale, pOutScaleData, 2 * remainingOutCh);
    IVP_LAVNX16_XP(outScaleDataH, vaScale, pOutScaleData, 2 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
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
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        hvecSum1LL = hvecSum1LH = hvecSum1HL = hvecSum1HH = 0;
        hvecSum2LL = hvecSum2LH = hvecSum2HL = hvecSum2HH = 0;
        hvecSum3LL = hvecSum3LH = hvecSum3HL = hvecSum3HH = 0;
        hvecSum4LL = hvecSum4LH = hvecSum4HL = hvecSum4HH = 0;
        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;


        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

          /* (Input Channels * kWidth) loops combined */
          for (j = 0; j < kWidthU * numInCh; j += 508) /* Emulation: To avoid 24 bit overflow 2^23-1 / 128 / 128 = 511.99 */
          {
            xb_vec2Nx24 daccSum1 = 0, daccSum2 = 0, daccSum3 = 0, daccSum4 = 0;
            int32_t numIter      = XT_MIN(508, kWidthU * numInCh - j);
#ifdef __XCC__
#pragma loop_count min=1
#endif
            for (k = 0; k < numIter; k += 4)
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
            }   /* End for (k = 0; k < row; k += 4) */

            hvecSum1LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum1), hvecSum1LL);
            hvecSum1LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum1), hvecSum1LH);
            hvecSum1HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum1), hvecSum1HL);
            hvecSum1HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum1), hvecSum1HH);

            hvecSum2LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum2), hvecSum2LL);
            hvecSum2LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum2), hvecSum2LH);
            hvecSum2HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum2), hvecSum2HL);
            hvecSum2HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum2), hvecSum2HH);

            hvecSum3LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum3), hvecSum3LL);
            hvecSum3LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum3), hvecSum3LH);
            hvecSum3HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum3), hvecSum3HL);
            hvecSum3HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum3), hvecSum3HH);

            hvecSum4LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum4), hvecSum4LL);
            hvecSum4LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum4), hvecSum4LH);
            hvecSum4HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum4), hvecSum4HL);
            hvecSum4HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum4), hvecSum4HH);
          } /* End Kernel Height * Width */
        }   /* End for (k = 0; k < row; k += 4)*/

        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          xb_vecN_2x32v hvecBiasLL, hvecBiasLH, hvecBiasHL, hvecBiasHH;
          valign vaBias = IVP_LAN_2X32_PP(phvecBias);
          IVP_LAVN_2X32_XP(hvecBiasLL, vaBias, phvecBias, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecBiasLH, vaBias, phvecBias, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecBiasHL, vaBias, phvecBias, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecBiasHH, vaBias, phvecBias, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);

          hvecSum1LL = IVP_ADDN_2X32(hvecSum1LL, hvecBiasLL);
          hvecSum1LH = IVP_ADDN_2X32(hvecSum1LH, hvecBiasLH);
          hvecSum1HL = IVP_ADDN_2X32(hvecSum1HL, hvecBiasHL);
          hvecSum1HH = IVP_ADDN_2X32(hvecSum1HH, hvecBiasHH);

          hvecSum2LL = IVP_ADDN_2X32(hvecSum2LL, hvecBiasLL);
          hvecSum2LH = IVP_ADDN_2X32(hvecSum2LH, hvecBiasLH);
          hvecSum2HL = IVP_ADDN_2X32(hvecSum2HL, hvecBiasHL);
          hvecSum2HH = IVP_ADDN_2X32(hvecSum2HH, hvecBiasHH);

          hvecSum3LL = IVP_ADDN_2X32(hvecSum3LL, hvecBiasLL);
          hvecSum3LH = IVP_ADDN_2X32(hvecSum3LH, hvecBiasLH);
          hvecSum3HL = IVP_ADDN_2X32(hvecSum3HL, hvecBiasHL);
          hvecSum3HH = IVP_ADDN_2X32(hvecSum3HH, hvecBiasHH);

          hvecSum4LL = IVP_ADDN_2X32(hvecSum4LL, hvecBiasLL);
          hvecSum4LH = IVP_ADDN_2X32(hvecSum4LH, hvecBiasLH);
          hvecSum4HL = IVP_ADDN_2X32(hvecSum4HL, hvecBiasHL);
          hvecSum4HH = IVP_ADDN_2X32(hvecSum4HH, hvecBiasHH);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);

          hvecSum1LL = IVP_ADDN_2X32(hvecSum1LL, hvecAcc1LL);
          hvecSum1LH = IVP_ADDN_2X32(hvecSum1LH, hvecAcc1LH);
          hvecSum1HL = IVP_ADDN_2X32(hvecSum1HL, hvecAcc1HL);
          hvecSum1HH = IVP_ADDN_2X32(hvecSum1HH, hvecAcc1HH);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);

          hvecSum2LL = IVP_ADDN_2X32(hvecSum2LL, hvecAcc2LL);
          hvecSum2LH = IVP_ADDN_2X32(hvecSum2LH, hvecAcc2LH);
          hvecSum2HL = IVP_ADDN_2X32(hvecSum2HL, hvecAcc2HL);
          hvecSum2HH = IVP_ADDN_2X32(hvecSum2HH, hvecAcc2HH);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          hvecSum3LL = IVP_ADDN_2X32(hvecSum3LL, hvecAcc3LL);
          hvecSum3LH = IVP_ADDN_2X32(hvecSum3LH, hvecAcc3LH);
          hvecSum3HL = IVP_ADDN_2X32(hvecSum3HL, hvecAcc3HL);
          hvecSum3HH = IVP_ADDN_2X32(hvecSum3HH, hvecAcc3HH);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          hvecSum4LL = IVP_ADDN_2X32(hvecSum4LL, hvecAcc4LL);
          hvecSum4LH = IVP_ADDN_2X32(hvecSum4LH, hvecAcc4LH);
          hvecSum4HL = IVP_ADDN_2X32(hvecSum4HL, hvecAcc4HL);
          hvecSum4HH = IVP_ADDN_2X32(hvecSum4HH, hvecAcc4HH);
        }


        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;


#ifdef DILATED_VQ_CONV_PARTIAL
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut1L, dvecOut1H, hvecSum1LL, hvecSum1LH, hvecSum1HL, hvecSum1HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut2L, dvecOut2H, hvecSum2LL, hvecSum2LH, hvecSum2HL, hvecSum2HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut3L, dvecOut3H, hvecSum3LL, hvecSum3LH, hvecSum3HL, hvecSum3HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut4L, dvecOut4H, hvecSum4LL, hvecSum4LH, hvecSum4HL, hvecSum4HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
#else
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut1L, dvecOut1H, hvecSum1LL, hvecSum1LH, hvecSum1HL, hvecSum1HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut2L, dvecOut2H, hvecSum2LL, hvecSum2LH, hvecSum2HL, hvecSum2HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut3L, dvecOut3H, hvecSum3LL, hvecSum3LH, hvecSum3HL, hvecSum3HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut4L, dvecOut4H, hvecSum4LL, hvecSum4LH, hvecSum4HL, hvecSum4HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
          /* Store the output dvecOut1 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
          IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut2 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * numX * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
          IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut3 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * numY * bytesPerPixel);
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
        }
        else /* Store to accumulator tile*/
        {
          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecSum1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecSum1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1) * numX);
          IVP_SAVN_2X32_XP(hvecSum2LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX);
          IVP_SAVN_2X32_XP(hvecSum2LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum2HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum2HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch2) * numY);
          IVP_SAVN_2X32_XP(hvecSum3LL, vaOutData, phvecAcc, 4 * remainingOutCh * numY);
          IVP_SAVN_2X32_XP(hvecSum3LH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum3HL, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum3HH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * numX + accDataPitch2 * numY));
          IVP_SAVN_2X32_XP(hvecSum4LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY);
          IVP_SAVN_2X32_XP(hvecSum4LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum4HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum4HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_MOD_DWH_QM32_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_S8S8IXCa2_MOD_DWH_QM32_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

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
  int32_t outCh, x, y, ky, k, j;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;
  xb_vecN_2x32v* restrict phvecAcc;

  xb_vecN_2x32v hvecSum1LL, hvecSum1LH, hvecSum1HL, hvecSum1HH;
  xb_vecN_2x32v hvecSum2LL, hvecSum2LH, hvecSum2HL, hvecSum2HH;
  xb_vecN_2x32v hvecSum3LL, hvecSum3LH, hvecSum3HL, hvecSum3HH;
  xb_vecN_2x32v hvecSum4LL, hvecSum4LH, hvecSum4HL, hvecSum4HH;

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
#ifdef DILATED_VQ_CONV_PARTIAL
    xb_vecNx16U outScaleDataL, outScaleDataH;
    /*Load output scale values*/
    valign vaScale = IVP_LANX16U_PP(pOutScaleData);
    IVP_LAVNX16_XP(outScaleDataL, vaScale, pOutScaleData, 2 * remainingOutCh);
    IVP_LAVNX16_XP(outScaleDataH, vaScale, pOutScaleData, 2 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
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
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        hvecSum1LL = hvecSum1LH = hvecSum1HL = hvecSum1HH = 0;
        hvecSum2LL = hvecSum2LH = hvecSum2HL = hvecSum2HH = 0;
        hvecSum3LL = hvecSum3LH = hvecSum3HL = hvecSum3HH = 0;
        hvecSum4LL = hvecSum4LH = hvecSum4HL = hvecSum4HH = 0;
        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;


        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * numX);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideY * inDataPitch2 * numY);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + ky * coeffPitch3);

          /* Primes for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

          for (j = 0; j < kWidthU * numInCh; j += 508)
          {
            xb_vec2Nx24 daccSum1 = 0, daccSum2 = 0, daccSum3 = 0, daccSum4 = 0;
            int32_t numIter      = XT_MIN(508, kWidthU * numInCh - j);
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
            if (k < numIter)
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
            }   /* End if( k < numIter)*/

            hvecSum1LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum1), hvecSum1LL);
            hvecSum1LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum1), hvecSum1LH);
            hvecSum1HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum1), hvecSum1HL);
            hvecSum1HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum1), hvecSum1HH);

            hvecSum2LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum2), hvecSum2LL);
            hvecSum2LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum2), hvecSum2LH);
            hvecSum2HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum2), hvecSum2HL);
            hvecSum2HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum2), hvecSum2HH);

            hvecSum3LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum3), hvecSum3LL);
            hvecSum3LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum3), hvecSum3LH);
            hvecSum3HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum3), hvecSum3HL);
            hvecSum3HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum3), hvecSum3HH);

            hvecSum4LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum4), hvecSum4LL);
            hvecSum4LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum4), hvecSum4LH);
            hvecSum4HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum4), hvecSum4HL);
            hvecSum4HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum4), hvecSum4HH);
          }
        } /* End Kernel Height * Width */
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          xb_vecN_2x32v hvecBiasLL, hvecBiasLH, hvecBiasHL, hvecBiasHH;
          valign vaBias = IVP_LAN_2X32_PP(phvecBias);
          IVP_LAVN_2X32_XP(hvecBiasLL, vaBias, phvecBias, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecBiasLH, vaBias, phvecBias, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecBiasHL, vaBias, phvecBias, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecBiasHH, vaBias, phvecBias, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);

          hvecSum1LL = IVP_ADDN_2X32(hvecSum1LL, hvecBiasLL);
          hvecSum1LH = IVP_ADDN_2X32(hvecSum1LH, hvecBiasLH);
          hvecSum1HL = IVP_ADDN_2X32(hvecSum1HL, hvecBiasHL);
          hvecSum1HH = IVP_ADDN_2X32(hvecSum1HH, hvecBiasHH);

          hvecSum2LL = IVP_ADDN_2X32(hvecSum2LL, hvecBiasLL);
          hvecSum2LH = IVP_ADDN_2X32(hvecSum2LH, hvecBiasLH);
          hvecSum2HL = IVP_ADDN_2X32(hvecSum2HL, hvecBiasHL);
          hvecSum2HH = IVP_ADDN_2X32(hvecSum2HH, hvecBiasHH);

          hvecSum3LL = IVP_ADDN_2X32(hvecSum3LL, hvecBiasLL);
          hvecSum3LH = IVP_ADDN_2X32(hvecSum3LH, hvecBiasLH);
          hvecSum3HL = IVP_ADDN_2X32(hvecSum3HL, hvecBiasHL);
          hvecSum3HH = IVP_ADDN_2X32(hvecSum3HH, hvecBiasHH);

          hvecSum4LL = IVP_ADDN_2X32(hvecSum4LL, hvecBiasLL);
          hvecSum4LH = IVP_ADDN_2X32(hvecSum4LH, hvecBiasLH);
          hvecSum4HL = IVP_ADDN_2X32(hvecSum4HL, hvecBiasHL);
          hvecSum4HH = IVP_ADDN_2X32(hvecSum4HH, hvecBiasHH);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);

          hvecSum1LL = IVP_ADDN_2X32(hvecSum1LL, hvecAcc1LL);
          hvecSum1LH = IVP_ADDN_2X32(hvecSum1LH, hvecAcc1LH);
          hvecSum1HL = IVP_ADDN_2X32(hvecSum1HL, hvecAcc1HL);
          hvecSum1HH = IVP_ADDN_2X32(hvecSum1HH, hvecAcc1HH);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);

          hvecSum2LL = IVP_ADDN_2X32(hvecSum2LL, hvecAcc2LL);
          hvecSum2LH = IVP_ADDN_2X32(hvecSum2LH, hvecAcc2LH);
          hvecSum2HL = IVP_ADDN_2X32(hvecSum2HL, hvecAcc2HL);
          hvecSum2HH = IVP_ADDN_2X32(hvecSum2HH, hvecAcc2HH);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          hvecSum3LL = IVP_ADDN_2X32(hvecSum3LL, hvecAcc3LL);
          hvecSum3LH = IVP_ADDN_2X32(hvecSum3LH, hvecAcc3LH);
          hvecSum3HL = IVP_ADDN_2X32(hvecSum3HL, hvecAcc3HL);
          hvecSum3HH = IVP_ADDN_2X32(hvecSum3HH, hvecAcc3HH);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          hvecSum4LL = IVP_ADDN_2X32(hvecSum4LL, hvecAcc4LL);
          hvecSum4LH = IVP_ADDN_2X32(hvecSum4LH, hvecAcc4LH);
          hvecSum4HL = IVP_ADDN_2X32(hvecSum4HL, hvecAcc4HL);
          hvecSum4HH = IVP_ADDN_2X32(hvecSum4HH, hvecAcc4HH);
        }

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut1L, dvecOut1H, hvecSum1LL, hvecSum1LH, hvecSum1HL, hvecSum1HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut2L, dvecOut2H, hvecSum2LL, hvecSum2LH, hvecSum2HL, hvecSum2HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut3L, dvecOut3H, hvecSum3LL, hvecSum3LH, hvecSum3HL, hvecSum3HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut4L, dvecOut4H, hvecSum4LL, hvecSum4LH, hvecSum4HL, hvecSum4HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
#else
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut1L, dvecOut1H, hvecSum1LL, hvecSum1LH, hvecSum1HL, hvecSum1HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut2L, dvecOut2H, hvecSum2LL, hvecSum2LH, hvecSum2HL, hvecSum2HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut3L, dvecOut3H, hvecSum3LL, hvecSum3LH, hvecSum3HL, hvecSum3HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut4L, dvecOut4H, hvecSum4LL, hvecSum4LH, hvecSum4HL, hvecSum4HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
          /* Store the output dvecOut1 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
          IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut2 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * numX * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
          IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut3 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * numY * bytesPerPixel);
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
        }
        else /* Store to accumulator tile*/
        {
          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecSum1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecSum1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1) * numX);
          IVP_SAVN_2X32_XP(hvecSum2LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX);
          IVP_SAVN_2X32_XP(hvecSum2LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum2HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum2HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch2) * numY);
          IVP_SAVN_2X32_XP(hvecSum3LL, vaOutData, phvecAcc, 4 * remainingOutCh * numY);
          IVP_SAVN_2X32_XP(hvecSum3LH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum3HL, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum3HH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * numX + accDataPitch2 * numY));
          IVP_SAVN_2X32_XP(hvecSum4LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY);
          IVP_SAVN_2X32_XP(hvecSum4LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum4HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum4HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_QM32(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH_QM32(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

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
  int32_t inCh, outCh, x, y, k, j;
  valign vaOutData = IVP_ZALIGN();

  xb_vecN_2x32v* restrict phvecBias;
  xb_vec2Nx8* restrict pdvecCoeff;
  xb_vec2Nx8* restrict pdvecData1;
  xb_vec2Nx8* restrict pdvecData2;
  xb_vec2Nx8* restrict pdvecData3;
  xb_vec2Nx8* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;
  xb_vecN_2x32v* restrict phvecAcc;

  xb_vecN_2x32v hvecSum1LL, hvecSum1LH, hvecSum1HL, hvecSum1HH;
  xb_vecN_2x32v hvecSum2LL, hvecSum2LH, hvecSum2HL, hvecSum2HH;
  xb_vecN_2x32v hvecSum3LL, hvecSum3LH, hvecSum3HL, hvecSum3HH;
  xb_vecN_2x32v hvecSum4LL, hvecSum4LH, hvecSum4HL, hvecSum4HH;
  /* Loops Start */
  for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
  { /* walk across the kernels */
    /* To handle corner case when number of output channels
     * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
    int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV_PARTIAL
    xb_vecNx16U outScaleDataL, outScaleDataH;
    /*Load output scale values*/
    valign vaScale = IVP_LANX16U_PP(pOutScaleData);
    IVP_LAVNX16_XP(outScaleDataL, vaScale, pOutScaleData, 2 * remainingOutCh);
    IVP_LAVNX16_XP(outScaleDataH, vaScale, pOutScaleData, 2 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
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
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        hvecSum1LL = hvecSum1LH = hvecSum1HL = hvecSum1HH = 0;
        hvecSum2LL = hvecSum2LH = hvecSum2HL = hvecSum2HH = 0;
        hvecSum3LL = hvecSum3LH = hvecSum3HL = hvecSum3HH = 0;
        hvecSum4LL = hvecSum4LH = hvecSum4HL = hvecSum4HH = 0;
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
          pdvecData4 = (xb_vec2Nx8 *) (pData + inAddrOff + (strideX * inDataPitch1 + strideY * inDataPitch2) * numX * numY);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

          for (j = 0; j < numInCh; j += 508) /* Emulation: To avoid 24 bit overflow 2^23-1 / 128 / 128 = 511.99 */
          {
            xb_vec2Nx24 daccSum1 = 0, daccSum2 = 0, daccSum3 = 0, daccSum4 = 0;
            int32_t numIter      = XT_MIN(508, numInCh - j);
            for (inCh = 0; inCh < numIter - 3; inCh += 4)
            {
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
            } /* End for (inCh = 0; inCh < row - 3; inCh += 4) */

            /* Corner Case Handling if number of input channels not multiple of 4 */
            if (inCh < numIter)
            {
              int32_t remInCh = numIter - inCh;

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

            hvecSum1LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum1), hvecSum1LL);
            hvecSum1LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum1), hvecSum1LH);
            hvecSum1HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum1), hvecSum1HL);
            hvecSum1HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum1), hvecSum1HH);

            hvecSum2LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum2), hvecSum2LL);
            hvecSum2LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum2), hvecSum2LH);
            hvecSum2HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum2), hvecSum2HL);
            hvecSum2HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum2), hvecSum2HH);

            hvecSum3LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum3), hvecSum3LL);
            hvecSum3LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum3), hvecSum3LH);
            hvecSum3HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum3), hvecSum3HL);
            hvecSum3HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum3), hvecSum3HH);

            hvecSum4LL = IVP_ADDN_2X32(IVP_CVT32S2NX24LL(daccSum4), hvecSum4LL);
            hvecSum4LH = IVP_ADDN_2X32(IVP_CVT32S2NX24LH(daccSum4), hvecSum4LH);
            hvecSum4HL = IVP_ADDN_2X32(IVP_CVT32S2NX24HL(daccSum4), hvecSum4HL);
            hvecSum4HH = IVP_ADDN_2X32(IVP_CVT32S2NX24HH(daccSum4), hvecSum4HH);
          }  /* End for(j = 0; j < numInCh; j += 508)*/
        }   /* End Kernel Height * Width */

        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          xb_vecN_2x32v hvecBiasLL, hvecBiasLH, hvecBiasHL, hvecBiasHH;
          valign vaBias = IVP_LAN_2X32_PP(phvecBias);
          IVP_LAVN_2X32_XP(hvecBiasLL, vaBias, phvecBias, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecBiasLH, vaBias, phvecBias, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecBiasHL, vaBias, phvecBias, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecBiasHH, vaBias, phvecBias, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);

          hvecSum1LL = IVP_ADDN_2X32(hvecSum1LL, hvecBiasLL);
          hvecSum1LH = IVP_ADDN_2X32(hvecSum1LH, hvecBiasLH);
          hvecSum1HL = IVP_ADDN_2X32(hvecSum1HL, hvecBiasHL);
          hvecSum1HH = IVP_ADDN_2X32(hvecSum1HH, hvecBiasHH);

          hvecSum2LL = IVP_ADDN_2X32(hvecSum2LL, hvecBiasLL);
          hvecSum2LH = IVP_ADDN_2X32(hvecSum2LH, hvecBiasLH);
          hvecSum2HL = IVP_ADDN_2X32(hvecSum2HL, hvecBiasHL);
          hvecSum2HH = IVP_ADDN_2X32(hvecSum2HH, hvecBiasHH);

          hvecSum3LL = IVP_ADDN_2X32(hvecSum3LL, hvecBiasLL);
          hvecSum3LH = IVP_ADDN_2X32(hvecSum3LH, hvecBiasLH);
          hvecSum3HL = IVP_ADDN_2X32(hvecSum3HL, hvecBiasHL);
          hvecSum3HH = IVP_ADDN_2X32(hvecSum3HH, hvecBiasHH);

          hvecSum4LL = IVP_ADDN_2X32(hvecSum4LL, hvecBiasLL);
          hvecSum4LH = IVP_ADDN_2X32(hvecSum4LH, hvecBiasLH);
          hvecSum4HL = IVP_ADDN_2X32(hvecSum4HL, hvecBiasHL);
          hvecSum4HH = IVP_ADDN_2X32(hvecSum4HH, hvecBiasHH);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);

          hvecSum1LL = IVP_ADDN_2X32(hvecSum1LL, hvecAcc1LL);
          hvecSum1LH = IVP_ADDN_2X32(hvecSum1LH, hvecAcc1LH);
          hvecSum1HL = IVP_ADDN_2X32(hvecSum1HL, hvecAcc1HL);
          hvecSum1HH = IVP_ADDN_2X32(hvecSum1HH, hvecAcc1HH);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);

          hvecSum2LL = IVP_ADDN_2X32(hvecSum2LL, hvecAcc2LL);
          hvecSum2LH = IVP_ADDN_2X32(hvecSum2LH, hvecAcc2LH);
          hvecSum2HL = IVP_ADDN_2X32(hvecSum2HL, hvecAcc2HL);
          hvecSum2HH = IVP_ADDN_2X32(hvecSum2HH, hvecAcc2HH);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          hvecSum3LL = IVP_ADDN_2X32(hvecSum3LL, hvecAcc3LL);
          hvecSum3LH = IVP_ADDN_2X32(hvecSum3LH, hvecAcc3LH);
          hvecSum3HL = IVP_ADDN_2X32(hvecSum3HL, hvecAcc3HL);
          hvecSum3HH = IVP_ADDN_2X32(hvecSum3HH, hvecAcc3HH);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * numX + accDataPitch2 * numY);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          hvecSum4LL = IVP_ADDN_2X32(hvecSum4LL, hvecAcc4LL);
          hvecSum4LH = IVP_ADDN_2X32(hvecSum4LH, hvecAcc4LH);
          hvecSum4HL = IVP_ADDN_2X32(hvecSum4HL, hvecAcc4HL);
          hvecSum4HH = IVP_ADDN_2X32(hvecSum4HH, hvecAcc4HH);
        }

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut1L, dvecOut1H, hvecSum1LL, hvecSum1LH, hvecSum1HL, hvecSum1HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut2L, dvecOut2H, hvecSum2LL, hvecSum2LH, hvecSum2HL, hvecSum2HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut3L, dvecOut3H, hvecSum3LL, hvecSum3LH, hvecSum3HL, hvecSum3HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut4L, dvecOut4H, hvecSum4LL, hvecSum4LH, hvecSum4HL, hvecSum4HH, \
                                                packShiftAccU, outScaleDataL, outScaleDataH, outShiftU, minLim, maxLim, typeFlag);
#else
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut1L, dvecOut1H, hvecSum1LL, hvecSum1LH, hvecSum1HL, hvecSum1HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut2L, dvecOut2H, hvecSum2LL, hvecSum2LH, hvecSum2HL, hvecSum2HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut3L, dvecOut3H, hvecSum3LL, hvecSum3LH, hvecSum3HL, hvecSum3HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut4L, dvecOut4H, hvecSum4LL, hvecSum4LH, hvecSum4HL, hvecSum4HH, \
                                             packShiftAccU, outScale, outShiftU, minLim, maxLim, typeFlag);
#endif
          /* Store the output dvecOut1 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + outCh * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut1L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh);
          IVP_SAV2NX8_XP(dvecOut1H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut2 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch1) * numX * bytesPerPixel);
          IVP_SAV2NX8_XP(dvecOut2L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * numX);
          IVP_SAV2NX8_XP(dvecOut2H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * numX);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);

          /* Store the output dvecOut3 along the output depth */
          pdvecOut = (xb_vec2Nx8 *) (pOut + (outCh + outDataPitch2) * numY * bytesPerPixel);
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
        }
        else /* Store to accumulator tile*/
        {
          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecSum1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecSum1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1) * numX);
          IVP_SAVN_2X32_XP(hvecSum2LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX);
          IVP_SAVN_2X32_XP(hvecSum2LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum2HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum2HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch2) * numY);
          IVP_SAVN_2X32_XP(hvecSum3LL, vaOutData, phvecAcc, 4 * remainingOutCh * numY);
          IVP_SAVN_2X32_XP(hvecSum3LH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum3HL, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum3HH, vaOutData, phvecAcc, 4 * remainingOutCh * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * numX + accDataPitch2 * numY));
          IVP_SAVN_2X32_XP(hvecSum4LL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY);
          IVP_SAVN_2X32_XP(hvecSum4LH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum4HL, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecSum4HH, vaOutData, phvecAcc, 4 * remainingOutCh * numX * numY - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/*****************************************************************************
*  xaiPartialConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH   \
*  xaiPartialConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D partial */
/*               dilated convolution function and MxN MOD_DWH 3D VQ partial */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Accumulator Tile, Output Tile                              */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Accumulated value will be within 24bit range               */
/****************************************************************************/
#ifdef DILATED_VQ_CONV_PARTIAL
XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
    XAI_CHECK_POINTER(biasArray);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE,   \
                    "\nKernel height = %d and width = %d\nKernel width and height should be less than or equal to 64", \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
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
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV_PARTIAL
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif

    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);

    if (XAI_CNN_CONV_GET_FLAG_INPUT(param))
    {
      XAI_CHECK_ARRAY_S32(biasArray);
    }
    if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
    {
      XAI_CHECK_TILE3D_S32(accTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, accTile);
      XAI_CHECK_TILE3D_DATA_ORDER(accTile, XAI_DWH);
      XAI_CHECK_TILE3D_SIZE_EQ(accTile, outTile);
    }
    if (XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
    {
      XAI_CHECK_TILE3D(outTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
      if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param)))
      {
        XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(accTile, outTile);
      }
    }
  }
#ifndef DILATED_VQ_CONV_PARTIAL
  if ((XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0) && \
      XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Calling further optimized function if dilation = 1 and (no edges along depth or kernelWidth = 1)*/
  if ((XAI_CNN_CONV_GET_DILATIONX(param) == 1) &&                            \
      ((XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile)) || \
       (XAI_TILE4D_GET_DIM3(coeffTile) == 1)))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, \
                                                                         coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_S8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, \
                                                                       coeffTile, biasArray, accTile, outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                                      coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_S8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                                    coeffTile, biasArray, accTile, outTile, param);
#endif
    }
  }
  else
  {
#ifdef DILATED_VQ_CONV_PARTIAL
    partialConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH(inTile, \
                                                 coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
    partialConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH(inTile, \
                                               coeffTile, biasArray, accTile, outTile, param);
#endif
  }
  return(XAI_ERROR_STATUS());
}

/*****************************************************************************
*  xaiPartialConvolved3D_S_MxN_U8S8IXCa2_MOD_DWH   \
*  xaiPartialConvolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D partial */
/*               dilated convolution function and MxN MOD_DWH 3D VQ partial */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Accumulator Tile, Output Tile                              */
/* Assumptions : InData are U8, CoeffData are S8                            */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Accumulated value will be within 24bit range               */
/****************************************************************************/
#ifdef DILATED_VQ_CONV_PARTIAL
XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_U8S8IXCa2_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
    XAI_CHECK_POINTER(biasArray);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE,   \
                    "\nKernel height = %d and width = %d\nKernel width and height should be less than or equal to 64", \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
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
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV_PARTIAL
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif

    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);

    if (XAI_CNN_CONV_GET_FLAG_INPUT(param))
    {
      XAI_CHECK_ARRAY_S32(biasArray);
    }
    if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
    {
      XAI_CHECK_TILE3D_S32(accTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, accTile);
      XAI_CHECK_TILE3D_DATA_ORDER(accTile, XAI_DWH);
      XAI_CHECK_TILE3D_SIZE_EQ(accTile, outTile);
    }
    if (XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
    {
      XAI_CHECK_TILE3D(outTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
      if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param)))
      {
        XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(accTile, outTile);
      }
    }
  }
#ifndef DILATED_VQ_CONV_PARTIAL
  if ((XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0) && \
      XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Calling further optimized function if dilation = 1 and (no edges along depth or kernelWidth = 1)*/
  if ((XAI_CNN_CONV_GET_DILATIONX(param) == 1) &&                            \
      ((XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile)) || \
       (XAI_TILE4D_GET_DIM3(coeffTile) == 1)))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, \
                                                                         coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_U8S8IXCa2_MOD_DWH_contiguous_depth_x4(inTile, \
                                                                       coeffTile, biasArray, accTile, outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_U8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                                      coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_U8S8IXCa2_MOD_DWH_contiguous_depth(inTile, \
                                                                    coeffTile, biasArray, accTile, outTile, param);
#endif
    }
  }
  else
  {
#ifdef DILATED_VQ_CONV_PARTIAL
    partialConvolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH(inTile, \
                                                 coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
    partialConvolved3D_S_MxN_U8S8IXCa2_MOD_DWH(inTile, \
                                               coeffTile, biasArray, accTile, outTile, param);
#endif
  }
  return(XAI_ERROR_STATUS());
}

/**********partialConvolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH************/
/**********partialConvolve3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH   ************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                                const xai_pTile4D coeffTile,
                                                                                const xai_pArray biasArray,
                                                                                const xai_pArray outputScaleArray,
                                                                                xai_pTile3D accTile,
                                                                                xai_pTile3D outTile,
                                                                                const xai_cnn_conv_params *param
                                                                                )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                              const xai_pTile4D coeffTile,
                                                                              const xai_pArray biasArray,
                                                                              xai_pTile3D accTile,
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
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

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
  xb_vec2Nx8U* restrict pdvecData1;
  xb_vec2Nx8U* restrict pdvecData2;
  xb_vec2Nx8U* restrict pdvecData3;
  xb_vec2Nx8U* restrict pdvecData4;
  xb_vec2Nx8* restrict pdvecOut;
  xb_vecN_2x32v* restrict phvecAcc;

  /* Loops Start */
  for (y = 0; y < outH; y++) /* Image Height */
  {
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV_PARTIAL
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
      for (x = 0; x < outW; x += 4) /* Image Width */
      {                             /* walk across the columns */
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        int32_t enable4thWidth = XT_SALT(3, outW - x);

        /* Output Data pointer */
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * enable2ndWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 2 * enable3rdWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 3 * enable4thWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
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
          pdvecData2 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1 * 2 * enable3rdWidth);
          pdvecData4 = (xb_vec2Nx8U *) (pData + inAddrOff + strideX * inDataPitch1 * 3 * enable4thWidth);

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8U_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8U_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8U_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8U_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1;
            xb_vec2Nx8 dvecData2;
            xb_vec2Nx8 dvecData3;
            xb_vec2Nx8 dvecData4;

            dvecData1 = IVP_SUB2NX8U(dvecInp1, 128);
            dvecData2 = IVP_SUB2NX8U(dvecInp2, 128);
            dvecData3 = IVP_SUB2NX8U(dvecInp3, 128);
            dvecData4 = IVP_SUB2NX8U(dvecInp4, 128);

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
            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          } /* End Input Channels */

          /* Corner Case Handling if number of input channels not multiple of 4 */
          if (inCh < numInCh)
          {
            int32_t remInCh = numInCh - inCh;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, remInCh);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            xb_vec2Nx8 dvecData3 = 0;
            xb_vec2Nx8 dvecData4 = 0;

            IVP_SUB2NX8UT(dvecData1, dvecInp1, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData2, dvecInp2, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData3, dvecInp3, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData4, dvecInp4, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));

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
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          } /* End Corner case handling */
        }   /* End Kernel Height * Width */

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
          IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable4thWidth);
          IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable4thWidth);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * enable2ndWidth);
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 2 * enable3rdWidth);
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 3 * enable4thWidth);
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

  int32_t numIter = kWidthU * numInCh;

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
  xb_vecN_2x32v* restrict phvecAcc;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (y = 0; y < outH; y++)  /* Image Height */
  {                           /* walk down the rows */
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV_PARTIAL
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
      for (x = 0; x < outW; x += 4) /* Image Width */
      {                             /* walk across the columns */
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        int32_t enable4thWidth = XT_SALT(3, outW - x);
        /* Output Data pointer */
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * enable2ndWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 2 * enable3rdWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 3 * enable4thWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;


        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * 2 * enable3rdWidth);
          pdvecData4 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * 3 * enable4thWidth);

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
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1;
            xb_vec2Nx8 dvecData2;
            xb_vec2Nx8 dvecData3;
            xb_vec2Nx8 dvecData4;

            dvecData1 = IVP_SUB2NX8U(dvecInp1, 128);
            dvecData2 = IVP_SUB2NX8U(dvecInp2, 128);
            dvecData3 = IVP_SUB2NX8U(dvecInp3, 128);
            dvecData4 = IVP_SUB2NX8U(dvecInp4, 128);

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
            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          }   /* End Input Channels */
        } /* End Kernel Height * Width */

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * enable2ndWidth));
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 2 * enable3rdWidth));
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 3 * enable4thWidth));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData is U8, CoeffData is S8                              */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  uint8_t *pInData   = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

  int32_t numIter = kWidthU * numInCh;

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
  xb_vecN_2x32v* restrict phvecAcc;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (y = 0; y < outH; y++)  /* Image Height */
  {                           /* walk down the rows */
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV_PARTIAL
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif

      for (x = 0; x < outW; x += 4) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        int32_t enable4thWidth = XT_SALT(3, outW - x);

        /* Output Data pointer */
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * enable2ndWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 2 * enable3rdWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 3 * enable4thWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        uint8_t *pData = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * 2 * enable3rdWidth);
          pdvecData4 = (xb_vec2Nx8U *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * 3 * enable4thWidth);

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
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, 4);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, 4);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, 4);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, 4);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1;
            xb_vec2Nx8 dvecData2;
            xb_vec2Nx8 dvecData3;
            xb_vec2Nx8 dvecData4;

            dvecData1 = IVP_SUB2NX8U(dvecInp1, 128);
            dvecData2 = IVP_SUB2NX8U(dvecInp2, 128);
            dvecData3 = IVP_SUB2NX8U(dvecInp3, 128);
            dvecData4 = IVP_SUB2NX8U(dvecInp4, 128);

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
            IVP_MULQA2N8XR8(daccSum1, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, dvecCoeff4, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          }   /* End Input Channels */
          /* Corner case handling as numIter is not a multiple of 4 */
          if (k < numIter)
          {
            int32_t remInCh = numIter - k;

            /* Aligning variable vector load of pixels */
            xb_vec2Nx8U dvecInp1; IVP_LAV2NX8U_XP(dvecInp1, vaData1, pdvecData1, remInCh);
            xb_vec2Nx8U dvecInp2; IVP_LAV2NX8U_XP(dvecInp2, vaData2, pdvecData2, remInCh);
            xb_vec2Nx8U dvecInp3; IVP_LAV2NX8U_XP(dvecInp3, vaData3, pdvecData3, remInCh);
            xb_vec2Nx8U dvecInp4; IVP_LAV2NX8U_XP(dvecInp4, vaData4, pdvecData4, remInCh);

#ifdef IVP_MULSUQA2N8XR8
            /* Extracting first 4 bytes of vector into address register */
            /* Scalar integers to be used for QMUL                      */
            int32_t qmulScalar1 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp1)), 0);
            int32_t qmulScalar2 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp2)), 0);
            int32_t qmulScalar3 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp3)), 0);
            int32_t qmulScalar4 = IVP_EXTRN_2X32(IVP_MOVN_2X32_FROMNX16 \
                                                   (IVP_MOVNX16_FROM2NX8U(dvecInp4)), 0);
#else
            xb_vec2Nx8 dvecData1 = 0;
            xb_vec2Nx8 dvecData2 = 0;
            xb_vec2Nx8 dvecData3 = 0;
            xb_vec2Nx8 dvecData4 = 0;

            IVP_SUB2NX8UT(dvecData1, dvecInp1, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData2, dvecInp2, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData3, dvecInp3, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));
            IVP_SUB2NX8UT(dvecData4, dvecInp4, 128, IVP_LT2NX8(IVP_SEQ2NX8U(), remInCh));

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
#endif
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
            IVP_MULQA2N8XR8(daccSum1, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar1);
            IVP_MULQA2N8XR8(daccSum2, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar2);
            IVP_MULQA2N8XR8(daccSum3, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar3);
            IVP_MULQA2N8XR8(daccSum4, 0, dvecCoeff3, dvecCoeff2, dvecCoeff1, qmulScalar4);
#endif
          } /* End Corner case handling */
        }   /* End Kernel Height * Width */

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * enable2ndWidth));
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 2 * enable3rdWidth));
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 3 * enable4thWidth));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/*****************************************************************************
*  xaiPartialConvolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH   \
*  xaiPartialConvolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D partial */
/*               dilated convolution function and MxN MOD_DWH 3D VQ partial */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Accumulator Tile, Output Tile                              */
/* Assumptions : InData are U8, CoeffData are S8                            */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Accumulated value will be within 24bit range               */
/****************************************************************************/
#ifdef DILATED_VQ_CONV_PARTIAL
XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
    XAI_CHECK_POINTER(biasArray);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE,   \
                    "\nKernel height = %d and width = %d\nKernel width and height should be less than or equal to 64", \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
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
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV_PARTIAL
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif

    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);

    if (XAI_CNN_CONV_GET_FLAG_INPUT(param))
    {
      XAI_CHECK_ARRAY_S32(biasArray);
    }
    if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
    {
      XAI_CHECK_TILE3D_S32(accTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, accTile);
      XAI_CHECK_TILE3D_DATA_ORDER(accTile, XAI_DWH);
      XAI_CHECK_TILE3D_SIZE_EQ(accTile, outTile);
    }
    if (XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
    {
      XAI_CHECK_TILE3D(outTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
      if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param)))
      {
        XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(accTile, outTile);
      }
    }
  }
#ifndef DILATED_VQ_CONV_PARTIAL
  if ((XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0) && \
      XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Calling further optimized function if dilation = 1 and (no edges along depth or kernelWidth = 1)*/
  if ((XAI_CNN_CONV_GET_DILATIONX(param) == 1) &&                            \
      ((XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile)) || \
       (XAI_TILE4D_GET_DIM3(coeffTile) == 1)))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(inTile, \
                                                                                   coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(inTile, \
                                                                                 coeffTile, biasArray, accTile, outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(inTile, \
                                                                                coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_U8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(inTile, \
                                                                              coeffTile, biasArray, accTile, outTile, param);
#endif
    }
  }
  else
  {
#ifdef DILATED_VQ_CONV_PARTIAL
    partialConvolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(inTile, \
                                                           coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
    partialConvolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(inTile, \
                                                         coeffTile, biasArray, accTile, outTile, param);
#endif
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D partial */
/*               dilated convolution function and MxN MOD_DWH 3D VQ partial */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Accumulator Tile, Output Tile                              */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Accumulated value will be within 32bit range               */
/****************************************************************************/
#ifdef DILATED_VQ_CONV_PARTIAL
XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_QM32(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH_QM32(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
    XAI_CHECK_POINTER(biasArray);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE,   \
                    "\nKernel height = %d and width = %d\nKernel width and height should be less than or equal to 64", \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
    XAI_CHECK_ERROR(((XAI_CNN_CONV_GET_STRIDEX(param) > 0) && (XAI_CNN_CONV_GET_STRIDEY(param) > 0)) &&                                      \
                    ((XAI_CNN_CONV_GET_STRIDEX(param) <= 64) && (XAI_CNN_CONV_GET_STRIDEY(param) <= 64)), XAI_ERR_BADARG,                    \
                    "\nStrideX = %hhu, StrideY = %hhu\nStride along width and height should be greater than 0 and less than or equal to 64", \
                    XAI_CNN_CONV_GET_STRIDEX(param), XAI_CNN_CONV_GET_STRIDEY(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONX(param) == 1) ||                                                           \
                    ((XAI_CNN_CONV_GET_DILATIONX(param) >= 1) &&                                                          \
                     (XAI_CNN_CONV_GET_STRIDEX(param) == 1)), XAI_ERR_BADARG,                                             \
                    "\nDilationX = %hhu\nDilationX should be 1. It can be greater than 1 only when stride is equal to 1", \
                    XAI_CNN_CONV_GET_DILATIONX(param));
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATIONY(param) == 1) ||                                                           \
                    ((XAI_CNN_CONV_GET_DILATIONY(param) >= 1) &&                                                          \
                     (XAI_CNN_CONV_GET_STRIDEY(param) == 1)), XAI_ERR_BADARG,                                             \
                    "\nDilationY = %hhu\nDilationY should be 1. It can be greater than 1 only when stride is equal to 1", \
                    XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV_PARTIAL
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif

    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);

    if (XAI_CNN_CONV_GET_FLAG_INPUT(param))
    {
      XAI_CHECK_ARRAY_S32(biasArray);
    }
    if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
    {
      XAI_CHECK_TILE3D_S32(accTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, accTile);
      XAI_CHECK_TILE3D_DATA_ORDER(accTile, XAI_DWH);
      XAI_CHECK_TILE3D_SIZE_EQ(accTile, outTile);
    }
    if (XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
    {
      XAI_CHECK_TILE3D(outTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
      if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param)))
      {
        XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(accTile, outTile);
      }
    }
  }
#ifndef DILATED_VQ_CONV_PARTIAL
  if ((XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0) && \
      XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif

  /* Calling further optimized function if dilation = 1 and (no edges along depth or kernelWidth = 1)*/
  if ((XAI_CNN_CONV_GET_DILATIONX(param) == 1) &&                            \
      ((XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile)) || \
       (XAI_TILE4D_GET_DIM3(coeffTile) == 1)))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_MOD_DWH_QM32_contiguous_depth_x4(inTile, \
                                                                              coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_S8S8IXCa2_MOD_DWH_QM32_contiguous_depth_x4(inTile, \
                                                                            coeffTile, biasArray, accTile, outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_MOD_DWH_QM32_contiguous_depth(inTile, \
                                                                           coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_S8S8IXCa2_MOD_DWH_QM32_contiguous_depth(inTile, \
                                                                         coeffTile, biasArray, accTile, outTile, param);
#endif
    }
  }
  else
  {
#ifdef DILATED_VQ_CONV_PARTIAL
    partialConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_QM32(inTile, \
                                                      coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
    partialConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH_QM32(inTile, \
                                                    coeffTile, biasArray, accTile, outTile, param);
#endif
  }
  return(XAI_ERROR_STATUS());
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

  int32_t numIter = kWidthU * numInCh;

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
  xb_vecN_2x32v* restrict phvecAcc;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (y = 0; y < outH; y++)  /* Image Height */
  {                           /* walk down the rows */
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV_PARTIAL
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
      for (x = 0; x < outW; x += 4) /* Image Width */
      {                             /* walk across the columns */
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        int32_t enable4thWidth = XT_SALT(3, outW - x);
        /* Output Data pointer */
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * enable2ndWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 2 * enable3rdWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 3 * enable4thWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;


        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * 2 * enable3rdWidth);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * 3 * enable4thWidth);

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

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * enable2ndWidth));
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 2 * enable3rdWidth));
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 3 * enable4thWidth));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/

#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxNd1_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t dilationX     = 1;
  const uint8_t dilationY     = XAI_CNN_CONV_GET_DILATIONY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

  int32_t numIter = kWidthU * numInCh;

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
  xb_vecN_2x32v* restrict phvecAcc;

  /*
   * inCh and kWidth loops are combined. Assumed that the
   * edges along Depth dimension of input data is zero and also
   * edges along depth dimension of coefficient data is zero.
   */

  /* Loops Start */
  for (y = 0; y < outH; y++)  /* Image Height */
  {                           /* walk down the rows */
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV_PARTIAL
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif

      for (x = 0; x < outW; x += 4) /* Image Width */
      {                             /* walk across the columns */
        /* Variable to handle corner case when width is odd */
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        int32_t enable4thWidth = XT_SALT(3, outW - x);

        /* Output Data pointer */
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * enable2ndWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 2 * enable3rdWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 3 * enable4thWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

        /* Input Data and Coeff Data Pointers */
        int8_t *pData  = pInData + x * strideX * inDataPitch1 + y * strideY * inDataPitch2;
        int8_t *pCoeff = pCoeffData + outCh;

#ifdef __XCC__
#pragma loop_count min=1
#endif
        for (ky = 0; ky < kHeightU; ky++) /* Kernel Height */
        {
          /* Pointers for Input Data Loads */
          pdvecData1 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY);
          pdvecData2 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * 2 * enable3rdWidth);
          pdvecData4 = (xb_vec2Nx8 *) (pData + ky * inDataPitch2 * dilationY + strideX * inDataPitch1 * 3 * enable4thWidth);

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
          if (k < numIter)
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
          }   /* End if( k < numIter)*/
        } /* End Kernel Height * Width */

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * enable2ndWidth));
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 2 * enable3rdWidth));
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 3 * enable4thWidth));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/****************************************************************************/
/* Description : P6 optimized implementation of 3D partial convolution      */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               CNN convolution params structure                           */
/* InOuts      : Output Tile                                                */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Edges along Depth dimension in inTile and coeffTile        */
/*               are zero.                                                  */
/****************************************************************************/
#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

  /* Data Pointers of input, output, coefficient and bias data */
  int8_t *pInData    = (int8_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  int8_t *pOutData   = (int8_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int8_t *pCoeffData = (int8_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  int32_t *pBiasData = (int32_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  int32_t * pAccData = NULL;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData = (int32_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
  }

#ifdef DILATED_VQ_CONV_PARTIAL
  uint16_t *pScale = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
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

  /* Pitch of AccTile Data (DWH) in dim1 and dim2 */
  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;
  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

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
  xb_vecN_2x32v* restrict phvecAcc;

  /* Loops Start */
  for (y = 0; y < outH; y++) /* Image Height */
  {                          /* walk down the rows */
    for (outCh = 0; outCh < numOutCh; outCh += 2 * XCHAL_IVPN_SIMD_WIDTH)
    { /* walk across the kernels */
      /* To handle corner case when number of output channels
       * is not a multiple of  2 * XCHAL_IVPN_SIMD_WIDTH*/
      int32_t remainingOutCh = numOutCh - outCh;
#ifdef DILATED_VQ_CONV_PARTIAL
      xb_vecNx16U outScaleDataEven, outScaleDataOdd;
      /*Load output scale values*/
      xb_vecNx16U* restrict pOutScaleData = (xb_vecNx16U *) (pScale + outCh);
      VQ_INIT_OUTSCALE(pOutScaleData, remainingOutCh, outScaleDataEven, outScaleDataOdd);
#endif
      for (x = 0; x < outW; x += 4) /* Image Width */
      {                             /* walk across the columns */
        int32_t enable2ndWidth = XT_SALT(1, outW - x);
        int32_t enable3rdWidth = XT_SALT(2, outW - x);
        int32_t enable4thWidth = XT_SALT(3, outW - x);
        /* Output Data pointer */
        int8_t *pOut  = pOutData + (x * outDataPitch1 + y * outDataPitch2) * bytesPerPixel;
        int32_t *pAcc = pAccData + (x * accDataPitch1 + y * accDataPitch2);

        /* Initialize accumulators with bias values */
        xb_vec2Nx24 daccSum1, daccSum2, daccSum3, daccSum4;
        if (inputFlag) /* Bias Values */
        {
          phvecBias = (xb_vecN_2x32v *) (pBiasData + outCh);
          ACC_INIT_BIAS(phvecBias, remainingOutCh, daccSum1, daccSum2, daccSum3, daccSum4);
        }
        else  /* Accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL, hvecAcc1LH, hvecAcc1HL, hvecAcc1HH;
          xb_vecN_2x32v hvecAcc2LL, hvecAcc2LH, hvecAcc2HL, hvecAcc2HH;
          xb_vecN_2x32v hvecAcc3LL, hvecAcc3LH, hvecAcc3HL, hvecAcc3HH;
          xb_vecN_2x32v hvecAcc4LL, hvecAcc4LH, hvecAcc4HL, hvecAcc4HH;

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          valign vaAcc = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc1LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc1LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc1HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum1 = IVP_CVT24UNX32L(hvecAcc1LH, hvecAcc1LL);
          IVP_CVT24UNX32H(daccSum1, hvecAcc1HH, hvecAcc1HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * enable2ndWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc2LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc2LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc2HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum2 = IVP_CVT24UNX32L(hvecAcc2LH, hvecAcc2LL);
          IVP_CVT24UNX32H(daccSum2, hvecAcc2HH, hvecAcc2HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 2 * enable3rdWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc3LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc3LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc3HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum3 = IVP_CVT24UNX32L(hvecAcc3LH, hvecAcc3LL);
          IVP_CVT24UNX32H(daccSum3, hvecAcc3HH, hvecAcc3HL);

          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh + accDataPitch1 * 3 * enable4thWidth);
          vaAcc    = IVP_LAN_2X32_PP(phvecAcc);
          IVP_LAVN_2X32_XP(hvecAcc4LL, vaAcc, phvecAcc, 4 * remainingOutCh);
          IVP_LAVN_2X32_XP(hvecAcc4LH, vaAcc, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HL, vaAcc, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_LAVN_2X32_XP(hvecAcc4HH, vaAcc, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          daccSum4 = IVP_CVT24UNX32L(hvecAcc4LH, hvecAcc4LL);
          IVP_CVT24UNX32H(daccSum4, hvecAcc4HH, hvecAcc4HL);
        }

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
          pdvecData2 = (xb_vec2Nx8 *) (pData + inAddrOff + strideX * inDataPitch1 * enable2ndWidth);
          pdvecData3 = (xb_vec2Nx8 *) (pData + inAddrOff + strideX * inDataPitch1 * 2 * enable3rdWidth);
          pdvecData4 = (xb_vec2Nx8 *) (pData + inAddrOff + (strideX * inDataPitch1 * 3 * enable4thWidth));

          /* Pointer for Coefficient Load */
          pdvecCoeff = (xb_vec2Nx8 *) (pCoeff + coeffAddrOff);

          /* Primes registers for Aligning Load */
          valign vaData1 = IVP_LA2NX8_PP(pdvecData1);
          valign vaData2 = IVP_LA2NX8_PP(pdvecData2);
          valign vaData3 = IVP_LA2NX8_PP(pdvecData3);
          valign vaData4 = IVP_LA2NX8_PP(pdvecData4);

          for (inCh = 0; inCh < numInCh - 3; inCh += 4) /* Input Channels */
          {
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

        if (outputFlag)  /* Store to ouput Tile*/
        {
          /* Pack, Output Scale, Output Shift and clamping */
          xb_vec2Nx8 dvecOut1L, dvecOut2L, dvecOut3L, dvecOut4L;
          xb_vec2Nx8 dvecOut1H, dvecOut2H, dvecOut3H, dvecOut4H;
#ifdef DILATED_VQ_CONV_PARTIAL
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
          IVP_SAV2NX8_XP(dvecOut4L, vaOutData, pdvecOut, bytesPerPixel * remainingOutCh * enable4thWidth);
          IVP_SAV2NX8_XP(dvecOut4H, vaOutData, pdvecOut, typeFlag * 2 * \
                         (remainingOutCh - XCHAL_IVPN_SIMD_WIDTH) * enable4thWidth);
          IVP_SAPOS2NX8_FP(vaOutData, pdvecOut);
        }
        else /* Store to accumulator tile*/
        {
          xb_vecN_2x32v hvecAcc1LL = IVP_CVT32S2NX24LL(daccSum1);
          xb_vecN_2x32v hvecAcc1LH = IVP_CVT32S2NX24LH(daccSum1);
          xb_vecN_2x32v hvecAcc1HL = IVP_CVT32S2NX24HL(daccSum1);
          xb_vecN_2x32v hvecAcc1HH = IVP_CVT32S2NX24HH(daccSum1);

          xb_vecN_2x32v hvecAcc2LL = IVP_CVT32S2NX24LL(daccSum2);
          xb_vecN_2x32v hvecAcc2LH = IVP_CVT32S2NX24LH(daccSum2);
          xb_vecN_2x32v hvecAcc2HL = IVP_CVT32S2NX24HL(daccSum2);
          xb_vecN_2x32v hvecAcc2HH = IVP_CVT32S2NX24HH(daccSum2);

          xb_vecN_2x32v hvecAcc3LL = IVP_CVT32S2NX24LL(daccSum3);
          xb_vecN_2x32v hvecAcc3LH = IVP_CVT32S2NX24LH(daccSum3);
          xb_vecN_2x32v hvecAcc3HL = IVP_CVT32S2NX24HL(daccSum3);
          xb_vecN_2x32v hvecAcc3HH = IVP_CVT32S2NX24HH(daccSum3);

          xb_vecN_2x32v hvecAcc4LL = IVP_CVT32S2NX24LL(daccSum4);
          xb_vecN_2x32v hvecAcc4LH = IVP_CVT32S2NX24LH(daccSum4);
          xb_vecN_2x32v hvecAcc4HL = IVP_CVT32S2NX24HL(daccSum4);
          xb_vecN_2x32v hvecAcc4HH = IVP_CVT32S2NX24HH(daccSum4);


          /* Store the hvecAcc1 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + outCh);
          IVP_SAVN_2X32_XP(hvecAcc1LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc1LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc1HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc2 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * enable2ndWidth));
          IVP_SAVN_2X32_XP(hvecAcc2LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc2LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc2HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the hvecAcc3 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 2 * enable3rdWidth));
          IVP_SAVN_2X32_XP(hvecAcc3LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc3LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc3HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);

          /* Store the  hvecAcc4 along the accTile depth */
          phvecAcc = (xb_vecN_2x32v *) (pAcc + (outCh + accDataPitch1 * 3 * enable4thWidth));
          IVP_SAVN_2X32_XP(hvecAcc4LL, vaOutData, phvecAcc, 4 * remainingOutCh);
          IVP_SAVN_2X32_XP(hvecAcc4LH, vaOutData, phvecAcc, 4 * remainingOutCh - 2 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HL, vaOutData, phvecAcc, 4 * remainingOutCh - 4 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAVN_2X32_XP(hvecAcc4HH, vaOutData, phvecAcc, 4 * remainingOutCh - 6 * XCHAL_IVPN_SIMD_WIDTH);
          IVP_SAPOSN_2X32_FP(vaOutData, phvecAcc);
        }
      } /* End image width */
    }   /* End image height */
  }     /* End Output Channels */
}

/*****************************************************************************
*  xaiPartialConvolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH   \
*  xaiPartialConvolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH
*  **************************************************************************/

/****************************************************************************/
/* Description : P6 optimized generic implementation for MxN MOD_DWH        */
/*               3D convolution. Based on pre-processor specifiers. Code    */
/*               implementation is generated during preprocessing stage.    */
/*               This method can be used to generate MxN MOD_DWH 3D partial */
/*               dilated convolution function and MxN MOD_DWH 3D VQ partial */
/*               dilated convolution function                               */
/*               Stride values = 1, 2 and 4 are supported                   */
/*               Implementation also supports dilation >= 1 for stride = 1  */
/*               and dilation = 1 for stride = 2, 4                         */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array,              */
/*               Output scale array, CNN convolution params structure       */
/* Outputs     : XI Error Code                                              */
/* InOuts      : Accumulator Tile, Output Tile                              */
/* Assumptions : InData, CoeffData are S8                                   */
/*               biasArray is signed 32b, value not exceeding signed 24b    */
/*               Output scale array is U16                                  */
/*               OutData is S8 / U8 / S16                                   */
/*               Kernel Size is MxNxDxNk. M and N sizes are less than or    */
/*               equal to 16.                                               */
/*               Input and Output are in DWH format                         */
/*               Coeff is in NDWH format                                    */
/*               CoeffDim1Pitch is aligned to 2N (Ca2)                      */
/*               Accumulated value will be within 24bit range               */
/****************************************************************************/
#ifdef DILATED_VQ_CONV_PARTIAL
XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  const xai_pArray outputScaleArray,
  xai_pTile3D accTile,
  xai_pTile3D outTile,
  const xai_cnn_conv_params *param
  )
#else
XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(
  const xai_pTile3D inTile,
  const xai_pTile4D coeffTile,
  const xai_pArray biasArray,
  xai_pTile3D accTile,
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
    XAI_CHECK_POINTER(biasArray);
    XAI_CHECK_POINTER(param);
    XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(inTile);
    XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(coeffTile);
    XAI_CHECK_ERROR((XAI_TILE4D_GET_DIM3(coeffTile) <= 64) && (XAI_TILE4D_GET_DIM4(coeffTile) <= 64), XAI_ERR_KSIZE,   \
                    "\nKernel height = %d and width = %d\nKernel width and height should be less than or equal to 64", \
                    XAI_TILE4D_GET_DIM4(coeffTile), XAI_TILE4D_GET_DIM3(coeffTile));
    XAI_CHECK_EDGES_MOD_DWH(inTile, coeffTile, param);
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
    XAI_CHECK_TILE4D_IALIGNMENT_2NX8(coeffTile);
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 24,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 24", \
                    XAI_CNN_CONV_GET_ACCUM_SHIFT(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_OUTPUT_SHIFT(param) < 32,                               \
                    XAI_ERR_NORM, "\nThe output shift = %hhu, value should be less than 32", \
                    XAI_CNN_CONV_GET_OUTPUT_SHIFT(param));
    XAI_CHECK_CONV_RELU_LIMITS_IX(param, outTile);
#ifdef DILATED_VQ_CONV_PARTIAL
    XAI_CHECK_ARRAY_U16(outputScaleArray);
    XAI_CHECK_ERROR(XAI_ARRAY_GET_WIDTH(outputScaleArray) >= XAI_TILE4D_GET_DIM1(coeffTile), XAI_ERR_DATASIZE,                                                      \
                    "\nWidth of Output Scale Array = %d, Number of Kernels = %d\nWidth of Output Scale Array should be greater than or equal to Number of Kernels", \
                    XAI_ARRAY_GET_WIDTH(outputScaleArray), XAI_TILE4D_GET_DIM1(coeffTile));
#endif

    XAI_CHECK_CONSISTENCY_MOD_DWH(inTile, coeffTile, biasArray, outTile, param);

    if (XAI_CNN_CONV_GET_FLAG_INPUT(param))
    {
      XAI_CHECK_ARRAY_S32(biasArray);
    }
    if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
    {
      XAI_CHECK_TILE3D_S32(accTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, accTile);
      XAI_CHECK_TILE3D_DATA_ORDER(accTile, XAI_DWH);
      XAI_CHECK_TILE3D_SIZE_EQ(accTile, outTile);
    }
    if (XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
    {
      XAI_CHECK_TILE3D(outTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
      if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param)))
      {
        XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(accTile, outTile);
      }
    }
  }
#ifndef DILATED_VQ_CONV_PARTIAL
  if ((XAI_CNN_CONV_GET_OUTPUT_SCALE(param) == 0) && \
      XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
  {
    int32_t fillValue;
    int32_t reluFlag = XAI_CNN_CONV_GET_FLAG_RELU(param);
    fillValue = reluFlag ? (CLAMP(0, XAI_CNN_CONV_GET_RELU_MIN(param), XAI_CNN_CONV_GET_RELU_MAX(param))) : 0;
    return(xaiFillTile3D(outTile, fillValue, 0));
  }
#endif
  /* Calling further optimized function if dilation = 1 and (no edges along depth or kernelWidth = 1)*/
  if ((XAI_CNN_CONV_GET_DILATIONX(param) == 1) &&                            \
      ((XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile)) || \
       (XAI_TILE4D_GET_DIM3(coeffTile) == 1)))
  {
    if ((XAI_TILE3D_GET_DIM1(inTile) * XAI_TILE4D_GET_DIM3(coeffTile)) % 4 == 0)
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(inTile, \
                                                                                   coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth_x4(inTile, \
                                                                                 coeffTile, biasArray, accTile, outTile, param);
#endif
    }
    else
    {
#ifdef DILATED_VQ_CONV_PARTIAL
      partialConvolvedVQ3D_S_MxNd1_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(inTile, \
                                                                                coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
      partialConvolved3D_S_MxNd1_S8S8IXCa2_noUnrollH_MOD_DWH_contiguous_depth(inTile, \
                                                                              coeffTile, biasArray, accTile, outTile, param);
#endif
    }
  }
  else
  {
#ifdef DILATED_VQ_CONV_PARTIAL
    partialConvolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(inTile, \
                                                           coeffTile, biasArray, outputScaleArray, accTile, outTile, param);
#else
    partialConvolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(inTile, \
                                                         coeffTile, biasArray, accTile, outTile, param);
#endif
  }
  return(XAI_ERROR_STATUS());
}
#endif /*#if ((XCHAL_VISION_TYPE >= 6))*/
