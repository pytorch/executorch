/*
 * Copyright (c) 2023 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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

/********* partialConvolvedVQ3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth ***********/
/********** partialConvolved3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth ************/
/***********************************************************************************/
/* Description : Specialized optimized implementation for partial 3D convolution   */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array, Output Scale Array, */
/*               CNN convolution params structure                                  */
/* Outputs     :                                                                   */
/* InOuts      : Accumulator Tile, Output Tile                                     */
/* Assumptions : InData, CoeffData are S16                                         */
/*               OutData is U16 / S16                                              */
/*               Input is in DWH and Output is in DWH format                       */
/*               Coeff is in NDWH format                                           */
/*               Input does not have edges along the depth dimension               */
/*               dilationX = dilationY = 1 always                                  */
/*               Accumulated value will be within 48-bit range                     */
/***********************************************************************************/
#ifdef DILATED_VQ_CONV_PARTIAL
static _XAI_INLINE_ void partialConvolvedVQ3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth(const xai_pTile3D inTile,
                                                                                       const xai_pTile4D coeffTile,
                                                                                       const xai_pArray biasArray,
                                                                                       const xai_pArray outputScaleArray,
                                                                                       xai_pTile3D accTile,
                                                                                       xai_pTile3D outTile,
                                                                                       const xai_cnn_conv_params *param)
#else
static _XAI_INLINE_ void partialConvolved3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth(const xai_pTile3D inTile,
                                                                                     const xai_pTile4D coeffTile,
                                                                                     const xai_pArray biasArray,
                                                                                     xai_pTile3D accTile,
                                                                                     xai_pTile3D outTile,
                                                                                     const xai_cnn_conv_params *param)
#endif
{
  /* Getting parameters from the tile structures */
  const int32_t numInCh         = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh        = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t outWidth        = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outHeight       = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t inDataPitch1    = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2    = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outDataPitch1   = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2   = XAI_TILE3D_GET_DIM2_PITCH(outTile);
  const int32_t coeffDataPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffDataPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);
  const int32_t kWidthU         = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU        = XAI_TILE4D_GET_DIM4(coeffTile);

  /* Convolution params */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

#ifdef DILATED_VQ_CONV_PARTIAL
  const uint16_t *pOutputScaleData = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  /* Data Pointers of input, coefficient, biasData */
  const int16_t *pInData    = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  const int16_t *pCoeffData = (int16_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  const int64_t *pBiasData  = (int64_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  /* Data Pointers of output and scratch buffer data */
  int16_t *pOutData = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int64_t *pAccData = NULL;

  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;

  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData      = (int64_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

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

  /* move to start of edge data only when input is already padded. */
  pInData = &pInData[-(int32_t) ((topEdge) * inDataPitch2 + (leftEdge) * inDataPitch1)];

  /* Setting the limits for output data according to ReLu is enabled or not*/
  int32_t minLim, maxLim;

  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MIN : 0);
    maxLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX : USHRT_MAX);
  }

  int32_t outCh, x, y, ky, numIter, iter;

  numIter = (numInCh * kWidthU);

  xb_vecN_2x32v *restrict phvecIn1;
  xb_vecN_2x32v *restrict phvecIn2;
  xb_vecN_2x32v *restrict phvecIn3;
  xb_vecN_2x32v *restrict phvecIn4;
  xb_vecNx16 *restrict pvecCoeff;
  xb_vec2Nx8 *restrict pdvecBias;
  xb_vec2Nx8 *restrict pdvecAccData;
  xb_vecNx16 *restrict pvecOut;

  xb_vecNx48 vecAcc1 = 0, vecAcc2 = 0, vecAcc3 = 0, vecAcc4 = 0, vecBias = 0;
  xb_vecN_2x32v hvecIn1, hvecIn2, hvecIn3, hvecIn4;
  xb_vecNx16 vecCoeff1, vecCoeff2;
  xb_vecNx16 vecOut1, vecOut2, vecOut3, vecOut4;
  xb_vec2Nx8 dvecAccLL, dvecAccLH, dvecAccHL, dvecAccHH;

  valign vaIn1, vaIn2, vaIn3, vaIn4, vaBias, vaAcc;

#ifdef DILATED_VQ_CONV_PARTIAL
  xb_vecNx16U vecOutScaleU;
  xb_vecNx16U *restrict pvecOutScaleData = (xb_vecNx16U *) (pOutputScaleData);
  valign vaScale                         = IVP_LANX16U_PP(pvecOutScaleData);
#endif

  pdvecBias = (xb_vec2Nx8 *) (pBiasData);
  vaBias    = IVP_LA2NX8_PP(pdvecBias);
  valign vaOut = IVP_ZALIGN();

  for (outCh = 0; outCh < numOutCh; outCh += XCHAL_IVPN_SIMD_WIDTH)
  {
    int32_t remOutCh = (numOutCh - outCh);
    /* Initially the accumulators with the 48-bit bias values */
    if (inputFlag) // Biases will be loaded only when "inputFlag" is set
    {
      ACC_INIT_BIAS64_MOD_ONEACC(pdvecBias, vaBias, remOutCh, vecBias);
    }

#ifdef DILATED_VQ_CONV_PARTIAL
    IVP_LAVNX16U_XP(vecOutScaleU, vaScale, pvecOutScaleData, 2 * remOutCh);
#endif

    for (y = 0; y < outHeight; y += 2)
    {
      // Calculating "remY" for integrated tail-handling purpose
      int32_t remY = XT_MIN(1, outHeight - y - 1);
      for (x = 0; x < outWidth; x += 2)
      {
        // Calculating "remX" for integrated tail-handling purpose
        int32_t remX    = XT_MIN(1, outWidth - x - 1);
        int16_t *pData1 = (int16_t *) (pInData + (x * strideX * inDataPitch1) + (y * strideY * inDataPitch2));
        int64_t *pAcc   = (int64_t *) (pAccData + outCh + (x * accDataPitch1) + (y * accDataPitch2));

        if (inputFlag) // if "inputFlag" is set, then initialize the accumulators with the bias values
        {
          /* Initializing all the 4 accumulators with bias values before accumulating for every spatial location */
          vecAcc4 = vecAcc3 = vecAcc2 = vecAcc1 = vecBias;
        }
        else // if "inputFlag" is not-set, then initialize the accumulators with the values stored in the accTile
        {
          // Loading accumulated values from W = 0, H = 0 spatial location initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc);
          vaAcc        = IVP_LA2NX8_PP(pdvecAccData);
          IVP_LAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_LAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          vecAcc1 = IVP_CVT48UN_2X64L(dvecAccLH, dvecAccLL);
          IVP_CVT48UN_2X64H(vecAcc1, dvecAccHH, dvecAccHL);

          // Loading accumulated values form W = 1, H = 0 spatial location initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remX * accDataPitch1));
          vaAcc        = IVP_LA2NX8_PP(pdvecAccData);
          IVP_LAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_LAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          vecAcc2 = IVP_CVT48UN_2X64L(dvecAccLH, dvecAccLL);
          IVP_CVT48UN_2X64H(vecAcc2, dvecAccHH, dvecAccHL);

          // Loading accumulated values form W = 0, H = 1 spatial location initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remY * accDataPitch2));
          vaAcc        = IVP_LA2NX8_PP(pdvecAccData);
          IVP_LAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_LAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          vecAcc3 = IVP_CVT48UN_2X64L(dvecAccLH, dvecAccLL);
          IVP_CVT48UN_2X64H(vecAcc3, dvecAccHH, dvecAccHL);

          // Loading accumulated values form W = 1, H = 1 spatial location initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remX * accDataPitch1) + (remY * accDataPitch2));
          vaAcc        = IVP_LA2NX8_PP(pdvecAccData);
          IVP_LAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_LAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          vecAcc4 = IVP_CVT48UN_2X64L(dvecAccLH, dvecAccLL);
          IVP_CVT48UN_2X64H(vecAcc4, dvecAccHH, dvecAccHL);
        }

        for (ky = 0; ky < kHeightU; ky++)
        {
          // Adjusting the coefficient data pointer
          pvecCoeff = (xb_vecNx16 *) (pCoeffData + outCh + (ky * coeffDataPitch3));
          int16_t *pData2 = (int16_t *) (pData1 + (ky * inDataPitch2));
          // phvecIn1 initially points to W = 0, H = 0 spatial location
          phvecIn1 = (xb_vecN_2x32v *) (pData2);
          // phvecIn2 initially points to W = 1, H = 0 spatial location
          phvecIn2 = (xb_vecN_2x32v *) (pData2 + (remX * strideX * inDataPitch1));
          // phvecIn3 initially points to W = 0, H = 1 spatial location
          phvecIn3 = (xb_vecN_2x32v *) (pData2 + (remY * strideY * inDataPitch2));
          // phvecIn4 initially points to W = 1, H = 1 spatial location
          phvecIn4 = (xb_vecN_2x32v *) (pData2 + (remX * strideX * inDataPitch1) + (remY * strideY * inDataPitch2));

          vaIn1 = IVP_LAN_2X32_PP(phvecIn1);
          vaIn2 = IVP_LAN_2X32_PP(phvecIn2);
          vaIn3 = IVP_LAN_2X32_PP(phvecIn3);
          vaIn4 = IVP_LAN_2X32_PP(phvecIn4);

          for (iter = 0; iter < (numIter - 1); iter += 2)
          {
            // hvecIn1 contains 4 bytes or 2 elements along D from W = 0, H = 0 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn1, vaIn1, phvecIn1, 4);
            // hvecIn2 contains 4 bytes or 2 elements along D from W = 1, H = 0 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn2, vaIn2, phvecIn2, 4);
            // hvecIn2 contains 4 bytes or 2 elements along D from W = 0, H = 1 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn3, vaIn3, phvecIn3, 4);
            // hvecIn2 contains 4 bytes or 2 elements along D from W = 1, H = 1 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn4, vaIn4, phvecIn4, 4);

            // vecCoeff1 contains 64 bytes or 32 elements along output depth (N) from initial input depth (D = 0) initially
            IVP_L2UNX16_XP(vecCoeff1, pvecCoeff, (2 * coeffDataPitch1));
            // vecCoeff2 contains 64 bytes or 32 elements along output depth (N) from next input depth (D = 1) initially
            IVP_L2UNX16_XP(vecCoeff2, pvecCoeff, (2 * coeffDataPitch1));

            // vecAcc1 contains 64 bytes or 32 elements along output depth (N) from W = 0, H = 0 spatial location initially
            IVP_MULPAN16XR16(vecAcc1, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecIn1, 0));
            // vecAcc2 contains 64 bytes or 32 elements along output depth (N) from W = 1, H = 0 spatial location initially
            IVP_MULPAN16XR16(vecAcc2, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecIn2, 0));
            // vecAcc3 contains 64 bytes or 32 elements along output depth (N) from W = 0, H = 1 spatial location initially
            IVP_MULPAN16XR16(vecAcc3, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecIn3, 0));
            // vecAcc4 contains 64 bytes or 32 elements along output depth (N) from W = 1, H = 1 spatial location initially
            IVP_MULPAN16XR16(vecAcc4, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecIn4, 0));
          } // End of for (iter = 0; iter < (numIter - 1); iter += 2)
          if (iter < numIter)
          {
            // hvecIn1 contains 2 bytes or 1 element along D from W = 0, H = 0 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn1, vaIn1, phvecIn1, 2);
            // hvecIn2 contains 2 bytes or 1 element along D from W = 1, H = 0 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn2, vaIn2, phvecIn2, 2);
            // hvecIn2 contains 2 bytes or 1 element along D from W = 0, H = 1 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn3, vaIn3, phvecIn3, 2);
            // hvecIn2 contains 2 bytes or 1 element along D from W = 1, H = 1 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn4, vaIn4, phvecIn4, 2);

            // vecCoeff1 contains 64 bytes or 32 elements along output depth (N) from initial input depth (D = 0)
            IVP_L2UNX16_XP(vecCoeff1, pvecCoeff, (2 * coeffDataPitch1));

            // vecAcc1 contains 64 bytes or 32 elements along output depth (N) from W = 0, H = 0 spatial location initially
            IVP_MULPAN16XR16(vecAcc1, 0, vecCoeff1, IVP_EXTRN_2X32(hvecIn1, 0));
            // vecAcc2 contains 64 bytes or 32 elements along output depth (N) from W = 1, H = 0 spatial location initially
            IVP_MULPAN16XR16(vecAcc2, 0, vecCoeff1, IVP_EXTRN_2X32(hvecIn2, 0));
            // vecAcc3 contains 64 bytes or 32 elements along output depth (N) from W = 0, H = 1 spatial location initially
            IVP_MULPAN16XR16(vecAcc3, 0, vecCoeff1, IVP_EXTRN_2X32(hvecIn3, 0));
            // vecAcc4 contains 64 bytes or 32 elements along output depth (N) from W = 1, H = 1 spatial location initially
            IVP_MULPAN16XR16(vecAcc4, 0, vecCoeff1, IVP_EXTRN_2X32(hvecIn4, 0));
          }
        } // End of for (ky = 0; ky < kHeightU; ky++)

        if (outputFlag) // if "outputFlag" is set, apply pack, scale, shift, clamp logic on accumulated values and store the output
        {
          /* Pack, scale, shift, clamp logic to follow */
#ifdef DILATED_VQ_CONV_PARTIAL
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut1, vecAcc1, packShiftAccU, vecOutScaleU, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut2, vecAcc2, packShiftAccU, vecOutScaleU, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut3, vecAcc3, packShiftAccU, vecOutScaleU, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut4, vecAcc4, packShiftAccU, vecOutScaleU, outShiftU, minLim, maxLim);
#else
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1, vecAcc1, packShiftAccU, outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2, vecAcc2, packShiftAccU, outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3, vecAcc3, packShiftAccU, outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4, vecAcc4, packShiftAccU, outScale, outShiftU, minLim, maxLim);
#endif
          // Storing 64 bytes or 32 elements along output depth (N) at W = 0, H = 0 initially
          pvecOut = (xb_vecNx16 *) (pOutData + outCh + (x * outDataPitch1) + (y * outDataPitch2));
          IVP_SAVNX16_XP(vecOut1, vaOut, pvecOut, (2 * remOutCh));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);

          // Storing 64 bytes or 32 elements along output depth (N) at W = 1, H = 0 initially
          pvecOut = (xb_vecNx16 *) (pOutData + outCh + ((x + remX) * outDataPitch1) + (y * outDataPitch2));
          IVP_SAVNX16_XP(vecOut2, vaOut, pvecOut, (2 * remOutCh) * remX);
          IVP_SAPOSNX16_FP(vaOut, pvecOut);

          // Storing 64 bytes or 32 elements along output depth (N) at W = 0, H = 1 initially
          pvecOut = (xb_vecNx16 *) (pOutData + outCh + (x * outDataPitch1) + ((y + remY) * outDataPitch2));
          IVP_SAVNX16_XP(vecOut3, vaOut, pvecOut, (2 * remOutCh) * remY);
          IVP_SAPOSNX16_FP(vaOut, pvecOut);

          // Storing 64 bytes or 32 elements along output depth (N) at W = 1, H = 1 initially
          pvecOut = (xb_vecNx16 *) (pOutData + outCh + ((x + remX) * outDataPitch1) + ((y + remY) * outDataPitch2));
          IVP_SAVNX16_XP(vecOut4, vaOut, pvecOut, (2 * remOutCh) * remX * remY);
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
        else // if "outputFlag" is not-set, store the accumulated values to the accTile
        {
          vaAcc     = IVP_ZALIGN();
          dvecAccLL = IVP_CVT64SNX48LL(vecAcc1);
          dvecAccLH = IVP_CVT64SNX48LH(vecAcc1);
          dvecAccHL = IVP_CVT64SNX48HL(vecAcc1);
          dvecAccHH = IVP_CVT64SNX48HH(vecAcc1);
          // Storing 32 elements at W = 0, H = 0 initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc);
          IVP_SAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_SAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaAcc, pdvecAccData);

          dvecAccLL = IVP_CVT64SNX48LL(vecAcc2);
          dvecAccLH = IVP_CVT64SNX48LH(vecAcc2);
          dvecAccHL = IVP_CVT64SNX48HL(vecAcc2);
          dvecAccHH = IVP_CVT64SNX48HH(vecAcc2);
          // Storing 32 elements at W = 1, H = 0 initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remX * accDataPitch1));
          IVP_SAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_SAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaAcc, pdvecAccData);

          dvecAccLL = IVP_CVT64SNX48LL(vecAcc3);
          dvecAccLH = IVP_CVT64SNX48LH(vecAcc3);
          dvecAccHL = IVP_CVT64SNX48HL(vecAcc3);
          dvecAccHH = IVP_CVT64SNX48HH(vecAcc3);
          // Storing 32 elements at W = 0, H = 1 initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remY * accDataPitch2));
          IVP_SAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_SAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaAcc, pdvecAccData);

          dvecAccLL = IVP_CVT64SNX48LL(vecAcc4);
          dvecAccLH = IVP_CVT64SNX48LH(vecAcc4);
          dvecAccHL = IVP_CVT64SNX48HL(vecAcc4);
          dvecAccHH = IVP_CVT64SNX48HH(vecAcc4);
          // Storing 32 elements at W = 1, H = 1 initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remX * accDataPitch1) + (remY * accDataPitch2));
          IVP_SAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_SAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaAcc, pdvecAccData);
        }
      } // End of for (x = 0; x < outWidth; x += 2)
    }   // End of for (y = 0; y < outHeight; y += 2)
  }     // End of for (outCh = 0; outCh < numOutCh; outCh += XCHAL_IVPN_SIMD_WIDTH)
}

/***************** xaiPartialConvolvedVQ3D_S_MxN_S16S16I16_MOD_DWH *****************/
/****************** xaiPartialConvolved3D_S_MxN_S16S16I16_MOD_DWH ******************/
/***********************************************************************************/
/* Description : Optimized implementation for partial 3D convolution               */
/* Inputs      : Input Data Tile, Coeff Data Tile, Bias Array, Output Scale Array, */
/*               CNN convolution params structure                                  */
/* Outputs     : XI Error Code                                                     */
/* InOuts      : Accumulator Tile, Output Tile                                     */
/* Assumptions : InData, CoeffData are S16                                         */
/*               OutData is U16 / S16                                              */
/*               Input is in DWH and Output is in DWH format                       */
/*               Coeff is in NDWH format                                           */
/*               Accumulated value will be within 48-bit range                     */
/***********************************************************************************/
#ifdef DILATED_VQ_CONV_PARTIAL
XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             const xai_pArray outputScaleArray,
                                                             xai_pTile3D accTile,
                                                             xai_pTile3D outTile,
                                                             const xai_cnn_conv_params *param)
#else
XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                           const xai_pTile4D coeffTile,
                                                           const xai_pArray biasArray,
                                                           xai_pTile3D accTile,
                                                           xai_pTile3D outTile,
                                                           const xai_cnn_conv_params *param)
#endif
{
  XAI_ERROR_CHECKS()
  {
    XAI_CHECK_TILE3D_S16(inTile);
    XAI_CHECK_TILE4D_S16(coeffTile);
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
    XAI_CHECK_ERROR((XAI_CNN_CONV_GET_DILATION(param) == 1) ||                                                          \
                    ((XAI_CNN_CONV_GET_DILATION(param) >= 1) &&                                                         \
                     (XAI_CNN_CONV_GET_STRIDEX(param) == 1) && (XAI_CNN_CONV_GET_STRIDEY(param) == 1)), XAI_ERR_BADARG, \
                    "\nDilation = %hhu\nDilation should be 1. It can be greater than 1 only when stride is equal to 1", \
                    XAI_CNN_CONV_GET_DILATION(param));
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_DILATIONX(param) == XAI_CNN_CONV_GET_DILATIONY(param),                                             \
                    XAI_ERR_BADARG, "\nDilation along width = %hhu and height = %hhu\nDilation along width and height should be equal", \
                    XAI_CNN_CONV_GET_DILATIONX(param), XAI_CNN_CONV_GET_DILATIONY(param));
    XAI_CHECK_TILE3D_DATA_ORDER(inTile, XAI_DWH);
    XAI_CHECK_TILE4D_DATA_ORDER(coeffTile, XAI_NDWH);
    XAI_CHECK_TILE3D_DATA_ORDER(outTile, XAI_DWH);
    XAI_CHECK_ERROR(XAI_CNN_CONV_GET_ACCUM_SHIFT(param) < 32,                                     \
                    XAI_ERR_NORM, "\nThe accumulator shift = %hhu, value should be less than 32", \
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
      XAI_CHECK_ARRAY_S64(biasArray);
    }
    if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
    {
      XAI_CHECK_TILE3D_S64(accTile);
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, accTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, accTile);
      XAI_CHECK_TILE3D_DATA_ORDER(accTile, XAI_DWH);
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM1(accTile) >= XAI_TILE3D_GET_DIM1(outTile)), XAI_ERR_DATASIZE,         \
                      "\ndim1Size of accTile = %d, should be greater than or equal to %d(dim1Size of outTile)", \
                      XAI_TILE3D_GET_DIM1(accTile), XAI_TILE3D_GET_DIM1(outTile));
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM2(accTile) >= XAI_TILE3D_GET_DIM2(outTile)), XAI_ERR_DATASIZE,         \
                      "\ndim2Size of accTile = %d, should be greater than or equal to %d(dim2Size of outTile)", \
                      XAI_TILE3D_GET_DIM2(accTile), XAI_TILE3D_GET_DIM2(outTile));
      XAI_CHECK_ERROR((XAI_TILE3D_GET_DIM3(accTile) >= XAI_TILE3D_GET_DIM3(outTile)), XAI_ERR_DATASIZE,         \
                      "\ndim3Size of accTile = %d, should be greater than or equal to %d(dim3Size of outTile)", \
                      XAI_TILE3D_GET_DIM3(accTile), XAI_TILE3D_GET_DIM3(outTile));
    }
    if (XAI_CNN_CONV_GET_FLAG_OUTPUT(param))
    {
      XAI_CHECK_ERROR(XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) || XAI_TILE3D_CHECK_TYPE(outTile, XAI_U16), \
                      XAI_ERR_DATATYPE, "\nOutTile data type need to be either XAI_S16 or XAI_U16");
      XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(inTile, outTile);
      XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(coeffTile, outTile);
      if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param)))
      {
        XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(accTile, outTile);
      }
    }
  }

  const uint8_t dilationU = XAI_CNN_CONV_GET_DILATION(param);

  /* Calling further optimized variant based on certain conditions */
  if ((XAI_TILE3D_GET_DIM1(inTile) == XAI_TILE3D_GET_DIM1_PITCH(inTile)) && (dilationU == 1))
  {
#ifdef DILATED_VQ_CONV_PARTIAL
    partialConvolvedVQ3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth(inTile, coeffTile, biasArray, outputScaleArray, \
                                                                  accTile, outTile, param);
#else
    partialConvolved3D_S_MxN_S16S16I16_MOD_DWH_contiguous_depth(inTile, coeffTile, biasArray, \
                                                                accTile, outTile, param);
#endif

    return(XAI_ERROR_STATUS());
  }

  /* Getting parameters from the tile structures */
  const int32_t numInCh         = XAI_TILE3D_GET_DIM1(inTile);
  const int32_t numOutCh        = XAI_TILE3D_GET_DIM1(outTile);
  const int32_t outWidth        = XAI_TILE3D_GET_DIM2(outTile);
  const int32_t outHeight       = XAI_TILE3D_GET_DIM3(outTile);
  const int32_t inDataPitch1    = XAI_TILE3D_GET_DIM1_PITCH(inTile);
  const int32_t inDataPitch2    = XAI_TILE3D_GET_DIM2_PITCH(inTile);
  const int32_t outDataPitch1   = XAI_TILE3D_GET_DIM1_PITCH(outTile);
  const int32_t outDataPitch2   = XAI_TILE3D_GET_DIM2_PITCH(outTile);
  const int32_t coeffDataPitch1 = XAI_TILE4D_GET_DIM1_PITCH(coeffTile);
  const int32_t coeffDataPitch2 = XAI_TILE4D_GET_DIM2_PITCH(coeffTile);
  const int32_t coeffDataPitch3 = XAI_TILE4D_GET_DIM3_PITCH(coeffTile);
  const int32_t kWidthU         = XAI_TILE4D_GET_DIM3(coeffTile);
  const int32_t kHeightU        = XAI_TILE4D_GET_DIM4(coeffTile);

  /* Convolution params */
  const uint8_t packShiftAccU = XAI_CNN_CONV_GET_ACCUM_SHIFT(param);
  const uint8_t outShiftU     = XAI_CNN_CONV_GET_OUTPUT_SHIFT(param);
  const uint8_t enableReLu    = XAI_CNN_CONV_GET_FLAG_RELU(param);
  const uint8_t strideX       = XAI_CNN_CONV_GET_STRIDEX(param);
  const uint8_t strideY       = XAI_CNN_CONV_GET_STRIDEY(param);
  const uint8_t leftEdgeFlag  = XAI_CNN_CONV_GET_FLAG_LEFTEDGE(param);
  const uint8_t topEdgeFlag   = XAI_CNN_CONV_GET_FLAG_TOPEDGE(param);
  const uint8_t inputFlag     = XAI_CNN_CONV_GET_FLAG_INPUT(param);
  const uint8_t outputFlag    = XAI_CNN_CONV_GET_FLAG_OUTPUT(param);

#ifdef DILATED_VQ_CONV_PARTIAL
  const uint16_t *pOutputScaleData = (uint16_t *) XAI_ARRAY_GET_DATA_PTR(outputScaleArray);
#else
  const uint16_t outScale = XAI_CNN_CONV_GET_OUTPUT_SCALE(param);
#endif

  /* Data Pointers of input, coefficient, biasData */
  const int16_t *pInData    = (int16_t *) XAI_TILE3D_GET_DATA_PTR(inTile);
  const int16_t *pCoeffData = (int16_t *) XAI_TILE4D_GET_DATA_PTR(coeffTile);
  const int64_t *pBiasData  = (int64_t *) XAI_ARRAY_GET_DATA_PTR(biasArray);

  /* Data Pointers of output and scratch buffer data */
  int16_t *pOutData = (int16_t *) XAI_TILE3D_GET_DATA_PTR(outTile);
  int64_t *pAccData = NULL;

  int32_t accDataPitch1 = 0;
  int32_t accDataPitch2 = 0;

  if (!(XAI_CNN_CONV_GET_FLAG_INPUT(param) && XAI_CNN_CONV_GET_FLAG_OUTPUT(param)))
  {
    pAccData      = (int64_t *) XAI_TILE3D_GET_DATA_PTR(accTile);
    accDataPitch1 = XAI_TILE3D_GET_DIM1_PITCH(accTile);
    accDataPitch2 = XAI_TILE3D_GET_DIM2_PITCH(accTile);
  }

  int32_t dilatedKWidthU  = dilationU * (kWidthU - 1) + 1;
  int32_t dilatedKHeightU = dilationU * (kHeightU - 1) + 1;
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

  /* move to start of edge data only when input is already padded. */
  pInData = &pInData[-(int32_t) ((topEdge) * inDataPitch2 + (leftEdge) * inDataPitch1)];

  /* Setting the limits for output data according to ReLu is enabled or not*/
  int32_t minLim, maxLim;

  if (enableReLu)
  {
    minLim = XAI_CNN_CONV_GET_RELU_MIN(param);
    maxLim = XAI_CNN_CONV_GET_RELU_MAX(param);
  }
  else
  {
    minLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MIN : 0);
    maxLim = (XAI_TILE3D_CHECK_TYPE(outTile, XAI_S16) ? SHRT_MAX : USHRT_MAX);
  }

  int32_t inCh, outCh, x, y, k;

  xb_vecN_2x32v *restrict phvecIn1;
  xb_vecN_2x32v *restrict phvecIn2;
  xb_vecN_2x32v *restrict phvecIn3;
  xb_vecN_2x32v *restrict phvecIn4;
  xb_vecNx16 *restrict pvecCoeff;
  xb_vec2Nx8 *restrict pdvecBias;
  xb_vec2Nx8 *restrict pdvecAccData;
  xb_vecNx16 *restrict pvecOut;

  xb_vecNx48 vecAcc1 = 0, vecAcc2 = 0, vecAcc3 = 0, vecAcc4 = 0, vecBias = 0;
  xb_vecN_2x32v hvecIn1, hvecIn2, hvecIn3, hvecIn4;
  xb_vecNx16 vecCoeff1, vecCoeff2;
  xb_vecNx16 vecOut1, vecOut2, vecOut3, vecOut4;
  xb_vec2Nx8 dvecAccLL, dvecAccLH, dvecAccHL, dvecAccHH;

  valign vaIn1, vaIn2, vaIn3, vaIn4, vaBias, vaAcc;

#ifdef DILATED_VQ_CONV_PARTIAL
  xb_vecNx16U vecOutScaleU;
  xb_vecNx16U *restrict pvecOutScaleData = (xb_vecNx16U *) (pOutputScaleData);
  valign vaScale                         = IVP_LANX16U_PP(pvecOutScaleData);
#endif

  pdvecBias = (xb_vec2Nx8 *) (pBiasData);
  vaBias    = IVP_LA2NX8_PP(pdvecBias);
  valign vaOut = IVP_ZALIGN();

  for (outCh = 0; outCh < numOutCh; outCh += XCHAL_IVPN_SIMD_WIDTH)
  {
    int32_t remOutCh = (numOutCh - outCh);
    /* Initially the accumulators with the 48-bit bias values */
    if (inputFlag) // Biases will be loaded only when "inputFlag" is set
    {
      ACC_INIT_BIAS64_MOD_ONEACC(pdvecBias, vaBias, remOutCh, vecBias);
    }

#ifdef DILATED_VQ_CONV_PARTIAL
    IVP_LAVNX16U_XP(vecOutScaleU, vaScale, pvecOutScaleData, 2 * remOutCh);
#endif

    for (y = 0; y < outHeight; y += 2)
    {
      // Calculating "remY" for integrated tail-handling purpose
      int32_t remY = XT_MIN(1, outHeight - y - 1);
      for (x = 0; x < outWidth; x += 2)
      {
        // Calculating "remX" for integrated tail-handling purpose
        int32_t remX    = XT_MIN(1, outWidth - x - 1);
        int16_t *pData1 = (int16_t *) (pInData + (x * strideX * inDataPitch1) + (y * strideY * inDataPitch2));
        int64_t *pAcc   = (int64_t *) (pAccData + outCh + (x * accDataPitch1) + (y * accDataPitch2));

        if (inputFlag) // if "inputFlag" is set, then initialize the accumulators with the bias values
        {
          /* Initializing all the 4 accumulators with bias values before accumulating for every spatial location */
          vecAcc4 = vecAcc3 = vecAcc2 = vecAcc1 = vecBias;
        }
        else // if "inputFlag" is not-set, then initialize the accumulators with the values stored in the accTile
        {
          // Loading accumulated values from W = 0, H = 0 spatial location initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc);
          vaAcc        = IVP_LA2NX8_PP(pdvecAccData);
          IVP_LAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_LAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          vecAcc1 = IVP_CVT48UN_2X64L(dvecAccLH, dvecAccLL);
          IVP_CVT48UN_2X64H(vecAcc1, dvecAccHH, dvecAccHL);

          // Loading accumulated values form W = 1, H = 0 spatial location initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remX * accDataPitch1));
          vaAcc        = IVP_LA2NX8_PP(pdvecAccData);
          IVP_LAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_LAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          vecAcc2 = IVP_CVT48UN_2X64L(dvecAccLH, dvecAccLL);
          IVP_CVT48UN_2X64H(vecAcc2, dvecAccHH, dvecAccHL);

          // Loading accumulated values form W = 0, H = 1 spatial location initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remY * accDataPitch2));
          vaAcc        = IVP_LA2NX8_PP(pdvecAccData);
          IVP_LAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_LAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          vecAcc3 = IVP_CVT48UN_2X64L(dvecAccLH, dvecAccLL);
          IVP_CVT48UN_2X64H(vecAcc3, dvecAccHH, dvecAccHL);

          // Loading accumulated values form W = 1, H = 1 spatial location initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remX * accDataPitch1) + (remY * accDataPitch2));
          vaAcc        = IVP_LA2NX8_PP(pdvecAccData);
          IVP_LAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_LAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_LAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          vecAcc4 = IVP_CVT48UN_2X64L(dvecAccLH, dvecAccLL);
          IVP_CVT48UN_2X64H(vecAcc4, dvecAccHH, dvecAccHL);
        }

        for (k = 0; k < kWidthU * kHeightU; k++)
        {
          // Adjusting the coefficient data pointer
          pvecCoeff = (xb_vecNx16 *) (pCoeffData + outCh + ((k % kWidthU) * coeffDataPitch2) + ((k / kWidthU) * coeffDataPitch3));
          int16_t *pData2 = (int16_t *) (pData1 + (((k % kWidthU) * dilationU) * inDataPitch1) + (((k / kWidthU) * dilationU) * inDataPitch2));
          // phvecIn1 initially points to W = 0, H = 0 spatial location
          phvecIn1 = (xb_vecN_2x32v *) (pData2);
          // phvecIn2 initially points to W = 1, H = 0 spatial location
          phvecIn2 = (xb_vecN_2x32v *) (pData2 + (remX * strideX * inDataPitch1));
          // phvecIn3 initially points to W = 0, H = 1 spatial location
          phvecIn3 = (xb_vecN_2x32v *) (pData2 + (remY * strideY * inDataPitch2));
          // phvecIn4 initially points to W = 1, H = 1 spatial location
          phvecIn4 = (xb_vecN_2x32v *) (pData2 + (remX * strideX * inDataPitch1) + (remY * strideY * inDataPitch2));

          vaIn1 = IVP_LAN_2X32_PP(phvecIn1);
          vaIn2 = IVP_LAN_2X32_PP(phvecIn2);
          vaIn3 = IVP_LAN_2X32_PP(phvecIn3);
          vaIn4 = IVP_LAN_2X32_PP(phvecIn4);

          for (inCh = 0; inCh < (numInCh - 1); inCh += 2)
          {
            // hvecIn1 contains 4 bytes or 2 elements along D from W = 0, H = 0 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn1, vaIn1, phvecIn1, 4);
            // hvecIn2 contains 4 bytes or 2 elements along D from W = 1, H = 0 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn2, vaIn2, phvecIn2, 4);
            // hvecIn2 contains 4 bytes or 2 elements along D from W = 0, H = 1 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn3, vaIn3, phvecIn3, 4);
            // hvecIn2 contains 4 bytes or 2 elements along D from W = 1, H = 1 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn4, vaIn4, phvecIn4, 4);

            // vecCoeff1 contains 64 bytes or 32 elements along output depth (N) from initial input depth (D = 0) initially
            IVP_L2UNX16_XP(vecCoeff1, pvecCoeff, (2 * coeffDataPitch1));
            // vecCoeff2 contains 64 bytes or 32 elements along output depth (N) from next input depth (D = 1) initially
            IVP_L2UNX16_XP(vecCoeff2, pvecCoeff, (2 * coeffDataPitch1));

            // vecAcc1 contains 64 bytes or 32 elements along output depth (N) from W = 0, H = 0 spatial location initially
            IVP_MULPAN16XR16(vecAcc1, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecIn1, 0));
            // vecAcc2 contains 64 bytes or 32 elements along output depth (N) from W = 1, H = 0 spatial location initially
            IVP_MULPAN16XR16(vecAcc2, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecIn2, 0));
            // vecAcc3 contains 64 bytes or 32 elements along output depth (N) from W = 0, H = 1 spatial location initially
            IVP_MULPAN16XR16(vecAcc3, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecIn3, 0));
            // vecAcc4 contains 64 bytes or 32 elements along output depth (N) from W = 1, H = 1 spatial location initially
            IVP_MULPAN16XR16(vecAcc4, vecCoeff2, vecCoeff1, IVP_EXTRN_2X32(hvecIn4, 0));
          } // End of for (inCh = 0; inCh < numInCh; inCh += 2)

          if (inCh < numInCh)
          {
            // hvecIn1 contains 2 bytes or 1 element along D from W = 0, H = 0 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn1, vaIn1, phvecIn1, 2);
            // hvecIn2 contains 2 bytes or 1 element along D from W = 1, H = 0 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn2, vaIn2, phvecIn2, 2);
            // hvecIn2 contains 2 bytes or 1 element along D from W = 0, H = 1 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn3, vaIn3, phvecIn3, 2);
            // hvecIn2 contains 2 bytes or 1 element along D from W = 1, H = 1 spatial location initially
            IVP_LAVN_2X32_XP(hvecIn4, vaIn4, phvecIn4, 2);

            // vecCoeff1 contains 64 bytes or 32 elements along output depth (N) from initial input depth (D = 0)
            IVP_L2UNX16_XP(vecCoeff1, pvecCoeff, (2 * coeffDataPitch1));

            // vecAcc1 contains 64 bytes or 32 elements along output depth (N) from W = 0, H = 0 spatial location initially
            IVP_MULPAN16XR16(vecAcc1, 0, vecCoeff1, IVP_EXTRN_2X32(hvecIn1, 0));
            // vecAcc2 contains 64 bytes or 32 elements along output depth (N) from W = 1, H = 0 spatial location initially
            IVP_MULPAN16XR16(vecAcc2, 0, vecCoeff1, IVP_EXTRN_2X32(hvecIn2, 0));
            // vecAcc3 contains 64 bytes or 32 elements along output depth (N) from W = 0, H = 1 spatial location initially
            IVP_MULPAN16XR16(vecAcc3, 0, vecCoeff1, IVP_EXTRN_2X32(hvecIn3, 0));
            // vecAcc4 contains 64 bytes or 32 elements along output depth (N) from W = 1, H = 1 spatial location initially
            IVP_MULPAN16XR16(vecAcc4, 0, vecCoeff1, IVP_EXTRN_2X32(hvecIn4, 0));
          }
        }   // End of for (k = 0; k < kWidthU * kHeightU; k++)

        if (outputFlag) // if "outputFlag" is set, apply pack, scale, shift, clamp logic on accumulated values and store the output
        {
          /* Pack, scale, shift, clamp logic to follow */
#ifdef DILATED_VQ_CONV_PARTIAL
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut1, vecAcc1, packShiftAccU, vecOutScaleU, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut2, vecAcc2, packShiftAccU, vecOutScaleU, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut3, vecAcc3, packShiftAccU, vecOutScaleU, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut4, vecAcc4, packShiftAccU, vecOutScaleU, outShiftU, minLim, maxLim);
#else
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut1, vecAcc1, packShiftAccU, outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut2, vecAcc2, packShiftAccU, outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut3, vecAcc3, packShiftAccU, outScale, outShiftU, minLim, maxLim);
          PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut4, vecAcc4, packShiftAccU, outScale, outShiftU, minLim, maxLim);
#endif
          // Storing 64 bytes or 32 elements along output depth (N) at W = 0, H = 0 initially
          pvecOut = (xb_vecNx16 *) (pOutData + outCh + (x * outDataPitch1) + (y * outDataPitch2));
          IVP_SAVNX16_XP(vecOut1, vaOut, pvecOut, (2 * remOutCh));
          IVP_SAPOSNX16_FP(vaOut, pvecOut);

          // Storing 64 bytes or 32 elements along output depth (N) at W = 1, H = 0 initially
          pvecOut = (xb_vecNx16 *) (pOutData + outCh + ((x + remX) * outDataPitch1) + (y * outDataPitch2));
          IVP_SAVNX16_XP(vecOut2, vaOut, pvecOut, (2 * remOutCh) * remX);
          IVP_SAPOSNX16_FP(vaOut, pvecOut);

          // Storing 64 bytes or 32 elements along output depth (N) at W = 0, H = 1 initially
          pvecOut = (xb_vecNx16 *) (pOutData + outCh + (x * outDataPitch1) + ((y + remY) * outDataPitch2));
          IVP_SAVNX16_XP(vecOut3, vaOut, pvecOut, (2 * remOutCh) * remY);
          IVP_SAPOSNX16_FP(vaOut, pvecOut);

          // Storing 64 bytes or 32 elements along output depth (N) at W = 1, H = 1 initially
          pvecOut = (xb_vecNx16 *) (pOutData + outCh + ((x + remX) * outDataPitch1) + ((y + remY) * outDataPitch2));
          IVP_SAVNX16_XP(vecOut4, vaOut, pvecOut, (2 * remOutCh) * remX * remY);
          IVP_SAPOSNX16_FP(vaOut, pvecOut);
        }
        else // if "outputFlag" is not-set, store the accumulated values to the accTile
        {
          vaAcc     = IVP_ZALIGN();
          dvecAccLL = IVP_CVT64SNX48LL(vecAcc1);
          dvecAccLH = IVP_CVT64SNX48LH(vecAcc1);
          dvecAccHL = IVP_CVT64SNX48HL(vecAcc1);
          dvecAccHH = IVP_CVT64SNX48HH(vecAcc1);
          // Storing 32 elements at W = 0, H = 0 initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc);
          IVP_SAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_SAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaAcc, pdvecAccData);

          dvecAccLL = IVP_CVT64SNX48LL(vecAcc2);
          dvecAccLH = IVP_CVT64SNX48LH(vecAcc2);
          dvecAccHL = IVP_CVT64SNX48HL(vecAcc2);
          dvecAccHH = IVP_CVT64SNX48HH(vecAcc2);
          // Storing 32 elements at W = 1, H = 0 initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remX * accDataPitch1));
          IVP_SAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_SAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaAcc, pdvecAccData);

          dvecAccLL = IVP_CVT64SNX48LL(vecAcc3);
          dvecAccLH = IVP_CVT64SNX48LH(vecAcc3);
          dvecAccHL = IVP_CVT64SNX48HL(vecAcc3);
          dvecAccHH = IVP_CVT64SNX48HH(vecAcc3);
          // Storing 32 elements at W = 0, H = 1 initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remY * accDataPitch2));
          IVP_SAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_SAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaAcc, pdvecAccData);

          dvecAccLL = IVP_CVT64SNX48LL(vecAcc4);
          dvecAccLH = IVP_CVT64SNX48LH(vecAcc4);
          dvecAccHL = IVP_CVT64SNX48HL(vecAcc4);
          dvecAccHH = IVP_CVT64SNX48HH(vecAcc4);
          // Storing 32 elements at W = 1, H = 1 initially
          pdvecAccData = (xb_vec2Nx8 *) (pAcc + (remX * accDataPitch1) + (remY * accDataPitch2));
          IVP_SAV2NX8_XP(dvecAccLL, vaAcc, pdvecAccData, (8 * remOutCh));
          IVP_SAV2NX8_XP(dvecAccLH, vaAcc, pdvecAccData, (8 * remOutCh) - (2 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHL, vaAcc, pdvecAccData, (8 * remOutCh) - (4 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAV2NX8_XP(dvecAccHH, vaAcc, pdvecAccData, (8 * remOutCh) - (6 * XCHAL_IVPN_SIMD_WIDTH));
          IVP_SAPOS2NX8_FP(vaAcc, pdvecAccData);
        }
      } // End of for (x = 0; x < outWidth; x += 2)
    }   // End of for (y = 0; y < outHeight; y += 2)
  }     // End of for (outCh = 0; outCh < numOutCh; outCh += XCHAL_IVPN_SIMD_WIDTH)

  return(XAI_ERROR_STATUS());
}
#endif // #if ((XCHAL_VISION_TYPE >= 6))
