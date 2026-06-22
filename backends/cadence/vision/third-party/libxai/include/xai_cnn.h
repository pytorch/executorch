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

#ifndef __XAI_CNN_H__
#define __XAI_CNN_H__

#include "xai_cnn_api.h"
#include "xai_cnn_common.h"
#include "xai_tile_manager.h"
#include "xai_core.h"
#include "limits.h"

/****************************************************************************/
/* MACROS :                                                                 */
/* Macro for Packing the accumulator output after convolution, scaling it,  */
/* shifting and clamping the final output between min and max limits        */
/****************************************************************************/
#define PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1, dvecOut2, daccSum, packSA, outScaleDataEven, outScaleDataOdd, outSh, min, max, flag)  { \
    xb_vecNx16 m_outEven = IVP_PACKVR2NX24_0(daccSum, packSA);                                                                             \
    xb_vecNx16 m_outOdd  = IVP_PACKVR2NX24_1(daccSum, packSA);                                                                             \
    xb_vecNx48 m_wvec    = IVP_MULUSNX16(outScaleDataEven, m_outEven);                                                                     \
    m_outEven = IVP_PACKVRNX48(m_wvec, outSh);                                                                                             \
    m_wvec    = IVP_MULUSNX16(outScaleDataOdd, m_outOdd);                                                                                  \
    m_outOdd  = IVP_PACKVRNX48(m_wvec, outSh);                                                                                             \
    m_outEven = IVP_MAXNX16(IVP_MINNX16(m_outEven, (xb_vecNx16) max), (xb_vecNx16) min);                                                   \
    m_outOdd  = IVP_MAXNX16(IVP_MINNX16(m_outOdd, (xb_vecNx16) max), (xb_vecNx16) min);                                                    \
    xb_vec2Nx8 m_dvec = IVP_SEL2NX8I(IVP_MOV2NX8_FROMNX16(m_outOdd),                                                                       \
                                     IVP_MOV2NX8_FROMNX16(m_outEven),                                                                      \
                                     IVP_SELI_8B_INTERLEAVE_1_EVEN);                                                                       \
    IVP_DSEL2NX8I(dvecOut2, dvecOut1, IVP_MOV2NX8_FROMNX16(m_outOdd),                                                                      \
                  IVP_MOV2NX8_FROMNX16(m_outEven),                                                                                         \
                  IVP_DSELI_INTERLEAVE_1);                                                                                                 \
    dvecOut1 = IVP_MOV2NX8T(dvecOut1, m_dvec, IVP_EQ2NX8((xb_vec2Nx8) flag, 1));                                                           \
}

#define PACK_SCALE_SHIFT_CLAMP_LIMITS(dvecOut1, dvecOut2, daccSum, packSA, outSc, outSh, min, max, flag) \
  PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ(dvecOut1, dvecOut2, daccSum, packSA, outSc, outSc, outSh, min, max, flag)

/****************************************************************************/
/* MACROS :                                                                 */
/* Macro for Packing the accumulator output after convolution, scaling it,  */
/* shifting and clamping the final output between min and max limits        */
/****************************************************************************/
#define PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut, accSum, packSA, outSc, outSh, min, max)  {       \
    vecOut = IVP_PACKVRNX48(accSum, packSA);                                                       \
    xb_vecNx48 m_wvec       = IVP_MULUSNX16(outSc, vecOut);                                        \
    xb_vecN_2x32v m_outEven = IVP_PACKVRNX48_0(m_wvec, outSh);                                     \
    xb_vecN_2x32v m_outOdd  = IVP_PACKVRNX48_1(m_wvec, outSh);                                     \
    m_outEven = IVP_MAXN_2X32(IVP_MINN_2X32(m_outEven, (xb_vecN_2x32v) max), (xb_vecN_2x32v) min); \
    m_outOdd  = IVP_MAXN_2X32(IVP_MINN_2X32(m_outOdd, (xb_vecN_2x32v) max), (xb_vecN_2x32v) min);  \
    vecOut    = IVP_SELNX16I(IVP_MOVNX16_FROMN_2X32(m_outOdd),                                     \
                             IVP_MOVNX16_FROMN_2X32(m_outEven),                                    \
                             IVP_SELI_INTERLEAVE_1_EVEN);                                          \
}

#define PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_S16(vecOut, accSum, packSA, vecScaleData, outSh, min, max) \
  PACK_SCALE_SHIFT_CLAMP_LIMITS_S16(vecOut, accSum, packSA, vecScaleData, outSh, min, max)

#define PACK_SCALE_SHIFT_CLAMP_LIMITS_QM32(dvecOut1, dvecOut2, hvecSumLL, hvecSumLH, hvecSumHL, hvecSumHH, packSA, outSc, outSh, min, max, flag)                            { \
    xb_vecNx48 vecSumL = IVP_CVT48SNX32(hvecSumLH, hvecSumLL);                                                                                                                \
    xb_vecNx48 vecSumH = IVP_CVT48SNX32(hvecSumHH, hvecSumHL);                                                                                                                \
    xb_vecNx16 m_outL  = IVP_PACKVRNX48(vecSumL, packSA);                                                                                                                     \
    xb_vecNx16 m_outH  = IVP_PACKVRNX48(vecSumH, packSA);                                                                                                                     \
    xb_vecNx48 m_wvec  = IVP_MULUSNX16((xb_vecNx16U) outSc, m_outL);                                                                                                          \
    m_outL = IVP_PACKVRNX48(m_wvec, outSh);                                                                                                                                   \
    m_outL = IVP_MAXNX16(IVP_MINNX16(m_outL, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);                                                                                      \
    m_wvec = IVP_MULUSNX16((xb_vecNx16U) outSc, m_outH);                                                                                                                      \
    m_outH = IVP_PACKVRNX48(m_wvec, outSh);                                                                                                                                   \
    m_outH = IVP_MAXNX16(IVP_MINNX16(m_outH, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);                                                                                      \
    xb_vec2Nx8 m_dvec = IVP_SEL2NX8I(IVP_MOV2NX8_FROMNX16(m_outH), IVP_MOV2NX8_FROMNX16(m_outL), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);                                           \
    dvecOut1 = IVP_MOV2NX8_FROMNX16(m_outL);                                                                                                                                  \
    dvecOut1 = IVP_MOV2NX8T(dvecOut1, m_dvec, IVP_EQ2NX8(flag, 1));                                                                                                           \
    dvecOut2 = IVP_MOV2NX8_FROMNX16(m_outH);                                                                                                                                  \
}

#define PACK_SCALE_SHIFT_CLAMP_LIMITS_VQ_QM32(dvecOut1, dvecOut2, hvecSumLL, hvecSumLH, hvecSumHL, hvecSumHH, packSA, outScaleDataL, outScaleDataH, outSh, min, max, flag)  { \
    xb_vecNx48 vecSumL = IVP_CVT48SNX32(hvecSumLH, hvecSumLL);                                                                                                                \
    xb_vecNx48 vecSumH = IVP_CVT48SNX32(hvecSumHH, hvecSumHL);                                                                                                                \
    xb_vecNx16 m_outL  = IVP_PACKVRNX48(vecSumL, packSA);                                                                                                                     \
    xb_vecNx16 m_outH  = IVP_PACKVRNX48(vecSumH, packSA);                                                                                                                     \
    xb_vecNx48 m_wvec  = IVP_MULUSNX16((xb_vecNx16U) outScaleDataL, m_outL);                                                                                                  \
    m_outL = IVP_PACKVRNX48(m_wvec, outSh);                                                                                                                                   \
    m_outL = IVP_MAXNX16(IVP_MINNX16(m_outL, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);                                                                                      \
    m_wvec = IVP_MULUSNX16((xb_vecNx16U) outScaleDataH, m_outH);                                                                                                              \
    m_outH = IVP_PACKVRNX48(m_wvec, outSh);                                                                                                                                   \
    m_outH = IVP_MAXNX16(IVP_MINNX16(m_outH, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);                                                                                      \
    xb_vec2Nx8 m_dvec = IVP_SEL2NX8I(IVP_MOV2NX8_FROMNX16(m_outH), IVP_MOV2NX8_FROMNX16(m_outL), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);                                           \
    dvecOut1 = IVP_MOV2NX8_FROMNX16(m_outL);                                                                                                                                  \
    dvecOut1 = IVP_MOV2NX8T(dvecOut1, m_dvec, IVP_EQ2NX8(flag, 1));                                                                                                           \
    dvecOut2 = IVP_MOV2NX8_FROMNX16(m_outH);                                                                                                                                  \
}

/****************************************************************************/
/* MACROS :                                                                 */
/* Macro for Packing the accumulator output after convolution, scaling it,  */
/* shifting and clamping the final output between min and max limits        */
/****************************************************************************/
#define PACK_SCALE_SHIFT_CLAMP_LIMITS_IXS16(dvecOut1, dvecOut2, hvecSumLL, hvecSumLH, hvecSumHL, hvecSumHH, packSA, outScaleDataL, outScaleDataH, outSh, min, max, flag, sel)  { \
    xb_vecNx16 hvecSum1, hvecSum2, hvecSum3, hvecSum4;                                                                                                                           \
    IVP_DSELNX16(hvecSum3, hvecSum1, IVP_MOVNX16_FROMN_2X32(hvecSumLH), IVP_MOVNX16_FROMN_2X32(hvecSumLL), sel);                                                                 \
    IVP_DSELNX16(hvecSum4, hvecSum2, IVP_MOVNX16_FROMN_2X32(hvecSumHH), IVP_MOVNX16_FROMN_2X32(hvecSumHL), sel);                                                                 \
    xb_vecNx48 vecSumL = IVP_CVT48SNX32(IVP_MOVN_2X32_FROMNX16(hvecSum2), IVP_MOVN_2X32_FROMNX16(hvecSum1));                                                                     \
    xb_vecNx48 vecSumH = IVP_CVT48SNX32(IVP_MOVN_2X32_FROMNX16(hvecSum4), IVP_MOVN_2X32_FROMNX16(hvecSum3));                                                                     \
    xb_vecNx16 m_outL  = IVP_PACKVRNX48(vecSumL, packSA);                                                                                                                        \
    xb_vecNx16 m_outH  = IVP_PACKVRNX48(vecSumH, packSA);                                                                                                                        \
    xb_vecNx48 m_wvec  = IVP_MULUSNX16((xb_vecNx16U) outScaleDataL, m_outL);                                                                                                     \
    m_outL = IVP_PACKVRNX48(m_wvec, outSh);                                                                                                                                      \
    m_outL = IVP_MAXNX16(IVP_MINNX16(m_outL, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);                                                                                         \
    m_wvec = IVP_MULUSNX16((xb_vecNx16U) outScaleDataH, m_outH);                                                                                                                 \
    m_outH = IVP_PACKVRNX48(m_wvec, outSh);                                                                                                                                      \
    m_outH = IVP_MAXNX16(IVP_MINNX16(m_outH, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);                                                                                         \
    xb_vec2Nx8 m_dvec = IVP_SEL2NX8I(IVP_MOV2NX8_FROMNX16(m_outH), IVP_MOV2NX8_FROMNX16(m_outL), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);                                              \
    dvecOut1 = IVP_MOV2NX8_FROMNX16(m_outL);                                                                                                                                     \
    dvecOut1 = IVP_MOV2NX8T(dvecOut1, m_dvec, IVP_EQ2NX8(flag, 1));                                                                                                              \
    dvecOut2 = IVP_MOV2NX8_FROMNX16(m_outH);                                                                                                                                     \
}

#define PACK_SCALE_SHIFT_CLAMP_LIMITS_S16S8(dvecOut1, dvecOut2, hvecSumLL, hvecSumLH, hvecSumHL, hvecSumHH, packSA, outScaleDataL, outScaleDataH, outSh, min, max, flag)       { \
    xb_vecNx48 vecSumL = IVP_CVT48SNX32(hvecSumLH, hvecSumLL);                                                                                                                   \
    xb_vecNx48 vecSumH = IVP_CVT48SNX32(hvecSumHH, hvecSumHL);                                                                                                                   \
    xb_vecNx16 m_outL  = IVP_PACKVRNX48(vecSumL, packSA);                                                                                                                        \
    xb_vecNx16 m_outH  = IVP_PACKVRNX48(vecSumH, packSA);                                                                                                                        \
    xb_vecNx48 m_wvec  = IVP_MULUSNX16((xb_vecNx16U) outScaleDataL, m_outL);                                                                                                     \
    m_outL = IVP_PACKVRNX48(m_wvec, outSh);                                                                                                                                      \
    m_outL = IVP_MAXNX16(IVP_MINNX16(m_outL, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);                                                                                         \
    m_wvec = IVP_MULUSNX16((xb_vecNx16U) outScaleDataH, m_outH);                                                                                                                 \
    m_outH = IVP_PACKVRNX48(m_wvec, outSh);                                                                                                                                      \
    m_outH = IVP_MAXNX16(IVP_MINNX16(m_outH, (xb_vecNx16) maxLim), (xb_vecNx16) minLim);                                                                                         \
    xb_vec2Nx8 m_dvec = IVP_SEL2NX8I(IVP_MOV2NX8_FROMNX16(m_outH), IVP_MOV2NX8_FROMNX16(m_outL), IVP_SELI_8B_EXTRACT_1_OF_2_OFF_0);                                              \
    dvecOut1 = IVP_MOV2NX8_FROMNX16(m_outL);                                                                                                                                     \
    dvecOut1 = IVP_MOV2NX8T(dvecOut1, m_dvec, IVP_EQ2NX8(flag, 1));                                                                                                              \
    dvecOut2 = IVP_MOV2NX8_FROMNX16(m_outH);                                                                                                                                     \
}

/****************************************************************************/
/* MACROS :                                                                 */
/* Macro for Packing the 24- bit accumulator output to 16-bit               */
/* shifting and clamping the final output between min and max limits        */
/****************************************************************************/

#define PACK_SCALE_SHIFT_S24_S16(accdotProd, scale1, accShift1,                           \
                                 vecClampL, vecClampH, vecScale1L, vecScale1H, shift1)  { \
    xb_vecN_2x32v vecaccHH = IVP_CVT32S2NX24HH(accdotProd);                               \
    xb_vecN_2x32v vecaccHL = IVP_CVT32S2NX24HL(accdotProd);                               \
    xb_vecN_2x32v vecaccLH = IVP_CVT32S2NX24LH(accdotProd);                               \
    xb_vecN_2x32v vecaccLL = IVP_CVT32S2NX24LL(accdotProd);                               \
    xb_vecN_2x64w haccA, haccB, haccC, haccD;                                             \
    haccA = IVP_MULN_2X16X32_0(scale1, vecaccLL);                                         \
    haccB = IVP_MULN_2X16X32_0(scale1, vecaccLH);                                         \
    haccC = IVP_MULN_2X16X32_0(scale1, vecaccHL);                                         \
    haccD = IVP_MULN_2X16X32_0(scale1, vecaccHH);                                         \
    xb_vecN_2x32v hvec0LL = IVP_PACKVRN_2X64W(haccA, accShift1);                          \
    xb_vecN_2x32v hvec0LH = IVP_PACKVRN_2X64W(haccB, accShift1);                          \
    xb_vecN_2x32v hvec0HL = IVP_PACKVRN_2X64W(haccC, accShift1);                          \
    xb_vecN_2x32v hvec0HH = IVP_PACKVRN_2X64W(haccD, accShift1);                          \
    xb_vecNx48 accA       = IVP_CVT48SNX32(hvec0LH, hvec0LL);                             \
    xb_vecNx48 accB       = IVP_CVT48SNX32(hvec0HH, hvec0HL);                             \
    vecClampL  = IVP_PACKVRNX48(accA, 0);                                                 \
    vecClampH  = IVP_PACKVRNX48(accB, 0);                                                 \
    accdotProd = IVP_CVT24S2NX16(vecClampH, vecClampL);                                   \
    xb_vecNx16U vecScaleLL = IVP_SELNX16UI(0, vecScale1L, IVP_SELI_INTERLEAVE_1_LO);      \
    xb_vecNx16U vecScaleLH = IVP_SELNX16UI(0, vecScale1L, IVP_SELI_INTERLEAVE_1_HI);      \
    xb_vecNx16U vecScaleHL = IVP_SELNX16UI(0, vecScale1H, IVP_SELI_INTERLEAVE_1_LO);      \
    xb_vecNx16U vecScaleHH = IVP_SELNX16UI(0, vecScale1H, IVP_SELI_INTERLEAVE_1_HI);      \
    vecaccHH  = IVP_CVT32S2NX24HH(accdotProd);                                            \
    vecaccHL  = IVP_CVT32S2NX24HL(accdotProd);                                            \
    vecaccLH  = IVP_CVT32S2NX24LH(accdotProd);                                            \
    vecaccLL  = IVP_CVT32S2NX24LL(accdotProd);                                            \
    haccA     = IVP_MULUSN_2X16X32_0(vecScaleLL, vecaccLL);                               \
    haccB     = IVP_MULUSN_2X16X32_0(vecScaleLH, vecaccLH);                               \
    haccC     = IVP_MULUSN_2X16X32_0(vecScaleHL, vecaccHL);                               \
    haccD     = IVP_MULUSN_2X16X32_0(vecScaleHH, vecaccHH);                               \
    hvec0LL   = IVP_PACKVRN_2X64W(haccA, shift1);                                         \
    hvec0LH   = IVP_PACKVRN_2X64W(haccB, shift1);                                         \
    hvec0HL   = IVP_PACKVRN_2X64W(haccC, shift1);                                         \
    hvec0HH   = IVP_PACKVRN_2X64W(haccD, shift1);                                         \
    accA      = IVP_CVT48SNX32(hvec0LH, hvec0LL);                                         \
    accB      = IVP_CVT48SNX32(hvec0HH, hvec0HL);                                         \
    vecClampL = IVP_PACKVRNX48(accA, 0);                                                  \
    vecClampH = IVP_PACKVRNX48(accB, 0);                                                  \
}

#define PACK_SCALE_SHIFT_S48_S8(accProd, accShift2, scale2L, shift2, vecRescale)        {                        \
    xb_vecN_2x64w wvecAccL = IVP_CVT96UN_2X64(IVP_CVT64SNX48LH(accProd), IVP_CVT64SNX48LL(accProd));             \
    xb_vecN_2x64w wvecAccH = IVP_CVT96UN_2X64(IVP_CVT64SNX48HH(accProd), IVP_CVT64SNX48HL(accProd));             \
    accProd    = IVP_CVT48SNX32(IVP_PACKVRN_2X64W(wvecAccH, accShift2), IVP_PACKVRN_2X64W(wvecAccL, accShift2)); \
    vecRescale = IVP_PACKVRNX48(accProd, 0);                                                                     \
    accProd    = IVP_MULUSNX16(scale2L, vecRescale);                                                             \
    vecRescale = IVP_PACKVRNX48(accProd, shift2);                                                                \
}

#define PACK_SCALE_SHIFT_S32_S8(inReg, inReg1, scale2, shift2, seq1, dvecOut)           { \
    xb_vecN_2x64w m_wvec = IVP_MULUSN_2X16X32_0((xb_vecNx16U) scale2, inReg);             \
    xb_vecN_2x32v m_outL = IVP_PACKVRN_2X64W(m_wvec, shift2);                             \
    m_outL = IVP_MAXN_2X32(IVP_MINN_2X32(m_outL, SCHAR_MAX), SCHAR_MIN);                  \
    m_wvec = IVP_MULUSN_2X16X32_1((xb_vecNx16U) scale2, inReg1);                          \
    xb_vecN_2x32v m_outH = IVP_PACKVRN_2X64W(m_wvec, shift2);                             \
    m_outH  = IVP_MAXN_2X32(IVP_MINN_2X32(m_outH, SCHAR_MAX), SCHAR_MIN);                 \
    dvecOut = IVP_SEL2NX8(IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(m_outH)),           \
                          IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(m_outL)), seq1);    \
}

#define PACK_SCALE_SHIFT_S32_S8(inReg, inReg1, scale2, shift2, seq1, dvecOut)           { \
    xb_vecN_2x64w m_wvec = IVP_MULUSN_2X16X32_0((xb_vecNx16U) scale2, inReg);             \
    xb_vecN_2x32v m_outL = IVP_PACKVRN_2X64W(m_wvec, shift2);                             \
    m_outL = IVP_MAXN_2X32(IVP_MINN_2X32(m_outL, SCHAR_MAX), SCHAR_MIN);                  \
    m_wvec = IVP_MULUSN_2X16X32_1((xb_vecNx16U) scale2, inReg1);                          \
    xb_vecN_2x32v m_outH = IVP_PACKVRN_2X64W(m_wvec, shift2);                             \
    m_outH  = IVP_MAXN_2X32(IVP_MINN_2X32(m_outH, SCHAR_MAX), SCHAR_MIN);                 \
    dvecOut = IVP_SEL2NX8(IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(m_outH)),           \
                          IVP_MOV2NX8_FROMNX16(IVP_MOVNX16_FROMN_2X32(m_outL)), seq1);    \
}

#define ACC_INIT_BIAS(phvecBias, numBias, daccSum0, daccSum1, daccSum2, daccSum3)       {    \
    xb_vecN_2x32v hvecBias1, hvecBias2;                                                      \
    valign vaBias = IVP_LAN_2X32_PP(phvecBias);                                              \
    IVP_LAVN_2X32_XP(hvecBias1, vaBias, phvecBias, 4 * numBias);                             \
    IVP_LAVN_2X32_XP(hvecBias2, vaBias, phvecBias, 4 * numBias - 2 * XCHAL_IVPN_SIMD_WIDTH); \
    daccSum0 = IVP_CVT24UNX32L(hvecBias2, hvecBias1);                                        \
    daccSum1 = IVP_CVT24UNX32L(hvecBias2, hvecBias1);                                        \
    daccSum2 = IVP_CVT24UNX32L(hvecBias2, hvecBias1);                                        \
    daccSum3 = IVP_CVT24UNX32L(hvecBias2, hvecBias1);                                        \
    IVP_LAVN_2X32_XP(hvecBias1, vaBias, phvecBias, 4 * numBias - 4 * XCHAL_IVPN_SIMD_WIDTH); \
    IVP_LAVN_2X32_XP(hvecBias2, vaBias, phvecBias, 4 * numBias - 6 * XCHAL_IVPN_SIMD_WIDTH); \
    IVP_CVT24UNX32H(daccSum0, hvecBias2, hvecBias1);                                         \
    IVP_CVT24UNX32H(daccSum1, hvecBias2, hvecBias1);                                         \
    IVP_CVT24UNX32H(daccSum2, hvecBias2, hvecBias1);                                         \
    IVP_CVT24UNX32H(daccSum3, hvecBias2, hvecBias1);                                         \
}

#define ACC_INIT_BIAS64_MOD_ONEACC(pdvecBias, vaBias, numBias, accSum64)                {    \
    xb_vec2Nx8 m_dvecBias1, m_dvecBias2, m_dvecBias3, m_dvecBias4;                           \
    IVP_LAV2NX8_XP(m_dvecBias1, vaBias, pdvecBias, numBias * 8);                             \
    IVP_LAV2NX8_XP(m_dvecBias2, vaBias, pdvecBias, numBias * 8 - 2 * XCHAL_IVPN_SIMD_WIDTH); \
    IVP_LAV2NX8_XP(m_dvecBias3, vaBias, pdvecBias, numBias * 8 - 4 * XCHAL_IVPN_SIMD_WIDTH); \
    IVP_LAV2NX8_XP(m_dvecBias4, vaBias, pdvecBias, numBias * 8 - 6 * XCHAL_IVPN_SIMD_WIDTH); \
    accSum64 = IVP_CVT48UN_2X64L(m_dvecBias2, m_dvecBias1);                                  \
    IVP_CVT48UN_2X64H(accSum64, m_dvecBias4, m_dvecBias3);                                   \
}

#define ACC_INIT_BIAS64_MOW_ONEACC(pBias64, vaBias, wvecAcc, flag)                        \
  {                                                                                       \
    xb_vec2Nx8 m_dvecBias64; IVP_LAV2NX8_XP(m_dvecBias64, vaBias, pdvecBias64, flag * 8); \
    m_dvecBias64 = IVP_SHFL2NX8I(m_dvecBias64, IVP_SHFLI_REP_0X4);                        \
    wvecAcc      = IVP_CVT48UN_2X64L(m_dvecBias64, m_dvecBias64);                         \
    IVP_CVT48UN_2X64H(wvecAcc, m_dvecBias64, m_dvecBias64);                               \
  }

#define VQ_INIT_OUTSCALE(pOutScale, numOutScale, vecDataEven, vecDataOdd)  {                   \
    xb_vecNx16U vecDataL, vecDataH;                                                            \
    valign vaScale = IVP_LANX16U_PP(pOutScale);                                                \
    IVP_LAVNX16_XP(vecDataL, vaScale, pOutScale, 2 * numOutScale);                             \
    IVP_LAVNX16_XP(vecDataH, vaScale, pOutScale, 2 * numOutScale - 2 * XCHAL_IVPN_SIMD_WIDTH); \
    vecDataEven = IVP_SELNX16UI(vecDataH, vecDataL, IVP_SELI_16B_EXTRACT_1_OF_2_OFF_0);        \
    vecDataOdd  = IVP_SELNX16UI(vecDataH, vecDataL, IVP_SELI_16B_EXTRACT_1_OF_2_OFF_1);        \
}
#endif
