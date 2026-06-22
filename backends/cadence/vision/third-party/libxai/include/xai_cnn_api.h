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

#ifndef __XAI_CNN_API_H__
#define __XAI_CNN_API_H__

#include "xai_cnn_api_params.h"
#include "xai_config_api.h"
#include "xai_core_api.h"
#include "xai_tile_manager.h"
#include <math.h>
#include <stdbool.h>


#if ((XCHAL_VISION_TYPE >= 6))
/***************************************************************************************************/
/******************************  Fixed Point routines declaration  *********************************/
/***************************************************************************************************/

/* Convolution wrappper functions */
_XAI_API_ XAI_ERR_TYPE xaiConvolve3D(const xai_pTile3D inTile,
                                     const xai_pTile4D coeffTile,
                                     const xai_pArray biasArray,
                                     xai_pTile3D outTile,
                                     xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE *xaiGetConvolve3DVariant(const xai_pTile3D inTile,
                                                const xai_pTile4D coeffTile,
                                                const xai_pArray biasArray,
                                                xai_pTile3D outTile,
                                                xai_cnn_conv_params *param);

/* Convolution MOW*/
_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_1x1j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_1x1j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_3x3j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_3x3j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_5x5j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_5x5j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_7x7j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_7x7j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxNj1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxNj1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_1x1j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_1x1j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_3x3j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_3x3j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_5x5j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_5x5j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_7x7j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_7x7j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxNj2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxNj2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_1x1j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_1x1j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_3x3j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_3x3j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_5x5j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_5x5j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_7x7j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_7x7j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxNj4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxNj4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            xai_cnn_conv_params *param);

/* Convolution MOD */
_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 xai_pTile3D outTile,
                                                                 xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 xai_pTile3D outTile,
                                                                 xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 xai_pTile3D outTile,
                                                                 xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 xai_pTile3D outTile,
                                                                 xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 xai_pTile3D outTile,
                                                                 xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_1x1_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             xai_pTile3D outTile,
                                                             xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_1x1_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             xai_pTile3D outTile,
                                                             xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_3x3_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             xai_pTile3D outTile,
                                                             xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_3x3_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             xai_pTile3D outTile,
                                                             xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_5x5_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             xai_pTile3D outTile,
                                                             xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_7x7_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             xai_pTile3D outTile,
                                                             xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxN_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             xai_pTile3D outTile,
                                                             xai_cnn_conv_params *param);

/* Convolution SO */
_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxN_S8S8IX_SO_DWH(const xai_pTile3D inTile,
                                                         const xai_pTile4D coeffTile,
                                                         const xai_pArray biasArray,
                                                         xai_pTile3D outTile,
                                                         xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolve3D_S_MxN_U8S8IX_SO_DWH(const xai_pTile3D inTile,
                                                         const xai_pTile4D coeffTile,
                                                         const xai_pArray biasArray,
                                                         xai_pTile3D outTile,
                                                         xai_cnn_conv_params *param);

/* Convolution Fully connected */
_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3D(const xai_pTile3D inTile,
                                           const xai_pTile4D coeffTile,
                                           const xai_pArray biasArray,
                                           xai_pTile3D outTile,
                                           const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3D_S_S8S8IX(const xai_pTile3D inTile,
                                                    const xai_pTile4D coeffTile,
                                                    const xai_pArray biasArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3D_S_U8S8IX(const xai_pTile3D inTile,
                                                    const xai_pTile4D coeffTile,
                                                    const xai_pArray biasArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3D_S_S16S16I16(const xai_pTile3D inTile,
                                                       const xai_pTile4D coeffTile,
                                                       const xai_pArray biasArray,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnected3D(const xai_pTile3D inTile,
                                                  const xai_pTile4D coeffTile,
                                                  const xai_pArray biasArray,
                                                  xai_pTile3D accArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnected3D_S_S8S8IXCa2(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D accTile,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnected3D_S_U8S8IXCa2(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D accTile,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnected3D_S_S16S16I16Ca2(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 xai_pTile3D accTile,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnected3D_S_S8S8IXCa2_QM32(const xai_pTile3D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pTile3D accTile,
                                                                   xai_pTile3D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnected3D_S_U8S8IXCa2_QM32(const xai_pTile3D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pTile3D accTile,
                                                                   xai_pTile3D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching(const xai_pTile4D inTile,
                                                       const xai_pTile4D coeffTile,
                                                       const xai_pArray biasArray,
                                                       xai_pArray accArray,
                                                       xai_pTile4D outTile,
                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching_S_S8S8IXCa2(const xai_pTile4D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pArray accArray,
                                                                   xai_pTile4D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching_S_U8S8IXCa2(const xai_pTile4D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pArray accArray,
                                                                   xai_pTile4D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching_S_S8U8IXCa2(const xai_pTile4D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pArray accArray,
                                                                   xai_pTile4D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching_S_U8U8IXCa2(const xai_pTile4D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pArray accArray,
                                                                   xai_pTile4D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching_S_U8S8IXCa2_NoBU(const xai_pTile4D inTile,
                                                                        const xai_pTile4D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pArray accArray,
                                                                        xai_pTile4D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching_S_S16S16I16(const xai_pTile4D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pArray accArray,
                                                                   xai_pTile4D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching_S_U16S16I16(const xai_pTile4D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pArray accArray,
                                                                   xai_pTile4D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching_S_S16U16I16(const xai_pTile4D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pArray accArray,
                                                                   xai_pTile4D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3D2_S_S8(const xai_pTile3D inTile,
                                                 const xai_pTile4D coeffTile,
                                                 const xai_pArray biasArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3D2_S_U8S8U8(const xai_pTile3D inTile,
                                                     const xai_pTile4D coeffTile,
                                                     const xai_pArray biasArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3D2_S_U8(const xai_pTile3D inTile,
                                                 const xai_pTile4D coeffTile,
                                                 const xai_pArray biasArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_conv_params *param);

/* Dilated Convolution wrapper function */
_XAI_API_ XAI_ERR_TYPE xaiConvolved3D(const xai_pTile3D inTile,
                                      const xai_pTile4D coeffTile,
                                      const xai_pArray biasArray,
                                      xai_pTile3D outTile,
                                      const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE *xaiGetConvolved3DVariant(const xai_pTile3D inTile,
                                                 const xai_pTile4D coeffTile,
                                                 const xai_pArray biasArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_conv_params *param);

/* Dilated Convolution MOW, dilation = 1 */
_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_2x2j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_2x2j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_4x4j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_4x4j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj1d1_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1j2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1j2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj2d1_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1j4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1j4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj4d1_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

/* Dilated Convolution MOW, dilation = 2*/
_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

/* Dilated Convolution MOW, dilation = 4 */
_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);
_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                               const xai_pTile4D coeffTile,
                                                               const xai_pArray biasArray,
                                                               xai_pTile3D outTile,
                                                               const xai_cnn_conv_params *param);

/* Dilated Convolution MOD*/
_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_2x2_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_4x4_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                  const xai_pTile4D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_2x2_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_4x4_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_5x5_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_7x7_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxN_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

/* Partial convolution */
_XAI_API_ XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D accTile,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D accTile,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_S8S8IXCa2_MOD_DWH_QM32(const xai_pTile3D inTile,
                                                                          const xai_pTile4D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          xai_pTile3D accTile,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_conv_params *param);

/* MOD_DWH S16S16 Partial Convolution variant */
_XAI_API_ XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D accTile,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

/* Dilated Convolution SO*/
_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxN_S8S8IX_SO_DWH(const xai_pTile3D inTile,
                                                          const xai_pTile4D coeffTile,
                                                          const xai_pArray biasArray,
                                                          xai_pTile3D outTile,
                                                          const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxN_U8S8IX_SO_DWH(const xai_pTile3D inTile,
                                                          const xai_pTile4D coeffTile,
                                                          const xai_pArray biasArray,
                                                          xai_pTile3D outTile,
                                                          const xai_cnn_conv_params *param);

/* Depthwise Convolution wrappper function */
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D(const xai_pTile3D inTile,
                                              const xai_pTile3D coeffTile,
                                              const xai_pArray biasArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE *xaiGetDepthwiseConvolve2DVariant(const xai_pTile3D inTile,
                                                         const xai_pTile3D coeffTile,
                                                         const xai_pArray biasArray,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_conv_params *param);

/* Depthwise Convolutions MOW */
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_7x7j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_7x7j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_7x7j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_7x7j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_7x7j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_7x7j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

/* Depthwise MOW Convolution MOW 16-bit Variants */

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj1_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj2_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj4_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);
/* Depthwise MOW Convolution VQ variants*/

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_3x3j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_3x3j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_5x5j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_5x5j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_7x7j1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_7x7j1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxNj1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxNj1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_3x3j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_3x3j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_5x5j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_5x5j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_7x7j2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_7x7j2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxNj2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxNj2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_3x3j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_3x3j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_5x5j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_5x5j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_7x7j4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_7x7j4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxNj4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxNj4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

/* Depthwise MOW Convolution MOW 16-bit Variants */

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxNj1_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                          const xai_pTile3D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxNj2_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                          const xai_pTile3D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxNj4_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                          const xai_pTile3D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_conv_params *param);

/* Depthwise Convolutions MOD */
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_7x7_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxN_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxN_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_conv_params *param);
/* Depthwise MOD16 Variants */

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D pinTile,
                                                                      const xai_pTile3D pcoeffTile,
                                                                      const xai_pArray pbiasArray,
                                                                      xai_pTile3D poutTile,
                                                                      const xai_cnn_conv_params *pconvParam);

/* Depthwise MOD VQ Convolution variants */

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_3x3_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        const xai_pArray outputScaleArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_3x3_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        const xai_pArray outputScaleArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_5x5_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        const xai_pArray outputScaleArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_5x5_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        const xai_pArray outputScaleArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_7x7_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        const xai_pArray outputScaleArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxN_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        const xai_pArray outputScaleArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxN_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        const xai_pArray outputScaleArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);
/* Depthwise MOD16 Variants */

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D pinTile,
                                                                        const xai_pTile3D pcoeffTile,
                                                                        const xai_pArray pbiasArray,
                                                                        const xai_pArray poutputScaleArray,
                                                                        xai_pTile3D poutTile,
                                                                        const xai_cnn_conv_params *pconvParam);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolveVQ2D(const xai_pTile3D inTile,
                                                const xai_pTile3D coeffTile,
                                                const xai_pArray biasArray,
                                                const xai_pArray outputScaleArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE *xaiGetDepthwiseConvolveVQ2DVariant(const xai_pTile3D inTile,
                                                           const xai_pTile3D coeffTile,
                                                           const xai_pArray biasArray,
                                                           const xai_pArray outputScaleArray,
                                                           xai_pTile3D outTile,
                                                           const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D(const xai_pTile3D inTile,
                                                 const xai_pTile3D coeffTile,
                                                 const xai_pArray biasArray,
                                                 const xai_pArray outputScaleArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE *xaiGetDepthwiseConvolvedVQ2DVariant(const xai_pTile3D inTile,
                                                            const xai_pTile3D coeffTile,
                                                            const xai_pArray biasArray,
                                                            const xai_pArray outputScaleArray,
                                                            xai_pTile3D outTile,
                                                            const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_MxNj1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                          const xai_pTile3D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_MxNj1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                          const xai_pTile3D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_depthwiseDilatedConv_params *param);


/*Depthwise dilated wrapper function*/
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D(const xai_pTile3D inTile,
                                               const xai_pTile3D coeffTile,
                                               const xai_pArray biasArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_MxNj1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                          const xai_pTile3D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_MxNj1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                          const xai_pTile3D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE *xaiGetDepthwiseConvolved2DVariant(const xai_pTile3D inTile,
                                                          const xai_pTile3D coeffTile,
                                                          const xai_pArray biasArray,
                                                          xai_pTile3D outTile,
                                                          const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_MxNj1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_MxNj1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_depthwiseDilatedConv_params *param);

/*_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_3x3j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_3x3j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);
 */
/*Depthwise dilated MOW convolution variants*/
/*_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_3x3j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_3x3j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_5x5j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_5x5j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_7x7j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_7x7j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);
 */
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_MxNj1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_MxNj1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_depthwiseDilatedConv_params *param);

/*
   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_5x5j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_5x5j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_7x7j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_7x7j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                     const xai_pTile3D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_depthwiseDilatedConv_params *param);

 */
/*Depthwise Dilated MOD convolution variants*/
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_3x3_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile3D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_5x5_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile3D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_7x7_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile3D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_depthwiseDilatedConv_params *param);


_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_MxN_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile3D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_MxN_U8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile3D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_5x5_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_3x3_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

/*
   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_3x3j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_3x3j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_5x5j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_5x5j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_7x7j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_7x7j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_3x3j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_3x3j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_5x5j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_5x5j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_7x7j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);

   _XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_7x7j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                       const xai_pTile3D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_depthwiseDilatedConv_params *param);
 */


/* VQ variants */
/*Depthwise Dilated MOD convolution variants*/

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_3x3_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      const xai_pArray outputScaleArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_5x5_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      const xai_pArray outputScaleArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_7x7_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      const xai_pArray outputScaleArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_MxN_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      const xai_pArray outputScaleArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_MxN_U8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      const xai_pArray outputScaleArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                         const xai_pTile3D coeffTile,
                                                                         const xai_pArray biasArray,
                                                                         const xai_pArray outputScaleArray,
                                                                         xai_pTile3D outTile,
                                                                         const xai_cnn_depthwiseDilatedConv_params *param);
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_3x3_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                         const xai_pTile3D coeffTile,
                                                                         const xai_pArray biasArray,
                                                                         const xai_pArray outputScaleArray,
                                                                         xai_pTile3D outTile,
                                                                         const xai_cnn_depthwiseDilatedConv_params *param);
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolvedVQ2D_S_5x5_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                         const xai_pTile3D coeffTile,
                                                                         const xai_pArray biasArray,
                                                                         const xai_pArray outputScaleArray,
                                                                         xai_pTile3D outTile,
                                                                         const xai_cnn_depthwiseDilatedConv_params *param);

/* Depthwise DM MOD convolve */
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseDMConvolved2D_S_MxN_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseDMConvolved2D_S_MxN_U8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseDMConvolved2D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                         const xai_pTile3D coeffTile,
                                                                         const xai_pArray biasArray,
                                                                         xai_pTile3D outTile,
                                                                         const xai_cnn_depthwiseDilatedConv_params *param);

/* Depthwise DM MOD convole VQ */
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseDMConvolvedVQ2D_S_MxN_S8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        const xai_pArray outputScaleArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseDMConvolvedVQ2D_S_MxN_U8S8IX_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile3D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        const xai_pArray outputScaleArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseDMConvolvedVQ2D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                           const xai_pTile3D coeffTile,
                                                                           const xai_pArray biasArray,
                                                                           const xai_pArray outputScaleArray,
                                                                           xai_pTile3D outTile,
                                                                           const xai_cnn_depthwiseDilatedConv_params *param);

/*_XAI_API_ XAI_ERR_TYPE xaiDepthwiseDMConvolvedReorderCoeff2D_MOD(const xai_pTile3D srcTile, const xai_pArray biasArray,
                                                                xai_pTile3D dstTile, xai_pArray biasArrayReOrder,
                                                                const int32_t inDepth, const int32_t depthMultiplier);
 */

/* VQ variants */

/* Dilated convolution wrapper */

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D(const xai_pTile3D inTile,
                                        const xai_pTile4D coeffTile,
                                        const xai_pArray biasArray,
                                        const xai_pArray outputScaleArray,
                                        xai_pTile3D outTile,
                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE *xaiGetConvolvedVQ3DVariant(const xai_pTile3D inTile,
                                                   const xai_pTile4D coeffTile,
                                                   const xai_pArray biasArray,
                                                   const xai_pArray outputScaleArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

/* Dilated MOD VQ Convolution variants */
_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);
_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_2x2_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_4x4_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);

/* Dilated Convolution MOW, dilation = 1 */

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_2x2j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_2x2j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);



_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_4x4j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_4x4j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj1d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj1d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj1d1_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1j2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1j2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj2d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj2d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj2d1_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1j4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1j4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj4d1_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj4d1_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj4d1_S16S16I16_MOW_WHD(const xai_pTile3D inTile,
                                                                    const xai_pTile4D coeffTile,
                                                                    const xai_pArray biasArray,
                                                                    const xai_pArray outputScaleArray,
                                                                    xai_pTile3D outTile,
                                                                    const xai_cnn_conv_params *param);

/* Dilated Convolutions MOW, dilation = 2*/

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj1d2_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj1d2_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

/* Dilated Convolutions MOW, dilation = 4 */

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7j1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj1d4_S8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxNj1d4_U8S8IX_MOW_WHD(const xai_pTile3D inTile,
                                                                 const xai_pTile4D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 const xai_pArray outputScaleArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_conv_params *param);

/* Dilated Convolution MOD_DWH - VQ variants */

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_2x2_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_4x4_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_5x5_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_7x7_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                          const xai_pTile4D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile4D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_1x1_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_3x3_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                              const xai_pTile4D coeffTile,
                                                              const xai_pArray biasArray,
                                                              xai_pTile3D outTile,
                                                              const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                          const xai_pTile4D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile3D outTile,
                                                                          const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                        const xai_pTile4D coeffTile,
                                                                        const xai_pArray biasArray,
                                                                        xai_pTile3D outTile,
                                                                        const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

/*Bias update function*/
_XAI_API_ XAI_ERR_TYPE xaiConvolvedBiasUpdate_S8S32(const xai_pTile4D coeffTile,
                                                    xai_pArray biasArray);

/* Reorder a 4D tile to IN64DWH format */
_XAI_API_ XAI_ERR_TYPE xaiReOrder4DToIN64DWH_I8(xai_pTile4D coeffTileIn, xai_pTile4D coeffTileOut);

/* Reorder a 4D tile to IN32DWH format */
_XAI_API_ XAI_ERR_TYPE xaiReOrder4DToIN32DWH_I16(xai_pTile4D coeffTileIn, xai_pTile4D coeffTileOut);

/* Partial convolution */
_XAI_API_ XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                       const xai_pTile4D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D accTile,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                                 const xai_pTile4D coeffTile,
                                                                                 const xai_pArray biasArray,
                                                                                 const xai_pArray outputScaleArray,
                                                                                 xai_pTile3D accTile,
                                                                                 xai_pTile3D outTile,
                                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_S8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                               const xai_pTile4D coeffTile,
                                                                               const xai_pArray biasArray,
                                                                               xai_pTile3D accTile,
                                                                               xai_pTile3D outTile,
                                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_U8S8IXCa2_MOD_DWH(const xai_pTile3D inTile,
                                                                       const xai_pTile4D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D accTile,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                                 const xai_pTile4D coeffTile,
                                                                                 const xai_pArray biasArray,
                                                                                 const xai_pArray outputScaleArray,
                                                                                 xai_pTile3D accTile,
                                                                                 xai_pTile3D outTile,
                                                                                 const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialConvolved3D_S_MxN_U8S8IXCa2_noUnrollH_MOD_DWH(const xai_pTile3D inTile,
                                                                               const xai_pTile4D coeffTile,
                                                                               const xai_pArray biasArray,
                                                                               xai_pTile3D accTile,
                                                                               xai_pTile3D outTile,
                                                                               const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_S8S8IXCa2_MOD_DWH_QM32(const xai_pTile3D inTile,
                                                                            const xai_pTile4D coeffTile,
                                                                            const xai_pArray biasArray,
                                                                            const xai_pArray outputScaleArray,
                                                                            xai_pTile3D accTile,
                                                                            xai_pTile3D outTile,
                                                                            const xai_cnn_conv_params *param);

/* MOD_DWH S16S16 Partial Convolution VQ variant */
_XAI_API_ XAI_ERR_TYPE xaiPartialConvolvedVQ3D_S_MxN_S16S16I16_MOD_DWH(const xai_pTile3D inTile,
                                                                       const xai_pTile4D coeffTile,
                                                                       const xai_pArray biasArray,
                                                                       const xai_pArray outputScaleArray,
                                                                       xai_pTile3D accTile,
                                                                       xai_pTile3D outTile,
                                                                       const xai_cnn_conv_params *param);

/* MxN SO VQ variants */

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_S8S8IX_SO_DWH(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            const xai_pArray outputScaleArray,
                                                            xai_pTile3D outTile,
                                                            const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolvedVQ3D_S_MxN_U8S8IX_SO_DWH(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            const xai_pArray outputScaleArray,
                                                            xai_pTile3D outTile,
                                                            const xai_cnn_conv_params *param);


/* MxN Fully Connected VQ variants */

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3D(const xai_pTile3D inTile,
                                             const xai_pTile4D coeffTile,
                                             const xai_pArray biasArray,
                                             const xai_pArray outputScaleArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3D_S_S8S8IX(const xai_pTile3D inTile,
                                                      const xai_pTile4D coeffTile,
                                                      const xai_pArray biasArray,
                                                      const xai_pArray outputScaleArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3D_S_U8S8IX(const xai_pTile3D inTile,
                                                      const xai_pTile4D coeffTile,
                                                      const xai_pArray biasArray,
                                                      const xai_pArray outputScaleArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3D_S_S16S16I16(const xai_pTile3D inTile,
                                                         const xai_pTile4D coeffTile,
                                                         const xai_pArray biasArray,
                                                         const xai_pArray outputScaleArray,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnectedVQ3D(const xai_pTile3D inTile,
                                                    const xai_pTile4D coeffTile,
                                                    const xai_pArray biasArray,
                                                    const xai_pArray outputScaleArray,
                                                    xai_pTile3D accTile,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnectedVQ3D_S_S8S8IXCa2(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D accTile,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnectedVQ3D_S_U8S8IXCa2(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                const xai_pArray outputScaleArray,
                                                                xai_pTile3D accTile,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnectedVQ3D_S_S16S16I16Ca2(const xai_pTile3D inTile,
                                                                   const xai_pTile4D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   const xai_pArray outputScaleArray,
                                                                   xai_pTile3D accTile,
                                                                   xai_pTile3D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnectedVQ3D_S_S8S8IXCa2_QM32(const xai_pTile3D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     const xai_pArray outputScaleArray,
                                                                     xai_pTile3D accTile,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiPartialFullyConnectedVQ3D_S_U8S8IXCa2_QM32(const xai_pTile3D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     const xai_pArray outputScaleArray,
                                                                     xai_pTile3D accTile,
                                                                     xai_pTile3D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3DWithBatching(const xai_pTile4D inTile,
                                                         const xai_pTile4D coeffTile,
                                                         const xai_pArray biasArray,
                                                         xai_pArray accArray,
                                                         const xai_pArray outputScaleArray,
                                                         xai_pTile4D outTile,
                                                         const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3DWithBatching_S_S8S8IXCa2(const xai_pTile4D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pArray accArray,
                                                                     const xai_pArray outputScaleArray,
                                                                     xai_pTile4D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3DWithBatching_S_U8S8IXCa2(const xai_pTile4D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pArray accArray,
                                                                     const xai_pArray outputScaleArray,
                                                                     xai_pTile4D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3DWithBatching_S_S8U8IXCa2(const xai_pTile4D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pArray accArray,
                                                                     const xai_pArray outputScaleArray,
                                                                     xai_pTile4D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3DWithBatching_S_U8U8IXCa2(const xai_pTile4D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pArray accArray,
                                                                     const xai_pArray outputScaleArray,
                                                                     xai_pTile4D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3DWithBatching_S_U8S8IXCa2_NoBU(const xai_pTile4D inTile,
                                                                          const xai_pTile4D coeffTile,
                                                                          const xai_pArray biasArray,
                                                                          xai_pArray accArray,
                                                                          const xai_pArray outputScaleArray,
                                                                          xai_pTile4D outTile,
                                                                          const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3DWithBatching_S_S16S16I16(const xai_pTile4D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pArray accArray,
                                                                     const xai_pArray outputScaleArray,
                                                                     xai_pTile4D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3DWithBatching_S_U16S16I16(const xai_pTile4D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pArray accArray,
                                                                     const xai_pArray outputScaleArray,
                                                                     xai_pTile4D outTile,
                                                                     const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedVQ3DWithBatching_S_S16U16I16(const xai_pTile4D inTile,
                                                                     const xai_pTile4D coeffTile,
                                                                     const xai_pArray biasArray,
                                                                     xai_pArray accArray,
                                                                     const xai_pArray outputScaleArray,
                                                                     xai_pTile4D outTile,
                                                                     const xai_cnn_conv_params *param);

/* Max Pool */
_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D(const xai_pTile3D inTile,
                                    xai_pTile3D outTile,
                                    const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxNj1_S8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxNj1_U8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxNj1_S16_WHD(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxNj2_S8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxNj2_U8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxNj2_S16_WHD(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxN_S8_WHD(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxN_U8_WHD(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxN_S16_WHD(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxN_S8_DWH(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxN_U8_DWH(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxN_S16_DWH(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param);

/* MaxPoolWithIdx Variants */

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           xai_pTile3D idxTile,
                                           const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxNj1_S8_WHD(const xai_pTile3D inTile,
                                                        xai_pTile3D outTile,
                                                        xai_pTile3D idxTile,
                                                        const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxNj2_S8_WHD(const xai_pTile3D inTile,
                                                        xai_pTile3D outTile,
                                                        xai_pTile3D idxTile,
                                                        const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxN_S8_DWH(const xai_pTile3D inTile,
                                                      xai_pTile3D outTile,
                                                      xai_pTile3D idxTile,
                                                      const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxNj1_U8_WHD(const xai_pTile3D inTile,
                                                        xai_pTile3D outTile,
                                                        xai_pTile3D idxTile,
                                                        const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxNj2_U8_WHD(const xai_pTile3D inTile,
                                                        xai_pTile3D outTile,
                                                        xai_pTile3D idxTile,
                                                        const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxN_U8_DWH(const xai_pTile3D inTile,
                                                      xai_pTile3D outTile,
                                                      xai_pTile3D idxTile,
                                                      const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxNj1_S16_WHD(const xai_pTile3D inTile,
                                                         xai_pTile3D outTile,
                                                         xai_pTile3D idxTile,
                                                         const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxNj2_S16_WHD(const xai_pTile3D inTile,
                                                         xai_pTile3D outTile,
                                                         xai_pTile3D idxTile,
                                                         const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxN_S16_DWH(const xai_pTile3D inTile,
                                                       xai_pTile3D outTile,
                                                       xai_pTile3D idxTile,
                                                       const xai_cnn_pooling_params *param);


/* MaxUnPool Variants */


_XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D(const xai_pTile3D inTile,
                                      const xai_pTile3D idxTile,
                                      xai_pTile3D outTile,
                                      const xai_cnn_pooling_params *param);
/*
   _XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxNj1_S8_WHD(const xai_pTile3D inTile,
                                                const xai_pTile3D idxTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param);
 */
_XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxNj2_S8_WHD(const xai_pTile3D inTile,
                                                   const xai_pTile3D idxTile,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxN_S8_DWH(const xai_pTile3D inTile,
                                                 const xai_pTile3D idxTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);
/*
   _XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxNj1_U8_WHD(const xai_pTile3D inTile,
                                                const xai_pTile3D idxTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param);
 */
_XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxNj2_U8_WHD(const xai_pTile3D inTile,
                                                   const xai_pTile3D idxTile,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxN_U8_DWH(const xai_pTile3D inTile,
                                                 const xai_pTile3D idxTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

/*
   _XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxNj1_S16_WHD(const xai_pTile3D inTile,
                                                 const xai_pTile3D idxTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);
 */
_XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxNj2_S16_WHD(const xai_pTile3D inTile,
                                                    const xai_pTile3D idxTile,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxNj2_F16_WHD(const xai_pTile3D inTile,
                                                    const xai_pTile3D idxTile,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxN_S16_DWH(const xai_pTile3D inTile,
                                                  const xai_pTile3D idxTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxUnPool3D_MxN_F16_DWH(const xai_pTile3D inTile,
                                                  const xai_pTile3D idxTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);
/* RoI Max Pool Variants */
_XAI_API_ XAI_ERR_TYPE xaiRoiMaxPool3D(const xai_pTile3D inTile,
                                       const xai_pArray RoIParam,
                                       xai_pTile4D outTile,
                                       const xai_cnn_roi_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiRoiMaxPool3D_U8_DWH(const xai_pTile3D inTile,
                                              const xai_pArray RoIParam,
                                              xai_pTile4D outTile,
                                              const xai_cnn_roi_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiRoiMaxPool3D_S8_DWH(const xai_pTile3D inTile,
                                              const xai_pArray RoIParam,
                                              xai_pTile4D outTile,
                                              const xai_cnn_roi_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiRoiMaxPool3D_S16_DWH(const xai_pTile3D inTile,
                                               const xai_pArray RoIParam,
                                               xai_pTile4D outTile,
                                               const xai_cnn_roi_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiRoiMaxPoolWithIdx3D(const xai_pTile3D inTile,
                                              const xai_pArray RoIParam,
                                              xai_pTile4D outTile,
                                              xai_pTile4D idxTile,
                                              const xai_cnn_roi_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiRoiMaxPoolWithIdx3D_U8_DWH(const xai_pTile3D inTile,
                                                     const xai_pArray RoIParam,
                                                     xai_pTile4D outTile,
                                                     xai_pTile4D idxTile,
                                                     const xai_cnn_roi_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiRoiMaxPoolWithIdx3D_S8_DWH(const xai_pTile3D inTile,
                                                     const xai_pArray RoIParam,
                                                     xai_pTile4D outTile,
                                                     xai_pTile4D idxTile,
                                                     const xai_cnn_roi_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiRoiMaxPoolWithIdx3D_S16_DWH(const xai_pTile3D inTile,
                                                      const xai_pArray RoIParam,
                                                      xai_pTile4D outTile,
                                                      xai_pTile4D idxTile,
                                                      const xai_cnn_roi_pooling_params *param);

/* Average Pool */
_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D(const xai_pTile3D inTile,
                                    xai_pArray bufArray,
                                    xai_pTile3D outTile,
                                    const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj1_S8_WHD(const xai_pTile3D inTile,
                                                 xai_pArray bufArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj1_U8_WHD(const xai_pTile3D inTile,
                                                 xai_pArray bufArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj1_S16_WHD(const xai_pTile3D inTile,
                                                  xai_pArray bufArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);


_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj1_S8U8_WHD(const xai_pTile3D inTile,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj1_S8S16_WHD(const xai_pTile3D inTile,
                                                    xai_pArray bufArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj1_U8S8_WHD(const xai_pTile3D inTile,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj1_U8S16_WHD(const xai_pTile3D inTile,
                                                    xai_pArray bufArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_S8_WHD(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_U8_WHD(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_S16_WHD(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param);


_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_S8U8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_S8S16_WHD(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_U8S8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_U8S16_WHD(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj2_S8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj2_U8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj2_S16_WHD(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj2_S8U8_WHD(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj2_S8S16_WHD(const xai_pTile3D inTile,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj2_U8S8_WHD(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxNj2_U8S16_WHD(const xai_pTile3D inTile,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_S8_DWH(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_U8_DWH(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_S16_DWH(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_S8U8_DWH(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);


_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_S8S16_DWH(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_U8S16_DWH(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_U8S8_DWH(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param);

/* Global Average Pool */
_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D(const xai_pTile3D inTile,
                                          xai_pArray bufferArray,
                                          xai_pTile3D outTile,
                                          const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_S8_WHD(const xai_pTile3D inTile,
                                                 xai_pArray bufferArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_U8_WHD(const xai_pTile3D inTile,
                                                 xai_pArray bufferArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_S8U8_WHD(const xai_pTile3D inTile,
                                                   xai_pArray bufferArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_S8S16_WHD(const xai_pTile3D inTile,
                                                    xai_pArray bufferArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_U8S8_WHD(const xai_pTile3D inTile,
                                                   xai_pArray bufferArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_U8S16_WHD(const xai_pTile3D inTile,
                                                    xai_pArray bufferArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_S16_WHD(const xai_pTile3D inTile,
                                                  xai_pArray bufferArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_S8_DWH(const xai_pTile3D inTile,
                                                 xai_pArray bufferArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_U8_DWH(const xai_pTile3D inTile,
                                                 xai_pArray bufferArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_S8U8_DWH(const xai_pTile3D inTile,
                                                   xai_pArray bufferArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_S8S16_DWH(const xai_pTile3D inTile,
                                                    xai_pArray bufferArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_U8S8_DWH(const xai_pTile3D inTile,
                                                   xai_pArray bufferArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_U8S16_DWH(const xai_pTile3D inTile,
                                                    xai_pArray bufferArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_global_pooling_params* param);

_XAI_API_ XAI_ERR_TYPE xaiGlobalAvgPool3D_S16_DWH(const xai_pTile3D inTile,
                                                  xai_pArray bufferArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_global_pooling_params* param);

/* Average pooling CNNA Variants */

_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D(const xai_pTile3D inTile,
                                     xai_pArray bufArray,
                                     xai_pTile3D outTile,
                                     const xai_cnn_pooling_params *param,
                                     const xai_size3D frame3DSize);


_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D_MxN_U8_DWH(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param,
                                                const xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D_MxN_S8_DWH(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param,
                                                const xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D_MxN_S16_DWH(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param,
                                                 const xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D_MxN_U8S8_DWH(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param,
                                                  const xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D_MxN_U8S16_DWH(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_pooling_params *param,
                                                   const xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D_MxN_S8U8_DWH(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_pooling_params *param,
                                                  const xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D_MxN_S8S16_DWH(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_pooling_params *param,
                                                   const xai_size3D frame3DSize);

/*Adaptive Average Pool*/
_XAI_API_ XAI_ERR_TYPE xaiAdaptiveAvgPool3D_S8_DWH(const xai_pTile3D inTile,
                                                   const xai_pArray inTileIndexArray,
                                                   xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiAdaptiveAvgPool3D_IX(const xai_pTile3D inTile,
                                               const xai_pArray inTileIndexArray,
                                               xai_pTile3D outTile);

/*Adaptive MaxPool*/
_XAI_API_ XAI_ERR_TYPE xaiAdaptiveMaxPool3D_S8_DWH(const xai_pTile3D inTile,
                                                   const xai_pArray inTileIndexArray,
                                                   xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiAdaptiveMaxPool3D_IX(const xai_pTile3D inTile,
                                               const xai_pArray inTileIndexArray,
                                               xai_pTile3D outTile);

/* LRN */
_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D(const xai_pTile3D inTile,
                                     const xai_pArray lutArray,
                                     xai_pTile3D outTile,
                                     const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D(const xai_pTile3D inTile,
                                       const xai_pArray lutArray,
                                       xai_pTile3D outTile,
                                       const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_3_U8S8_WHD(const xai_pTile3D inTile,
                                                  const xai_pArray lutArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_5_U8S8_WHD(const xai_pTile3D inTile,
                                                  const xai_pArray lutArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_N_U8S8_WHD(const xai_pTile3D inTile,
                                                  const xai_pArray lutArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_3_U8S8_DWH(const xai_pTile3D inTile,
                                                  const xai_pArray lutArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_5_U8S8_DWH(const xai_pTile3D inTile,
                                                  const xai_pArray lutArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_N_U8S8_DWH(const xai_pTile3D inTile,
                                                  const xai_pArray lutArray,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_3x3_U8S8_WHD(const xai_pTile3D inTile,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_5x5_U8S8_WHD(const xai_pTile3D inTile,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_MxN_U8S8_WHD(const xai_pTile3D inTile,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_3x3_U8S8_DWH(const xai_pTile3D inTile,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_5x5_U8S8_DWH(const xai_pTile3D inTile,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_MxN_U8S8_DWH(const xai_pTile3D inTile,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_3_U8_WHD(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_5_U8_WHD(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_N_U8_WHD(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_3_U8_DWH(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_5_U8_DWH(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_N_U8_DWH(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_3x3_U8_WHD(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_5x5_U8_WHD(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_MxN_U8_WHD(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_3x3_U8_DWH(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_5x5_U8_DWH(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_MxN_U8_DWH(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_3_S8_WHD(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_5_S8_WHD(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_N_S8_WHD(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_3_S8_DWH(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_5_S8_DWH(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_S_N_S8_DWH(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_lrn_depth_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_3x3_S8_WHD(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_5x5_S8_WHD(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_MxN_S8_WHD(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_3x3_S8_DWH(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_5x5_S8_DWH(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_S_MxN_S8_DWH(const xai_pTile3D inTile,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_lrn_spatial_params *param);

/* LUT APIs */

_XAI_API_ XAI_ERR_TYPE xaiLUT3D(const xai_pTile3D inTile,
                                const xai_pArray lutArray,
                                xai_pTile3D outTile,
                                const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Oddsym_S8S8(const xai_pTile3D inTile,
                                            const xai_pArray lutArray,
                                            xai_pTile3D outTile,
                                            const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Evensym_S8I8(const xai_pTile3D inTile,
                                             const xai_pArray lutArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Normal_S8I8(const xai_pTile3D inTile,
                                            const xai_pArray lutArray,
                                            xai_pTile3D outTile,
                                            const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Oddsym_S8S16(const xai_pTile3D inTile,
                                             const xai_pArray lutArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Evensym_S8I16(const xai_pTile3D inTile,
                                              const xai_pArray lutArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Normal_S8I16(const xai_pTile3D inTile,
                                             const xai_pArray lutArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Oddsym_S16S8(const xai_pTile3D inTile,
                                             const xai_pArray lutArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Evensym_S16I8(const xai_pTile3D inTile,
                                              const xai_pArray lutArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Normal_S16I8(const xai_pTile3D inTile,
                                             const xai_pArray lutArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Oddsym_S16S16(const xai_pTile3D inTile,
                                              const xai_pArray lutArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Evensym_S16I16(const xai_pTile3D inTile,
                                               const xai_pArray lutArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Normal_S16I16(const xai_pTile3D inTile,
                                              const xai_pArray lutArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_lut_params *params);

/* Partial Dual LUT APIs */

_XAI_API_ XAI_ERR_TYPE xaiPartialDualLUT3D_S16I16(const xai_pTile3D inTile,
                                                  const xai_pArray lut1Array,
                                                  const xai_pArray lut2Array,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPartialDualLUT3D_Oddsym_S16S16(const xai_pTile3D inTile,
                                                         const xai_pArray lut1Array,
                                                         const xai_pArray lut2Array,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPartialDualLUT3D_Evensym_S16I16(const xai_pTile3D inTile,
                                                          const xai_pArray lut1Array,
                                                          const xai_pArray lut2Array,
                                                          xai_pTile3D outTile,
                                                          const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPartialDualLUT3D_Normal_S16I16(const xai_pTile3D inTile,
                                                         const xai_pArray lut1Array,
                                                         const xai_pArray lut2Array,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_lut_params *params);

/* FillTile */
_XAI_API_ XAI_ERR_TYPE xaiFillTile3D(xai_pTile3D dstTile,
                                     const int32_t value,
                                     xai_bool fillEdgeExtension);

_XAI_API_ XAI_ERR_TYPE xaiFillTile3D_I8(xai_pTile3D dstTile,
                                        const int32_t value,
                                        xai_bool fill_edge_extension);

_XAI_API_ XAI_ERR_TYPE xaiFillTile3D_I16(xai_pTile3D dstTile,
                                         const int32_t value,
                                         xai_bool fill_edge_extension);

/* Extend Edge */
_XAI_API_ XAI_ERR_TYPE xaiExtendEdgesConst3D(xai_pTile3D dstTile,
                                             const int32_t value,
                                             xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiExtendEdgesConst3D_I8(xai_pTile3D dstTile,
                                                const int32_t value,
                                                xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiExtendEdgesConst3D_I16(xai_pTile3D dstTile,
                                                 const int32_t value,
                                                 xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiExtendEdges3D(xai_pTile3D dstTile,
                                        const xai_pArray pArray,
                                        xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiExtendEdges3D_I8(xai_pTile3D dstTile,
                                           const xai_pArray pArray,
                                           xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiExtendEdges3D_I16(xai_pTile3D dstTile,
                                            const xai_pArray pArray,
                                            xai_size3D frame3DSize);

/* Copy Tile */
_XAI_API_ XAI_ERR_TYPE xaiCopyTile3D(const xai_pTile3D inTile,
                                     xai_pTile3D outTile,
                                     xai_bool copy_edge_extension);

/* Transpose */
_XAI_API_ XAI_ERR_TYPE xaiTranspose3D(const xai_pTile3D inTile,
                                      xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D_I8_WHD_DWH(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D_I8_WHD_DWH_Depth3(const xai_pTile3D inTile,
                                                        xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D_I8_DWH_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D_I8_DWH_WHD_Depth3(const xai_pTile3D inTile,
                                                        xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D_I16_WHD_DWH(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D_I16_DWH_WHD(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D_I32_WHD_DWH(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D_I32_DWH_WHD(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile);
/*
   _XAI_API_ XAI_ERR_TYPE xaiTranspose_I32(const xai_pArray srcArray,
                                     xai_pArray dstArray);
 */
_XAI_API_ XAI_ERR_TYPE xaiTranspose3D2(const xai_pTile3D inTile,
                                       xai_pArray bufArray,
                                       xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D2_I8_DWH_WHD(const xai_pTile3D inTile,
                                                  xai_pArray bufArray,
                                                  xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D2_I16_DWH_WHD(const xai_pTile3D inTile,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D2_I32_DWH_WHD(const xai_pTile3D inTile,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D outTile);
_XAI_API_ XAI_ERR_TYPE xaiTranspose3D2_I8_WHD_DWH(const xai_pTile3D inTile,
                                                  xai_pArray bufArray,
                                                  xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D2_I16_WHD_DWH(const xai_pTile3D inTile,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTranspose3D2_I32_WHD_DWH(const xai_pTile3D inTile,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D outTile);

/* Unsigned to Signed */
_XAI_API_ XAI_ERR_TYPE xaiUnsignedToSigned3D_U8S8(xai_pTile3D inTile,
                                                  xai_pTile3D outTile);

/* Data Conversions */
_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const uint16_t scale,
                                           const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S32S8(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S32U8(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S32S16(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const uint16_t scale,
                                                  const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S32U16(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const uint16_t scale,
                                                  const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S8S16(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S8I32(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S16I32(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const uint16_t scale,
                                                  const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S16I64(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const uint16_t scale,
                                                  const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U8I32(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U16I32(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const uint16_t scale,
                                                  const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U16I64(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const uint16_t scale,
                                                  const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S8I64(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U8S16(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U8I64(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U8U16(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S16I8(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U16I8(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S16(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const uint16_t scale,
                                               const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U16S16(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const uint16_t scale,
                                                  const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U8S8(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const uint16_t scale,
                                                const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S16U16(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const uint16_t scale,
                                                  const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S8U8(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const uint16_t scale,
                                                const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_FLOATIX(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const float scale);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_FLOATS8(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const float scale);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_FLOATS16(const xai_pTile3D inTile,
                                                    xai_pTile3D outTile,
                                                    const float scale);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_FLOATU16(const xai_pTile3D inTile,
                                                    xai_pTile3D outTile,
                                                    const float scale);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_FLOATU8(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const float scale);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_IXFLOAT(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const float scale);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S8FLOAT(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const float scale);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_S16FLOAT(const xai_pTile3D inTile,
                                                    xai_pTile3D outTile,
                                                    const float scale);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U16FLOAT(const xai_pTile3D inTile,
                                                    xai_pTile3D outTile,
                                                    const float scale);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U8FLOAT(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const float scale);

/* Data Conversions with Asymmetric Quantization */
_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const int16_t zeroPoint,
                                                 const uint16_t scale,
                                                 const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S8S8(const xai_pTile3D inTile,
                                                      xai_pTile3D outTile,
                                                      const int16_t fixUp,
                                                      const uint16_t scale,
                                                      const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S8U8(const xai_pTile3D inTile,
                                                      xai_pTile3D outTile,
                                                      const int16_t fixUp,
                                                      const uint16_t scale,
                                                      const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S8S16(const xai_pTile3D inTile,
                                                       xai_pTile3D outTile,
                                                       const int16_t fixUp,
                                                       const uint16_t scale,
                                                       const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S8U16(const xai_pTile3D inTile,
                                                       xai_pTile3D outTile,
                                                       const int16_t fixUp,
                                                       const uint16_t scale,
                                                       const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S8I32(const xai_pTile3D inTile,
                                                       xai_pTile3D outTile,
                                                       const int16_t zeroIn,
                                                       const uint16_t scale,
                                                       const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S8I64(const xai_pTile3D inTile,
                                                       xai_pTile3D outTile,
                                                       const int16_t zeroIn,
                                                       const uint16_t scale,
                                                       const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_U8S8(const xai_pTile3D inTile,
                                                      xai_pTile3D outTile,
                                                      const int16_t zeroOut,
                                                      const uint16_t scale,
                                                      const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S16S8(const xai_pTile3D inTile,
                                                       xai_pTile3D outTile,
                                                       const int16_t zeroOut,
                                                       const uint16_t scale,
                                                       const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_U16S8(const xai_pTile3D inTile,
                                                       xai_pTile3D outTile,
                                                       const int16_t zeroOut,
                                                       const uint16_t scale,
                                                       const uint8_t shift);

// Temporary prototype definition, to be removed later
_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_U16AS8(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const int16_t zeroOut,
                                                  const uint16_t scale,
                                                  const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S32S8(const xai_pTile3D inTile,
                                                       xai_pTile3D outTile,
                                                       const int16_t zeroOut,
                                                       const uint16_t scale,
                                                       const uint8_t shift);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_S8FLOAT(const xai_pTile3D inTile,
                                                         xai_pTile3D outTile,
                                                         const float scale,
                                                         const int16_t zeroPoint);

_XAI_API_ XAI_ERR_TYPE xaiDataConversion3D_AsymQ_FLOATS8(const xai_pTile3D inTile,
                                                         xai_pTile3D outTile,
                                                         const float scale,
                                                         const int16_t zeroPoint);
/* ReOrg */
_XAI_API_ XAI_ERR_TYPE xaiReOrg3D(const xai_pTile3D inTile,
                                  xai_pTile3D outTile,
                                  const xai_cnn_reorg_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReOrg3D_I8_WHD(const xai_pTile3D inTile,
                                         xai_pTile3D outTile,
                                         const xai_cnn_reorg_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReOrg3D_I16_WHD(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reorg_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReOrg3D_I8_DWH(const xai_pTile3D inTile,
                                         xai_pTile3D outTile,
                                         const xai_cnn_reorg_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReOrg3D_I16_DWH(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reorg_params *params);

/* ReOrg4D */
_XAI_API_ XAI_ERR_TYPE xaiReOrg4DBatchSpace(const xai_pTile4D inTile,
                                            xai_pTile4D outTile,
                                            const xai_cnn_reorg4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReOrg4DBatchSpace_I8_WHDN(const xai_pTile4D inTile,
                                                    xai_pTile4D outTile,
                                                    const xai_cnn_reorg4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReOrg4DBatchSpace_I16_WHDN(const xai_pTile4D inTile,
                                                     xai_pTile4D outTile,
                                                     const xai_cnn_reorg4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReOrg4DBatchSpace_I8_DWHN(const xai_pTile4D inTile,
                                                    xai_pTile4D outTile,
                                                    const xai_cnn_reorg4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReOrg4DBatchSpace_I16_DWHN(const xai_pTile4D inTile,
                                                     xai_pTile4D outTile,
                                                     const xai_cnn_reorg4D_params *params);

/* ReOrg Caffe*/
/*_XAI_API_ XAI_ERR_TYPE xaiReOrgCaffe3D_I8_DWH(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_reorg_params *params);
   _XAI_API_ XAI_ERR_TYPE xaiReOrgCaffe3D_I16_DWH(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_reorg_params *params);
 */
/* Renormalisation */
_XAI_API_ XAI_ERR_TYPE xaiRenorm3D(const xai_pTile3D inTile,
                                   xai_pTile3D outTile,
                                   const uint16_t renormScale,
                                   const uint8_t renormShift);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D_S8(const xai_pTile3D inTile,
                                      xai_pTile3D outTile,
                                      const uint16_t renormScale,
                                      const uint8_t renormShift);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D_U8(const xai_pTile3D inTile,
                                      xai_pTile3D outTile,
                                      const uint16_t renormScale,
                                      const uint8_t renormShift);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D_S16(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const uint16_t renormScale,
                                       const uint8_t renormShift);

_XAI_API_ XAI_ERR_TYPE xaiRenormVQ3D_S16_WHD(const xai_pTile3D inTile,
                                             const xai_pArray scaleArray,
                                             xai_pTile3D outTile,
                                             const uint8_t renormShift);


_XAI_API_ XAI_ERR_TYPE xaiRenormVQ3D_S16_DWH(const xai_pTile3D inTile,
                                             const xai_pArray scaleArray,
                                             xai_pTile3D outTile,
                                             const uint8_t renormShift);
_XAI_API_ XAI_ERR_TYPE xaiRenorm3D_U16(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const uint16_t renormScale,
                                       const uint8_t renormShift);

_XAI_API_ XAI_ERR_TYPE xaiRenormVQ3D_U16_WHD(const xai_pTile3D inTile,
                                             const xai_pArray scaleArray,
                                             xai_pTile3D outTile,
                                             const uint8_t renormShift);


_XAI_API_ XAI_ERR_TYPE xaiRenormVQ3D_U16_DWH(const xai_pTile3D inTile,
                                             const xai_pArray scaleArray,
                                             xai_pTile3D outTile,
                                             const uint8_t renormShift);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D2(const xai_pTile3D inTile,
                                    xai_pTile3D outTile,
                                    const xai_cnn_renorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D2_S8(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const xai_cnn_renorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D2_U8(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const xai_cnn_renorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D2_S16(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const xai_cnn_renorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D2_U16(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const xai_cnn_renorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D2_AsymQ_S8(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const xai_cnn_renorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D2_AsymQ_U8S8(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_renorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiRenorm3D2_AsymQ_S8U8(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               const xai_cnn_renorm_params *params);
/* Interp Variants */

_XAI_API_ XAI_ERR_TYPE xaiInterp3D(const xai_pTile3D inTile,
                                   xai_pTile3D outTile,
                                   const xai_cnn_interp3D_params *params);


_XAI_API_ XAI_ERR_TYPE xaiInterp3D_U8_WHD(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_interp3D_params *params);


_XAI_API_ XAI_ERR_TYPE xaiInterp3D_S8_WHD(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_S16_WHD(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_U8S16_WHD(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_S8S16_WHD(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_U8_DWH(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_S8_DWH(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_S16_DWH(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_U8S16_DWH(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_S8S16_DWH(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_SetTileParams(const xai_size3D *inFrame3DSize,
                                                 const xai_size3D *outFrame3DSize,
                                                 const xai_cnn_data_order dataOrder,
                                                 int32_t half_pixel_flag,
                                                 xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_SetTileParams(const xai_size3D *inFrame3DSize,
                                                        const xai_size3D *outFrame3DSize,
                                                        const xai_cnn_data_order dataOrder,
                                                        xai_cnn_resize_nearest3D_params *params);

/* ResizeNearest variants */

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_resize_nearest3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_S8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_resize_nearest3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_S16_WHD(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_resize_nearest3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_U8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_resize_nearest3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_S8U8_WHD(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_resize_nearest3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_S8_DWH(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_resize_nearest3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_S16_DWH(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_resize_nearest3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_U8_DWH(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_resize_nearest3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_S8U8_DWH(const xai_pTile3D inTile,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_resize_nearest3D_params *params);

/* RELU */
_XAI_API_ XAI_ERR_TYPE xaiLeakyRELU(const xai_pTile3D inTile,
                                    xai_pTile3D outTile,
                                    const XAI_Q15 slope);

_XAI_API_ XAI_ERR_TYPE xaiLeakyRELU_S8(const xai_pTile3D inTile,
                                       xai_pTile3D outTile,
                                       const XAI_Q15 slope);

_XAI_API_ XAI_ERR_TYPE xaiLeakyRELU_S16(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const XAI_Q15 slope);

_XAI_API_ XAI_ERR_TYPE xaiLeakyRELU_S16S8(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const XAI_Q15 slope);


_XAI_API_ XAI_ERR_TYPE xaiRELU(const xai_pTile3D inTile,
                               xai_pTile3D outTile,
                               const uint8_t minVal,
                               const uint8_t maxVal);

_XAI_API_ XAI_ERR_TYPE xaiRELU_U8(const xai_pTile3D inTile,
                                  xai_pTile3D outTile,
                                  const uint8_t minVal,
                                  const uint8_t maxVal);

_XAI_API_ XAI_ERR_TYPE xaiRELU_S8U8(const xai_pTile3D inTile,
                                    xai_pTile3D outTile,
                                    const uint8_t minVal,
                                    const uint8_t maxVal);

/* PRELU*/
_XAI_API_ XAI_ERR_TYPE xaiPRELU3D_S8_WHD(const xai_pTile3D inTile,
                                         const xai_pTile3D slopeArray,
                                         xai_pTile3D outTile,
                                         const uint8_t outputShift);

_XAI_API_ XAI_ERR_TYPE xaiPRELU3D_S16_WHD(const xai_pTile3D inTile,
                                          const xai_pTile3D slopeArray,
                                          xai_pTile3D outTile,
                                          const uint8_t outputShift);

_XAI_API_ XAI_ERR_TYPE xaiPRELU3D_S8_DWH(const xai_pTile3D inTile,
                                         const xai_pTile3D slopeArray,
                                         xai_pTile3D outTile,
                                         const uint8_t outputShift);

_XAI_API_ XAI_ERR_TYPE xaiPRELU3D_S16_DWH(const xai_pTile3D inTile,
                                          const xai_pTile3D slopeArray,
                                          xai_pTile3D outTile,
                                          const uint8_t outputShift);

_XAI_API_ XAI_ERR_TYPE xaiPRELU3D(const xai_pTile3D inTile,
                                  const xai_pTile3D slopeArray,
                                  xai_pTile3D outTile,
                                  const uint8_t outputShift);

_XAI_API_ XAI_ERR_TYPE xaiRELUScale(const xai_pTile3D inTile,
                                    xai_pTile3D outTile,
                                    const int16_t scale,
                                    const uint8_t shift,
                                    const int8_t offset,
                                    const uint8_t minVal,
                                    const uint8_t maxVal);

_XAI_API_ XAI_ERR_TYPE xaiRELUScale_S8U8(const xai_pTile3D inTile,
                                         xai_pTile3D outTile,
                                         const int16_t scale,
                                         const uint8_t shift,
                                         const int8_t offset,
                                         const uint8_t minVal,
                                         const uint8_t maxVal);

_XAI_API_ XAI_ERR_TYPE xaiRELU16(const xai_pTile3D inTile,
                                 xai_pTile3D outTile,
                                 const int32_t minVal,
                                 const int32_t maxVal);

_XAI_API_ XAI_ERR_TYPE xaiRELU16_S16I16(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const int32_t minVal,
                                        const int32_t maxVal);

_XAI_API_ XAI_ERR_TYPE xaiRELU16_U16I16(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const int32_t minVal,
                                        const int32_t maxVal);

/* Modified Relu for BN + Depthwise Clip operation */
_XAI_API_ XAI_ERR_TYPE xaiChannelwiseClip(const xai_pTile3D inTile,
                                          const xai_pArray thresholdMax,
                                          const xai_pArray thresholdMin,
                                          xai_pTile3D outTile,
                                          const xai_cnn_relu_params *params);

_XAI_API_ XAI_ERR_TYPE xaiChannelwiseClip_S8_DWH(const xai_pTile3D inTile,
                                                 const xai_pArray thresholdMax,
                                                 const xai_pArray thresholdMin,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_relu_params *params);

_XAI_API_ XAI_ERR_TYPE xaiChannelwiseClip_S8_WHD(const xai_pTile3D inTile,
                                                 const xai_pArray thresholdMax,
                                                 const xai_pArray thresholdMin,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_relu_params *params);

_XAI_API_ XAI_ERR_TYPE xaiChannelwiseClip_S16_DWH(const xai_pTile3D inTile,
                                                  const xai_pArray thresholdMax,
                                                  const xai_pArray thresholdMin,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_relu_params *params);

_XAI_API_ XAI_ERR_TYPE xaiChannelwiseClip_S16_WHD(const xai_pTile3D inTile,
                                                  const xai_pArray thresholdMax,
                                                  const xai_pArray thresholdMin,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_relu_params *params);

/* Batchnorm */

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S8_WHD(const xai_pTile3D inTile,
                                             const xai_pArray alphaArray,
                                             const xai_pArray betaArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_U8_WHD(const xai_pTile3D inTile,
                                             const xai_pArray alphaArray,
                                             const xai_pArray betaArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S8S16_WHD(const xai_pTile3D inTile,
                                                const xai_pArray alphaArray,
                                                const xai_pArray betaArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_U8S8_WHD(const xai_pTile3D inTile,
                                               const xai_pArray alphaArray,
                                               const xai_pArray betaArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_U8S16_WHD(const xai_pTile3D inTile,
                                                const xai_pArray alphaArray,
                                                const xai_pArray betaArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S16_WHD(const xai_pTile3D inTile,
                                              const xai_pArray alphaArray,
                                              const xai_pArray betaArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S8_DWH(const xai_pTile3D inTile,
                                             const xai_pArray alphaArray,
                                             const xai_pArray betaArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_U8_DWH(const xai_pTile3D inTile,
                                             const xai_pArray alphaArray,
                                             const xai_pArray betaArray,
                                             xai_pTile3D outTile,
                                             const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_U8S8_DWH(const xai_pTile3D inTile,
                                               const xai_pArray alphaArray,
                                               const xai_pArray betaArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_U8S16_DWH(const xai_pTile3D inTile,
                                                const xai_pArray alphaArray,
                                                const xai_pArray betaArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S8S16_DWH(const xai_pTile3D inTile,
                                                const xai_pArray alphaArray,
                                                const xai_pArray betaArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S16_DWH(const xai_pTile3D inTile,
                                              const xai_pArray alphaArray,
                                              const xai_pArray betaArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D(const xai_pTile3D inTile,
                                      const xai_pArray alphaArray,
                                      const xai_pArray betaArray,
                                      xai_pTile3D outTile,
                                      const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S8_Dim2(const xai_pTile3D inTile,
                                              const xai_pArray alphaArray,
                                              const xai_pArray betaArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S16_Dim2(const xai_pTile3D inTile,
                                               const xai_pArray alphaArray,
                                               const xai_pArray betaArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S8U8_Dim2(const xai_pTile3D inTile,
                                                const xai_pArray alphaArray,
                                                const xai_pArray betaArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_S8S16_Dim2(const xai_pTile3D inTile,
                                                 const xai_pArray alphaArray,
                                                 const xai_pArray betaArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_U8_Dim2(const xai_pTile3D inTile,
                                              const xai_pArray alphaArray,
                                              const xai_pArray betaArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_U8S8_Dim2(const xai_pTile3D inTile,
                                                const xai_pArray alphaArray,
                                                const xai_pArray betaArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_U8S16_Dim2(const xai_pTile3D inTile,
                                                 const xai_pArray alphaArray,
                                                 const xai_pArray betaArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_Dim2(const xai_pTile3D inTile,
                                           const xai_pArray alphaArray,
                                           const xai_pArray betaArray,
                                           xai_pTile3D outTile,
                                           const xai_cnn_batchnorm_params *params);

/* ArgMax */
_XAI_API_ XAI_ERR_TYPE xaiArgmax_S8(const xai_pTile3D inTile,
                                    xai_pTile3D outTileIdx,
                                    xai_pTile3D outTileVal,
                                    xai_pTile2D extraValCnt,
                                    xai_pArray sortedIdxArr,
                                    xai_pArray sortedValArr,
                                    const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin_S8(const xai_pTile3D inTile,
                                    xai_pTile3D outTileIdx,
                                    xai_pTile3D outTileVal,
                                    xai_pTile2D extraValCnt,
                                    xai_pArray sortedIdxArr,
                                    xai_pArray sortedValArr,
                                    const uint16_t numSmallestVal);

_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_dim1(const xai_pTile3D inTile,
                                        xai_pArray bufArray,
                                        xai_pTile3D outTileIdx,
                                        xai_pTile3D outTileVal,
                                        const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_dim1(const xai_pTile3D inTile,
                                        xai_pArray bufArray,
                                        xai_pTile3D outTileIdx,
                                        xai_pTile3D outTileVal,
                                        const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_S8_dim1(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_S8_dim1(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numSmallestVal);

_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_U8_dim1(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_U8_dim1(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_S16_dim1(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_S16_dim1(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_U16_dim1(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_U16_dim1(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_dim2(const xai_pTile3D inTile,
                                        xai_pArray bufArray,
                                        xai_pTile3D outTileIdx,
                                        xai_pTile3D outTileVal,
                                        const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_dim2(const xai_pTile3D inTile,
                                        xai_pArray bufArray,
                                        xai_pTile3D outTileIdx,
                                        xai_pTile3D outTileVal,
                                        const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_S8_dim2(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_S8_dim2(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_U8_dim2(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_U8_dim2(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numSmallestVal);

_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_S16_dim2(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_S16_dim2(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_U16_dim2(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_U16_dim2(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_dim3(const xai_pTile3D inTile,
                                        xai_pArray bufArray,
                                        xai_pTile3D outTileIdx,
                                        xai_pTile3D outTileVal,
                                        const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_dim3(const xai_pTile3D inTile,
                                        xai_pArray bufArray,
                                        xai_pTile3D outTileIdx,
                                        xai_pTile3D outTileVal,
                                        const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_S8_dim3(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_S8_dim3(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_U8_dim3(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_U8_dim3(const xai_pTile3D inTile,
                                           xai_pArray bufArray,
                                           xai_pTile3D outTileIdx,
                                           xai_pTile3D outTileVal,
                                           const uint16_t numSmallestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_S16_dim3(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_S16_dim3(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numSmallestVal);

_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_U16_dim3(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_U16_dim3(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numSmallestVal);
/*argmax merger variants*/
_XAI_API_ XAI_ERR_TYPE xaiMergeTopKArgmax3D_S8_dim1(const xai_pTile3D inTileIdx,
                                                    const xai_pTile3D inTileVal,
                                                    const xai_pArray inPtrOffsetArr,
                                                    xai_pArray bufArray,
                                                    xai_pTile3D outTileIdx,
                                                    xai_pTile3D outTileVal,
                                                    const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiMergeTopKArgmax3D_U8_dim1(const xai_pTile3D inTileIdx,
                                                    const xai_pTile3D inTileVal,
                                                    const xai_pArray inPtrOffsetArr,
                                                    xai_pArray bufArray,
                                                    xai_pTile3D outTileIdx,
                                                    xai_pTile3D outTileVal,
                                                    const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiMergeTopKArgmax3D_S16_dim1(const xai_pTile3D inTileIdx,
                                                     const xai_pTile3D inTileVal,
                                                     const xai_pArray inPtrOffsetArr,
                                                     xai_pArray bufArray,
                                                     xai_pTile3D outTileIdx,
                                                     xai_pTile3D outTileVal,
                                                     const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiMergeTopKArgmax3D_U16_dim1(const xai_pTile3D inTileIdx,
                                                     const xai_pTile3D inTileVal,
                                                     const xai_pArray inPtrOffsetArr,
                                                     xai_pArray bufArray,
                                                     xai_pTile3D outTileIdx,
                                                     xai_pTile3D outTileVal,
                                                     const uint16_t numLargestVal);
_XAI_API_ XAI_ERR_TYPE xaiMergeTopKArgmax3D_dim1(const xai_pTile3D inTileIdx,
                                                 const xai_pTile3D inTileVal,
                                                 const xai_pArray inPtrOffsetArr,
                                                 xai_pArray bufArray,
                                                 xai_pTile3D outTileIdx,
                                                 xai_pTile3D outTileVal,
                                                 const uint16_t numLargestVal);
/* SoftMax Variants */

/* 1D variant */
_XAI_API_ XAI_ERR_TYPE xaiSoftmax_S16U16(const xai_pArray input,
                                         const xai_pArray lutArray,
                                         xai_pArray output,
                                         const xai_cnn_softmax_params *params);

/* 3D variant */

_XAI_API_ XAI_ERR_TYPE xaiCalcMaxval3D_S8(const xai_pTile3D inTile,
                                          xai_pArray maxValArr,
                                          xai_cnn_maxval_params *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcMaxval3D_S16(const xai_pTile3D inTile,
                                           xai_pArray maxValArr,
                                           xai_cnn_maxval_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Dim1(const xai_pTile3D input,
                                         const xai_pArray lutArray, xai_pArray buffArray, xai_pTile3D output,
                                         const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8U8_Dim1(const xai_pTile3D inTile,
                                              const xai_pArray lutArray,
                                              xai_pArray buffArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8AS8_Dim1(const xai_pTile3D inTile,
                                               const xai_pArray lutArray,
                                               xai_pArray buffArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8AS8_Dim2(const xai_pTile3D input,
                                               const xai_pArray lutArray,
                                               xai_pArray bufArray,
                                               xai_pTile3D output,
                                               const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8AS8_Dim3(const xai_pTile3D inTile,
                                               const xai_pArray lutArray,
                                               xai_pArray bufArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_softmax_params  *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8AS8(const xai_pTile3D inTile,
                                          const xai_pArray lutArray,
                                          xai_pArray buffArray,
                                          xai_pTile3D outTile,
                                          const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S16U16_Dim1(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S16U16_Dim2(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_softmax_params  *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8U8_Dim3(const xai_pTile3D inTile,
                                              const xai_pArray lutArray,
                                              xai_pArray bufArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_softmax_params  *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Dim3(const xai_pTile3D inTile,
                                         const xai_pArray lutArray,
                                         xai_pArray bufArray,
                                         xai_pTile3D outTile,
                                         const xai_cnn_softmax_params  *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S16U16_Dim3(const xai_pTile3D inTile,
                                                const xai_pArray lutArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_softmax_params  *params);

/* Faster performing immplementation of Softmax3D along Dim1*/
_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Dim1_fast(const xai_pTile3D inTile,
                                              const xai_pArray lutArray,
                                              xai_pArray buffArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_softmax_params *params);

/* Faster performing immplementation of Softmax3D along Dim2*/
_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Dim2_fast(const xai_pTile3D inTile,
                                              const xai_pArray lutArray,
                                              xai_pArray buffArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_softmax_params *params);

/* Faster performing immplementation of Softmax3D along Dim3*/
_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Dim3_fast(const xai_pTile3D inTile,
                                              const xai_pArray lutArray,
                                              xai_pArray buffArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8U8_Dim1_fast(const xai_pTile3D inTile,
                                                   const xai_pArray lutArray,
                                                   xai_pArray buffArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_softmax_params *params);

/* Faster performing immplementation of Softmax3D along Dim1*/
_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S16U16_Dim1_fast(const xai_pTile3D inTile,
                                                     const xai_pArray lutArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_softmax_params *params);

/* Faster performing immplementation of Softmax3D along Dim2*/
_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S16U16_Dim2_fast(const xai_pTile3D inTile,
                                                     const xai_pArray lutArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_softmax_params  *params);

/* Faster performing immplementation of Softmax3D along Dim3*/
_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S16U16_Dim3_fast(const xai_pTile3D inTile,
                                                     const xai_pArray lutArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_softmax_params  *params);

/* Input 3D MxN */
_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Mclasses(const xai_pTile3D input,
                                             const xai_pArray lutArray,
                                             xai_pArray buffArray,
                                             xai_pTile3D output,
                                             const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Mclasses_S16U16(const xai_pTile3D input,
                                                    const xai_pArray lutArray,
                                                    xai_pTile3D output,
                                                    const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Mclasses_S8U8(const xai_pTile3D input,
                                                  const xai_pArray lutArray,
                                                  xai_pArray buffArray,
                                                  xai_pTile3D output,
                                                  const xai_cnn_softmax_params *params);

/* Input 3D NxM */
_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Ndata(const xai_pTile3D input,
                                          const xai_pArray lutArray,
                                          xai_pArray bufArray,
                                          xai_pTile3D output,
                                          const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Ndata_S8U8(const xai_pTile3D input,
                                               const xai_pArray lutArray,
                                               xai_pArray bufArray,
                                               xai_pTile3D output,
                                               const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S8U8_Dim2(const xai_pTile3D input,
                                                   const xai_pArray lutArray,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D output,
                                                   const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8U8_Dim2(const xai_pTile3D input,
                                              const xai_pArray lutArray,
                                              xai_pArray bufArray,
                                              xai_pTile3D output,
                                              const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Dim2(const xai_pTile3D input,
                                         const xai_pArray lutArray,
                                         xai_pArray bufArray,
                                         xai_pTile3D output,
                                         const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8U8_Dim2_fast(const xai_pTile3D input,
                                                   const xai_pArray lutArray,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D output,
                                                   const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_S8U8_Dim3_fast(const xai_pTile3D input,
                                                   const xai_pArray lutArray,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D output,
                                                   const xai_cnn_softmax_params *params);

/* Input 3D NxM */

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_Dim1(const xai_pTile3D input,
                                              const xai_pArray lutArray, xai_pArray buffArray, xai_pTile3D output,
                                              const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_Dim2(const xai_pTile3D input,
                                              const xai_pArray lutArray, xai_pArray buffArray, xai_pTile3D output,
                                              const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_Dim3(const xai_pTile3D input,
                                              const xai_pArray lutArray, xai_pArray buffArray, xai_pTile3D output,
                                              const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_Ndata_S16U16(const xai_pTile3D input,
                                                 const xai_pArray lutArray,
                                                 xai_pTile3D output,
                                                 const xai_cnn_softmax_params *params);


_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S8U8_Dim1(const xai_pTile3D inTile,
                                                   const xai_pArray lutArray,
                                                   xai_pArray bufArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S16U16_Dim1(const xai_pTile3D inTile,
                                                     const xai_pArray lutArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S16U16_Dim2(const xai_pTile3D inTile,
                                                     const xai_pArray lutArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_softmax_params  *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S8U8_Dim3(const xai_pTile3D inTile,
                                                   const xai_pArray lutArray,
                                                   xai_pArray buffArray,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S16U16_Dim3(const xai_pTile3D inTile,
                                                     const xai_pArray lutArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_softmax_params  *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_Dim1_fast(const xai_pTile3D input,
                                                   const xai_pArray lutArray, xai_pArray buffArray, xai_pTile3D output,
                                                   const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_Dim2_fast(const xai_pTile3D input,
                                                   const xai_pArray lutArray, xai_pArray buffArray, xai_pTile3D output,
                                                   const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_Dim3_fast(const xai_pTile3D input,
                                                   const xai_pArray lutArray, xai_pArray buffArray, xai_pTile3D output,
                                                   const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S8U8_Dim1_fast(const xai_pTile3D inTile,
                                                        const xai_pArray lutArray,
                                                        xai_pArray buffArray,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S16U16_Dim1_fast(const xai_pTile3D inTile,
                                                          const xai_pArray lutArray,
                                                          xai_pTile3D outTile,
                                                          const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S8U8_Dim2_fast(const xai_pTile3D inTile,
                                                        const xai_pArray lutArray,
                                                        xai_pArray buffArray,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S16U16_Dim2_fast(const xai_pTile3D inTile,
                                                          const xai_pArray lutArray,
                                                          xai_pTile3D outTile,
                                                          const xai_cnn_softmax_params  *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S8U8_Dim3_fast(const xai_pTile3D inTile,
                                                        const xai_pArray lutArray,
                                                        xai_pArray buffArray,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_S16U16_Dim3_fast(const xai_pTile3D inTile,
                                                          const xai_pArray lutArray,
                                                          xai_pTile3D outTile,
                                                          const xai_cnn_softmax_params  *params);

/* Input 3D MxN */

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_Mclasses_S16U16(const xai_pTile3D input,
                                                         const xai_pArray lutArray,
                                                         xai_pTile3D output,
                                                         const xai_cnn_softmax_params *params);

/* Input 3D NxM */

_XAI_API_ XAI_ERR_TYPE xaiSoftmax3D_gMax_Ndata_S16U16(const xai_pTile3D input,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D output,
                                                      const xai_cnn_softmax_params *params);

/* Softmax 8-bit variant */
/* 1D variant */
_XAI_API_ XAI_ERR_TYPE xaiSoftmax_S8U8(const xai_pArray input,
                                       const xai_pArray lutArray,
                                       xai_pArray buffArray,
                                       xai_pArray output,
                                       const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax(const xai_pArray input,
                                  const xai_pArray lutArray,
                                  xai_pArray buffArray,
                                  xai_pArray output,
                                  const xai_cnn_softmax_params *params);


_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D_S8U8_Dim1(const xai_pTile3D inTile,
                                                    const xai_pTile3D maskTile,
                                                    const xai_pArray lutArray,
                                                    xai_pArray buffArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D_S8AS8_Dim1(const xai_pTile3D inTile,
                                                     const xai_pTile3D maskTile,
                                                     const xai_pArray lutArray,
                                                     xai_pArray buffArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D_S16U16_Dim1(const xai_pTile3D inTile,
                                                      const xai_pTile3D maskTile,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D_S8U8_Dim2(const xai_pTile3D inTile,
                                                    const xai_pTile3D maskTile,
                                                    const xai_pArray lutArray,
                                                    xai_pArray buffArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D_S8AS8_Dim2(const xai_pTile3D inTile,
                                                     const xai_pTile3D maskTile,
                                                     const xai_pArray lutArray,
                                                     xai_pArray buffArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D_S16U16_Dim2(const xai_pTile3D inTile,
                                                      const xai_pTile3D maskTile,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D_S8U8_Dim3(const xai_pTile3D inTile,
                                                    const xai_pTile3D maskTile,
                                                    const xai_pArray lutArray,
                                                    xai_pArray buffArray,
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D_S8AS8_Dim3(const xai_pTile3D inTile,
                                                     const xai_pTile3D maskTile,
                                                     const xai_pArray lutArray,
                                                     xai_pArray buffArray,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D_S16U16_Dim3(const xai_pTile3D inTile,
                                                      const xai_pTile3D maskTile,
                                                      const xai_pArray lutArray,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_softmax_params *params);

_XAI_API_ XAI_ERR_TYPE xaiMaskedSoftmax3D(const xai_pTile3D inTile,
                                          const xai_pTile3D maskTile,
                                          const xai_pArray lutArray,
                                          xai_pArray buffArray,
                                          xai_pTile3D outTile,
                                          const xai_cnn_softmax_params *params);


/*Sigmoid3D functions*/

_XAI_API_ XAI_ERR_TYPE xaiSigmoid3D(const xai_pTile3D inTile,
                                    const xai_pArray lutArray,
                                    xai_pTile3D outTile,
                                    const int16_t shift,
                                    const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiSigmoid3D_S8U8(const xai_pTile3D inTile,
                                         const xai_pArray lutArray,
                                         xai_pTile3D outTile,
                                         const int16_t shift,
                                         const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiSigmoid3D_S8AS8(const xai_pTile3D inTile,
                                          const xai_pArray lutArray,
                                          xai_pTile3D outTile,
                                          const int16_t shift,
                                          const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiSigmoid3D_S8(const xai_pTile3D inTile,
                                       const xai_pArray lutArray,
                                       xai_pTile3D outTile,
                                       const int16_t shift,
                                       const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiSigmoid3D_S16U8(const xai_pTile3D inTile,
                                          const xai_pArray lutArray,
                                          xai_pTile3D outTile,
                                          const int16_t shift,
                                          const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiSigmoid3D_S16S8(const xai_pTile3D inTile,
                                          const xai_pArray lutArray,
                                          xai_pTile3D outTile,
                                          const int16_t shift,
                                          const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiSigmoid3D_S16U16(const xai_pTile3D inTile,
                                           const xai_pArray lutArray,
                                           xai_pTile3D outTile,
                                           const int16_t shift,
                                           const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiSigmoid3D_S16(const xai_pTile3D inTile,
                                        const xai_pArray lutArray,
                                        xai_pTile3D outTile,
                                        const int16_t shift,
                                        const int16_t scale);

/*Tanh3D hyperbolic functions*/


_XAI_API_ XAI_ERR_TYPE xaiTanh3D(const xai_pTile3D inTile,
                                 const xai_pArray lutArray,
                                 xai_pTile3D outTile,
                                 const int16_t shift,
                                 const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiTanh3D_S8(const xai_pTile3D inTile,
                                    const xai_pArray lutArray,
                                    xai_pTile3D outTile,
                                    const int16_t shift,
                                    const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiTanh3D_S16S8(const xai_pTile3D inTile,
                                       const xai_pArray lutArray,
                                       xai_pTile3D outTile,
                                       const int16_t shift,
                                       const int16_t scale);

_XAI_API_ XAI_ERR_TYPE xaiTanh3D_S16(const xai_pTile3D inTile,
                                     const xai_pArray lutArray,
                                     xai_pTile3D outTile,
                                     const int16_t shift,
                                     const int16_t scale);


/* Eltwise Add */
_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D(const xai_pTile3D inTile1,
                                       const xai_pTile3D inTile2,
                                       xai_pTile3D outTile,
                                       const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j1_S8I8(const xai_pTile3D inTile1,
                                               const xai_pTile3D inTile2,
                                               xai_pTile3D outTile,
                                               const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j1_U8(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile,
                                             const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j1_S8U8I8(const xai_pTile3D inTile1,
                                                 const xai_pTile3D inTile2,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j1_S16I16(const xai_pTile3D inTile1,
                                                 const xai_pTile3D inTile2,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_eltwise_params *param);

/* Eltwise Add j2 variants */

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j2_S8I8_DWH(const xai_pTile3D inTile1,
                                                   const xai_pTile3D inTile2,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j2_U8_DWH(const xai_pTile3D inTile1,
                                                 const xai_pTile3D inTile2,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j2_S16I16_DWH(const xai_pTile3D inTile1,
                                                     const xai_pTile3D inTile2,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_eltwise_params *param);

/* Eltwise Add j1j2 variants */

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j1j2_S8I8_DWH(const xai_pTile3D inTile1,
                                                     const xai_pTile3D inTile2,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j1j2_U8_DWH(const xai_pTile3D inTile1,
                                                   const xai_pTile3D inTile2,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_j1j2_S16I16_DWH(const xai_pTile3D inTile1,
                                                       const xai_pTile3D inTile2,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_eltwise_params *param);

/* Eltwise Subtraction */

_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D(const xai_pTile3D inTile1,
                                       const xai_pTile3D inTile2,
                                       xai_pTile3D outTile,
                                       const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_j1_S8I8(const xai_pTile3D inTile1,
                                               const xai_pTile3D inTile2,
                                               xai_pTile3D outTile,
                                               const xai_cnn_eltwise_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_j1_S16I16(const xai_pTile3D inTile1,
                                                 const xai_pTile3D inTile2,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_eltwise_params *param);

/* Eltwise Multiply */
_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D(const xai_pTile3D inTile1,
                                       const xai_pTile3D inTile2,
                                       xai_pTile3D outTile,
                                       const xai_cnn_eltwiseMul_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_S8(const xai_pTile3D inTile1,
                                          const xai_pTile3D inTile2,
                                          xai_pTile3D outTile,
                                          const xai_cnn_eltwiseMul_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_S8S16(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile,
                                             const xai_cnn_eltwiseMul_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_S8U8S8(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile,
                                              const xai_cnn_eltwiseMul_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_S8U8U8(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile,
                                              const xai_cnn_eltwiseMul_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_S8U8S16(const xai_pTile3D inTile1,
                                               const xai_pTile3D inTile2,
                                               xai_pTile3D outTile,
                                               const xai_cnn_eltwiseMul_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_U8I8(const xai_pTile3D inTile1,
                                            const xai_pTile3D inTile2,
                                            xai_pTile3D outTile,
                                            const xai_cnn_eltwiseMul_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_U8S16(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile,
                                             const xai_cnn_eltwiseMul_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_S16(const xai_pTile3D inTile1,
                                           const xai_pTile3D inTile2,
                                           xai_pTile3D outTile,
                                           const xai_cnn_eltwiseMul_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_U16(const xai_pTile3D inTile1,
                                           const xai_pTile3D inTile2,
                                           xai_pTile3D outTile,
                                           const xai_cnn_eltwiseMul_params *param);

/* Eltwise Exponent */

_XAI_API_ XAI_ERR_TYPE xaiExp3D(const xai_pTile3D inTile,
                                const xai_pArray lutArray,
                                xai_pTile3D outTile,
                                const xai_cnn_exponent_params *params);

_XAI_API_ XAI_ERR_TYPE xaiExp3D_S16(const xai_pTile3D inTile,
                                    const xai_pArray lutArray,
                                    xai_pTile3D outTile,
                                    const xai_cnn_exponent_params *params);

_XAI_API_ XAI_ERR_TYPE xaiExp3D_S16U16(const xai_pTile3D inTile,
                                       const xai_pArray lutArray,
                                       xai_pTile3D outTile,
                                       const xai_cnn_exponent_params *params);


/* Maxout */
_XAI_API_ XAI_ERR_TYPE xaiMaxout3D(const xai_pTile3D inTile,
                                   xai_pTile3D outTile,
                                   const uint16_t kSize);

_XAI_API_ XAI_ERR_TYPE xaiMaxout3D_S8_WHD(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const uint16_t kSize);

_XAI_API_ XAI_ERR_TYPE xaiMaxout3D_S8_DWH(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const uint16_t kSize);

/* Mean subtraction */
_XAI_API_ XAI_ERR_TYPE xaiMeanSubtraction3D_U8S8(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const uint8_t mean,
                                                 const uint16_t scale,
                                                 const uint8_t shift);
/* Generate LUT for LRN */
_XAI_API_ XAI_ERR_TYPE xaiLRNSpatial3D_generateLut(xai_pArray lutArray,
                                                   xai_cnn_lrn_spatial_params *params,
                                                   float alpha,
                                                   float beta,
                                                   float kValue,
                                                   int32_t maxSumOfSquares,
                                                   float qIn,
                                                   float qOut);

_XAI_API_ XAI_ERR_TYPE xaiLRNDepth3D_generateLut(xai_pArray lutArray,
                                                 xai_cnn_lrn_depth_params *params,
                                                 float alpha,
                                                 float beta,
                                                 float kValue,
                                                 int32_t maxSumOfSquares,
                                                 float qIn,
                                                 float qOut);

/* Generate LUT*/
_XAI_API_ XAI_ERR_TYPE xaiTanh_generateLut(xai_pArray lutArray,
                                           const int32_t inpDataType,
                                           const uint8_t lutQfactor,
                                           const float qIn);

_XAI_API_ XAI_ERR_TYPE xaiTanh3D_generateLut(const xai_pTile3D inTile,
                                             xai_pArray lutArray,
                                             const uint16_t tanh_cutoff);

_XAI_API_ XAI_ERR_TYPE xaiSigmoid3D_generateLut(const xai_pTile3D inTile,
                                                xai_pArray lutArray,
                                                const uint16_t sigmoidCutoff);

_XAI_API_ XAI_ERR_TYPE xaiSigmoid_generateLut(xai_pArray lutArray,
                                              const int32_t inpDataType,
                                              const uint8_t lutQfactor,
                                              const float qIn);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax_generateLut_S16(xai_pArray lutArray,
                                                  xai_cnn_softmax_params *params,
                                                  const uint16_t qFactorLUT,
                                                  const float qIn);

_XAI_API_ XAI_ERR_TYPE xaiSoftmax_generateLut_S8(xai_pArray lutArray,
                                                 xai_cnn_softmax_params *params,
                                                 const uint16_t qFactorLUT,
                                                 const float qIn
                                                 );

_XAI_API_ XAI_ERR_TYPE xaiSoftmax_generateLut(xai_pArray lutArray,
                                              const xai_pTile3D input, xai_cnn_softmax_params *params,
                                              const uint16_t qFactorLUT,
                                              const float qIn
                                              );
_XAI_API_ XAI_ERR_TYPE xaiExp_generateLUT(float inScaleF, int inQPDepth, float outScaleF,
                                          int outQPDepth, xai_pArray tables,
                                          xai_cnn_exponent_params *params, const xai_pArray qXBits,
                                          const xai_pArray qYBits);

_XAI_API_ XAI_ERR_TYPE xaiStdDevRecip_generateLut(xai_pArray rSqrtTable,
                                                  const xai_dataType dataType);

/* Wrappper Functions */
_XAI_API_ XAI_ERR_TYPE xaiWrapper3D_TYPE_1(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           void *function2DPtr);

_XAI_API_ XAI_ERR_TYPE xaiWrapper3D_TYPE_2(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           int32_t value,
                                           void *function2DPtr);

_XAI_API_ XAI_ERR_TYPE xaiWrapper3D_TYPE_3(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           xai_pArray pArray,
                                           void *function2DPtr);

_XAI_API_ XAI_ERR_TYPE xaiWrapper3D_TYPE_4(const xai_pTile3D inTile0,
                                           const xai_pTile3D inTile1,
                                           xai_pTile3D outTile,
                                           void *function2DPtr);

_XAI_API_ XAI_ERR_TYPE xaiWrapper3D_TYPE_5(const xai_pTile3D inTile0,
                                           const xai_pTile3D inTile1,
                                           xai_pTile3D outTile,
                                           int32_t value,
                                           void *function2DPtr);

_XAI_API_ XAI_ERR_TYPE xaiWrapper3D_TYPE_6(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           xai_pTile2D tmpTile,
                                           int32_t value,
                                           void *function2DPtr);

_XAI_API_ XAI_ERR_TYPE xaiWrapper3D_TYPE_7(const xai_pTile3D inTile,
                                           int32_t *counter,
                                           void *function2DPtr);

_XAI_API_ XAI_ERR_TYPE xaiWrapper3D_TYPE_8(const xai_pTile3D inTile,
                                           int32_t value,
                                           int32_t *counter,
                                           void *function2DPtr);

_XAI_API_ XAI_ERR_TYPE xaiWrapper3D_TYPE_9(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           XAI_Q13_18 xscale,
                                           XAI_Q13_18 yscale,
                                           XAI_Q13_18 xshift,
                                           XAI_Q13_18 yshift,
                                           void *function2DPtr);


_XAI_API_ XAI_ERR_TYPE xaiDeConvGetDim4D_WHDN(const xai_pTile4D coeffTile,
                                              xai_pTile4D subCoeffInfo[],
                                              uint16_t *numSubKernels,
                                              const uint8_t strideX,
                                              const uint8_t strideY,
                                              const uint8_t getNumKernelsFlag);

_XAI_API_ XAI_ERR_TYPE xaiDeConvGetDim3D_WHD(const xai_pTile3D coeffTile,
                                             xai_pTile3D subCoeffInfo[],
                                             uint16_t *numSubKernels,
                                             const uint8_t strideX,
                                             const uint8_t strideY,
                                             const uint8_t getNumKernelsFlag);

_XAI_API_ XAI_ERR_TYPE xaiDeConvReOrder4D_I8_WHDN(const xai_pTile4D inTile,
                                                  xai_pTile4D subCoeffs[],
                                                  const xai_cnn_conv_params *param,
                                                  const uint8_t transposeCoeffsFlag);

_XAI_API_ XAI_ERR_TYPE xaiDeConvReOrder3D_I8_WHD(const xai_pTile3D inTile,
                                                 xai_pTile3D subCoeffs[],
                                                 const xai_cnn_depthwiseDilatedConv_params *param,
                                                 const uint8_t transposeCoeffsFlag);

_XAI_API_ XAI_ERR_TYPE xaiDeConvInterleave3D_I8_WHD(const xai_pTile3D inTile[],
                                                    xai_pTile3D outTile,
                                                    const xai_cnn_conv_params *convParams);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseDeConvInterleave3D_I8_WHD(const xai_pTile3D inTile[],
                                                             xai_pTile3D outTile,
                                                             const xai_cnn_depthwiseDilatedConv_params *convParams);

_XAI_API_ XAI_ERR_TYPE xaiDeConvInterleave3D_I16_WHD(const xai_pTile3D inTile[],
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_conv_params *convParams);

_XAI_API_ XAI_ERR_TYPE xaiBiasExtend_S32_MOD(const xai_pArray inBiasArray,
                                             xai_pArray outBiasArray);

_XAI_API_ XAI_ERR_TYPE xaiOutScaleExtend_U16_MOD(const xai_pArray outScaleArray,
                                                 xai_pArray extendedOutScaleArray);

_XAI_API_ XAI_ERR_TYPE xaiDeConvGetDim4D_NDWH(const xai_pTile4D coeffTile,
                                              xai_pTile4D subCoeffInfo[],
                                              xai_pTile4D superCoeffInfo[],
                                              uint16_t *numSubKernels,
                                              uint16_t *numSuperKernels,
                                              const uint8_t strideX,
                                              const uint8_t strideY,
                                              const uint8_t getNumKernelsFlag);

_XAI_API_ XAI_ERR_TYPE xaiDeConvGetDim3D_DWH(const xai_pTile3D coeffTile,
                                             xai_pTile3D subCoeffInfo[],
                                             uint16_t *numSubKernels,
                                             const uint8_t strideX,
                                             const uint8_t strideY,
                                             const uint8_t getNumKernelsFlag);

_XAI_API_ XAI_ERR_TYPE xaiDeConvReOrder4D_I8_NDWH(const xai_pTile4D inTile,
                                                  xai_pTile4D subCoeffs[],
                                                  xai_pTile4D superCoeffs[],
                                                  const xai_cnn_conv_params *param,
                                                  const uint8_t transposeCoeffsFlag);

_XAI_API_ XAI_ERR_TYPE xaiDeConvReOrder3D_I8_DWH(const xai_pTile3D inTile,
                                                 xai_pTile3D subCoeffs[],
                                                 const xai_cnn_depthwiseDilatedConv_params *param,
                                                 const uint8_t transposeCoeffsFlag);

/*Permute Functions*/

_XAI_API_ XAI_ERR_TYPE xaiPermute4D(const xai_pTile4D inTile,
                                    xai_pTile4D outTile,
                                    const xai_cnn_permute4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPermute4D_I8(const xai_pTile4D inTile,
                                       xai_pTile4D outTile,
                                       const xai_cnn_permute4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPermute4D_I16(const xai_pTile4D inTile,
                                        xai_pTile4D outTile,
                                        const xai_cnn_permute4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPermute4D_I32(const xai_pTile4D inTile,
                                        xai_pTile4D outTile,
                                        const xai_cnn_permute4D_params* params);

_XAI_API_ XAI_ERR_TYPE xaiPermute4D2(const xai_pTile4D inTile,
                                     xai_pArray bufArray,
                                     xai_pTile4D outTile,
                                     const xai_cnn_permute4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPermute4D2_I8(const xai_pTile4D inTile,
                                        xai_pArray bufArray,
                                        xai_pTile4D outTile,
                                        const xai_cnn_permute4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPermute4D2_I16(const xai_pTile4D inTile,
                                         xai_pArray bufArray,
                                         xai_pTile4D outTile,
                                         const xai_cnn_permute4D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPermute4D2_I32(const xai_pTile4D inTile,
                                         xai_pArray bufArray,
                                         xai_pTile4D outTile,
                                         const xai_cnn_permute4D_params *params);

/*Shuffle variants*/

_XAI_API_ XAI_ERR_TYPE xaiShuffle3D(const xai_pTile3D inTile,
                                    xai_pTile3D outTile,
                                    const xai_cnn_shuffle3D_params *shuffParams);

_XAI_API_ XAI_ERR_TYPE xaiShuffle3D_I8_DWH(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_shuffle3D_params *shuffParams);

_XAI_API_ XAI_ERR_TYPE xaiShuffle3D_I16_DWH(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            const xai_cnn_shuffle3D_params *shuffParams);


_XAI_API_ XAI_ERR_TYPE xaiShuffle3D_I8_WHD(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_shuffle3D_params *shuffParams);


_XAI_API_ XAI_ERR_TYPE xaiShuffle3D_I16_WHD(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            const xai_cnn_shuffle3D_params *shuffParams);


/* Calc Normalize Wrapper Function */
_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_I8(const xai_pTile3D pInTile,
                                                   const xai_pArray rSqrtTable,
                                                   const xai_pArray recipTable,
                                                   xai_pArray buffArrSoS,
                                                   xai_pArray buffNSAShiftArray,
                                                   xai_pArray pNormScaleArr,
                                                   const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_S16(const xai_pTile3D pInTile,
                                                    const xai_pArray rSqrtTable,
                                                    xai_pArray buffArrSoS,
                                                    xai_pArray buffNSAShiftArray,
                                                    xai_pArray pNormScaleArr,
                                                    const xai_cnn_normalize3D_params *params);

/* Calc Normalize Variants*/
_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_S8_WHD(const xai_pTile3D pInTile,
                                                       const xai_pArray rSqrtTable,
                                                       const xai_pArray recipTable,
                                                       xai_pArray buffArrSoS,
                                                       xai_pArray buffNSAShiftArray,
                                                       xai_pArray pNormScaleArr,
                                                       const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_U8_WHD(const xai_pTile3D pInTile,
                                                       const xai_pArray rSqrtTable,
                                                       const xai_pArray recipTable,
                                                       xai_pArray buffArrSoS,
                                                       xai_pArray buffNSAShiftArray,
                                                       xai_pArray pNormScaleArr,
                                                       const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_S16_WHD(const xai_pTile3D pInTile,
                                                        const xai_pArray rSqrtTable,
                                                        xai_pArray buffArrSoS,
                                                        xai_pArray buffNSAShiftArray,
                                                        xai_pArray pNormScaleArr,
                                                        const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_S8_DWH(const xai_pTile3D pInTile,
                                                       const xai_pArray rSqrtTable,
                                                       const xai_pArray recipTable,
                                                       xai_pArray buffArrSoS,
                                                       xai_pArray buffNSAShiftArray,
                                                       xai_pArray pNormScaleArr,
                                                       const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_U8_DWH(const xai_pTile3D pInTile,
                                                       const xai_pArray rSqrtTable,
                                                       const xai_pArray recipTable,
                                                       xai_pArray buffArrSoS,
                                                       xai_pArray buffNSAShiftArray,
                                                       xai_pArray pNormScaleArr,
                                                       const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_S16_DWH(const xai_pTile3D pInTile,
                                                        const xai_pArray rSqrtTable,
                                                        xai_pArray buffArrSoS,
                                                        xai_pArray buffNSAShiftArray,
                                                        xai_pArray pNormScaleArr,
                                                        const xai_cnn_normalize3D_params *params);

/* Apply Scale Wrapper Function */
_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_I8(const xai_pTile3D InTile,
                                          const xai_pArray pNormScaleArr,
                                          const xai_pArray pQuantScaleTable,
										  const xai_pArray buffNSAShiftArray,
                                          xai_pTile3D pOutTile,
                                          const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_S16(const xai_pTile3D InTile,
                                           const xai_pArray pNormScaleArr,
                                           const xai_pArray pQuantScaleTable,
                                           const xai_pArray buffNSAShiftArray,
                                           xai_pTile3D pOutTile,
                                           const xai_cnn_normalize3D_params *params);

/* Apply Scale Variants */
_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_S8_WHD(const xai_pTile3D inTile,
                                              const xai_pArray pNormScaleArr,
                                              const xai_pArray pQuantScaleTable,
											  const xai_pArray buffNSAShiftArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_U8_WHD(const xai_pTile3D inTile,
                                              const xai_pArray pNormScaleArr,
                                              const xai_pArray pQuantScaleTable,
											  const xai_pArray buffNSAShiftArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_S16_WHD(const xai_pTile3D inTile,
                                               const xai_pArray pNormScaleArr,
                                               const xai_pArray pQuantScaleTable,
                                               const xai_pArray buffNSAShiftArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_S8_DWH(const xai_pTile3D inTile,
                                              const xai_pArray pNormScaleArr,
                                              const xai_pArray pQuantScaleTable,
											  const xai_pArray buffNSAShiftArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_U8_DWH(const xai_pTile3D inTile,
                                              const xai_pArray pNormScaleArr,
                                              const xai_pArray pQuantScaleTable,
											  const xai_pArray buffNSAShiftArray,
                                              xai_pTile3D outTile,
                                              const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_S16_DWH(const xai_pTile3D inTile,
                                               const xai_pArray pNormScaleArr,
                                               const xai_pArray pQuantScaleTable,
                                               const xai_pArray buffNSAShiftArray,
                                               xai_pTile3D outTile,
                                               const xai_cnn_normalize3D_params *params);

/*Generate LUT for normalize variants*/
_XAI_API_ XAI_ERR_TYPE xaiNormalize3D_generateLut(xai_pArray rSqrtTable,
                                                  xai_pArray recipTable,
                                                  const xai_cnn_normalize3D_params *params,
                                                  const xai_dataType dataType);

_XAI_API_ XAI_ERR_TYPE xaiNormalize3D_generateLut_S16(xai_pArray rSqrtTable,
                                                      const xai_cnn_normalize3D_params *params,
                                                      const xai_dataType dataType);

/* Instance Normalization API ref */

/* calcInstanceNorm APIs */

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D(const xai_pTile3D inTile,
                                                   xai_pArray meanArr,
                                                   xai_pArray recipArr,
                                                   xai_pArray buffArr,
                                                   xai_pArray buffArrSoS,
                                                   const xai_pArray rSqrtTable,
                                                   const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S8_WHD(const xai_pTile3D inTile,
                                                          xai_pArray meanArr,
                                                          xai_pArray recipArr,
                                                          xai_pArray buffArr,
                                                          xai_pArray buffArrSoS,
                                                          const xai_pArray rSqrtTable,
                                                          const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_U8_WHD(const xai_pTile3D inTile,
                                                          xai_pArray meanArr,
                                                          xai_pArray recipArr,
                                                          xai_pArray buffArr,
                                                          xai_pArray buffArrSoS,
                                                          const xai_pArray rSqrtTable,
                                                          const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S16_WHD(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_pArray rSqrtTable,
                                                           const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S8_DWH(const xai_pTile3D inTile,
                                                          xai_pArray meanArr,
                                                          xai_pArray recipArr,
                                                          xai_pArray buffArr,
                                                          xai_pArray buffArrSoS,
                                                          const xai_pArray rSqrtTable,
                                                          const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_U8_DWH(const xai_pTile3D inTile,
                                                          xai_pArray meanArr,
                                                          xai_pArray recipArr,
                                                          xai_pArray buffArr,
                                                          xai_pArray buffArrSoS,
                                                          const xai_pArray rSqrtTable,
                                                          const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S16_DWH(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_pArray rSqrtTable,
                                                           const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S8_Dim1(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_pArray rSqrtTable,
                                                           const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S8_Dim2(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_pArray rSqrtTable,
                                                           const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S8_Dim3(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_pArray rSqrtTable,
                                                           const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_U8_Dim1(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_pArray rSqrtTable,
                                                           const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_U8_Dim2(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_pArray rSqrtTable,
                                                           const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_U8_Dim3(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_pArray rSqrtTable,
                                                           const xai_cnn_instance_norm_param *params);


_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S16_Dim1(const xai_pTile3D inTile,
                                                            xai_pArray meanArr,
                                                            xai_pArray recipArr,
                                                            xai_pArray buffArr,
                                                            xai_pArray buffArrSoS,
                                                            const xai_pArray rSqrtTable,
                                                            const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S16_Dim2(const xai_pTile3D inTile,
                                                            xai_pArray meanArr,
                                                            xai_pArray recipArr,
                                                            xai_pArray buffArr,
                                                            xai_pArray buffArrSoS,
                                                            const xai_pArray rSqrtTable,
                                                            const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_S16_Dim3(const xai_pTile3D inTile,
                                                            xai_pArray meanArr,
                                                            xai_pArray recipArr,
                                                            xai_pArray buffArr,
                                                            xai_pArray buffArrSoS,
                                                            const xai_pArray rSqrtTable,
                                                            const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_Dim1(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        xai_pArray buffArr,
                                                        xai_pArray buffArrSoS,
                                                        const xai_pArray rSqrtTable,
                                                        const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_Dim2(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        xai_pArray buffArr,
                                                        xai_pArray buffArrSoS,
                                                        const xai_pArray rSqrtTable,
                                                        const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_Dim3(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        xai_pArray buffArr,
                                                        xai_pArray buffArrSoS,
                                                        const xai_pArray rSqrtTable,
                                                        const xai_cnn_instance_norm_param *params);

/* applyInstanceNorm APIs */

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D(const xai_pTile3D inTile,
                                              xai_pArray meanArr,
                                              xai_pArray recipArr,
                                              const xai_pArray alphaArr,
                                              const xai_pArray betaArr,
                                              xai_pTile3D outTile,
                                              const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8_WHD(const xai_pTile3D inTile,
                                                     xai_pArray meanArr,
                                                     xai_pArray recipArr,
                                                     const xai_pArray alphaArr,
                                                     const xai_pArray betaArr,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8_WHD(const xai_pTile3D inTile,
                                                     xai_pArray meanArr,
                                                     xai_pArray recipArr,
                                                     const xai_pArray alphaArr,
                                                     const xai_pArray betaArr,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8U8_WHD(const xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S8_WHD(const xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8S16_WHD(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S16_WHD(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S16_WHD(const xai_pTile3D inTile,
                                                      xai_pArray meanArr,
                                                      xai_pArray recipArr,
                                                      const xai_pArray alphaArr,
                                                      const xai_pArray betaArr,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8_DWH(const xai_pTile3D inTile,
                                                     xai_pArray meanArr,
                                                     xai_pArray recipArr,
                                                     const xai_pArray alphaArr,
                                                     const xai_pArray betaArr,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8_DWH(const xai_pTile3D inTile,
                                                     xai_pArray meanArr,
                                                     xai_pArray recipArr,
                                                     const xai_pArray alphaArr,
                                                     const xai_pArray betaArr,
                                                     xai_pTile3D outTile,
                                                     const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8U8_DWH(const xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S8_DWH(const xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8S16_DWH(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S16_DWH(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S16_DWH(const xai_pTile3D inTile,
                                                      xai_pArray meanArr,
                                                      xai_pArray recipArr,
                                                      const xai_pArray alphaArr,
                                                      const xai_pArray betaArr,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8_Dim1(const xai_pTile3D inTile,
                                                      xai_pArray meanArr,
                                                      xai_pArray recipArr,
                                                      const xai_pArray alphaArr,
                                                      const xai_pArray betaArr,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8_Dim1(const xai_pTile3D inTile,
                                                      xai_pArray meanArr,
                                                      xai_pArray recipArr,
                                                      const xai_pArray alphaArr,
                                                      const xai_pArray betaArr,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8U8_Dim1(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8S16_Dim1(const xai_pTile3D inTile,
                                                         xai_pArray meanArr,
                                                         xai_pArray recipArr,
                                                         const xai_pArray alphaArr,
                                                         const xai_pArray betaArr,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S8_Dim1(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S16_Dim1(const xai_pTile3D inTile,
                                                         xai_pArray meanArr,
                                                         xai_pArray recipArr,
                                                         const xai_pArray alphaArr,
                                                         const xai_pArray betaArr,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S16_Dim1(const xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8_Dim2(const xai_pTile3D inTile,
                                                      xai_pArray meanArr,
                                                      xai_pArray recipArr,
                                                      const xai_pArray alphaArr,
                                                      const xai_pArray betaArr,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8_Dim2(const xai_pTile3D inTile,
                                                      xai_pArray meanArr,
                                                      xai_pArray recipArr,
                                                      const xai_pArray alphaArr,
                                                      const xai_pArray betaArr,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8U8_Dim2(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8S16_Dim2(const xai_pTile3D inTile,
                                                         xai_pArray meanArr,
                                                         xai_pArray recipArr,
                                                         const xai_pArray alphaArr,
                                                         const xai_pArray betaArr,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S8_Dim2(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S16_Dim2(const xai_pTile3D inTile,
                                                         xai_pArray meanArr,
                                                         xai_pArray recipArr,
                                                         const xai_pArray alphaArr,
                                                         const xai_pArray betaArr,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S16_Dim2(const xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param * params);


_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8_Dim3(const xai_pTile3D inTile,
                                                      xai_pArray meanArr,
                                                      xai_pArray recipArr,
                                                      const xai_pArray alphaArr,
                                                      const xai_pArray betaArr,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8_Dim3(const xai_pTile3D inTile,
                                                      xai_pArray meanArr,
                                                      xai_pArray recipArr,
                                                      const xai_pArray alphaArr,
                                                      const xai_pArray betaArr,
                                                      xai_pTile3D outTile,
                                                      const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8U8_Dim3(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S8S16_Dim3(const xai_pTile3D inTile,
                                                         xai_pArray meanArr,
                                                         xai_pArray recipArr,
                                                         const xai_pArray alphaArr,
                                                         const xai_pArray betaArr,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S8_Dim3(const xai_pTile3D inTile,
                                                        xai_pArray meanArr,
                                                        xai_pArray recipArr,
                                                        const xai_pArray alphaArr,
                                                        const xai_pArray betaArr,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_U8S16_Dim3(const xai_pTile3D inTile,
                                                         xai_pArray meanArr,
                                                         xai_pArray recipArr,
                                                         const xai_pArray alphaArr,
                                                         const xai_pArray betaArr,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_S16_Dim3(const xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param * params);


/*Wrapper function for xaiApplyInstanceNorm3D_Dim*/

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_Dim1(const xai_pTile3D inTile,
                                                   xai_pArray meanArr,
                                                   xai_pArray recipArr,
                                                   const xai_pArray alphaArr,
                                                   const xai_pArray betaArr,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_Dim2(const xai_pTile3D inTile,
                                                   xai_pArray meanArr,
                                                   xai_pArray recipArr,
                                                   const xai_pArray alphaArr,
                                                   const xai_pArray betaArr,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_Dim3(const xai_pTile3D inTile,
                                                   xai_pArray meanArr,
                                                   xai_pArray recipArr,
                                                   const xai_pArray alphaArr,
                                                   const xai_pArray betaArr,
                                                   xai_pTile3D outTile,
                                                   const xai_cnn_instance_norm_param * params);

/*Channelwise Divide variants*/
_XAI_API_ XAI_ERR_TYPE xaiDivide3D(const xai_pTile3D inTile,
                                   const xai_pArray channelDivisor,
                                   xai_pTile3D outTile,
                                   const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_S8_WHD(const xai_pTile3D inTile,
                                          const xai_pArray channelDivisor,
                                          xai_pTile3D outTile,
                                          const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_U8_WHD(const xai_pTile3D inTile,
                                          const xai_pArray channelDivisor,
                                          xai_pTile3D outTile,
                                          const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_U8S8_WHD(const xai_pTile3D inTile,
                                            const xai_pArray channelDivisor,
                                            xai_pTile3D outTile,
                                            const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_S16_WHD(const xai_pTile3D inTile,
                                           const xai_pArray channelDivisor,
                                           xai_pTile3D outTile,
                                           const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_S16S8_WHD(const xai_pTile3D inTile,
                                             const xai_pArray channelDivisor,
                                             xai_pTile3D outTile,
                                             const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_S8_DWH(const xai_pTile3D inTile,
                                          const xai_pArray channelDivisor,
                                          xai_pTile3D outTile,
                                          const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_U8_DWH(const xai_pTile3D inTile,
                                          const xai_pArray channelDivisor,
                                          xai_pTile3D outTile,
                                          const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_U8S8_DWH(const xai_pTile3D inTile,
                                            const xai_pArray channelDivisor,
                                            xai_pTile3D outTile,
                                            const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_S16_DWH(const xai_pTile3D inTile,
                                           const xai_pArray channelDivisor,
                                           xai_pTile3D outTile,
                                           const xai_cnn_divide3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDivide3D_S16S8_DWH(const xai_pTile3D inTile,
                                             const xai_pArray channelDivisor,
                                             xai_pTile3D outTile,
                                             const xai_cnn_divide3D_params *params);

/* Crop and Resize variants */

_XAI_API_ XAI_ERR_TYPE xaiCropResize3D(const xai_pTile3D inTile,
                                       const xai_pArray ROIinfo,
                                       xai_pTile4D outTile,
                                       const xai_cnn_cropResize3D_params *params);


_XAI_API_ XAI_ERR_TYPE xaiCropResize3D_S8_DWH(const xai_pTile3D inTile,
                                              const xai_pArray ROIinfo,
                                              xai_pTile4D outTile,
                                              const xai_cnn_cropResize3D_params *params);


_XAI_API_ XAI_ERR_TYPE xaiCropResize3D_U8_DWH(const xai_pTile3D inTile,
                                              const xai_pArray ROIinfo,
                                              xai_pTile4D outTile,
                                              const xai_cnn_cropResize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiCropResize3D_S16_DWH(const xai_pTile3D inTile,
                                               const xai_pArray ROIinfo,
                                               xai_pTile4D outTile,
                                               const xai_cnn_cropResize3D_params *params);

/* ReduceSum3D variants */
// -----------------------------------------------------------------------------------
_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D(const xai_pTile3D inTile,
                                      xai_pArray bufferArray,
                                      xai_pTile3D outTile,
                                      const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_S8(const xai_pTile3D inTile,
                                         xai_pArray bufferArray,
                                         xai_pTile3D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_S8U8(const xai_pTile3D inTile,
                                           xai_pArray bufferArray,
                                           xai_pTile3D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_S8S16(const xai_pTile3D inTile,
                                            xai_pArray bufferArray,
                                            xai_pTile3D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_U8(const xai_pTile3D inTile,
                                         xai_pArray bufferArray,
                                         xai_pTile3D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_U8S8(const xai_pTile3D inTile,
                                           xai_pArray bufferArray,
                                           xai_pTile3D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_U8S16(const xai_pTile3D inTile,
                                            xai_pArray bufferArray,
                                            xai_pTile3D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_S16(const xai_pTile3D inTile,
                                          xai_pArray bufferArray,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_U16(const xai_pTile3D inTile,
                                          xai_pArray bufferArray,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_S32(const xai_pTile3D inTile,
                                          xai_pArray bufferArray,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum3D_U32(const xai_pTile3D inTile,
                                          xai_pArray bufferArray,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);
// -----------------------------------------------------------------------------------
/* ReduceSum4D variants */
// -----------------------------------------------------------------------------------
#ifndef GLOW_BUILD
_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D(const xai_pTile4D inTile,
                                      xai_pArray bufferArray,
                                      xai_pTile4D outTile,
                                      const xai_cnn_reduce_params *params);
#endif
_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_S8(const xai_pTile4D inTile,
                                         xai_pArray bufferArray,
                                         xai_pTile4D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_S8U8(const xai_pTile4D inTile,
                                           xai_pArray bufferArray,
                                           xai_pTile4D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_S8S16(const xai_pTile4D inTile,
                                            xai_pArray bufferArray,
                                            xai_pTile4D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_U8(const xai_pTile4D inTile,
                                         xai_pArray bufferArray,
                                         xai_pTile4D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_U8S8(const xai_pTile4D inTile,
                                           xai_pArray bufferArray,
                                           xai_pTile4D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_U8S16(const xai_pTile4D inTile,
                                            xai_pArray bufferArray,
                                            xai_pTile4D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_S16(const xai_pTile4D inTile,
                                          xai_pArray bufferArray,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_U16(const xai_pTile4D inTile,
                                          xai_pArray bufferArray,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_S32(const xai_pTile4D inTile,
                                          xai_pArray bufferArray,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSum4D_U32(const xai_pTile4D inTile,
                                          xai_pArray bufferArray,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);
// -----------------------------------------------------------------------------------
_XAI_API_ XAI_ERR_TYPE xaiReduceMax3D(const xai_pTile3D inTile,
                                      xai_pTile3D outTile,
                                      const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax3D_U8(const xai_pTile3D inTile,
                                         xai_pTile3D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax3D_S8(const xai_pTile3D inTile,
                                         xai_pTile3D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin3D_U8(const xai_pTile3D inTile,
                                         xai_pTile3D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin3D(const xai_pTile3D inTile,
                                      xai_pTile3D outTile,
                                      const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin3D_S8(const xai_pTile3D inTile,
                                         xai_pTile3D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax3D_S16(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin3D_S16(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax3D_U16(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin3D_U16(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax3D_S32(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin3D_S32(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax3D_U32(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin3D_U32(const xai_pTile3D inTile,
                                          xai_pTile3D outTile,
                                          const xai_cnn_reduce_params *params);
// -----------------------------------------------------------------------------------
#ifndef GLOW_BUILD
_XAI_API_ XAI_ERR_TYPE xaiReduceMax4D(const xai_pTile4D inTile,
                                      xai_pTile4D outTile,
                                      const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin4D(const xai_pTile4D inTile,
                                      xai_pTile4D outTile,
                                      const xai_cnn_reduce_params *params);
#endif
_XAI_API_ XAI_ERR_TYPE xaiReduceMax4D_U8(const xai_pTile4D inTile,
                                         xai_pTile4D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax4D_S8(const xai_pTile4D inTile,
                                         xai_pTile4D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin4D_U8(const xai_pTile4D inTile,
                                         xai_pTile4D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin4D_S8(const xai_pTile4D inTile,
                                         xai_pTile4D outTile,
                                         const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax4D_S16(const xai_pTile4D inTile,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin4D_S16(const xai_pTile4D inTile,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax4D_U16(const xai_pTile4D inTile,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin4D_U16(const xai_pTile4D inTile,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax4D_S32(const xai_pTile4D inTile,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin4D_S32(const xai_pTile4D inTile,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMax4D_U32(const xai_pTile4D inTile,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMin4D_U32(const xai_pTile4D inTile,
                                          xai_pTile4D outTile,
                                          const xai_cnn_reduce_params *params);
/* ReduceSAD3D variants */
_XAI_API_ XAI_ERR_TYPE xaiReduceSAD3D(const xai_pTile3D inTile1,
                                      const xai_pTile3D inTile2,
                                      xai_pArray buffArr,
                                      xai_pTile3D outTile,
                                      const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSAD3D_S16UX(const xai_pTile3D inTile1,
                                            const xai_pTile3D inTile2,
                                            xai_pArray buffArr,
                                            xai_pTile3D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSAD3D_S8U16(const xai_pTile3D inTile1,
                                            const xai_pTile3D inTile2,
                                            xai_pArray buffArr,
                                            xai_pTile3D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSAD3D_S8U8(const xai_pTile3D inTile1,
                                           const xai_pTile3D inTile2,
                                           xai_pArray buffArr,
                                           xai_pTile3D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSAD3D_U8U16(const xai_pTile3D inTile1,
                                            const xai_pTile3D inTile2,
                                            xai_pArray buffArr,
                                            xai_pTile3D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSAD3D_U8(const xai_pTile3D inTile1,
                                         const xai_pTile3D inTile2,
                                         xai_pArray buffArr,
                                         xai_pTile3D outTile,
                                         const xai_cnn_reduce_params *params);

//SVDF function
_XAI_API_ XAI_ERR_TYPE svdf_S8I8(const xai_pTile3D inTile, xai_pTile3D stateTile,
                                 const xai_pTile4D betaTile, const xai_pTile4D alphaTile,
                                 xai_pTile3D scratchTile, const xai_pArray biasTile,
                                 xai_pTile3D outTile, xai_cnn_svdf_params *svdfParams,
                                 const xai_pArray outputScaleArray1,
                                 const xai_pArray outputScaleArray2,
                                 const xai_pArray fixUpBiasBuf);

_XAI_API_ XAI_ERR_TYPE svdf_U8I8(const xai_pTile3D inTile, xai_pTile3D stateTile,
                                 const xai_pTile4D betaTile, const xai_pTile4D alphaTile,
                                 xai_pTile3D scratchTile, const xai_pArray biasTile,
                                 xai_pTile3D outTile, xai_cnn_svdf_params *svdfParams,
                                 const xai_pArray outputScaleArray1,
                                 const xai_pArray outputScaleArray2,
                                 const xai_pArray fixUpBiasBuf);

_XAI_API_ XAI_ERR_TYPE xaiSvdf_VQ(const xai_pTile3D inTile, xai_pTile3D stateTile,
                                  const xai_pTile4D betaTile, const xai_pTile4D alphaTile,
                                  xai_pTile3D scratchTile, const xai_pArray biasTile,
                                  xai_pTile3D outTile, xai_cnn_svdf_params *svdfParams,
                                  const xai_pArray outputScaleArray1,
                                  const xai_pArray outputScaleArray2,
                                  const xai_pArray fixUpBiasBuf);

//SVDF function
_XAI_API_ XAI_ERR_TYPE svdfAligned(const xai_pTile3D inTile, xai_pTile3D stateTile,
                                   const xai_pTile4D betaTile, const xai_pTile4D alphaTile,
                                   xai_pTile3D scratchTile, const xai_pArray biasTile,
                                   xai_pTile3D outTile, xai_cnn_svdf_params *svdfParams,
                                   const xai_pArray outputScaleArray1,
                                   const xai_pArray outputScaleArray2,
                                   const xai_pArray fixUpBiasBuf);

//SVDF function
_XAI_API_ XAI_ERR_TYPE xaiSvdf_S8U8_VQ(const xai_pTile3D inTile, xai_pTile3D stateTile,
                                       const xai_pTile4D betaTile, const xai_pTile4D alphaTile,
                                       xai_pTile3D scratchTile, const xai_pArray biasArray,
                                       xai_pTile3D outTile, xai_cnn_svdf_params *svdfParams,
                                       const xai_pArray outputScaleArray1,
                                       const xai_pArray outputScaleArray2,
                                       const xai_pArray fixUpBiasBuf);

_XAI_API_ XAI_ERR_TYPE xaiSvdf_U8U8_VQ(const xai_pTile3D inTile, xai_pTile3D stateTile,
                                       const xai_pTile4D betaTile, const xai_pTile4D alphaTile,
                                       xai_pTile3D scratchTile, const xai_pArray biasArray,
                                       xai_pTile3D outTile, xai_cnn_svdf_params *svdfParams,
                                       const xai_pArray outputScaleArray1,
                                       const xai_pArray outputScaleArray2,
                                       const xai_pArray fixUpBiasBuf);

//SVDF function
_XAI_API_ XAI_ERR_TYPE xaiSvdf_AS8AS8_VQ(const xai_pTile3D inTile, xai_pTile3D stateTile,
                                         const xai_pTile4D betaTile, const xai_pTile4D alphaTile,
                                         xai_pTile3D scratchTile, const xai_pArray biasArray,
                                         xai_pTile3D outTile, xai_cnn_svdf_params *svdfParams,
                                         const xai_pArray outputScaleArray1,
                                         const xai_pArray outputScaleArray2,
                                         const xai_pArray fixUpBiasBuf);

/*************************************************************************************************/
/* Quantize3D/4D (FP32 to fixed point) is declared in Fixed Point routines declaration           */
/* as it can be used for the non AO or non FP32 support Hardwares via REF code inside the opt    */
/*************************************************************************************************/


_XAI_API_ XAI_ERR_TYPE xaiQuantize3D_F32U8(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantize3D_F32S8(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantize3D_F32S16(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantize4D_F32U8(const xai_pTile4D inTile,
                                           xai_pTile4D outTile,
                                           const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantize4D_F32S8(const xai_pTile4D inTile,
                                           xai_pTile4D outTile,
                                           const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantize4D_F32S16(const xai_pTile4D inTile,
                                            xai_pTile4D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

/*************************************************************************************************/
/**************************   END of Fixed Point routines declaration  ***************************/

#if (XCHAL_HAVE_VISION_HP_VFPU == 1)
/***************************************************************************************************/
/******************************  FP16 routines declaration  ****************************************/
/***************************************************************************************************/

_XAI_API_ XAI_ERR_TYPE xaiBroadcastAddA3D_F16(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile,
                                              const xai_cnn_eltwise_params * params);

_XAI_API_ XAI_ERR_TYPE xaiBroadcastSub3D_F16(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile,
                                             const xai_cnn_eltwise_params * params);

_XAI_API_ XAI_ERR_TYPE xaiBroadcastEltwiseEqualA3D_F16(const xai_pTile3D inTile1,
                                                       const xai_pTile3D inTile2,
                                                       xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiBroadcastEltwiseNotEqualA3D_F16(const xai_pTile3D inTile1,
                                                          const xai_pTile3D inTile2,
                                                          xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiBroadcastMulA3D_F16(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile,
                                              const xai_cnn_eltwiseMul_params * params);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedA3D_F16(const xai_pTile3D inTile,
                                                const xai_pTile4D coeffTile,
                                                const xai_pArray biasArray,
                                                xai_pTile3D outTile,
                                                const xai_cnn_conv_params * params);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedA3D2_F16(const xai_pTile3D inTile,
                                                 const xai_pTile4D coeffTile,
                                                 const xai_pArray biasArray,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_conv_params * params);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedA3D2_F16_FOLD8(const xai_pTile3D inTile,
                                                       const xai_pTile4D coeffTile,
                                                       const xai_pArray biasArray,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_conv_params * params);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnectedA3D2_F16_FOLD16(const xai_pTile3D inTile,
                                                        const xai_pTile4D coeffTile,
                                                        const xai_pArray biasArray,
                                                        xai_pTile3D outTile,
                                                        const xai_cnn_conv_params * params);

_XAI_API_ XAI_ERR_TYPE xaiFullyConnected3DWithBatching_S_F16(const xai_pTile4D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             xai_pArray accArray,
                                                             xai_pTile4D outTile,
                                                             const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_MxNj1d1_F16_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            const xai_cnn_conv_params *params);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_1x1j1d1_F16_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            const xai_cnn_conv_params *params);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_2x2j1d1_F16_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j1d1_F16_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_S_3x3j2d1_F16_MOW_WHD(const xai_pTile3D inTile,
                                                            const xai_pTile4D coeffTile,
                                                            const xai_pArray biasArray,
                                                            xai_pTile3D outTile,
                                                            const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_MXN_F16Ca2_MOD_DWH(const xai_pTile3D inTile,
                                                         const xai_pTile4D coeffTile,
                                                         const xai_pArray biasArray,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_conv_params *params);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_1X1_F16Ca2_MOD_DWH(const xai_pTile3D inTile,
                                                         const xai_pTile4D coeffTile,
                                                         const xai_pArray biasArray,
                                                         xai_pTile3D outTile,
                                                         const xai_cnn_conv_params *params);

_XAI_API_ XAI_ERR_TYPE xaiConvolved3D_1X1_F16Ca2_MOD_WHD_DWH(const xai_pTile3D inTile,
                                                             const xai_pTile4D coeffTile,
                                                             const xai_pArray biasArray,
                                                             xai_pTile3D outTile,
                                                             const xai_cnn_conv_params *params);

_XAI_API_ XAI_ERR_TYPE xaiPartialConvolved3D_MXN_F16Ca2_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile4D coeffTile,
                                                                const xai_pArray biasArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *params);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolved2D_S_MxN_F16_MOD_DWH(const xai_pTile3D inTile,
                                                                 const xai_pTile3D coeffTile,
                                                                 const xai_pArray biasArray,
                                                                 xai_pTile3D outTile,
                                                                 const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE  xaiDepthwiseConvolved2D_S_MxNj1d2_F16_MOW_WHD(const xai_pTile3D inTile,
                                                                      const xai_pTile3D coeffTile,
                                                                      const xai_pArray biasArray,
                                                                      xai_pTile3D outTile,
                                                                      const xai_cnn_depthwiseDilatedConv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseLOGA3D_F16(const xai_pTile3D inTile,
                                            xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiExp3D_F16(const xai_pTile3D inTile,
                                    xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiReduceMaxA3D_F16(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_reduce_params *params);
_XAI_API_ XAI_ERR_TYPE xaiReduceMinA3D_F16(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMaxA4D_F16(const xai_pTile4D inTile,
                                           xai_pTile4D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMinA4D_F16(const xai_pTile4D inTile,
                                           xai_pTile4D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSumA3D_F16(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceSumA4D_F16(const xai_pTile4D inTile,
                                           xai_pTile4D outTile,
                                           const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMeanA3D_F16(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceMeanA4D_F16(const xai_pTile4D inTile,
                                            xai_pArray intermediateArray,
                                            xai_pTile4D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceProdA3D_F16(const xai_pTile3D inTile,
                                            xai_pArray intermediateArray,
                                            xai_pTile3D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiReduceProdA4D_F16(const xai_pTile4D inTile,
                                            xai_pArray intermediateArray,
                                            xai_pTile4D outTile,
                                            const xai_cnn_reduce_params *params);

_XAI_API_ XAI_ERR_TYPE xaiQuantizeA3D_F16U8(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantizeA3D_F16S8(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantize3D_F16S16(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantizeA4D_F16U8(const xai_pTile4D inTile,
                                            xai_pTile4D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantizeA4D_F16S8(const xai_pTile4D inTile,
                                            xai_pTile4D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantize4D_F16S16(const xai_pTile4D inTile,
                                            xai_pTile4D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeA3D_U8F16(const xai_pTile3D inTile,
                                              xai_pTile3D outTile,
                                              xai_pArray lut,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeA3D_S8F16(const xai_pTile3D inTile,
                                              xai_pTile3D outTile,
                                              xai_pArray lut,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantize3D_S16F16(const xai_pTile3D inTile,
                                              xai_pTile3D outTile,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeA4D_U8F16(const xai_pTile4D inTile,
                                              xai_pTile4D outTile,
                                              xai_pArray lut,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeA4D_S8F16(const xai_pTile4D inTile,
                                              xai_pTile4D outTile,
                                              xai_pArray lut,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantize4D_S16F16(const xai_pTile4D inTile,
                                              xai_pTile4D outTile,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeAVQ3D_S8F16(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                xai_pArray outScaleArray,
                                                const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeAVQ4D_S8F16(const xai_pTile4D inTile,
                                                xai_pTile4D outTile,
                                                xai_pArray outScaleArray,
                                                const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiSqrtA3D_F16(const xai_pTile3D inTile,
                                      xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiRSqrtA3D_F16(const xai_pTile3D inTile,
                                       xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiEltwisePOWA3D_F16(const xai_pTile3D baseTile,
                                            const xai_pTile3D exponentTile,
                                            xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiBroadcastEltwisePOWA3D_F16(const xai_pTile3D baseTile,
                                                     const xai_pTile3D exponentTile,
                                                     xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseFLOORA3D_F16(const xai_pTile3D inTile,
                                              xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseCEILA3D_F16(const xai_pTile3D inTile,
                                             xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseROUNDA3D_F16(const xai_pTile3D inTile,
                                              xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiDivA3D_F16(const xai_pTile3D numeratorTile,
                                     const xai_pTile3D denominatorTile,
                                     xai_pTile3D outTile,
                                     const xai_cnn_eltwise_params *params);

_XAI_API_ XAI_ERR_TYPE xaiBroadcastDivA3D_F16(const xai_pTile3D numeratorTile,
                                              const xai_pTile3D denominatorTile,
                                              xai_pTile3D outTile,
                                              const xai_cnn_eltwise_params *params);

_XAI_API_ XAI_ERR_TYPE xaiSoftMaxA3D_F16(const xai_pTile3D inTile,
                                         xai_pTile3D outTile,
                                         xai_cnn_softmaxA3D_F16_params * params);

_XAI_API_ XAI_ERR_TYPE xaiSoftMaxA3D_dim1_F16(const xai_pTile3D inTile,
                                              xai_pTile3D outTile,
                                              xai_cnn_softmaxA3D_F16_params * params);

_XAI_API_ XAI_ERR_TYPE xaiSoftMaxA3D_dim2_F16(const xai_pTile3D inTile,
                                              xai_pTile3D outTile,
                                              xai_cnn_softmaxA3D_F16_params * params);

_XAI_API_ XAI_ERR_TYPE xaiSoftMaxA3D_dim3_F16(const xai_pTile3D inTile,
                                              xai_pTile3D outTile,
                                              xai_cnn_softmaxA3D_F16_params * params);

_XAI_API_ XAI_ERR_TYPE xaiLogSoftMaxA3D_F16(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            xai_cnn_softmaxA3D_F16_params * params);

_XAI_API_ XAI_ERR_TYPE xaiSigmoidA3D_F16(const xai_pTile3D inTile,
                                         xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiTanh3D_F16(const xai_pTile3D inTile,
                                     xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_3x3_F16_DWH(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D_3x3_F16_DWH(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param,
                                                 const xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiAvgPoolA3D_MxN_F16_DWH(const xai_pTile3D inTile,
                                                 xai_pTile3D outTile,
                                                 const xai_cnn_pooling_params *param,
                                                 const xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiAvgPool3D_MxN_F16_DWH(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPool3D_MxN_F16_DWH(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                const xai_cnn_pooling_params * param);

_XAI_API_ XAI_ERR_TYPE xaiMaxPoolWithIdx3D_MxN_F16_DWH(const xai_pTile3D inTile,
                                                       xai_pTile3D outTile,
                                                       xai_pTile3D idxTile,
                                                       const xai_cnn_pooling_params *param);

_XAI_API_ XAI_ERR_TYPE xaiExtendEdges3D_F16(xai_pTile3D dstTile,
                                            const xai_pArray pArray,
                                            xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiExtendEdgesConst3D_F16(xai_pTile3D dstTile,
                                                 const xb_f16 value,
                                                 xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiFillTile3D_F16(xai_pTile3D dstTile,
                                         const xb_f16 value,
                                         xai_bool fill_edge_extension);

/*_XAI_API_ XAI_ERR_TYPE xaiBiasExtend_F16_MOD(const xai_pArray inBiasArray,
                                             xai_pArray outBiasArray);*/

/*_XAI_API_ XAI_ERR_TYPE xaiOutScaleExtend_F16_MOD(const xai_pArray outScaleArray,
                                                 xai_pArray extendedOutScaleArray);*/

/*_XAI_API_ XAI_ERR_TYPE xaiDeConvReOrder4D_F16_NDWH(const xai_pTile4D inTile,
                                                   xai_pTile4D subCoeffs[],
                                                   xai_pTile4D superCoeffs[],
                                                   const xai_cnn_conv_params *param,
                                                   const uint8_t transposeCoeffsFlag);*/

_XAI_API_ XAI_ERR_TYPE xaiResize3D_SetTileParams(const xai_size3D *inFrame3DSize,
                                                 const xai_size3D *outFrame3DSize,
                                                 const xai_cnn_data_order dataOrder,
                                                 int32_t half_pixel_flag,
                                                 xai_cnn_resizeA3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_F16_SetTileParams(const xai_size3D *inFrame3DSize,
                                                     const xai_size3D *outFrame3DSize,
                                                     const xai_cnn_data_order dataOrder,
                                                     int32_t half_pixel_flag,
                                                     xai_cnn_interp3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiInterp3D_F16_DWH(const xai_pTile3D inTile,
                                           xai_pTile3D outTile,
                                           const xai_cnn_interp3D_params * pparams);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_F16_DWH(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_resize_nearest3D_params * pparams);

_XAI_API_ XAI_ERR_TYPE xaiResizeNearest3D_F16_WHD(const xai_pTile3D inTile,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_resize_nearest3D_params *params);

/*hardSwish FP16*/
_XAI_API_ XAI_ERR_TYPE xaiHardSwish_F16(const xai_pTile3D inTile,
                                        xai_pTile3D outTile);
/*ArgMin ArgMax*/
_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_F16_dim1(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numLargestVal);

_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_F16_dim2(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numLargestVal);

_XAI_API_ XAI_ERR_TYPE xaiArgmax3D_F16_dim3(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numLargestVal);

_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_F16_dim1(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numSmallestVal);

_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_F16_dim2(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numSmallestVal);

_XAI_API_ XAI_ERR_TYPE xaiArgmin3D_F16_dim3(const xai_pTile3D inTile,
                                            xai_pArray bufArray,
                                            xai_pTile3D outTileIdx,
                                            xai_pTile3D outTileVal,
                                            const uint16_t numSmallestVal);

_XAI_API_ XAI_ERR_TYPE xaiMergeTopKArgmax3D_F16_dim1(const xai_pTile3D inTileIdx,
                                                     const xai_pTile3D inTileVal,
                                                     const xai_pArray inPtrOffsetArr,
                                                     xai_pArray bufArray,
                                                     xai_pTile3D outTileIdx,
                                                     xai_pTile3D outTileVal,
                                                     const uint16_t numLargestVal);

_XAI_API_ XAI_ERR_TYPE xaiMergeTopKArgmin3D_F16_dim1(const xai_pTile3D inTileIdx,
                                                     const xai_pTile3D inTileVal,
                                                     const xai_pArray inPtrOffsetArr,
                                                     xai_pArray bufArray,
                                                     xai_pTile3D outTileIdx,
                                                     xai_pTile3D outTileVal,
                                                     const uint16_t numSmallestVal);

/*prelu FP16*/
_XAI_API_ XAI_ERR_TYPE xaiPRELU3D_F16(const xai_pTile3D inTile,
                                      const xai_pTile3D slopeArray,
                                      xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiPRELU3D_F16_DWH(const xai_pTile3D inTile,
                                          const xai_pTile3D slopeArray,
                                          xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiPRELU3D_F16_WHD(const xai_pTile3D inTile,
                                          const xai_pTile3D slopeArray,
                                          xai_pTile3D outTile);

/*Leaky relu F16*/
_XAI_API_ XAI_ERR_TYPE xaiLeakyRELU_F16(const xai_pTile3D inTile,
                                        xai_pTile3D outTile,
                                        const xb_f16 slope);

/* LUT APIs */

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Oddsym_F16(const xai_pTile3D inTile,
                                           const xai_pArray lutArray,
                                           xai_pTile3D outTile,
                                           const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Evensym_F16(const xai_pTile3D inTile,
                                            const xai_pArray lutArray,
                                            xai_pTile3D outTile,
                                            const xai_cnn_lut_params *params);

_XAI_API_ XAI_ERR_TYPE xaiLUT3D_Normal_F16(const xai_pTile3D inTile,
                                           const xai_pArray lutArray,
                                           xai_pTile3D outTile,
                                           const xai_cnn_lut_params *params);

/*Depthwise Conv F16*/
_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxN_F16_MOD_DWH(const xai_pTile3D inTile,
                                                                const xai_pTile3D coeffTile,
                                                                const xai_pArray biasArray,
                                                                xai_pTile3D outTile,
                                                                const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj1_F16_MOW_WHD(const xai_pTile3D inTile,
                                                                  const xai_pTile3D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_MxNj2_F16_MOW_WHD(const xai_pTile3D inTile,
                                                                  const xai_pTile3D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3j1_F16_MOW_WHD(const xai_pTile3D inTile,
                                                                  const xai_pTile3D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3j2_F16_MOW_WHD(const xai_pTile3D inTile,
                                                                  const xai_pTile3D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5j1_F16_MOW_WHD(const xai_pTile3D inTile,
                                                                  const xai_pTile3D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5j2_F16_MOW_WHD(const xai_pTile3D inTile,
                                                                  const xai_pTile3D coeffTile,
                                                                  const xai_pArray biasArray,
                                                                  xai_pTile3D outTile,
                                                                  const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_3x3_F16Ca2_MOD_DWH(const xai_pTile3D inTile,
                                                                   const xai_pTile3D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pTile3D outTile,
                                                                   const xai_cnn_conv_params *param);

_XAI_API_ XAI_ERR_TYPE xaiDepthwiseConvolve2D_S_5x5_F16Ca2_MOD_DWH(const xai_pTile3D inTile,
                                                                   const xai_pTile3D coeffTile,
                                                                   const xai_pArray biasArray,
                                                                   xai_pTile3D outTile,
                                                                   const xai_cnn_conv_params *param);

/*Batchnorm f16*/
_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_F16_DWH(const xai_pTile3D inTile,
                                              const xai_pArray Alpha,
                                              const xai_pArray Beta,
                                              xai_pTile3D outTile,
                                              const xai_cnn_batchnorm_params *params);
_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_F16_WHD(const xai_pTile3D inTile,
                                              const xai_pArray Alpha,
                                              const xai_pArray Beta,
                                              xai_pTile3D outTile,
                                              const xai_cnn_batchnorm_params *params);
_XAI_API_ XAI_ERR_TYPE xaiBatchnorm3D_F16(const xai_pTile3D inTile,
                                          const xai_pArray Alpha,
                                          const xai_pArray Beta,
                                          xai_pTile3D outTile,
                                          const xai_cnn_batchnorm_params *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_f16_WHD(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_f16_DWH(const xai_pTile3D inTile,
                                                           xai_pArray meanArr,
                                                           xai_pArray recipArr,
                                                           xai_pArray buffArr,
                                                           xai_pArray buffArrSoS,
                                                           const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_F16_Dim1(const xai_pTile3D inTile,
                                                            xai_pArray meanArr,
                                                            xai_pArray recipArr,
                                                            xai_pArray buffArr,
                                                            xai_pArray buffArrSoS,
                                                            const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_F16_Dim2(const xai_pTile3D inTile,
                                                            xai_pArray meanArr,
                                                            xai_pArray recipArr,
                                                            xai_pArray buffArr,
                                                            xai_pArray buffArrSoS,
                                                            const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_F16_Dim3(const xai_pTile3D inTile,
                                                            xai_pArray meanArr,
                                                            xai_pArray recipArr,
                                                            xai_pArray buffArr,
                                                            xai_pArray buffArrSoS,
                                                            const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiCalcInstanceNormFactor3D_F16(const xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       xai_pArray buffArr,
                                                       xai_pArray buffArrSoS,
                                                       const xai_cnn_instance_norm_param * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_F16_Dim1(xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_F16_Dim2(xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_F16_Dim3(xai_pTile3D inTile,
                                                       xai_pArray meanArr,
                                                       xai_pArray recipArr,
                                                       const xai_pArray alphaArr,
                                                       const xai_pArray betaArr,
                                                       xai_pTile3D outTile,
                                                       const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyInstanceNorm3D_F16(xai_pTile3D inTile,
                                                  xai_pArray meanArr,
                                                  xai_pArray recipArr,
                                                  const xai_pArray alphaArr,
                                                  const xai_pArray betaArr,
                                                  xai_pTile3D outTile,
                                                  const xai_cnn_instance_norm_param *params);

_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_F16_WHD(const xai_pTile3D inTile,
                                                        xai_pArray buffArrSoS,
                                                        xai_pArray pNormScaleArr,
                                                        const xai_cnn_normalize3D_params * params);

_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_F16_DWH(const xai_pTile3D inTile,
                                                        xai_pArray buffArrSoS,
                                                        xai_pArray pNormScaleArr,
                                                        const xai_cnn_normalize3D_params * params);

_XAI_API_ XAI_ERR_TYPE xaiCalcNormalizeFactor3D_F16(const xai_pTile3D inTile,
                                                    xai_pArray buffArrSoS,
                                                    xai_pArray pNormScaleArr,
                                                    const xai_cnn_normalize3D_params * params);

_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_F16_WHD(xai_pTile3D inTile,
                                               const xai_pArray pNormScaleArr,
                                               xai_pTile3D pOutTile,
                                               const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_F16_DWH(xai_pTile3D inTile,
                                               const xai_pArray pNormScaleArr,
                                               xai_pTile3D pOutTile,
                                               const xai_cnn_normalize3D_params *params);

_XAI_API_ XAI_ERR_TYPE xaiApplyScale3D_F16(xai_pTile3D inTile,
                                           const xai_pArray pNormScaleArr,
                                           xai_pTile3D pOutTile,
                                           const xai_cnn_normalize3D_params *params);

/****************************  END of FP16 routines declaration  ************************************/
#endif // end of #if (XCHAL_HAVE_VISION_HP_VFPU == 1)


#if (XCHAL_HAVE_VISION_SP_VFPU == 1) && (XCHAL_HAVE_VISION_HP_VFPU == 1)
/***************************************************************************************************/
/******************************  Mixed FP16/FP32 routines declaration  *****************************/
/***************************************************************************************************/

_XAI_API_ XAI_ERR_TYPE xaiQuantize3D_F32F16(const xai_pTile3D inTile,
                                            xai_pTile3D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiQuantize4D_F32F16(const xai_pTile4D inTile,
                                            xai_pTile4D outTile,
                                            const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantize3D_F16F32(const xai_pTile3D inTile,
                                              xai_pTile3D outTile,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantize4D_F16F32(const xai_pTile4D inTile,
                                              xai_pTile4D outTile,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeVQ3D_F16F32(const xai_pTile3D inTile,
                                                xai_pTile3D outTile,
                                                xai_pArray outScaleArray,
                                                const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeVQ4D_F16F32(const xai_pTile4D inTile,
                                                xai_pTile4D outTile,
                                                xai_pArray outScaleArray,
                                                const xai_cnn_quantDequantA_params *pparams);

/****************************  END of Mixed FP16/FP32 routines declaration  *************************/
#endif //end of #if (XCHAL_HAVE_VISION_SP_VFPU == 1) && (XCHAL_HAVE_VISION_HP_VFPU == 1)


#if (XCHAL_HAVE_VISION_SP_VFPU == 1)
/***************************************************************************************************/
/******************************  FP32 routines declaration  ****************************************/
/***************************************************************************************************/
_XAI_API_ XAI_ERR_TYPE xaiDeQuantize3D_U8F32(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantize3D_S8F32(const xai_pTile3D inTile,
                                             xai_pTile3D outTile,
                                             const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantize3D_S16F32(const xai_pTile3D inTile,
                                              xai_pTile3D outTile,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantize4D_U8F32(const xai_pTile4D inTile,
                                             xai_pTile4D outTile,
                                             const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantize4D_S8F32(const xai_pTile4D inTile,
                                             xai_pTile4D outTile,
                                             const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantize4D_S16F32(const xai_pTile4D inTile,
                                              xai_pTile4D outTile,
                                              const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeVQ3D_S8F32(const xai_pTile3D inTile,
                                               xai_pTile3D outTile,
                                               xai_pArray outScaleArray,
                                               const xai_cnn_quantDequantA_params *pparams);

_XAI_API_ XAI_ERR_TYPE xaiDeQuantizeVQ4D_S8F32(const xai_pTile4D inTile,
                                               xai_pTile4D outTile,
                                               xai_pArray outScaleArray,
                                               const xai_cnn_quantDequantA_params *pparams);



_XAI_API_ XAI_ERR_TYPE xaiExtendEdges3D_F32(xai_pTile3D dstTile,
                                            const xai_pArray pArray,
                                            xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiExtendEdgesConst3D_F32(xai_pTile3D dstTile,
                                                 const float value,
                                                 xai_size3D frame3DSize);

_XAI_API_ XAI_ERR_TYPE xaiFillTile3D_F32(xai_pTile3D dstTile,
                                         const float value,
                                         xai_bool fill_edge_extension);
/****************************  END of FP32 routines declaration  ************************************/
#endif //end of #if (XCHAL_HAVE_VISION_SP_VFPU == 1)
#endif //if ((XCHAL_VISION_TYPE >= 6))
#endif // #ifndef __XAI_CNN_API_H__
