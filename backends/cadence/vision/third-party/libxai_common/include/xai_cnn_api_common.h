/*
 * Copyright (c) 2021 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
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

#ifndef __XAI_CNN_API_COMMON_H__
#define __XAI_CNN_API_COMMON_H__

#include "xai_cnn_api_params.h"
#include "xai_config_api.h"
#include "xai_core_api.h"
#include "xai_tile_manager.h"
#include <math.h>
#include <stdbool.h>


// ElementWise APIs
_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_AV(const xai_pTile3D inTile1,
                                          const xai_pTile3D inTile2,
                                          xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_S8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_U8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_S16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_U16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_S32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_U32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_AV(const xai_pTile3D inTile1,
                                          const xai_pTile3D inTile2,
                                          xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_S8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_U8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_S16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_U16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_S32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_U32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMul3D_S32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);

_XAI_API_ XAI_ERR_TYPE xaiEltwiseMax3D_AV(const xai_pTile3D inTile1,
                                          const xai_pTile3D inTile2,
                                          xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMax3D_S8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMax3D_U8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMax3D_S16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMax3D_U16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMax3D_S32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMax3D_U32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMin3D_AV(const xai_pTile3D inTile1,
                                          const xai_pTile3D inTile2,
                                          xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMin3D_S8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMin3D_U8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMin3D_S16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMin3D_U16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMin3D_S32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMin3D_U32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseOr3D_AV(const xai_pTile3D inTile1,
                                         const xai_pTile3D inTile2,
                                         xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseOr3D_S8_AV(const xai_pTile3D inTile1,
                                            const xai_pTile3D inTile2,
                                            xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseOr3D_U8_AV(const xai_pTile3D inTile1,
                                            const xai_pTile3D inTile2,
                                            xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseOr3D_S16_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseOr3D_U16_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseOr3D_S32_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseOr3D_U32_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);



_XAI_API_ XAI_ERR_TYPE xaiEltwiseAnd3D_AV(const xai_pTile3D inTile1,
                                          const xai_pTile3D inTile2,
                                          xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAnd3D_S8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAnd3D_U8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAnd3D_S16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAnd3D_U16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAnd3D_S32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseAnd3D_U32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);



_XAI_API_ XAI_ERR_TYPE xaiEltwiseXor3D_AV(const xai_pTile3D inTile1,
                                          const xai_pTile3D inTile2,
                                          xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseXor3D_S8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseXor3D_U8_AV(const xai_pTile3D inTile1,
                                             const xai_pTile3D inTile2,
                                             xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseXor3D_S16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseXor3D_U16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseXor3D_S32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseXor3D_U32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseEqual3D_AV(const xai_pTile3D inTile1,
                                            const xai_pTile3D inTile2,
                                            xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseEqual3D_S8_AV(const xai_pTile3D inTile1,
                                               const xai_pTile3D inTile2,
                                               xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseEqual3D_U8_AV(const xai_pTile3D inTile1,
                                               const xai_pTile3D inTile2,
                                               xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseEqual3D_S16_AV(const xai_pTile3D inTile1,
                                                const xai_pTile3D inTile2,
                                                xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseEqual3D_U16_AV(const xai_pTile3D inTile1,
                                                const xai_pTile3D inTile2,
                                                xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseEqual3D_S32_AV(const xai_pTile3D inTile1,
                                                const xai_pTile3D inTile2,
                                                xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseEqual3D_U32_AV(const xai_pTile3D inTile1,
                                                const xai_pTile3D inTile2,
                                                xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseGreaterThan3D_AV(const xai_pTile3D inTile1,
                                                  const xai_pTile3D inTile2,
                                                  xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseGreaterThan3D_S8_AV(const xai_pTile3D inTile1,
                                                     const xai_pTile3D inTile2,
                                                     xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseGreaterThan3D_U8_AV(const xai_pTile3D inTile1,
                                                     const xai_pTile3D inTile2,
                                                     xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseGreaterThan3D_S16_AV(const xai_pTile3D inTile1,
                                                      const xai_pTile3D inTile2,
                                                      xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseGreaterThan3D_U16_AV(const xai_pTile3D inTile1,
                                                      const xai_pTile3D inTile2,
                                                      xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseGreaterThan3D_S32_AV(const xai_pTile3D inTile1,
                                                      const xai_pTile3D inTile2,
                                                      xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseGreaterThan3D_U32_AV(const xai_pTile3D inTile1,
                                                      const xai_pTile3D inTile2,
                                                      xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseLessThan3D_AV(const xai_pTile3D inTile1,
                                               const xai_pTile3D inTile2,
                                               xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseLessThan3D_S8_AV(const xai_pTile3D inTile1,
                                                  const xai_pTile3D inTile2,
                                                  xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseLessThan3D_U8_AV(const xai_pTile3D inTile1,
                                                  const xai_pTile3D inTile2,
                                                  xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseLessThan3D_S16_AV(const xai_pTile3D inTile1,
                                                   const xai_pTile3D inTile2,
                                                   xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseLessThan3D_U16_AV(const xai_pTile3D inTile1,
                                                   const xai_pTile3D inTile2,
                                                   xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseLessThan3D_S32_AV(const xai_pTile3D inTile1,
                                                   const xai_pTile3D inTile2,
                                                   xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseLessThan3D_U32_AV(const xai_pTile3D inTile1,
                                                   const xai_pTile3D inTile2,
                                                   xai_pTile3D outTile);


#if XCHAL_HAVE_VISION_HP_VFPU == 1
_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_F16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_F16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMax3D_F16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMin3D_F16_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseEqual3D_F16_AV(const xai_pTile3D inTile1,
                                                const xai_pTile3D inTile2,
                                                xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseGreaterThan3D_F16_AV(const xai_pTile3D inTile1,
                                                      const xai_pTile3D inTile2,
                                                      xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseLessThan3D_F16_AV(const xai_pTile3D inTile1,
                                                   const xai_pTile3D inTile2,
                                                   xai_pTile3D outTile);
#endif //#if XCHAL_HAVE_VISION_HP_VFPU == 1


#if XCHAL_HAVE_VISION_SP_VFPU == 1

_XAI_API_ XAI_ERR_TYPE xaiEltwiseAdd3D_F32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseSub3D_F32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMax3D_F32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseMin3D_F32_AV(const xai_pTile3D inTile1,
                                              const xai_pTile3D inTile2,
                                              xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseEqual3D_F32_AV(const xai_pTile3D inTile1,
                                                const xai_pTile3D inTile2,
                                                xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseGreaterThan3D_F32_AV(const xai_pTile3D inTile1,
                                                      const xai_pTile3D inTile2,
                                                      xai_pTile3D outTile);


_XAI_API_ XAI_ERR_TYPE xaiEltwiseLessThan3D_F32_AV(const xai_pTile3D inTile1,
                                                   const xai_pTile3D inTile2,
                                                   xai_pTile3D outTile);
#endif //#if XCHAL_HAVE_VISION_SP_VFPU == 1

_XAI_API_ XAI_ERR_TYPE xaiCast3D(const xai_pTile3D inTile,
                                 xai_pTile3D outTile);
#endif //#ifndef __XAI_CNN_API_COMMON_H__
