/*
 * Copyright (c) 2013-2018 Tensilica Inc. ALL RIGHTS RESERVED.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __XAI_TILE_MANAGER_H__
#define __XAI_TILE_MANAGER_H__

#include <stdint.h>
#include <stddef.h>
#include "xai_config_api.h"

typedef struct xaiFrameStruct
{
  void     *pFrameBuff;
  uint32_t frameBuffSize;
  void     *pFrameData;
  int32_t  frameWidth;
  int32_t  frameHeight;
  int32_t  framePitch;
  uint8_t  pixelRes;
  uint8_t  pixelPackFormat;
} xai_frame, *xai_pFrame;

#define XAI_ARRAY_FIELDS \
  void *pBuffer;         \
  void *pData;           \
  uint32_t bufferSize;   \
  int32_t width;         \
  int32_t pitch;         \
  uint32_t status;       \
  uint16_t type;         \
  int32_t height;

typedef struct xaiArrayStruct
{
  XAI_ARRAY_FIELDS
} xai_array, *xai_pArray;

#define XAI_ARRAY_FIELDS_COEFF_32 \
  uintptr_t pBuffer;              \
  uintptr_t pData;                \
  uint64_t bufferSize;            \
  uint64_t width;                 \
  int64_t pitch;                  \
  uint32_t status;                \
  uint16_t type;                  \
  int32_t height;

typedef struct xaiArrayStruct_coeff_32
{
  XAI_ARRAY_FIELDS_COEFF_32
} xai_array_coeff_32, *xai_pArray_coeff_32;

#define XAI_ARRAY_FIELDS_COEFF_64 \
  uint64_t pBuffer;               \
  uint64_t pData;                 \
  uint64_t bufferSize;            \
  uint64_t width;                 \
  int64_t pitch;                  \
  uint32_t status;                \
  uint16_t type;                  \
  int32_t height;

typedef struct xaiArrayStruct_coeff_64
{
  XAI_ARRAY_FIELDS_COEFF_64
} xai_array_coeff_64, *xai_pArray_coeff_64;

typedef struct xaiTile2DStruct
{
  XAI_ARRAY_FIELDS
  xai_frame *pFrame;
  int32_t   x;
  int32_t   y;
  uint16_t  edgeWidth;
  uint16_t  edgeHeight;
} xai_tile2D, *xai_pTile2D;

/*****************************************
*   Data type definitions
*****************************************/

//** 16 bit data type, bit 0 - 3 for data encoded depth(bits/bytes), 5 - 7 free bits (reserved for future use),  bit 8 - 10 for encoded special type float
//** 11 bit for float (denotes whether float or not), 12 - 14 bit for encoded tile type, and 15 bit for data sign

#define XAI_TYPE_SIGNED_BIT          (1 << 15)

#define XAI_TYPE_ARRAY_BITS          (1 << 12)
#define XAI_TYPE_TILE2D_BITS         (2 << 12)
#define XAI_TYPE_TILE3D_BITS         (3 << 12)
#define XAI_TYPE_TILE4D_BITS         (4 << 12)
#define XAI_TYPE_TILE5D_BITS         (5 << 12)
#define XAI_TYPE_TILE6D_BITS         (6 << 12)
#define XAI_TYPE_TILE_BITS           3
#define XAI_TYPE_TILE_MASK           (((1 << XAI_TYPE_TILE_BITS) - 1) << 12)

#define XAI_TYPE_FLOAT_BIT           (1 << 11)
#define XAI_TYPE_SPECIAL_FLOAT_BITS  3
#define XAI_TYPE_SPECIAL_FLOAT_MASK  (((1 << XAI_TYPE_SPECIAL_FLOAT_BITS) - 1) << 8)
#define XAI_TYPE_BFLOAT_BIT          (XAI_TYPE_FLOAT_BIT | (1 << 8))

#define XAI_TYPE_ELEMENT_SIZE_BITS   4
#define XAI_TYPE_ELEMENT_SIZE_MASK   ((1 << XAI_TYPE_ELEMENT_SIZE_BITS) - 1)

#define XAI_MAKETYPE(flags, depth)            ((flags) | (depth))
#define XAI_CUSTOMTYPE(type)                  XAI_MAKETYPE(0, (sizeof(type) + 2))   //convert byte to representation sequence

#define XAI_TYPE_ELEMENT_SIZE_IN_BYTES(type)  (1 << (((type) & (XAI_TYPE_ELEMENT_SIZE_MASK)) - 3))
#define XAI_TYPE_ELEMENT_SIZE_IN_BITS(type)   (XAI_TYPE_ELEMENT_SIZE_IN_BYTES(type) << 3)
#define XAI_TYPE_ELEMENT_SIZE(type)           XAI_TYPE_ELEMENT_SIZE_IN_BYTES(type)
#define XAI_TYPE_ELEMENT_TYPE(type)           ((type) & (XAI_TYPE_SIGNED_BIT | XAI_TYPE_ELEMENT_SIZE_MASK | XAI_TYPE_FLOAT_BIT | XAI_TYPE_SPECIAL_FLOAT_MASK))
#define XAI_TYPE_IS_ARRAY(type)               (!(((type) & (XAI_TYPE_TILE_MASK)) ^ XAI_TYPE_ARRAY_BITS))
#define XAI_TYPE_IS_TILE2D(type)              (!(((type) & (XAI_TYPE_TILE_MASK)) ^ XAI_TYPE_TILE2D_BITS))
#define XAI_TYPE_IS_SIGNED(type)              ((type) & (XAI_TYPE_SIGNED_BIT))

// XAI_MAKETYPE accepts 2 parameters
// 1: Denotes whether the entity is a tile(XAI_TYPE_TILE2D_BITS, XAI_TYPE_TILE3D_BITS etc. flag set) or an array(XAI_TYPE_ARRAY_BITS flag set) ,
//    ,if the data is a signed or unsigned(XAI_TYPE_SIGNED_BIT) and also if data is float(XAI_TYPE_FLOAT_BIT) and float type(XAI_TYPE_BFLOAT_BIT etc.)
// 2: Denotes encoded number of bits/bytes
//    0 implies the data is bool, 1 implies the data is 2 bit, 2 implies the data is 4bit, 3  implies the data is 8bit, 4 implies the data is 16bit.
//    5 implies the data is 32bit, 6 implies the data is 64bit and 7 implies the data is 128bit

#define XAI_BOOL         XAI_MAKETYPE(0, 0)
#define XAI_U2           XAI_MAKETYPE(0, 1)
#define XAI_U4           XAI_MAKETYPE(0, 2)
#define XAI_U8           XAI_MAKETYPE(0, 3)
#define XAI_U16          XAI_MAKETYPE(0, 4)
#define XAI_U32          XAI_MAKETYPE(0, 5)
#define XAI_U64          XAI_MAKETYPE(0, 6)
#define XAI_U128         XAI_MAKETYPE(0, 7)

#define XAI_S2           XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT, 1)
#define XAI_S4           XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT, 2)
#define XAI_S8           XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT, 3)
#define XAI_S16          XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT, 4)
#define XAI_S32          XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT, 5)
#define XAI_S64          XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT, 6)
#define XAI_S128         XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT, 7)

#define XAI_F8           (XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT | XAI_TYPE_FLOAT_BIT, 3))
#define XAI_F16          (XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT | XAI_TYPE_FLOAT_BIT, 4))
#define XAI_F32          (XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT | XAI_TYPE_FLOAT_BIT, 5))
#define XAI_F64          (XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT | XAI_TYPE_FLOAT_BIT, 6))
#define XAI_F128         (XAI_MAKETYPE(XAI_TYPE_SIGNED_BIT | XAI_TYPE_FLOAT_BIT, 7))

#define XAI_ARRAY_BOOL   (XAI_BOOL | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_U4     (XAI_U4 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_U8     (XAI_U8 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_U16    (XAI_U16 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_U32    (XAI_U32 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_U64    (XAI_U64 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_U128   (XAI_U128 | XAI_TYPE_ARRAY_BITS)

#define XAI_ARRAY_S4     (XAI_S4 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_S8     (XAI_S8 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_S16    (XAI_S16 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_S32    (XAI_S32 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_S64    (XAI_S64 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_S128   (XAI_S128 | XAI_TYPE_ARRAY_BITS)

#define XAI_ARRAY_F8     (XAI_F8 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_F16    (XAI_F16 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_F32    (XAI_F32 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_F64    (XAI_F64 | XAI_TYPE_ARRAY_BITS)
#define XAI_ARRAY_F128   (XAI_F128 | XAI_TYPE_ARRAY_BITS)

#define XAI_TILE2D_BOOL  (XAI_BOOL | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_U4    (XAI_U4 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_U8    (XAI_U8 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_U16   (XAI_U16 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_U32   (XAI_U32 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_U64   (XAI_U64 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_U128  (XAI_U128 | XAI_TYPE_TILE2D_BITS)

#define XAI_TILE2D_S4    (XAI_S4 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_S8    (XAI_S8 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_S16   (XAI_S16 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_S32   (XAI_S32 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_S64   (XAI_S64 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_S128  (XAI_S128 | XAI_TYPE_TILE2D_BITS)

#define XAI_TILE2D_F8    (XAI_F8 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_F16   (XAI_F16 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_F32   (XAI_F32 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_F64   (XAI_F64 | XAI_TYPE_TILE2D_BITS)
#define XAI_TILE2D_F128  (XAI_F128 | XAI_TYPE_TILE2D_BITS)

/*****************************************
*                   Frame Access Macros
*****************************************/
#define XAI_FRAME_GET_BUFF_PTR(pFrame)                   ((pFrame)->pFrameBuff)
#define XAI_FRAME_SET_BUFF_PTR(pFrame, pBuff)            (pFrame)->pFrameBuff = ((void *) (pBuff))

#define XAI_FRAME_GET_BUFF_SIZE(pFrame)                  ((pFrame)->frameBuffSize)
#define XAI_FRAME_SET_BUFF_SIZE(pFrame, buffSize)        (pFrame)->frameBuffSize = ((uint32_t) (buffSize))

#define XAI_FRAME_GET_DATA_PTR(pFrame)                   ((pFrame)->pFrameData)
#define XAI_FRAME_SET_DATA_PTR(pFrame, pData)            (pFrame)->pFrameData = ((void *) (pData))

#define XAI_FRAME_GET_WIDTH(pFrame)                      ((pFrame)->frameWidth)
#define XAI_FRAME_SET_WIDTH(pFrame, width)               (pFrame)->frameWidth = ((int32_t) (width))

#define XAI_FRAME_GET_HEIGHT(pFrame)                     ((pFrame)->frameHeight)
#define XAI_FRAME_SET_HEIGHT(pFrame, height)             (pFrame)->frameHeight = ((int32_t) (height))

#define XAI_FRAME_GET_PITCH(pFrame)                      ((pFrame)->framePitch)
#define XAI_FRAME_SET_PITCH(pFrame, pitch)               (pFrame)->framePitch = ((int32_t) (pitch))

#define XAI_FRAME_GET_PIXEL_RES(pFrame)                  ((pFrame)->pixelRes)
#define XAI_FRAME_SET_PIXEL_RES(pFrame, pixRes)          (pFrame)->pixelRes = ((uint8_t) (pixRes))

#define XAI_FRAME_GET_PIXEL_FORMAT(pFrame)               ((pFrame)->pixelPackFormat)
#define XAI_FRAME_SET_PIXEL_FORMAT(pFrame, pixelFormat)  (pFrame)->pixelPackFormat = ((uint8_t) (pixelFormat))

/*****************************************
*                   Array Access Macros
*****************************************/
#define XAI_ARRAY_GET_BUFF_PTR(pArray)                    ((pArray)->pBuffer)
#define XAI_ARRAY_SET_BUFF_PTR(pArray, pBuff)             (pArray)->pBuffer = ((void *) (pBuff))

#define XAI_ARRAY_GET_BUFF_SIZE(pArray)                   ((pArray)->bufferSize)
#define XAI_ARRAY_SET_BUFF_SIZE(pArray, buffSize)         (pArray)->bufferSize = (buffSize)

#define XAI_ARRAY_GET_DATA_PTR(pArray)                    ((pArray)->pData)
#define XAI_ARRAY_SET_DATA_PTR(pArray, pArrayData)        (pArray)->pData = ((void *) (pArrayData))

#define XAI_ARRAY_SET_BUFF_PTR_COEFF(pArray, pBuff)       (pArray)->pBuffer = ((uint64_t) (pBuff))
#define XAI_ARRAY_SET_DATA_PTR_COEFF(pArray, pArrayData)  (pArray)->pData   = ((uint64_t) (pArrayData))

#define XAI_ARRAY_GET_WIDTH(pArray)                       ((pArray)->width)
#define XAI_ARRAY_SET_WIDTH(pArray, value)                (pArray)->width = ((int32_t) (value))
#define XAI_ARRAY_SET_WIDTH_COEFF(pArray, value)          (pArray)->width = ((uint64_t) (value))

#define XAI_ARRAY_GET_PITCH(pArray)                       ((pArray)->pitch)
#define XAI_ARRAY_SET_PITCH(pArray, value)                (pArray)->pitch = ((int32_t) (value))

#define XAI_ARRAY_GET_HEIGHT(pArray)                      ((pArray)->height)
#define XAI_ARRAY_SET_HEIGHT(pArray, value)               (pArray)->height = ((uint16_t) (value))

#define XAI_ARRAY_GET_STATUS_FLAGS(pArray)                ((pArray)->status)
#define XAI_ARRAY_SET_STATUS_FLAGS(pArray, value)         (pArray)->status = ((uint8_t) (value))

#define XAI_ARRAY_GET_TYPE(pArray)                        ((pArray)->type)
#define XAI_ARRAY_SET_TYPE(pArray, value)                 (pArray)->type = ((uint16_t) (value))

#define XAI_ARRAY_GET_CAPACITY(pArray)                    ((pArray)->pitch)
#define XAI_ARRAY_SET_CAPACITY(pArray, value)             (pArray)->pitch = ((int32_t) (value))
#define XAI_ARRAY_SET_CAPACITY_COEFF(pArray, value)       (pArray)->pitch = ((int64_t) (value))

#define XAI_ARRAY_GET_ELEMENT_TYPE(pArray)                (XAI_TYPE_ELEMENT_TYPE(XAI_ARRAY_GET_TYPE(pArray)))
#define XAI_ARRAY_GET_ELEMENT_SIZE(pArray)                (XAI_TYPE_ELEMENT_SIZE(XAI_ARRAY_GET_TYPE(pArray)))
#define XAI_ARRAY_IS_TILE2D(pArray)                       (!(((XAI_ARRAY_GET_TYPE(pArray)) & (XAI_TYPE_TILE_MASK)) ^ XAI_TYPE_TILE2D_BITS))

#define XAI_ARRAY_GET_AREA(pArray)                        (((pArray)->width) * ((int32_t) (pArray)->height))

/*****************************************
*                   Tile Access Macros
*****************************************/
#define XAI_TILE2D_GET_BUFF_PTR      XAI_ARRAY_GET_BUFF_PTR
#define XAI_TILE2D_SET_BUFF_PTR      XAI_ARRAY_SET_BUFF_PTR

#define XAI_TILE2D_GET_BUFF_SIZE     XAI_ARRAY_GET_BUFF_SIZE
#define XAI_TILE2D_SET_BUFF_SIZE     XAI_ARRAY_SET_BUFF_SIZE

#define XAI_TILE2D_GET_DATA_PTR      XAI_ARRAY_GET_DATA_PTR
#define XAI_TILE2D_SET_DATA_PTR      XAI_ARRAY_SET_DATA_PTR

#define XAI_TILE2D_GET_WIDTH         XAI_ARRAY_GET_WIDTH
#define XAI_TILE2D_SET_WIDTH         XAI_ARRAY_SET_WIDTH

#define XAI_TILE2D_GET_PITCH         XAI_ARRAY_GET_PITCH
#define XAI_TILE2D_SET_PITCH         XAI_ARRAY_SET_PITCH

#define XAI_TILE2D_GET_HEIGHT        XAI_ARRAY_GET_HEIGHT
#define XAI_TILE2D_SET_HEIGHT        XAI_ARRAY_SET_HEIGHT

#define XAI_TILE2D_GET_STATUS_FLAGS  XAI_ARRAY_GET_STATUS_FLAGS
#define XAI_TILE2D_SET_STATUS_FLAGS  XAI_ARRAY_SET_STATUS_FLAGS

#define XAI_TILE2D_GET_TYPE          XAI_ARRAY_GET_TYPE
#define XAI_TILE2D_SET_TYPE          XAI_ARRAY_SET_TYPE

#define XAI_TILE2D_GET_ELEMENT_TYPE  XAI_ARRAY_GET_ELEMENT_TYPE
#define XAI_TILE2D_GET_ELEMENT_SIZE  XAI_ARRAY_GET_ELEMENT_SIZE
#define XAI_TILE2D_IS_TILE2D         XAI_ARRAY_IS_TILE2D

#define XAI_TILE2D_GET_FRAME_PTR(pTile)             ((pTile)->pFrame)
#define XAI_TILE2D_SET_FRAME_PTR(pTile, ptrFrame)   (pTile)->pFrame = ((xai_frame *) (ptrFrame))

#define XAI_TILE2D_GET_X_COORD(pTile)               ((pTile)->x)
#define XAI_TILE2D_SET_X_COORD(pTile, xcoord)       (pTile)->x = ((int32_t) (xcoord))

#define XAI_TILE2D_GET_Y_COORD(pTile)               ((pTile)->y)
#define XAI_TILE2D_SET_Y_COORD(pTile, ycoord)       (pTile)->y = ((int32_t) (ycoord))

#define XAI_TILE2D_GET_EDGE_WIDTH(pTile)            ((pTile)->edgeWidth)
#define XAI_TILE2D_SET_EDGE_WIDTH(pTile, eWidth)    ((pTile)->edgeWidth = (uint16_t) eWidth)

#define XAI_TILE2D_GET_EDGE_HEIGHT(pTile)           ((pTile)->edgeHeight)
#define XAI_TILE2D_SET_EDGE_HEIGHT(pTile, eHeight)  ((pTile)->edgeHeight = (uint16_t) eHeight)

/***********************************
*              Other Marcos
***********************************/
#define XAI_TILE2D_CHECK_VIRTUAL_FRAME(pTile)  ((pTile)->pFrame->pFrameBuff == NULL)
#define XAI_FRAME_CHECK_VIRTUAL_FRAME(pFrame)  ((pFrame)->pFrameBuff == NULL)

typedef enum { XAI_WHD, XAI_DWH, XAI_ID4WH, XAI_ID16WH, XAI_ID32WH, XAI_WHDN, XAI_NWHD, XAI_NDWH, XAI_DWHN, XAI_IN64DWH, XAI_IN32DWH, XAI_RMOD, XAI_IN16DWH, XAI_MTILE, XAI_CMTILE, XAI_RMOD_DWH_ID16WH, XAI_RMOD_InOutDepth32X, XAI_RMOD_ID4WH, XAI_ID16WHN, XAI_ID32WHN, XAI_IN128DWH, XAI_RMOD_DWH_I16_ID16WH, XAI_RMOD_ID16WH, XAI_RMOD_InOutDepth64X, XAI_UNKNOWN }  xai_cnn_data_order;

/******************************************************************************************************************
*
*                    3D definitions - extension of 2D definitions
*
* ****************************************************************************************************************/
typedef struct xai_frame3DStruct
{
  void               *pFrameBuff;
  uint32_t           frameBuffSize;
  void               *pFrameData;
  int32_t            dim1Size;
  int32_t            dim2Size;
  int32_t            dim1Pitch;       // pitch in width dimension
  uint8_t            pixelRes;        // in bits
  uint8_t            pixelPackFormat; // not used in XI library
  uint16_t           dim1Edge1;
  uint16_t           dim1Edge2;
  uint16_t           dim2Edge1;
  uint16_t           dim2Edge2;
  uint16_t           dim3Edge1;
  uint16_t           dim3Edge2;
  uint8_t            paddingType;
  // new fields
  int32_t            dim2Pitch;
  int32_t            dim3Size;
  xai_cnn_data_order dataOrder; // WHD, DWH, etc.
} xai_frame3D, *xai_pFrame3D;

// new access macros
#define XAI_FRAME3D_GET_DIM1(x)                 ((x)->dim1Size)
#define XAI_FRAME3D_SET_DIM1(x, v)              ((x)->dim1Size = (v))
#define XAI_FRAME3D_GET_DIM1_PITCH(x)           ((x)->dim1Pitch)
#define XAI_FRAME3D_SET_DIM1_PITCH(x, v)        ((x)->dim1Pitch = (v))
#define XAI_FRAME3D_GET_DIM1_PITCH_IN_BYTES(x)  ((x)->dim1Pitch * ((x)->pixelRes / 8 + ((x)->pixelRes & 7 != 0)))
#define XAI_FRAME3D_GET_DIM2(x)                 ((x)->dim2Size)
#define XAI_FRAME3D_SET_DIM2(x, v)              ((x)->dim2Size = (v))
#define XAI_FRAME3D_GET_DIM2_PITCH(x)           ((x)->dim2Pitch)
#define XAI_FRAME3D_SET_DIM2_PITCH(x, v)        ((x)->dim2Pitch = (v))
#define XAI_FRAME3D_GET_DIM2_PITCH_IN_BYTES(x)  ((x)->dim2Pitch * ((x)->pixelRes / 8 + ((x)->pixelRes & 7 != 0)))
#define XAI_FRAME3D_GET_DIM3(x)                 ((x)->dim3Size)
#define XAI_FRAME3D_SET_DIM3(x, v)              ((x)->dim3Size = (v))
#define XAI_FRAME3D_GET_DIM1_EDGE1(x)           ((x)->dim1Edge1)
#define XAI_FRAME3D_SET_DIM1_EDGE1(x, v)        ((x)->dim1Edge1 = (v))
#define XAI_FRAME3D_GET_DIM1_EDGE2(x)           ((x)->dim1Edge2)
#define XAI_FRAME3D_SET_DIM1_EDGE2(x, v)        ((x)->dim1Edge2 = (v))
#define XAI_FRAME3D_GET_DIM2_EDGE1(x)           ((x)->dim2Edge1)
#define XAI_FRAME3D_SET_DIM2_EDGE1(x, v)        ((x)->dim2Edge1 = (v))
#define XAI_FRAME3D_GET_DIM2_EDGE2(x)           ((x)->dim2Edge2)
#define XAI_FRAME3D_SET_DIM2_EDGE2(x, v)        ((x)->dim2Edge2 = (v))
#define XAI_FRAME3D_GET_DIM3_EDGE1(x)           ((x)->dim3Edge1)
#define XAI_FRAME3D_SET_DIM3_EDGE1(x, v)        ((x)->dim3Edge1 = (v))
#define XAI_FRAME3D_GET_DIM3_EDGE2(x)           ((x)->dim3Edge2)
#define XAI_FRAME3D_SET_DIM3_EDGE2(x, v)        ((x)->dim3Edge2 = (v))
#define XAI_FRAME3D_GET_DATA_ORDER(x)           ((x)->dataOrder)
#define XAI_FRAME3D_SET_DATA_ORDER(x, v)        ((x)->dataOrder = (v))

typedef struct
{
  int32_t dim1Size;
  int32_t dim2Size;
  int32_t dim3Size;
} xai_size3D;

typedef struct
{
  int32_t dim1Size;
  int32_t dim2Size;
  int32_t dim3Size;
  int32_t dim4Size;
} xai_size4D;

typedef struct
{
  uint16_t dim1Edge1;
  uint16_t dim1Edge2;
  uint16_t dim2Edge1;
  uint16_t dim2Edge2;
  uint16_t dim3Edge1;
  uint16_t dim3Edge2;
} xai_edge3D;

typedef struct
{
  int32_t dataType;
} xai_dataType;

// 3D tile
#define XAI_TILE3D_FIELDS                                                                  \
  uint32_t bufferSize;                                                                     \
  int32_t dim1Size;                                                                        \
  int32_t dim1Pitch;                                                                       \
  uint32_t status; /* Currently not used, planned to be obsolete */                        \
  uint16_t type;                                                                           \
  int32_t dim2Size;                                                                        \
  xai_frame3D *pFrame; /* changed to 3D frame */                                           \
  int32_t dim1Loc;     /* dim1-loc of top-left active pixel in src frame */                \
  int32_t dim2Loc;     /* dim2-loc of top-left active pixel in src frame */                \
  uint16_t dim1Edge1;                                                                      \
  uint16_t dim2Edge1;                                                                      \
  uint16_t dim1Edge2;                                                                      \
  uint16_t dim2Edge2;                                                                      \
  /* new fields */                                                                         \
  int32_t dim2Pitch;                                                                       \
  int32_t dim3Size;                                                                        \
  xai_cnn_data_order dataOrder;                                                            \
  int32_t dim3Loc; /* dim3-loc of top-left active pixel in src frame */                    \
  uint16_t dim3Edge1;                                                                      \
  uint16_t dim3Edge2;                                                                      \
  /* Number of PTILES in a MEMTILE along a particular dimension. Used for MEMTILES only */ \
  int16_t numPtilesDim1;                                                                   \
  int16_t numPtilesDim2;                                                                   \
  int16_t numPtilesDim3;

typedef struct xai_tile3DStruct
{
  void *pBuffer;
  void *pData;
  XAI_TILE3D_FIELDS
} xai_tile3D, *xai_pTile3D;

typedef struct xai_tile3DStruct_64
{
  uint64_t pBuffer;
  uint64_t pData;
  XAI_TILE3D_FIELDS
} xai_tile3D_64, *xai_pTile3D_64;

#define XAI_TILE3D_GET_DIM1(x)           ((x)->dim1Size)
#define XAI_TILE3D_SET_DIM1(x, v)        ((x)->dim1Size = (v))
#define XAI_TILE3D_GET_DIM1_PITCH(x)     ((x)->dim1Pitch)
#define XAI_TILE3D_SET_DIM1_PITCH(x, v)  ((x)->dim1Pitch = (v))
#define XAI_TILE3D_GET_DIM2(x)           ((x)->dim2Size)
#define XAI_TILE3D_SET_DIM2(x, v)        ((x)->dim2Size = (v))
#define XAI_TILE3D_GET_DIM2_PITCH(x)     ((x)->dim2Pitch)
#define XAI_TILE3D_SET_DIM2_PITCH(x, v)  ((x)->dim2Pitch = (v))
#define XAI_TILE3D_GET_DIM3(x)           ((x)->dim3Size)
#define XAI_TILE3D_SET_DIM3(x, v)        ((x)->dim3Size = (v))
#define XAI_TILE3D_GET_DATA_ORDER(x)     ((x)->dataOrder)
#define XAI_TILE3D_SET_DATA_ORDER(x, v)  ((x)->dataOrder = (v))
#define XAI_TILE3D_GET_DIM1_COORD(x)     ((x)->dim1Loc)
#define XAI_TILE3D_SET_DIM1_COORD(x, v)  ((x)->dim1Loc = (v))
#define XAI_TILE3D_GET_DIM2_COORD(x)     ((x)->dim2Loc)
#define XAI_TILE3D_SET_DIM2_COORD(x, v)  ((x)->dim2Loc = (v))
#define XAI_TILE3D_GET_DIM3_COORD(x)     ((x)->dim3Loc)
#define XAI_TILE3D_SET_DIM3_COORD(x, v)  ((x)->dim3Loc = (v))
#define XAI_TILE3D_GET_DIM1_EDGE1(x)     ((x)->dim1Edge1)
#define XAI_TILE3D_SET_DIM1_EDGE1(x, v)  ((x)->dim1Edge1 = (v))
#define XAI_TILE3D_GET_DIM1_EDGE2(x)     ((x)->dim1Edge2)
#define XAI_TILE3D_SET_DIM1_EDGE2(x, v)  ((x)->dim1Edge2 = (v))
#define XAI_TILE3D_GET_DIM2_EDGE1(x)     ((x)->dim2Edge1)
#define XAI_TILE3D_SET_DIM2_EDGE1(x, v)  ((x)->dim2Edge1 = (v))
#define XAI_TILE3D_GET_DIM2_EDGE2(x)     ((x)->dim2Edge2)
#define XAI_TILE3D_SET_DIM2_EDGE2(x, v)  ((x)->dim2Edge2 = (v))
#define XAI_TILE3D_GET_DIM3_EDGE1(x)     ((x)->dim3Edge1)
#define XAI_TILE3D_SET_DIM3_EDGE1(x, v)  ((x)->dim3Edge1 = (v))
#define XAI_TILE3D_GET_DIM3_EDGE2(x)     ((x)->dim3Edge2)
#define XAI_TILE3D_SET_DIM3_EDGE2(x, v)  ((x)->dim3Edge2 = (v))

/*****************************************
*   Data type definitions
*****************************************/
#define XAI_TYPE_IS_TILE3D(type)  (!(((type) & (XAI_TYPE_TILE_MASK)) ^ XAI_TYPE_TILE3D_BITS))

#define XAI_TILE3D_U4    (XAI_U4 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_U8    (XAI_U8 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_U16   (XAI_U16 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_U32   (XAI_U32 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_U64   (XAI_U64 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_U128  (XAI_U128 | XAI_TYPE_TILE3D_BITS)

#define XAI_TILE3D_S4    (XAI_S4 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_S8    (XAI_S8 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_S16   (XAI_S16 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_S32   (XAI_S32 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_S64   (XAI_S64 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_S128  (XAI_S128 | XAI_TYPE_TILE3D_BITS)

#define XAI_TILE3D_F8    (XAI_F8 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_F16   (XAI_F16 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_F32   (XAI_F32 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_F64   (XAI_F64 | XAI_TYPE_TILE3D_BITS)
#define XAI_TILE3D_F128  (XAI_F128 | XAI_TYPE_TILE3D_BITS)

/*****************************************
*                   3D Frame Access Macros
*****************************************/
#define XAI_FRAME3D_GET_BUFF_PTR      XAI_FRAME_GET_BUFF_PTR
#define XAI_FRAME3D_SET_BUFF_PTR      XAI_FRAME_SET_BUFF_PTR

#define XAI_FRAME3D_GET_BUFF_SIZE     XAI_FRAME_GET_BUFF_SIZE
#define XAI_FRAME3D_SET_BUFF_SIZE     XAI_FRAME_SET_BUFF_SIZE

#define XAI_FRAME3D_GET_DATA_PTR      XAI_FRAME_GET_DATA_PTR
#define XAI_FRAME3D_SET_DATA_PTR      XAI_FRAME_SET_DATA_PTR

#define XAI_FRAME3D_GET_PIXEL_RES     XAI_FRAME_GET_PIXEL_RES
#define XAI_FRAME3D_SET_PIXEL_RES     XAI_FRAME_SET_PIXEL_RES

#define XAI_FRAME3D_GET_PIXEL_FORMAT  XAI_FRAME_GET_PIXEL_FORMAT
#define XAI_FRAME3D_SET_PIXEL_FORMAT  XAI_FRAME_SET_PIXEL_FORMAT

#define XAI_FRAME3D_GET_PADDING_TYPE  XAI_FRAME_GET_PADDING_TYPE
#define XAI_FRAME3D_SET_PADDING_TYPE  XAI_FRAME_SET_PADDING_TYPE

/*****************************************
*                   3D Tile Access Macros
*****************************************/
#define XAI_TILE3D_GET_BUFF_PTR        XAI_TILE2D_GET_BUFF_PTR
#define XAI_TILE3D_SET_BUFF_PTR        XAI_TILE2D_SET_BUFF_PTR
#define XAI_TILE3D_SET_BUFF_PTR_COEFF  XAI_TILE2D_SET_BUFF_PTR_COEFF

#define XAI_TILE3D_GET_BUFF_SIZE       XAI_TILE2D_GET_BUFF_SIZE
#define XAI_TILE3D_SET_BUFF_SIZE       XAI_TILE2D_SET_BUFF_SIZE

#define XAI_TILE3D_GET_DATA_PTR        XAI_TILE2D_GET_DATA_PTR
#define XAI_TILE3D_SET_DATA_PTR        XAI_TILE2D_SET_DATA_PTR
#define XAI_TILE3D_SET_DATA_PTR_COEFF  XAI_TILE2D_SET_DATA_PTR_COEFF

#define XAI_TILE3D_GET_STATUS_FLAGS    XAI_TILE2D_GET_STATUS_FLAGS
#define XAI_TILE3D_SET_STATUS_FLAGS    XAI_TILE2D_SET_STATUS_FLAGS

#define XAI_TILE3D_GET_TYPE            XAI_TILE2D_GET_TYPE
#define XAI_TILE3D_SET_TYPE            XAI_TILE2D_SET_TYPE

#define XAI_TILE3D_GET_ELEMENT_TYPE    XAI_TILE2D_GET_ELEMENT_TYPE
#define XAI_TILE3D_GET_ELEMENT_SIZE    XAI_TILE2D_GET_ELEMENT_SIZE
#define XAI_TILE3D_IS_TILE             XAI_TILE2D_IS_TILE2D

#define XAI_TILE3D_GET_FRAME_PTR(pTile3D)            ((pTile3D)->pFrame)
#define XAI_TILE3D_SET_FRAME_PTR(pTile3D, ptrFrame)  (pTile3D)->pFrame = ((xai_pFrame3D) (ptrFrame))

#define XAI_TILE3D_CHECK_STATUS_FLAGS_DMA_ONGOING  XAI_TILE2D_CHECK_STATUS_FLAGS_DMA_ONGOING

/***********************************
*              Other Marcos
***********************************/
#define XAI_TILE3D_CHECK_VIRTUAL_FRAME   XAI_TILE2D_CHECK_VIRTUAL_FRAME
#define XAI_FRAME3D_CHECK_VIRTUAL_FRAME  XAI_FRAME_CHECK_VIRTUAL_FRAME

typedef enum
{
  XAI_TILE_UNALIGNED,
  XAI_EDGE_ALIGNED_32,
  XAI_DATA_ALIGNED_32,
  XAI_EDGE_ALIGNED_64,
  XAI_DATA_ALIGNED_64,
  EDGE_ALIGNED_128,
  DATA_ALIGNED_128,
} xai_buffer_align_type_t;

// Only Q8, 240 and 341 uses alignment = 127. for P6,P1 and Q7 like dsps alignment = 127 is not supported
#define XAI_SETUP_TILE3D(type, pTile, pBuf, pFrame, bufSize, dim1Size, dim2Size, dim3Size, dim1Pitch, dim2Pitch,                 \
                         dim1Edge1, dim1Edge2, dim2Edge1, dim2Edge2, dim3Edge1, dim3Edge2, dim1Loc, dim2Loc, dim3Loc, dataOrder, \
                         alignType)                                                                                              \
  {                                                                                                                              \
    XAI_TILE3D_SET_TYPE(pTile, type);                                                                                            \
    XAI_TILE3D_SET_FRAME_PTR(pTile, pFrame);                                                                                     \
    XAI_TILE3D_SET_BUFF_PTR(pTile, pBuf);                                                                                        \
    XAI_TILE3D_SET_BUFF_SIZE(pTile, bufSize);                                                                                    \
    XAI_TILE3D_SET_DIM1(pTile, dim1Size);                                                                                        \
    XAI_TILE3D_SET_DIM2(pTile, dim2Size);                                                                                        \
    XAI_TILE3D_SET_DIM3(pTile, dim3Size);                                                                                        \
    XAI_TILE3D_SET_DIM1_PITCH(pTile, dim1Pitch);                                                                                 \
    XAI_TILE3D_SET_DIM2_PITCH(pTile, dim2Pitch);                                                                                 \
    uint8_t *edgePtr  = (uint8_t *) pBuf, *dataPtr;                                                                              \
    int32_t alignment = 127;                                                                                                     \
    if ((alignType == XAI_EDGE_ALIGNED_64) || (alignType == XAI_DATA_ALIGNED_64)) { alignment = 63; }                            \
    if ((alignType == XAI_EDGE_ALIGNED_32) || (alignType == XAI_DATA_ALIGNED_32)) { alignment = 31; }                            \
    if ((alignType == XAI_EDGE_ALIGNED_32) || (alignType == XAI_EDGE_ALIGNED_64) || (alignType == EDGE_ALIGNED_128))             \
    {                                                                                                                            \
      edgePtr = (uint8_t *) (((uintptr_t) (pBuf) + alignment) & (~alignment));                                                   \
    }                                                                                                                            \
    XAI_TILE3D_SET_DATA_PTR(pTile, edgePtr + ((dim3Edge1) * (dim2Pitch) +                                                        \
                                              (dim2Edge1) * (dim1Pitch) + (dim1Edge1)) * XAI_TILE3D_GET_ELEMENT_SIZE(pTile));    \
    if ((alignType == XAI_DATA_ALIGNED_32) || (alignType == XAI_DATA_ALIGNED_64) || (alignType == DATA_ALIGNED_128))             \
    {                                                                                                                            \
      dataPtr = (uint8_t *) XAI_TILE3D_GET_DATA_PTR(pTile);                                                                      \
      dataPtr = (uint8_t *) (((uintptr_t) (dataPtr) + alignment) & (~alignment));                                                \
      XAI_TILE3D_SET_DATA_PTR(pTile, dataPtr);                                                                                   \
    }                                                                                                                            \
    XAI_TILE3D_SET_DIM1_EDGE1(pTile, dim1Edge1);                                                                                 \
    XAI_TILE3D_SET_DIM1_EDGE2(pTile, dim1Edge2);                                                                                 \
    XAI_TILE3D_SET_DIM2_EDGE1(pTile, dim2Edge1);                                                                                 \
    XAI_TILE3D_SET_DIM2_EDGE2(pTile, dim2Edge2);                                                                                 \
    XAI_TILE3D_SET_DIM3_EDGE1(pTile, dim3Edge1);                                                                                 \
    XAI_TILE3D_SET_DIM3_EDGE2(pTile, dim3Edge2);                                                                                 \
    XAI_TILE3D_SET_DIM1_COORD(pTile, dim1Loc);                                                                                   \
    XAI_TILE3D_SET_DIM2_COORD(pTile, dim2Loc);                                                                                   \
    XAI_TILE3D_SET_DIM3_COORD(pTile, dim3Loc);                                                                                   \
    XAI_TILE3D_SET_DATA_ORDER(pTile, dataOrder);                                                                                 \
  }

#define XAI_SETUP_FRAME3D(pFrame, pFrameBuffer, bufSize, dim1Size, dim2Size, dim3Size, dim1Pitch, dim2Pitch,                    \
                          dim1Edge1, dim1Edge2, dim2Edge1, dim2Edge2, dim3Edge1, dim3Edge2, pixRes, pixPackFormat, paddingType, \
                          dataOrder)                                                                                            \
  {                                                                                                                             \
    XAI_FRAME3D_SET_BUFF_PTR(pFrame, pFrameBuffer);                                                                             \
    XAI_FRAME3D_SET_BUFF_SIZE(pFrame, bufSize);                                                                                 \
    XAI_FRAME3D_SET_DIM1(pFrame, dim1Size);                                                                                     \
    XAI_FRAME3D_SET_DIM2(pFrame, dim2Size);                                                                                     \
    XAI_FRAME3D_SET_DIM3(pFrame, dim3Size);                                                                                     \
    XAI_FRAME3D_SET_DIM1_PITCH(pFrame, dim1Pitch);                                                                              \
    XAI_FRAME3D_SET_DIM2_PITCH(pFrame, dim2Pitch);                                                                              \
    XAI_FRAME3D_SET_DATA_PTR(pFrame, pFrameBuffer + ((dim3Edge1) * (dim2Pitch) +                                                \
                                                     (dim2Edge1) * (dim1Pitch) + (dim1Edge1)) * pixRes);                        \
    XAI_FRAME3D_SET_DIM1_EDGE1(pFrame, dim1Edge1);                                                                              \
    XAI_FRAME3D_SET_DIM1_EDGE2(pFrame, dim1Edge2);                                                                              \
    XAI_FRAME3D_SET_DIM2_EDGE1(pFrame, dim2Edge1);                                                                              \
    XAI_FRAME3D_SET_DIM2_EDGE2(pFrame, dim2Edge2);                                                                              \
    XAI_FRAME3D_SET_DIM3_EDGE1(pFrame, dim3Edge1);                                                                              \
    XAI_FRAME3D_SET_DIM3_EDGE2(pFrame, dim3Edge2);                                                                              \
    XAI_FRAME3D_SET_PIXEL_RES(pFrame, pixRes);                                                                                  \
    XAI_FRAME3D_SET_PIXEL_FORMAT(pFrame, pixPackFormat);                                                                        \
    XAI_FRAME3D_SET_PADDING_TYPE(pFrame, paddingType);                                                                          \
    XAI_FRAME3D_SET_DATA_ORDER(pFrame, dataOrder);                                                                              \
  }

#define XAI_COPY_FRAME3D_TO_TILE3D(frame, tile)         {               \
    XAI_TILE3D_SET_DIM1(tile, XAI_FRAME3D_GET_DIM1(frame));             \
    XAI_TILE3D_SET_DIM1_PITCH(tile, XAI_FRAME3D_GET_DIM1_PITCH(frame)); \
    XAI_TILE3D_SET_DIM1_EDGE1(tile, XAI_FRAME3D_GET_DIM1_EDGE1(frame)); \
    XAI_TILE3D_SET_DIM1_EDGE2(tile, XAI_FRAME3D_GET_DIM1_EDGE2(frame)); \
    XAI_TILE3D_SET_DIM2(tile, XAI_FRAME3D_GET_DIM2(frame));             \
    XAI_TILE3D_SET_DIM2_PITCH(tile, XAI_FRAME3D_GET_DIM2_PITCH(frame)); \
    XAI_TILE3D_SET_DIM2_EDGE1(tile, XAI_FRAME3D_GET_DIM2_EDGE1(frame)); \
    XAI_TILE3D_SET_DIM2_EDGE2(tile, XAI_FRAME3D_GET_DIM2_EDGE2(frame)); \
    XAI_TILE3D_SET_DIM3(tile, XAI_FRAME3D_GET_DIM3(frame));             \
    XAI_TILE3D_SET_DIM3_EDGE1(tile, XAI_FRAME3D_GET_DIM3_EDGE1(frame)); \
    XAI_TILE3D_SET_DIM3_EDGE2(tile, XAI_FRAME3D_GET_DIM3_EDGE2(frame)); \
    XAI_TILE3D_SET_DATA_PTR(tile, XAI_FRAME3D_GET_DATA_PTR(frame));     \
    XAI_TILE3D_SET_DATA_ORDER(tile, XAI_FRAME3D_GET_DATA_ORDER(frame)); \
}

#define XAI_COPY_FRAME3D_TO_FRAME3D(frameIn, frameOut)  {                      \
    XAI_FRAME3D_SET_DIM1(frameOut, XAI_FRAME3D_GET_DIM1(frameIn));             \
    XAI_FRAME3D_SET_DIM1_PITCH(frameOut, XAI_FRAME3D_GET_DIM1_PITCH(frameIn)); \
    XAI_FRAME3D_SET_DIM1_EDGE1(frameOut, XAI_FRAME3D_GET_DIM1_EDGE1(frameIn)); \
    XAI_FRAME3D_SET_DIM1_EDGE2(frameOut, XAI_FRAME3D_GET_DIM1_EDGE2(frameIn)); \
    XAI_FRAME3D_SET_DIM2(frameOut, XAI_FRAME3D_GET_DIM2(frameIn));             \
    XAI_FRAME3D_SET_DIM2_PITCH(frameOut, XAI_FRAME3D_GET_DIM2_PITCH(frameIn)); \
    XAI_FRAME3D_SET_DIM2_EDGE1(frameOut, XAI_FRAME3D_GET_DIM2_EDGE1(frameIn)); \
    XAI_FRAME3D_SET_DIM2_EDGE2(frameOut, XAI_FRAME3D_GET_DIM2_EDGE2(frameIn)); \
    XAI_FRAME3D_SET_DIM3(frameOut, XAI_FRAME3D_GET_DIM3(frameIn));             \
    XAI_FRAME3D_SET_DIM3_EDGE1(frameOut, XAI_FRAME3D_GET_DIM2_EDGE1(frameIn)); \
    XAI_FRAME3D_SET_DIM3_EDGE2(frameOut, XAI_FRAME3D_GET_DIM2_EDGE2(frameIn)); \
    XAI_FRAME3D_SET_DATA_PTR(frameOut, XAI_FRAME3D_GET_DATA_PTR(frameIn));     \
    XAI_FRAME3D_SET_DATA_ORDER(frameOut, XAI_FRAME3D_GET_DATA_ORDER(frameIn)); \
    XAI_FRAME3D_SET_PIXEL_RES(frameOut, XAI_FRAME3D_GET_PIXEL_RES(frameIn));   \
}

#define XAI_COPY_TILE3D_TO_TILE3D(tileIn, tileOut)      {                  \
    XAI_TILE3D_SET_DIM1(tileOut, XAI_TILE3D_GET_DIM1(tileIn));             \
    XAI_TILE3D_SET_DIM1_PITCH(tileOut, XAI_TILE3D_GET_DIM1_PITCH(tileIn)); \
    XAI_TILE3D_SET_DIM1_EDGE1(tileOut, XAI_TILE3D_GET_DIM1_EDGE1(tileIn)); \
    XAI_TILE3D_SET_DIM1_EDGE2(tileOut, XAI_TILE3D_GET_DIM1_EDGE2(tileIn)); \
    XAI_TILE3D_SET_DIM2(tileOut, XAI_TILE3D_GET_DIM2(tileIn));             \
    XAI_TILE3D_SET_DIM2_PITCH(tileOut, XAI_TILE3D_GET_DIM2_PITCH(tileIn)); \
    XAI_TILE3D_SET_DIM2_EDGE1(tileOut, XAI_TILE3D_GET_DIM2_EDGE1(tileIn)); \
    XAI_TILE3D_SET_DIM2_EDGE2(tileOut, XAI_TILE3D_GET_DIM2_EDGE2(tileIn)); \
    XAI_TILE3D_SET_DIM3(tileOut, XAI_TILE3D_GET_DIM3(tileIn));             \
    XAI_TILE3D_SET_DIM3_EDGE1(tileOut, XAI_TILE3D_GET_DIM3_EDGE1(tileIn)); \
    XAI_TILE3D_SET_DIM3_EDGE2(tileOut, XAI_TILE3D_GET_DIM3_EDGE2(tileIn)); \
    XAI_TILE3D_SET_DATA_PTR(tileOut, XAI_TILE3D_GET_DATA_PTR(tileIn));     \
    XAI_TILE3D_SET_DATA_ORDER(tileOut, XAI_TILE3D_GET_DATA_ORDER(tileIn)); \
}

// Assumes 8 bit pixRes and Edge1 = Edge2
#define XAI_TILE3D_UPDATE_EDGE_DIM1(pTile, newEdgeSize)                  \
  {                                                                      \
    uint16_t currEdgeSize = (uint16_t) XAI_TILE3D_GET_DIM1_EDGE1(pTile); \
    uint32_t dim1Pitch    = (uint32_t) XAI_TILE3D_GET_DIM1_PITCH(pTile); \
    uintptr_t dataU32     = (uintptr_t) XAI_TILE3D_GET_DATA_PTR(pTile);  \
    dataU32 = dataU32 + newEdgeSize - currEdgeSize;                      \
    XAI_TILE3D_SET_DATA_PTR(pTile, (void *) dataU32);                    \
    XAI_TILE3D_SET_DIM1_EDGE1(pTile, newEdgeSize);                       \
    XAI_TILE3D_SET_DIM1_EDGE2(pTile, newEdgeSize);                       \
    XAI_TILE3D_SET_DIM1(pTile, dim1Pitch - 2 * newEdgeSize);             \
  }

// Assumes 8 bit pixRes and Edge1 = Edge2
#define XAI_TILE3D_UPDATE_EDGE_DIM2(pTile, newEdgeSize)                      \
  {                                                                          \
    uint16_t currEdgeSize = (uint16_t) XAI_TILE3D_GET_DIM2_EDGE1(pTile);     \
    uint32_t dim1Pitch    = (uint32_t) XAI_TILE3D_GET_DIM1_PITCH(pTile);     \
    uint16_t dim2Size     = (uint16_t) XAI_TILE3D_GET_DIM2(pTile);           \
    uintptr_t dataU32     = (uintptr_t) XAI_TILE3D_GET_DATA_PTR(pTile);      \
    dataU32 = dataU32 + dim1Pitch * (newEdgeSize - currEdgeSize);            \
    XAI_TILE3D_SET_DATA_PTR(pTile, (void *) dataU32);                        \
    XAI_TILE3D_SET_DIM2_EDGE1(pTile, newEdgeSize);                           \
    XAI_TILE3D_SET_DIM2_EDGE2(pTile, newEdgeSize);                           \
    XAI_TILE3D_SET_DIM2(pTile, dim2Size + 2 * (currEdgeSize - newEdgeSize)); \
  }

// Assumes 8 bit pixRes and Edge1 = Edge2
#define XAI_TILE3D_UPDATE_EDGE_DIM3(pTile, newEdgeSize)                      \
  {                                                                          \
    uint16_t currEdgeSize = (uint16_t) XAI_TILE3D_GET_DIM3_EDGE1(pTile);     \
    uint32_t dim2Pitch    = (uint32_t) XAI_TILE3D_GET_DIM2_PITCH(pTile);     \
    uint16_t dim3Size     = (uint16_t) XAI_TILE3D_GET_DIM3(pTile);           \
    uintptr_t dataU32     = (uintptr_t) XAI_TILE3D_GET_DATA_PTR(pTile);      \
    dataU32 = dataU32 + dim2Pitch * (newEdgeSize - currEdgeSize);            \
    XAI_TILE3D_SET_DATA_PTR(pTile, (void *) dataU32);                        \
    XAI_TILE3D_SET_DIM3_EDGE1(pTile, newEdgeSize);                           \
    XAI_TILE3D_SET_DIM3_EDGE2(pTile, newEdgeSize);                           \
    XAI_TILE3D_SET_DIM3(pTile, dim3Size + 2 * (currEdgeSize - newEdgeSize)); \
  }

#define XAI_TILE3D_UPDATE_DIMENSIONS(pTile, dim1Loc, dim2Loc, dim3Loc, dim1Size, dim2Size, dim3Size, \
                                     dim1Pitch, dim2Pitch)                                           \
  {                                                                                                  \
    XAI_TILE3D_SET_DIM1_COORD(pTile, dim1Loc);                                                       \
    XAI_TILE3D_SET_DIM2_COORD(pTile, dim2Loc);                                                       \
    XAI_TILE3D_SET_DIM3_COORD(pTile, dim3Loc);                                                       \
    XAI_TILE3D_SET_DIM1(pTile, dim1Size);                                                            \
    XAI_TILE3D_SET_DIM2(pTile, dim2Size);                                                            \
    XAI_TILE3D_SET_DIM3(pTile, dim3Size);                                                            \
    XAI_TILE3D_SET_DIM1_PITCH(pTile, dim1Pitch);                                                     \
    XAI_TILE3D_SET_DIM2_PITCH(pTile, dim2Pitch);                                                     \
  }

/******************************************************************************************************************
*
*                    4D definitions - extension of 3D definitions
*
* ****************************************************************************************************************/
typedef struct xai_frame4DStruct
{
  void               *pFrameBuff;
  uint32_t           frameBuffSize;
  void               *pFrameData;
  int32_t            dim1Size;
  int32_t            dim2Size;
  int32_t            dim1Pitch; // pitch in width dimension
  uint8_t            pixelRes;  // in bits
  uint8_t            pixelPackFormat;
  uint16_t           dim1Edge1;
  uint16_t           dim1Edge2;
  uint16_t           dim2Edge1;
  uint16_t           dim2Edge2;
  uint16_t           dim3Edge1;
  uint16_t           dim3Edge2;
  uint8_t            paddingType;
  // new fields
  int32_t            dim2Pitch;
  int32_t            dim3Size;
  xai_cnn_data_order dataOrder; // WHD, DWH, WHDN, NWHD, etc.
  // new fields
  int32_t            dim3Pitch;
  int32_t            dim4Size;
} xai_frame4D, *xai_pFrame4D;

// new access macros
#define XAI_FRAME4D_GET_DIM1                 XAI_FRAME3D_GET_DIM1
#define XAI_FRAME4D_SET_DIM1                 XAI_FRAME3D_SET_DIM1
#define XAI_FRAME4D_GET_DIM1_PITCH           XAI_FRAME3D_GET_DIM1_PITCH
#define XAI_FRAME4D_SET_DIM1_PITCH           XAI_FRAME3D_SET_DIM1_PITCH
#define XAI_FRAME4D_GET_DIM1_PITCH_IN_BYTES  XAI_FRAME3D_GET_DIM1_PITCH_IN_BYTES
#define XAI_FRAME4D_GET_DIM2                 XAI_FRAME3D_GET_DIM2
#define XAI_FRAME4D_SET_DIM2                 XAI_FRAME3D_SET_DIM2
#define XAI_FRAME4D_GET_DIM2_PITCH           XAI_FRAME3D_GET_DIM2_PITCH
#define XAI_FRAME4D_SET_DIM2_PITCH           XAI_FRAME3D_SET_DIM2_PITCH
#define XAI_FRAME4D_GET_DIM2_PITCH_IN_BYTES  XAI_FRAME3D_GET_DIM2_PITCH_IN_BYTES
#define XAI_FRAME4D_GET_DIM3                 XAI_FRAME3D_GET_DIM3
#define XAI_FRAME4D_SET_DIM3                 XAI_FRAME3D_SET_DIM3
#define XAI_FRAME4D_GET_DATA_ORDER           XAI_FRAME3D_GET_DATA_ORDER
#define XAI_FRAME4D_SET_DATA_ORDER           XAI_FRAME3D_SET_DATA_ORDER
#define XAI_FRAME4D_GET_DIM1_EDGE1           XAI_FRAME3D_GET_DIM1_EDGE1
#define XAI_FRAME4D_SET_DIM1_EDGE1           XAI_FRAME3D_SET_DIM1_EDGE1
#define XAI_FRAME4D_GET_DIM1_EDGE2           XAI_FRAME3D_GET_DIM1_EDGE2
#define XAI_FRAME4D_SET_DIM1_EDGE2           XAI_FRAME3D_SET_DIM1_EDGE2
#define XAI_FRAME4D_GET_DIM2_EDGE1           XAI_FRAME3D_GET_DIM2_EDGE1
#define XAI_FRAME4D_SET_DIM2_EDGE1           XAI_FRAME3D_SET_DIM2_EDGE1
#define XAI_FRAME4D_GET_DIM2_EDGE2           XAI_FRAME3D_GET_DIM2_EDGE2
#define XAI_FRAME4D_SET_DIM2_EDGE2           XAI_FRAME3D_SET_DIM2_EDGE2
#define XAI_FRAME4D_GET_DIM4(x)           ((x)->dim4Size)
#define XAI_FRAME4D_SET_DIM4(x, v)        ((x)->dim4Size = (v))
#define XAI_FRAME4D_GET_DIM3_PITCH(x)     ((x)->dim3Pitch)
#define XAI_FRAME4D_SET_DIM3_PITCH(x, v)  ((x)->dim3Pitch = (v))
#define XAI_FRAME4D_GET_DIM3_EDGE1(x)     ((x)->dim3Edge1)
#define XAI_FRAME4D_SET_DIM3_EDGE1(x, v)  ((x)->dim3Edge1 = (v))
#define XAI_FRAME4D_GET_DIM3_EDGE2(x)     ((x)->dim3Edge2)
#define XAI_FRAME4D_SET_DIM3_EDGE2(x, v)  ((x)->dim3Edge2 = (v))

// 4D tile
#define XAI_TILE4D_FIELDS                                                                  \
  uint32_t bufferSize;                                                                     \
  int32_t dim1Size;                                                                        \
  int32_t dim1Pitch;                                                                       \
  uint32_t status; /*Currently not used, planned to be obsolete*/                          \
  uint16_t type;                                                                           \
  int32_t dim2Size;                                                                        \
  xai_frame4D *pFrame;                                                                     \
  int32_t dim1Loc;  /* dim1-loc of top-left active pixel in src frame */                   \
  int32_t dim2Loc;  /* dim2-loc of top-left active pixel in src frame */                   \
  uint16_t dim1Edge1;                                                                      \
  uint16_t dim2Edge1;                                                                      \
  uint16_t dim1Edge2;                                                                      \
  uint16_t dim2Edge2;                                                                      \
  /* new fields */                                                                         \
  int32_t dim2Pitch;                                                                       \
  int32_t dim3Size;                                                                        \
  xai_cnn_data_order dataOrder;                                                            \
  int32_t dim3Loc;  /* dim3-loc of top-left active pixel in src frame */                   \
  uint16_t dim3Edge1;                                                                      \
  uint16_t dim3Edge2;                                                                      \
  /* new fields */                                                                         \
  int32_t dim3Pitch;                                                                       \
  int32_t dim4Size;  /* 4th dimension is num for lack of better term */                    \
  int32_t dim4Loc;   /* dim4-loc of top-left active pixel in src frame */                  \
  /* Number of PTILES in a MEMTILE along a particular dimension. Used for MEMTILES only */ \
  int16_t numPtilesDim1;                                                                   \
  int16_t numPtilesDim2;                                                                   \
  int16_t numPtilesDim3;

typedef struct xai_tile4DStruct
{
  void *pBuffer;
  void *pData;
  XAI_TILE4D_FIELDS
#ifdef GLOW_BUILD
  int8_t printFlag;
  const char *nodeName;
  const char *fileName;
#endif // GLOW_BUILD
} xai_tile4D, *xai_pTile4D;

typedef struct xai_tile4DStruct_64
{
  uint64_t pBuffer;
  uint64_t pData;
  XAI_TILE4D_FIELDS
#ifdef GLOW_BUILD
  int8_t printFlag;
  const char *nodeName;
  const char *fileName;
#endif // GLOW_BUILD
} xai_tile4D_64, *xai_pTile4D_64;

#define XAI_TILE4D_GET_DIM1        XAI_TILE3D_GET_DIM1
#define XAI_TILE4D_SET_DIM1        XAI_TILE3D_SET_DIM1
#define XAI_TILE4D_GET_DIM1_PITCH  XAI_TILE3D_GET_DIM1_PITCH
#define XAI_TILE4D_SET_DIM1_PITCH  XAI_TILE3D_SET_DIM1_PITCH
#define XAI_TILE4D_GET_DIM2        XAI_TILE3D_GET_DIM2
#define XAI_TILE4D_SET_DIM2        XAI_TILE3D_SET_DIM2
#define XAI_TILE4D_GET_DIM2_PITCH  XAI_TILE3D_GET_DIM2_PITCH
#define XAI_TILE4D_SET_DIM2_PITCH  XAI_TILE3D_SET_DIM2_PITCH
#define XAI_TILE4D_GET_DIM3        XAI_TILE3D_GET_DIM3
#define XAI_TILE4D_SET_DIM3        XAI_TILE3D_SET_DIM3
#define XAI_TILE4D_GET_DIM3_PITCH(x)     ((x)->dim3Pitch)
#define XAI_TILE4D_SET_DIM3_PITCH(x, v)  ((x)->dim3Pitch = (v))
#define XAI_TILE4D_GET_DIM4(x)           ((x)->dim4Size)
#define XAI_TILE4D_SET_DIM4(x, v)        ((x)->dim4Size = (v))
#define XAI_TILE4D_GET_DIM1_EDGE1  XAI_TILE3D_GET_DIM1_EDGE1
#define XAI_TILE4D_SET_DIM1_EDGE1  XAI_TILE3D_SET_DIM1_EDGE1
#define XAI_TILE4D_GET_DIM1_EDGE2  XAI_TILE3D_GET_DIM1_EDGE2
#define XAI_TILE4D_SET_DIM1_EDGE2  XAI_TILE3D_SET_DIM1_EDGE2
#define XAI_TILE4D_GET_DIM2_EDGE1  XAI_TILE3D_GET_DIM2_EDGE1
#define XAI_TILE4D_SET_DIM2_EDGE1  XAI_TILE3D_SET_DIM2_EDGE1
#define XAI_TILE4D_GET_DIM2_EDGE2  XAI_TILE3D_GET_DIM2_EDGE2
#define XAI_TILE4D_SET_DIM2_EDGE2  XAI_TILE3D_SET_DIM2_EDGE2
#define XAI_TILE4D_GET_DIM3_EDGE1  XAI_TILE3D_GET_DIM3_EDGE1
#define XAI_TILE4D_SET_DIM3_EDGE1  XAI_TILE3D_SET_DIM3_EDGE1
#define XAI_TILE4D_GET_DIM3_EDGE2  XAI_TILE3D_GET_DIM3_EDGE2
#define XAI_TILE4D_SET_DIM3_EDGE2  XAI_TILE3D_SET_DIM3_EDGE2
#define XAI_TILE4D_GET_DATA_ORDER  XAI_TILE3D_GET_DATA_ORDER
#define XAI_TILE4D_SET_DATA_ORDER  XAI_TILE3D_SET_DATA_ORDER
#define XAI_TILE4D_GET_DIM1_COORD  XAI_TILE3D_GET_DIM1_COORD
#define XAI_TILE4D_SET_DIM1_COORD  XAI_TILE3D_SET_DIM1_COORD
#define XAI_TILE4D_GET_DIM2_COORD  XAI_TILE3D_GET_DIM2_COORD
#define XAI_TILE4D_SET_DIM2_COORD  XAI_TILE3D_SET_DIM2_COORD
#define XAI_TILE4D_GET_DIM3_COORD  XAI_TILE3D_GET_DIM3_COORD
#define XAI_TILE4D_SET_DIM3_COORD  XAI_TILE3D_SET_DIM3_COORD
#define XAI_TILE4D_GET_DIM4_COORD(x)     ((x)->dim4Loc)
#define XAI_TILE4D_SET_DIM4_COORD(x, v)  ((x)->dim4Loc = (v))
#ifdef GLOW_BUILD
#define XAI_TILE4D_GET_PRINT_FLAG(x)     ((x)->printFlag)
#define XAI_TILE4D_SET_PRINT_FLAG(x, v)  ((x)->printFlag = (v))
#define XAI_TILE4D_GET_NODE_NAME(x)      ((x)->nodeName)
#define XAI_TILE4D_SET_NODE_NAME(x, v)   ((x)->nodeName = (v))
#define XAI_TILE4D_GET_FILE_NAME(x)      ((x)->fileName)
#define XAI_TILE4D_SET_FILE_NAME(x, v)   ((x)->fileName = (v))
#endif

/*****************************************
*   Data type definitions
*****************************************/
#define XAI_TYPE_IS_TILE4D(type)  (!(((type) & (XAI_TYPE_TILE_MASK)) ^ XAI_TYPE_TILE4D_BITS))

#define XAI_TILE4D_U4    (XAI_U4 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_U8    (XAI_U8 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_U16   (XAI_U16 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_U32   (XAI_U32 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_U64   (XAI_U64 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_U128  (XAI_U128 | XAI_TYPE_TILE4D_BITS)

#define XAI_TILE4D_S4    (XAI_S8 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_S8    (XAI_S8 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_S16   (XAI_S16 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_S32   (XAI_S32 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_S64   (XAI_S64 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_S128  (XAI_S128 | XAI_TYPE_TILE4D_BITS)

#define XAI_TILE4D_F8    (XAI_F8 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_F16   (XAI_F16 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_F32   (XAI_F32 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_F64   (XAI_F64 | XAI_TYPE_TILE4D_BITS)
#define XAI_TILE4D_F128  (XAI_F128 | XAI_TYPE_TILE4D_BITS)

/*****************************************
*                   4D Frame Access Macros
*****************************************/
#define XAI_FRAME4D_GET_BUFF_PTR      XAI_FRAME_GET_BUFF_PTR
#define XAI_FRAME4D_SET_BUFF_PTR      XAI_FRAME_SET_BUFF_PTR

#define XAI_FRAME4D_GET_BUFF_SIZE     XAI_FRAME_GET_BUFF_SIZE
#define XAI_FRAME4D_SET_BUFF_SIZE     XAI_FRAME_SET_BUFF_SIZE

#define XAI_FRAME4D_GET_DATA_PTR      XAI_FRAME_GET_DATA_PTR
#define XAI_FRAME4D_SET_DATA_PTR      XAI_FRAME_SET_DATA_PTR

#define XAI_FRAME4D_GET_PIXEL_RES     XAI_FRAME_GET_PIXEL_RES
#define XAI_FRAME4D_SET_PIXEL_RES     XAI_FRAME_SET_PIXEL_RES

#define XAI_FRAME4D_GET_PIXEL_FORMAT  XAI_FRAME_GET_PIXEL_FORMAT
#define XAI_FRAME4D_SET_PIXEL_FORMAT  XAI_FRAME_SET_PIXEL_FORMAT

#define XAI_FRAME4D_GET_PADDING_TYPE  XAI_FRAME_GET_PADDING_TYPE
#define XAI_FRAME4D_SET_PADDING_TYPE  XAI_FRAME_SET_PADDING_TYPE

/*****************************************
*                   4D Tile Access Macros
*****************************************/
#define XAI_TILE4D_GET_BUFF_PTR        XAI_TILE2D_GET_BUFF_PTR
#define XAI_TILE4D_SET_BUFF_PTR        XAI_TILE2D_SET_BUFF_PTR
#define XAI_TILE4D_SET_BUFF_PTR_COEFF  XAI_TILE2D_SET_BUFF_PTR_COEFF

#define XAI_TILE4D_GET_BUFF_SIZE       XAI_TILE2D_GET_BUFF_SIZE
#define XAI_TILE4D_SET_BUFF_SIZE       XAI_TILE2D_SET_BUFF_SIZE

#define XAI_TILE4D_GET_DATA_PTR        XAI_TILE2D_GET_DATA_PTR
#define XAI_TILE4D_SET_DATA_PTR        XAI_TILE2D_SET_DATA_PTR
#define XAI_TILE4D_SET_DATA_PTR_COEFF  XAI_TILE2D_SET_DATA_PTR_COEFF

#define XAI_TILE4D_GET_STATUS_FLAGS    XAI_TILE2D_GET_STATUS_FLAGS
#define XAI_TILE4D_SET_STATUS_FLAGS    XAI_TILE2D_SET_STATUS_FLAGS

#define XAI_TILE4D_GET_TYPE            XAI_TILE2D_GET_TYPE
#define XAI_TILE4D_SET_TYPE            XAI_TILE2D_SET_TYPE

#define XAI_TILE4D_GET_ELEMENT_TYPE    XAI_TILE2D_GET_ELEMENT_TYPE
#define XAI_TILE4D_GET_ELEMENT_SIZE    XAI_TILE2D_GET_ELEMENT_SIZE
#define XAI_TILE4D_IS_TILE             XAI_TILE2D_IS_TILE2D

#define XAI_TILE4D_GET_FRAME_PTR(pTile4D)            ((pTile4D)->pFrame)
#define XAI_TILE4D_SET_FRAME_PTR(pTile4D, ptrFrame)  (pTile4D)->pFrame = ((xai_pFrame4D) (ptrFrame))

#define XAI_TILE4D_CHECK_STATUS_FLAGS_DMA_ONGOING          XAI_TILE2D_CHECK_STATUS_FLAGS_DMA_ONGOING
#define XAI_TILE4D_CHECK_STATUS_FLAGS_EDGE_PADDING_NEEDED  XAI_TILE2D_CHECK_STATUS_FLAGS_EDGE_PADDING_NEEDED

/***********************************
*              Other Marcos
***********************************/
#define XAI_TILE4D_CHECK_VIRTUAL_FRAME   XAI_TILE2D_CHECK_VIRTUAL_FRAME
#define XAI_FRAME4D_CHECK_VIRTUAL_FRAME  XAI_FRAME_CHECK_VIRTUAL_FRAME

// Only Q8, 240 and 341 uses alignment = 127. for P6,P1 and Q7 like dsps alignment = 127 is not supported
#define XAI_SETUP_TILE4D(type, pTile, pBuf, pFrame, bufSize, dim1Size, dim2Size, dim3Size, dim4Size, dim1Pitch, dim2Pitch, \
                         dim3Pitch, dim1Edge1, dim1Edge2, dim2Edge1, dim2Edge2, dim3Edge1, dim3Edge2,                      \
                         dim1Loc, dim2Loc, dim3Loc, dim4Loc, dataOrder, alignType)                                         \
  {                                                                                                                        \
    XAI_TILE4D_SET_TYPE(pTile, type);                                                                                      \
    XAI_TILE4D_SET_FRAME_PTR(pTile, pFrame);                                                                               \
    XAI_TILE4D_SET_BUFF_PTR(pTile, pBuf);                                                                                  \
    XAI_TILE4D_SET_BUFF_SIZE(pTile, bufSize);                                                                              \
    XAI_TILE4D_SET_DIM1(pTile, dim1Size);                                                                                  \
    XAI_TILE4D_SET_DIM2(pTile, dim2Size);                                                                                  \
    XAI_TILE4D_SET_DIM3(pTile, dim3Size);                                                                                  \
    XAI_TILE4D_SET_DIM4(pTile, dim4Size);                                                                                  \
    XAI_TILE4D_SET_DIM1_PITCH(pTile, dim1Pitch);                                                                           \
    XAI_TILE4D_SET_DIM2_PITCH(pTile, dim2Pitch);                                                                           \
    XAI_TILE4D_SET_DIM3_PITCH(pTile, dim3Pitch);                                                                           \
    uint8_t *edgePtr  = (uint8_t *) pBuf, *dataPtr;                                                                        \
    int32_t alignment = 127;                                                                                               \
    if ((alignType == XAI_EDGE_ALIGNED_64) || (alignType == XAI_DATA_ALIGNED_64)) { alignment = 63; }                      \
    if ((alignType == XAI_EDGE_ALIGNED_32) || (alignType == XAI_DATA_ALIGNED_32)) { alignment = 31; }                      \
    if ((alignType == XAI_EDGE_ALIGNED_32) || (alignType == XAI_EDGE_ALIGNED_64) || (alignType == EDGE_ALIGNED_128))       \
    {                                                                                                                      \
      edgePtr = (uint8_t *) (((uintptr_t) (pBuf) + alignment) & (~alignment));                                             \
    }                                                                                                                      \
    XAI_TILE4D_SET_DATA_PTR(pTile, edgePtr + ((dim3Edge1) * (dim2Pitch) + (dim2Edge1) * (dim1Pitch) + (dim1Edge1))         \
                            * XAI_TILE4D_GET_ELEMENT_SIZE(pTile));                                                         \
    if ((alignType == XAI_DATA_ALIGNED_32) || (alignType == XAI_DATA_ALIGNED_64) || (alignType == DATA_ALIGNED_128))       \
    {                                                                                                                      \
      dataPtr = (uint8_t *) XAI_TILE4D_GET_DATA_PTR(pTile);                                                                \
      dataPtr = (uint8_t *) (((uintptr_t) (dataPtr) + alignment) & (~alignment));                                          \
      XAI_TILE4D_SET_DATA_PTR(pTile, dataPtr);                                                                             \
    }                                                                                                                      \
    XAI_TILE4D_SET_DIM1_EDGE1(pTile, dim1Edge1);                                                                           \
    XAI_TILE4D_SET_DIM1_EDGE2(pTile, dim1Edge2);                                                                           \
    XAI_TILE4D_SET_DIM2_EDGE1(pTile, dim2Edge1);                                                                           \
    XAI_TILE4D_SET_DIM2_EDGE2(pTile, dim2Edge2);                                                                           \
    XAI_TILE4D_SET_DIM3_EDGE1(pTile, dim3Edge1);                                                                           \
    XAI_TILE4D_SET_DIM3_EDGE2(pTile, dim3Edge2);                                                                           \
    XAI_TILE4D_SET_DIM1_COORD(pTile, dim1Loc);                                                                             \
    XAI_TILE4D_SET_DIM2_COORD(pTile, dim2Loc);                                                                             \
    XAI_TILE4D_SET_DIM3_COORD(pTile, dim3Loc);                                                                             \
    XAI_TILE4D_SET_DIM4_COORD(pTile, dim4Loc);                                                                             \
    XAI_TILE4D_SET_DATA_ORDER(pTile, dataOrder);                                                                           \
  }

#define XAI_SETUP_FRAME4D(pFrame, pFrameBuffer, bufSize, dim1Size, dim2Size, dim3Size, dim4Size, dim1Pitch, dim2Pitch, dim3Pitch,          \
                          dim1Edge1, dim1Edge2, dim2Edge1, dim2Edge2, dim3Edge1, dim3Edge2, pixRes, pixPackFormat, paddingType, dataOrder) \
  {                                                                                                                                        \
    XAI_FRAME4D_SET_BUFF_PTR(pFrame, pFrameBuffer);                                                                                        \
    XAI_FRAME4D_SET_BUFF_SIZE(pFrame, bufSize);                                                                                            \
    XAI_FRAME4D_SET_DIM1(pFrame, dim1Size);                                                                                                \
    XAI_FRAME4D_SET_DIM2(pFrame, dim2Size);                                                                                                \
    XAI_FRAME4D_SET_DIM3(pFrame, dim3Size);                                                                                                \
    XAI_FRAME4D_SET_DIM4(pFrame, dim4Size);                                                                                                \
    XAI_FRAME4D_SET_DIM1_PITCH(pFrame, dim1Pitch);                                                                                         \
    XAI_FRAME4D_SET_DIM2_PITCH(pFrame, dim2Pitch);                                                                                         \
    XAI_FRAME4D_SET_DIM3_PITCH(pFrame, dim3Pitch);                                                                                         \
    XAI_FRAME4D_SET_DATA_PTR(pFrame, pFrameBuffer + ((dim3Edge1) * (dim2Pitch) + (dim2Edge1) * (dim1Pitch) +                               \
                                                     (dim1Edge1)) * pixRes);                                                               \
    XAI_FRAME4D_SET_DIM1_EDGE1(pFrame, dim1Edge1);                                                                                         \
    XAI_FRAME4D_SET_DIM1_EDGE2(pFrame, dim1Edge2);                                                                                         \
    XAI_FRAME4D_SET_DIM2_EDGE1(pFrame, dim2Edge1);                                                                                         \
    XAI_FRAME4D_SET_DIM2_EDGE2(pFrame, dim2Edge2);                                                                                         \
    XAI_FRAME4D_SET_DIM3_EDGE1(pFrame, dim3Edge1);                                                                                         \
    XAI_FRAME4D_SET_DIM3_EDGE2(pFrame, dim3Edge2);                                                                                         \
    XAI_FRAME4D_SET_PIXEL_RES(pFrame, pixRes);                                                                                             \
    XAI_FRAME4D_SET_PIXEL_FORMAT(pFrame, pixPackFormat);                                                                                   \
    XAI_FRAME4D_SET_PADDING_TYPE(pFrame, paddingType);                                                                                     \
    XAI_FRAME4D_SET_DATA_ORDER(pTile, dataOrder);                                                                                          \
  }

#define XAI_COPY_FRAME4D_TO_TILE4D(frame, tile)         {               \
    XAI_TILE4D_SET_DIM1(tile, XAI_FRAME4D_GET_DIM1(frame));             \
    XAI_TILE4D_SET_DIM1_PITCH(tile, XAI_FRAME4D_GET_DIM1_PITCH(frame)); \
    XAI_TILE4D_SET_DIM1_EDGE1(tile, XAI_FRAME4D_GET_DIM1_EDGE1(frame)); \
    XAI_TILE4D_SET_DIM1_EDGE2(tile, XAI_FRAME4D_GET_DIM1_EDGE2(frame)); \
    XAI_TILE4D_SET_DIM2(tile, XAI_FRAME4D_GET_DIM2(frame));             \
    XAI_TILE4D_SET_DIM2_PITCH(tile, XAI_FRAME4D_GET_DIM2_PITCH(frame)); \
    XAI_TILE4D_SET_DIM2_EDGE1(tile, XAI_FRAME4D_GET_DIM2_EDGE1(frame)); \
    XAI_TILE4D_SET_DIM2_EDGE2(tile, XAI_FRAME4D_GET_DIM2_EDGE2(frame)); \
    XAI_TILE4D_SET_DIM3(tile, XAI_FRAME4D_GET_DIM3(frame));             \
    XAI_TILE4D_SET_DIM3_PITCH(tile, XAI_FRAME4D_GET_DIM3_PITCH(frame)); \
    XAI_TILE4D_SET_DIM3_EDGE1(tile, XAI_FRAME4D_GET_DIM3_EDGE1(frame)); \
    XAI_TILE4D_SET_DIM3_EDGE2(tile, XAI_FRAME4D_GET_DIM3_EDGE2(frame)); \
    XAI_TILE4D_SET_DIM4(tile, XAI_FRAME4D_GET_DIM4(frame));             \
    XAI_TILE4D_SET_DATA_PTR(tile, XAI_FRAME4D_GET_DATA_PTR(frame));     \
    XAI_TILE4D_SET_DATA_ORDER(tile, XAI_FRAME4D_GET_DATA_ORDER(frame)); \
}

#define XAI_COPY_FRAME4D_TO_FRAME4D(frameIn, frameOut)  {                      \
    XAI_FRAME4D_SET_DIM1(frameOut, XAI_FRAME4D_GET_DIM1(frameIn));             \
    XAI_FRAME4D_SET_DIM1_PITCH(frameOut, XAI_FRAME4D_GET_DIM1_PITCH(frameIn)); \
    XAI_FRAME4D_SET_DIM1_EDGE1(frameOut, XAI_FRAME4D_GET_DIM1_EDGE1(frameIn)); \
    XAI_FRAME4D_SET_DIM1_EDGE2(frameOut, XAI_FRAME4D_GET_DIM1_EDGE2(frameIn)); \
    XAI_FRAME4D_SET_DIM2(frameOut, XAI_FRAME4D_GET_DIM2(frameIn));             \
    XAI_FRAME4D_SET_DIM2_PITCH(frameOut, XAI_FRAME4D_GET_DIM2_PITCH(frameIn)); \
    XAI_FRAME4D_SET_DIM2_EDGE1(frameOut, XAI_FRAME4D_GET_DIM2_EDGE1(frameIn)); \
    XAI_FRAME4D_SET_DIM2_EDGE2(frameOut, XAI_FRAME4D_GET_DIM2_EDGE2(frameIn)); \
    XAI_FRAME4D_SET_DIM3(frameOut, XAI_FRAME4D_GET_DIM3(frameIn));             \
    XAI_FRAME4D_SET_DIM3_PITCH(frameOut, XAI_FRAME4D_GET_DIM3_PITCH(frameIn)); \
    XAI_FRAME4D_SET_DIM3_EDGE1(frameOut, XAI_FRAME4D_GET_DIM3_EDGE1(frameIn)); \
    XAI_FRAME4D_SET_DIM3_EDGE2(frameOut, XAI_FRAME4D_GET_DIM3_EDGE2(frameIn)); \
    XAI_FRAME4D_SET_DIM4(frameOut, XAI_FRAME4D_GET_DIM4(frameIn));             \
    XAI_FRAME4D_SET_DATA_PTR(frameOut, XAI_FRAME4D_GET_DATA_PTR(frameIn));     \
    XAI_FRAME4D_SET_DATA_ORDER(frameOut, XAI_FRAME4D_GET_DATA_ORDER(frameIn)); \
    XAI_FRAME4D_SET_PIXEL_RES(frameOut, XAI_FRAME4D_GET_PIXEL_RES(frameIn));   \
}

#define XAI_COPY_TILE4D_TO_TILE4D(tileIn, tileOut)      {                  \
    XAI_TILE4D_SET_DIM1(tileOut, XAI_TILE4D_GET_DIM1(tileIn));             \
    XAI_TILE4D_SET_DIM1_PITCH(tileOut, XAI_TILE4D_GET_DIM1_PITCH(tileIn)); \
    XAI_TILE4D_SET_DIM1_EDGE1(tileOut, XAI_TILE4D_GET_DIM1_EDGE1(tileIn)); \
    XAI_TILE4D_SET_DIM1_EDGE2(tileOut, XAI_TILE4D_GET_DIM1_EDGE2(tileIn)); \
    XAI_TILE4D_SET_DIM2(tileOut, XAI_TILE4D_GET_DIM2(tileIn));             \
    XAI_TILE4D_SET_DIM2_PITCH(tileOut, XAI_TILE4D_GET_DIM2_PITCH(tileIn)); \
    XAI_TILE4D_SET_DIM2_EDGE1(tileOut, XAI_TILE4D_GET_DIM2_EDGE1(tileIn)); \
    XAI_TILE4D_SET_DIM2_EDGE2(tileOut, XAI_TILE4D_GET_DIM2_EDGE2(tileIn)); \
    XAI_TILE4D_SET_DIM3(tileOut, XAI_TILE4D_GET_DIM3(tileIn));             \
    XAI_TILE4D_SET_DIM3_PITCH(tileOut, XAI_TILE4D_GET_DIM3_PITCH(tileIn)); \
    XAI_TILE4D_SET_DIM3_EDGE1(tileOut, XAI_TILE4D_GET_DIM3_EDGE1(tileIn)); \
    XAI_TILE4D_SET_DIM3_EDGE2(tileOut, XAI_TILE4D_GET_DIM3_EDGE2(tileIn)); \
    XAI_TILE4D_SET_DIM4(tileOut, XAI_TILE4D_GET_DIM4(tileIn));             \
    XAI_TILE4D_SET_DATA_PTR(tileOut, XAI_TILE4D_GET_DATA_PTR(tileIn));     \
    XAI_TILE4D_SET_DATA_ORDER(tileOut, XAI_TILE4D_GET_DATA_ORDER(tileIn)); \
}

// Assumes 8 bit pixRes and Edge1 = Edge2
#define XAI_TILE4D_UPDATE_EDGE_DIM1(pTile, newEdgeSize)                  \
  {                                                                      \
    uint16_t currEdgeSize = (uint16_t) XAI_TILE4D_GET_DIM1_EDGE1(pTile); \
    uint32_t dim1Pitch    = (uint32_t) XAI_TILE4D_GET_DIM1_PITCH(pTile); \
    uintptr_t dataU32     = (uintptr_t) XAI_TILE4D_GET_DATA_PTR(pTile);  \
    dataU32 = dataU32 + newEdgeSize - currEdgeSize;                      \
    XAI_TILE4D_SET_DATA_PTR(pTile, (void *) dataU32);                    \
    XAI_TILE4D_SET_DIM1_EDGE1(pTile, newEdgeSize);                       \
    XAI_TILE4D_SET_DIM1_EDGE2(pTile, newEdgeSize);                       \
    XAI_TILE4D_SET_DIM1(pTile, dim1Pitch - 2 * newEdgeSize);             \
  }

// Assumes 8 bit pixRes and Edge1 = Edge2
#define XAI_TILE4D_UPDATE_EDGE_DIM2(pTile, newEdgeSize)                      \
  {                                                                          \
    uint16_t currEdgeSize = (uint16_t) XAI_TILE4D_GET_DIM2_EDGE1(pTile);     \
    uint32_t dim1Pitch    = (uint32_t) XAI_TILE4D_GET_DIM1_PITCH(pTile);     \
    uint16_t dim2Size     = (uint16_t) XAI_TILE4D_GET_DIM2(pTile);           \
    uintptr_t dataU32     = (uintptr_t) XAI_TILE4D_GET_DATA_PTR(pTile);      \
    dataU32 = dataU32 + dim1Pitch * (newEdgeSize - currEdgeSize);            \
    XAI_TILE4D_SET_DATA_PTR(pTile, (void *) dataU32);                        \
    XAI_TILE4D_SET_DIM2_EDGE1(pTile, newEdgeSize);                           \
    XAI_TILE4D_SET_DIM2_EDGE2(pTile, newEdgeSize);                           \
    XAI_TILE4D_SET_DIM2(pTile, dim2Size + 2 * (currEdgeSize - newEdgeSize)); \
  }

// Assumes 8 bit pixRes and Edge1 = Edge2
#define XAI_TILE4D_UPDATE_EDGE_DIM3(pTile, newEdgeSize)                      \
  {                                                                          \
    uint16_t currEdgeSize = (uint16_t) XAI_TILE4D_GET_DIM3_EDGE1(pTile);     \
    uint32_t dim2Pitch    = (uint32_t) XAI_TILE4D_GET_DIM2_PITCH(pTile);     \
    uint16_t dim3Size     = (uint16_t) XAI_TILE4D_GET_DIM3(pTile);           \
    uintptr_t dataU32     = (uintptr_t) XAI_TILE4D_GET_DATA_PTR(pTile);      \
    dataU32 = dataU32 + dim2Pitch * (newEdgeSize - currEdgeSize);            \
    XAI_TILE4D_SET_DATA_PTR(pTile, (void *) dataU32);                        \
    XAI_TILE4D_SET_DIM3_EDGE1(pTile, newEdgeSize);                           \
    XAI_TILE4D_SET_DIM3_EDGE2(pTile, newEdgeSize);                           \
    XAI_TILE4D_SET_DIM3(pTile, dim3Size + 2 * (currEdgeSize - newEdgeSize)); \
  }

#define XAI_TILE4D_UPDATE_DIMENSIONS(pTile, dim1Loc, dim2Loc, dim3Loc, dim4Loc, dim1Size, dim2Size, dim3Size, dim4Size, \
                                     dim1Pitch, dim2Pitch, dim3Pitch)                                                   \
  {                                                                                                                     \
    XAI_TILE4D_SET_DIM1_COORD(pTile, dim1Loc);                                                                          \
    XAI_TILE4D_SET_DIM2_COORD(pTile, dim2Loc);                                                                          \
    XAI_TILE4D_SET_DIM3_COORD(pTile, dim3Loc);                                                                          \
    XAI_TILE4D_SET_DIM4_COORD(pTile, dim4Loc);                                                                          \
    XAI_TILE4D_SET_DIM1(pTile, dim1Size);                                                                               \
    XAI_TILE4D_SET_DIM2(pTile, dim2Size);                                                                               \
    XAI_TILE4D_SET_DIM3(pTile, dim3Size);                                                                               \
    XAI_TILE4D_SET_DIM4(pTile, dim4Size);                                                                               \
    XAI_TILE4D_SET_DIM1_PITCH(pTile, dim1Pitch);                                                                        \
    XAI_TILE4D_SET_DIM2_PITCH(pTile, dim2Pitch);                                                                        \
    XAI_TILE4D_SET_DIM3_PITCH(pTile, dim3Pitch);                                                                        \
  }

// 5D tile
#define XAI_TILE5D_FIELDS \
  uint32_t bufferSize;    \
  int32_t dim1Size;       \
  int32_t dim1Pitch;      \
  uint16_t type;          \
  int32_t dim2Size;       \
  int32_t dim2Pitch;      \
  int32_t dim3Size;       \
  int32_t dim3Pitch;      \
  int32_t dim4Size;       \
  int32_t dim4Pitch;      \
  int32_t dim5Size;       \
  xai_cnn_data_order dataOrder;

// 5D tile
typedef struct xai_tile5DStruct
{
  void *pBuffer;
  void *pData;
  XAI_TILE5D_FIELDS
} xai_tile5D, *xai_pTile5D;

/*****************************************
*                   5D Tile Access Macros
*****************************************/
#define XAI_TILE5D_GET_BUFF_PTR    XAI_TILE2D_GET_BUFF_PTR
#define XAI_TILE5D_SET_BUFF_PTR    XAI_TILE2D_SET_BUFF_PTR

#define XAI_TILE5D_GET_BUFF_SIZE   XAI_TILE2D_GET_BUFF_SIZE
#define XAI_TILE5D_SET_BUFF_SIZE   XAI_TILE2D_SET_BUFF_SIZE

#define XAI_TILE5D_GET_DATA_PTR    XAI_TILE2D_GET_DATA_PTR
#define XAI_TILE5D_SET_DATA_PTR    XAI_TILE2D_SET_DATA_PTR

#define XAI_TILE5D_GET_TYPE        XAI_TILE2D_GET_TYPE
#define XAI_TILE5D_SET_TYPE        XAI_TILE2D_SET_TYPE

#define XAI_TILE5D_GET_DIM1        XAI_TILE4D_GET_DIM1
#define XAI_TILE5D_SET_DIM1        XAI_TILE4D_SET_DIM1
#define XAI_TILE5D_GET_DIM1_PITCH  XAI_TILE4D_GET_DIM1_PITCH
#define XAI_TILE5D_SET_DIM1_PITCH  XAI_TILE4D_SET_DIM1_PITCH
#define XAI_TILE5D_GET_DIM2        XAI_TILE4D_GET_DIM2
#define XAI_TILE5D_SET_DIM2        XAI_TILE4D_SET_DIM2
#define XAI_TILE5D_GET_DIM2_PITCH  XAI_TILE4D_GET_DIM2_PITCH
#define XAI_TILE5D_SET_DIM2_PITCH  XAI_TILE4D_SET_DIM2_PITCH
#define XAI_TILE5D_GET_DIM3        XAI_TILE4D_GET_DIM3
#define XAI_TILE5D_SET_DIM3        XAI_TILE4D_SET_DIM3
#define XAI_TILE5D_GET_DIM3_PITCH  XAI_TILE4D_GET_DIM3_PITCH
#define XAI_TILE5D_SET_DIM3_PITCH  XAI_TILE4D_SET_DIM3_PITCH
#define XAI_TILE5D_GET_DIM4        XAI_TILE4D_GET_DIM4
#define XAI_TILE5D_SET_DIM4        XAI_TILE4D_SET_DIM4
#define XAI_TILE5D_GET_DIM4_PITCH(x)     ((x)->dim4Pitch)
#define XAI_TILE5D_SET_DIM4_PITCH(x, v)  ((x)->dim4Pitch = (v))
#define XAI_TILE5D_GET_DIM5(x)           ((x)->dim5Size)
#define XAI_TILE5D_SET_DIM5(x, v)        ((x)->dim5Size = (v))
#define XAI_TILE5D_GET_DATA_ORDER(x)     ((x)->dataOrder)
#define XAI_TILE5D_SET_DATA_ORDER(x, v)  ((x)->dataOrder = (v))
#define XAI_TILE5D_GET_ELEMENT_TYPE  XAI_TILE2D_GET_ELEMENT_TYPE
#define XAI_TILE5D_GET_ELEMENT_SIZE  XAI_TILE2D_GET_ELEMENT_SIZE

#if USE_64BIT_COEFF
#define xai_pArray_coeff   xai_pArray_coeff_64
#define xai_pTile3D_coeff  xai_pTile3D_64
#define xai_pTile4D_coeff  xai_pTile4D_64
#else
#define xai_pArray_coeff   xai_pArray_coeff_32
#define xai_pTile3D_coeff  xai_pTile3D
#define xai_pTile4D_coeff  xai_pTile4D
#endif // #if USE_64BIT_COEFF
#endif // #ifndef __XAI_TILE_MANAGER_H__
