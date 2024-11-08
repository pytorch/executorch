/*******************************************************************************
* Copyright (c) 2024 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/

#ifndef  __XA_TYPE_DEF_H__
#define  __XA_TYPE_DEF_H__

/* typed defs for datatypes used in the library */
typedef signed char             WORD8   ;
typedef signed char         *   pWORD8  ;
typedef unsigned char           UWORD8  ;
typedef unsigned char       *   pUWORD8 ;

typedef signed short            WORD16  ;
typedef signed short        *   pWORD16 ;
typedef unsigned short          UWORD16 ;
typedef unsigned short      *   pUWORD16;

typedef signed int              WORD32  ;
typedef signed int          *   pWORD32 ;
typedef unsigned int            UWORD32 ;
typedef unsigned int        *   pUWORD32;

typedef signed long long        WORD40  ;
typedef signed long long    *   pWORD40 ;
typedef unsigned long long      UWORD40 ;
typedef unsigned long long  *   pUWORD40;

typedef signed long long        WORD64  ;
typedef signed long long    *   pWORD64 ;
typedef unsigned long long      UWORD64 ;
typedef unsigned long long  *   pUWORD64;

typedef float                   FLOAT32 ;
typedef float               *   pFLOAT32;
typedef double                  FLOAT64 ;
typedef double              *   pFlOAT64;

typedef void                    VOID    ;
typedef void                *   pVOID   ;

/* variable size types: platform optimized implementation */
typedef signed int              BOOL    ;
typedef unsigned int            UBOOL   ;
typedef signed int              FLAG    ;
typedef unsigned int            UFLAG   ;
typedef signed int              LOOPIDX ;
typedef unsigned int            ULOOPIDX;
typedef signed int              WORD    ;
typedef unsigned int            UWORD   ;

typedef LOOPIDX                 LOOPINDEX;
typedef ULOOPIDX                ULOOPINDEX;

#define PLATFORM_INLINE __inline

typedef struct xa_codec_opaque { WORD32 _; } *xa_codec_handle_t;

typedef int XA_ERRORCODE;

typedef XA_ERRORCODE xa_codec_func_t(xa_codec_handle_t p_xa_module_obj,
                     WORD32            i_cmd,
                     WORD32            i_idx,
                     pVOID             pv_value);

#endif /* __XA_TYPE_DEF_H__ */
