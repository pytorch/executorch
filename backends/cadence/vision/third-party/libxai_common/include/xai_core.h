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

#ifndef __XAI_CORE_H__
#define __XAI_CORE_H__

#include "xai_core_api.h"

#if defined(_MSC_VER)
#define isfinite  _finite
#define __func__  __FUNCTION__
#endif

/* Linear congruential generator */
#define RND_A      1103515245
#define RND_LOG_M  31
#define RND_C      12345
#define GET_NEXT_RND(x_pr)  (((RND_A) *(x_pr) + (RND_C)) & ((unsigned int) (1 << (RND_LOG_M)) - 1))

/* return 0 on success or required memory size on failure */
_XAI_EXTERN_C_ size_t xaiFitArray_U8(const xai_pArray donor, xai_pArray rec, int width, int height, xai_bool aligned);
_XAI_EXTERN_C_ size_t xaiFitArray_U8S16(const xai_pArray donor, xai_pArray rec, int width, int height, xai_bool aligned);
_XAI_EXTERN_C_ size_t xaiFitArray_S16(const xai_pArray donor, xai_pArray rec, int width, int height, xai_bool aligned);
_XAI_EXTERN_C_ size_t xaiFitTile_U8(const xai_pTile2D donor, xai_pTile2D rec, int width, int height, xai_bool aligned);
_XAI_EXTERN_C_ size_t xaiFitTile_S16(const xai_pTile2D donor, xai_pTile2D rec, int width, int height, xai_bool aligned);

#define XAI_FIT_ALIGNED  1
#define XAI_FIT_ANY      0


// error check macro
#if XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_PRINT_ON_ERROR || XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_PRINT_AND_CONTINUE_ON_ERROR
#  include <stdio.h>
#endif

#if XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_TERMINATE_ON_ERROR
#  include <stdlib.h>
#endif

#define MARK_VAR_AS_USED(var)  (void) (var)

#if (XAI_ERROR_LEVEL != XAI_ERROR_LEVEL_NO_ERROR)
#  define XAI_ERROR_CHECKS()           XAI_ERR_TYPE __xai_local_err_code = XAI_ERR_OK;
#  define XAI_ERROR_CHECKS_CONTINUE()
#  define XAI_ERROR_STATUS()           __xai_local_err_code
#else
#  define XAI_ERROR_CHECKS()           while (0)
#  define XAI_ERROR_CHECKS_CONTINUE()  while (0)
#  define XAI_ERROR_STATUS()           XAI_ERR_OK
#endif

#if XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_TERMINATE_ON_ERROR
#  define XAI_CHECK_ERROR(condition, code, ...) \
  if (condition) {} else exit(-1)
#elif XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_RETURN_ON_ERROR
#  define XAI_CHECK_ERROR(condition, code, ...) \
  if (condition) {} else return (code)
#elif XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_CONTINUE_ON_ERROR
#  define XAI_CHECK_ERROR(condition, code, ...) \
  if (condition) {} else __xai_local_err_code = (code)
#elif XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_PRINT_ON_ERROR
#  define XAI_CHECK_ERROR(condition, code, ...)                                                                                           \
  do { if (!(condition)) { printf("%s:%d: Error #%d (%s) in function %s: ", __FILE__, __LINE__, (int) (code), xaiErrStr(code), __func__); \
                           printf(__VA_ARGS__);                                                                                           \
                           printf("\n");                                                                                                  \
                           fflush(stdout); return code; } } while (0)
#elif XAI_ERROR_LEVEL == XAI_ERROR_LEVEL_PRINT_AND_CONTINUE_ON_ERROR
#  define XAI_CHECK_ERROR(condition, code, ...)                                                                                           \
  do { if (!(condition)) { printf("%s:%d: Error #%d (%s) in function %s: ", __FILE__, __LINE__, (int) (code), xaiErrStr(code), __func__); \
                           printf(__VA_ARGS__);                                                                                           \
                           printf("\n");                                                                                                  \
                           fflush(stdout); return code; } } while (0)
#else
#  define XAI_CHECK_ERROR(condition, code, ...)
#endif

// helper macro
#define XAI_ARRAY_USEFUL_CAPACITY(array, ptr)  ((ptrdiff_t) XAI_ARRAY_GET_BUFF_SIZE(array) - ((uint8_t *) (ptr) - (uint8_t *) XAI_ARRAY_GET_BUFF_PTR(array)))

// macro for standard array/tile checks:

// check that array/tile data is placed in the DRAM
#if XAI_EMULATE_LOCAL_RAM && __XTENSA__
#if XCHAL_NUM_DATARAM == 2
#define XAI_ARRAY_STARTS_IN_DRAM(t)                                                   \
  (XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) >= ((uint32_t) XCHAL_DATARAM0_VADDR) || \
   (XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) >= ((uint32_t) XCHAL_DATARAM1_VADDR)))
#define XAI_ARRAY_ENDS_IN_DRAM(t)                                                                                                                       \
  (XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) + XAI_ARRAY_GET_BUFF_SIZE(t) <= (((uint32_t) XCHAL_DATARAM0_VADDR) + ((uint32_t) XCHAL_DATARAM0_SIZE)) || \
   (XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) + XAI_ARRAY_GET_BUFF_SIZE(t) <= (((uint32_t) XCHAL_DATARAM1_VADDR) + ((uint32_t) XCHAL_DATARAM1_SIZE))))
#define XAI_TILE2D_STARTS_IN_DRAM(t)                                                   \
  (XAI_PTR_TO_ADDR(XAI_TILE2D_GET_BUFF_PTR(t)) >= ((uint32_t) XCHAL_DATARAM0_VADDR) || \
   (XAI_PTR_TO_ADDR(XAI_TILE2D_GET_BUFF_PTR(t)) >= ((uint32_t) XCHAL_DATARAM1_VADDR)))
#define XAI_TILE2D_ENDS_IN_DRAM(t)                                                                                                                      \
  (XAI_PTR_TO_ADDR(XAI_TILE2D_GET_BUFF_PTR(t)) + XAI_TILE2D_GET_BUFF_SIZE(t) <= (((uint32_t) XCHAL_DATARAM0_VADDR) + ((uint32_t) XCHAL_DATARAM0_SIZE)) || \
   (XAI_PTR_TO_ADDR(XAI_TILE2D_GET_BUFF_PTR(t)) + XAI_TILE2D_GET_BUFF_SIZE(t) <= (((uint32_t) XCHAL_DATARAM1_VADDR) + ((uint32_t) XCHAL_DATARAM1_SIZE))))
#elif XCHAL_NUM_DATARAM == 1
#define XAI_ARRAY_STARTS_IN_DRAM(t) \
  (XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) >= ((uint32_t) XCHAL_DATARAM0_VADDR))
#define XAI_ARRAY_ENDS_IN_DRAM(t) \
  (XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) + XAI_ARRAY_GET_BUFF_SIZE(t) <= (((uint32_t) XCHAL_DATARAM0_VADDR) + ((uint32_t) XCHAL_DATARAM0_SIZE)))
#define XAI_TILE2D_STARTS_IN_DRAM(t) \
  (XAI_PTR_TO_ADDR(XAI_TILE2D_GET_BUFF_PTR(t)) >= ((uint32_t) XCHAL_DATARAM0_VADDR))
#define XAI_TILE2D_ENDS_IN_DRAM(t) \
  (XAI_PTR_TO_ADDR(XAI_TILE2D_GET_BUFF_PTR(t)) + XAI_TILE2D_GET_BUFF_SIZE(t) <= (((uint32_t) XCHAL_DATARAM0_VADDR) + ((uint32_t) XCHAL_DATARAM0_SIZE)))
#endif

#else //#XAI_EMULATE_LOCAL_RAM && __XTENSA__
#define XAI_ARRAY_STARTS_IN_DRAM(t)  1
#define XAI_ARRAY_ENDS_IN_DRAM(t)    1
#define XAI_TILE2D_STARTS_IN_DRAM(t)   1
#define XAI_TILE2D_ENDS_IN_DRAM(t)     1
#endif //#XAI_EMULATE_LOCAL_RAM && __XTENSA__

// check the minimal alignment requirements
#define XAI_ARRAY_IS_WIDTH_ALIGNED(t)       ((XAI_ARRAY_GET_WIDTH(t) & (XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_ARRAY_IS_WIDTH_ALIGNED2(t)      ((XAI_ARRAY_GET_WIDTH(t) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_ARRAY_IS_WIDTH_ALIGNED_2(t)     ((XAI_ARRAY_GET_WIDTH(t) & (XCHAL_IVPN_SIMD_WIDTH / 2 - 1)) == 0)
#define XAI_ARRAY_IS_STRIDE_ALIGNED(t)      ((XAI_ARRAY_GET_PITCH(t) & (XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_ARRAY_IS_STRIDE_ALIGNED2(t)     ((XAI_ARRAY_GET_PITCH(t) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_ARRAY_IS_STRIDE_ALIGNED_2(t)    ((XAI_ARRAY_GET_PITCH(t) & (XCHAL_IVPN_SIMD_WIDTH / 2 - 1)) == 0)
#define XAI_ARRAY_IS_PTR_ALIGNED_NX8(t)     ((XAI_PTR_TO_ADDR(XAI_ARRAY_GET_DATA_PTR(t)) & (XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_ARRAY_IS_PTR_ALIGNED_2NX8(t)    ((XAI_PTR_TO_ADDR(XAI_ARRAY_GET_DATA_PTR(t)) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_ARRAY_IS_PTR_ALIGNED_NX16(t)    ((XAI_PTR_TO_ADDR(XAI_ARRAY_GET_DATA_PTR(t)) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_ARRAY_IS_PTR_ALIGNED_N_2X32(t)  ((XAI_PTR_TO_ADDR(XAI_ARRAY_GET_DATA_PTR(t)) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)

#define XAI_ARRAY_IS_ALIGNED_NX8(t)         (XAI_ARRAY_IS_PTR_ALIGNED_NX8(t) && XAI_ARRAY_IS_WIDTH_ALIGNED(t) && XAI_ARRAY_IS_STRIDE_ALIGNED(t))
#define XAI_ARRAY_IS_ALIGNED_2NX8(t)        (XAI_ARRAY_IS_PTR_ALIGNED_2NX8(t) && XAI_ARRAY_IS_WIDTH_ALIGNED2(t) && XAI_ARRAY_IS_STRIDE_ALIGNED2(t))
#define XAI_ARRAY_IS_ALIGNED_NX16(t)        (XAI_ARRAY_IS_PTR_ALIGNED_NX16(t) && XAI_ARRAY_IS_WIDTH_ALIGNED(t) && XAI_ARRAY_IS_STRIDE_ALIGNED(t))
#define XAI_ARRAY_IS_ALIGNED_N_2X32(t)      (XAI_ARRAY_IS_PTR_ALIGNED_N_2X32(t) && XAI_ARRAY_IS_WIDTH_ALIGNED_2(t) && XAI_ARRAY_IS_STRIDE_ALIGNED_2(t))

#define XAI_TILE2D_IS_WIDTH_ALIGNED(t)        XAI_ARRAY_IS_WIDTH_ALIGNED(t)
#define XAI_TILE2D_IS_WIDTH_ALIGNED2(t)       XAI_ARRAY_IS_WIDTH_ALIGNED2(t)
#define XAI_TILE2D_IS_WIDTH_ALIGNED_2(t)      XAI_ARRAY_IS_WIDTH_ALIGNED_2(t)
#define XAI_TILE2D_IS_STRIDE_ALIGNED(t)       XAI_ARRAY_IS_STRIDE_ALIGNED(t)
#define XAI_TILE2D_IS_STRIDE_ALIGNED2(t)      XAI_ARRAY_IS_STRIDE_ALIGNED2(t)
#define XAI_TILE2D_IS_STRIDE_ALIGNED_2(t)     XAI_ARRAY_IS_STRIDE_ALIGNED_2(t)
#define XAI_TILE2D_IS_PTR_ALIGNED_NX8(t)      XAI_ARRAY_IS_PTR_ALIGNED_NX8(t)
#define XAI_TILE2D_IS_PTR_ALIGNED_2NX8(t)     XAI_ARRAY_IS_PTR_ALIGNED_2NX8(t)
#define XAI_TILE2D_IS_PTR_ALIGNED_NX16(t)     XAI_ARRAY_IS_PTR_ALIGNED_NX16(t)
#define XAI_TILE2D_IS_PTR_ALIGNED_N_2X32(t)   XAI_ARRAY_IS_PTR_ALIGNED_N_2X32(t)

// check array invariants
#define XAI_ARRAY_IS_1D(t)                     (XAI_ARRAY_GET_HEIGHT(t) == 1)

#define XAI_ARRAY_CHECK_TYPE(a, type)          (XAI_TYPE_ELEMENT_TYPE(XAI_ARRAY_GET_TYPE(a)) == type)

#define XAI_ARRAY_CHECK_ELEMENT_SIZE(a, size)  (XAI_ARRAY_GET_ELEMENT_SIZE(a) == (size))

#define XAI_ARRAY_SIZE_EQ(t1, t2)              (XAI_ARRAY_GET_WIDTH(t1) == XAI_ARRAY_GET_WIDTH(t2) && XAI_ARRAY_GET_HEIGHT(t1) == XAI_ARRAY_GET_HEIGHT(t2))

#define XAI_ARRAY_SIZE_GEQ(t1, t2)             (XAI_ARRAY_GET_WIDTH(t1) >= XAI_ARRAY_GET_WIDTH(t2) && XAI_ARRAY_GET_HEIGHT(t1) >= XAI_ARRAY_GET_HEIGHT(t2))

#define XAI_ARRAYS_ARE_NOT_OVERLAP(t1, t2)     (XAI_ARRAY_GET_DATA_PTR(t1) != XAI_ARRAY_GET_DATA_PTR(t2))

#define XAI_ARRAY_IS_CONSISTENT(a)                                                                                                                            \
  ((XAI_ARRAY_GET_PITCH(a) >= XAI_ARRAY_GET_WIDTH(a)) &&                                                                                                      \
   (XAI_ARRAY_GET_WIDTH(a) > 0) && (XAI_ARRAY_GET_HEIGHT(a) > 0) &&                                                                                           \
   ((uint8_t *) XAI_ARRAY_GET_DATA_PTR(a) >= (uint8_t *) XAI_ARRAY_GET_BUFF_PTR(a)) &&                                                                        \
   ((uint8_t *) XAI_ARRAY_GET_DATA_PTR(a) + (XAI_ARRAY_GET_PITCH(a) * (XAI_ARRAY_GET_HEIGHT(a) - 1) + XAI_ARRAY_GET_WIDTH(a)) * XAI_ARRAY_GET_ELEMENT_SIZE(a) \
    <= (uint8_t *) XAI_ARRAY_GET_BUFF_PTR(a) + XAI_ARRAY_GET_BUFF_SIZE(a)))

// common array error checks
#define XAI_CHECK_POINTER(pointer) \
  XAI_CHECK_ERROR(pointer != 0, XAI_ERR_NULLARG, "The pointer (" #pointer ") is NULL")

#if ((defined(XCHAL_VISION_TYPE) && (XCHAL_VISION_TYPE >= 6)) || (defined(XCHAL_HAVE_BBENEP) && (XCHAL_HAVE_BBENEP == 1)))

#define XAI_CHECK_BUFFER(array)                                                                                                \
  XAI_CHECK_POINTER(array);                                                                                                    \
  XAI_CHECK_ERROR(XAI_ARRAY_STARTS_IN_DRAM(array), XAI_ERR_MEMLOCAL, "The argument (" #array ") data does not start in DRAM"); \
  XAI_CHECK_ERROR(XAI_ARRAY_ENDS_IN_DRAM(array), XAI_ERR_MEMLOCAL, "Complete data for the argument  (" #array ")  does not lie in DRAM")

#else

#define XAI_CHECK_BUFFER(array) \
  XAI_CHECK_POINTER(array);
#endif

#define XAI_CHECK_ARRAY(array) \
  XAI_CHECK_BUFFER(array);     \
  XAI_CHECK_ERROR(XAI_ARRAY_IS_CONSISTENT(array), XAI_ERR_BADARG, "The argument (" #array ") is invalid")

#define XAI_CHECK_ARRAY_I(array, element_size)                           \
  XAI_CHECK_ARRAY(array);                                                \
  XAI_CHECK_ERROR(XAI_ARRAY_CHECK_ELEMENT_SIZE(array, element_size) &&   \
                  !((XAI_ARRAY_GET_TYPE(array)) & (XAI_TYPE_FLOAT_BIT)), \
                  XAI_ERR_DATATYPE, "The argument (" #array ") has wrong type")

#define XAI_CHECK_ARRAY_X(array, element_size)                       \
  XAI_CHECK_ARRAY(array);                                            \
  XAI_CHECK_ERROR(XAI_ARRAY_CHECK_ELEMENT_SIZE(array, element_size), \
                  XAI_ERR_DATATYPE, "The argument (" #array ") has wrong type")

#define XAI_CHECK_ARRAY_T(array, type) \
  XAI_CHECK_ARRAY(array);              \
  XAI_CHECK_ERROR(XAI_ARRAY_CHECK_TYPE(array, type), XAI_ERR_DATATYPE, "The argument (" #array ") has wrong type")

#define XAI_CHECK_ARRAY_I8(array)   XAI_CHECK_ARRAY_I(array, sizeof(int8_t))
#define XAI_CHECK_ARRAY_I16(array)  XAI_CHECK_ARRAY_I(array, sizeof(int16_t))
#define XAI_CHECK_ARRAY_I32(array)  XAI_CHECK_ARRAY_I(array, sizeof(int32_t))

#define XAI_CHECK_ARRAY_X16(array)  XAI_CHECK_ARRAY_X(array, sizeof(int16_t))
#define XAI_CHECK_ARRAY_X32(array)  XAI_CHECK_ARRAY_X(array, sizeof(int32_t))

#define XAI_CHECK_ARRAY_U8(array)   XAI_CHECK_ARRAY_T(array, XAI_U8)
#define XAI_CHECK_ARRAY_S8(array)   XAI_CHECK_ARRAY_T(array, XAI_S8)
#define XAI_CHECK_ARRAY_U16(array)  XAI_CHECK_ARRAY_T(array, XAI_U16)
#define XAI_CHECK_ARRAY_S16(array)  XAI_CHECK_ARRAY_T(array, XAI_S16)
#define XAI_CHECK_ARRAY_U32(array)  XAI_CHECK_ARRAY_T(array, XAI_U32)
#define XAI_CHECK_ARRAY_S32(array)  XAI_CHECK_ARRAY_T(array, XAI_S32)
#define XAI_CHECK_ARRAY_S64(array)  XAI_CHECK_ARRAY_T(array, XAI_S64)
#define XAI_CHECK_ARRAY_F16(array)  XAI_CHECK_ARRAY_T(array, XAI_F16)
#define XAI_CHECK_ARRAY_F32(array)  XAI_CHECK_ARRAY_T(array, XAI_F32)

#define XAI_CHECK_ARRAY_IS_1D(array) \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_HEIGHT(array) == 1, XAI_ERR_BADARG, "The argument (" #array ") must be a 1D array")

#define XAI_CHECK_ARRAYS_ARE_NOT_OVERLAP(array0, array1) \
  XAI_CHECK_ERROR(XAI_ARRAYS_ARE_NOT_OVERLAP(array0, array1), XAI_ERR_INPLACE, "Inplace operation is not supported")

#define XAI_CHECK_ARRAY_ELEMENT_SIZE_EQ(array0, array1)                                     \
  XAI_CHECK_ERROR(XAI_ARRAY_GET_ELEMENT_SIZE(array0) == XAI_ARRAY_GET_ELEMENT_SIZE(array1), \
                  XAI_ERR_DATATYPE, "The (" #array0 ") element size must be equal to the (" #array1 ") element size")

#define XAI_CHECK_ARRAY_SIZE_EQ(array0, array1) \
  XAI_CHECK_ERROR(XAI_ARRAY_SIZE_EQ(array0, array1), XAI_ERR_DATASIZE, "The (" #array0 ") argument size is not equal to the (" #array1 ") argument size")

#define XAI_CHECK_ARRAY_SIZE_GEQ(array0, array1) \
  XAI_CHECK_ERROR(XAI_ARRAY_SIZE_GEQ(array0, array1), XAI_ERR_DATASIZE, "The (" #array0 ") argument size is not equal to OR greater than the (" #array1 ") argument size")

#define XAI_CHECK_ARRAY_ALIGNMENT(array, DEPTH, ERR) \
  XAI_CHECK_ERROR(XAI_ARRAY_IS_ALIGNED_ ## DEPTH(array), XAI_ERR_ ## ERR, "The argument (" #array ") is not fully aligned")

#define XAI_CHECK_ARRAY_IALIGNMENT_NX8(array)     XAI_CHECK_ARRAY_ALIGNMENT(array, NX8, IALIGNMENT)
#define XAI_CHECK_ARRAY_IALIGNMENT_2NX8(array)    XAI_CHECK_ARRAY_ALIGNMENT(array, 2NX8, IALIGNMENT)
#define XAI_CHECK_ARRAY_IALIGNMENT_NX16(array)    XAI_CHECK_ARRAY_ALIGNMENT(array, NX16, IALIGNMENT)
#define XAI_CHECK_ARRAY_IALIGNMENT_N_2X32(array)  XAI_CHECK_ARRAY_ALIGNMENT(array, N_2X32, IALIGNMENT)
#define XAI_CHECK_ARRAY_OALIGNMENT_NX8(array)     XAI_CHECK_ARRAY_ALIGNMENT(array, NX8, OALIGNMENT)
#define XAI_CHECK_ARRAY_OALIGNMENT_2NX8(array)    XAI_CHECK_ARRAY_ALIGNMENT(array, 2NX8, OALIGNMENT)
#define XAI_CHECK_ARRAY_OALIGNMENT_NX16(array)    XAI_CHECK_ARRAY_ALIGNMENT(array, NX16, OALIGNMENT)
#define XAI_CHECK_ARRAY_OALIGNMENT_N_2X32(array)  XAI_CHECK_ARRAY_ALIGNMENT(array, N_2X32, OALIGNMENT)


// check tile invariants
#define XAI_TILE2D_IS_CONSISTENT(t)                                                                                                                                                                                   \
  ((XAI_TILE2D_GET_PITCH(t) >= XAI_TILE2D_GET_WIDTH(t) + XAI_TILE2D_GET_EDGE_WIDTH(t) * 2) &&                                                                                                                             \
   ((uint8_t *) XAI_TILE2D_GET_DATA_PTR(t) - (XAI_TILE2D_GET_EDGE_WIDTH(t) + XAI_TILE2D_GET_PITCH(t) * XAI_TILE2D_GET_EDGE_HEIGHT(t)) * XAI_TILE2D_GET_ELEMENT_SIZE(t)                                                        \
    >= (uint8_t *) XAI_TILE2D_GET_BUFF_PTR(t)) &&                                                                                                                                                                     \
   ((uint8_t *) XAI_TILE2D_GET_DATA_PTR(t) + (XAI_TILE2D_GET_PITCH(t) * (XAI_TILE2D_GET_HEIGHT(t) + XAI_TILE2D_GET_EDGE_HEIGHT(t) - 1) + XAI_TILE2D_GET_WIDTH(t) + XAI_TILE2D_GET_EDGE_WIDTH(t)) * XAI_TILE2D_GET_ELEMENT_SIZE(t) \
    <= (uint8_t *) XAI_TILE2D_GET_BUFF_PTR(t) + XAI_TILE2D_GET_BUFF_SIZE(t)))

// common tile error checks
#define XAI_CHECK_TILE2D(tile)                                                                                                \
  XAI_CHECK_POINTER(tile);                                                                                                  \
  XAI_CHECK_ERROR(XAI_TILE2D_IS_CONSISTENT(tile), XAI_ERR_BADARG, "The argument (" #tile ") is invalid");                     \
  XAI_CHECK_ERROR(XAI_TILE2D_IS_TILE2D(tile), XAI_ERR_BADARG, "The argument (" #tile ") is not a tile");                        \
  XAI_CHECK_ERROR(XAI_TILE2D_STARTS_IN_DRAM(tile), XAI_ERR_MEMLOCAL, "The argument (" #tile ") data does not start in DRAM"); \
  XAI_CHECK_ERROR(XAI_TILE2D_ENDS_IN_DRAM(tile), XAI_ERR_MEMLOCAL, "Complete data for the argument  (" #tile ")  does not lie in DRAM")

#define XAI_TILE2D_CHECK_TYPE(a, type) \
  ((XAI_TYPE_ELEMENT_TYPE(XAI_TILE2D_GET_TYPE(a)) == type) && (XAI_TILE2D_IS_TILE2D(a)))

#define XAI_CHECK_TILE2D_I(tile, element_size)                        \
  XAI_CHECK_TILE2D(tile);                                             \
  XAI_CHECK_ERROR(XAI_ARRAY_CHECK_ELEMENT_SIZE(tile, element_size), \
                  XAI_ERR_DATATYPE, "The argument (" #tile ") has wrong type")

#define XAI_CHECK_TILE2D_T(tile, type) \
  XAI_CHECK_TILE2D(tile);              \
  XAI_CHECK_ERROR(XAI_TILE2D_CHECK_TYPE(tile, type), XAI_ERR_DATATYPE, "The argument (" #tile ") has wrong type")

#define XAI_CHECK_TILE2D_I8(array)   XAI_CHECK_TILE2D_I(array, sizeof(int8_t))
#define XAI_CHECK_TILE2D_I16(array)  XAI_CHECK_TILE2D_I(array, sizeof(int16_t))
#define XAI_CHECK_TILE2D_I32(array)  XAI_CHECK_TILE2D_I(array, sizeof(int32_t))

#define XAI_CHECK_TILE2D_U8(array)   XAI_CHECK_TILE2D_T(array, XAI_U8)
#define XAI_CHECK_TILE2D_S8(array)   XAI_CHECK_TILE2D_T(array, XAI_S8)
#define XAI_CHECK_TILE2D_U16(array)  XAI_CHECK_TILE2D_T(array, XAI_U16)
#define XAI_CHECK_TILE2D_S16(array)  XAI_CHECK_TILE2D_T(array, XAI_S16)
#define XAI_CHECK_TILE2D_U32(array)  XAI_CHECK_TILE2D_T(array, XAI_U32)
#define XAI_CHECK_TILE2D_S32(array)  XAI_CHECK_TILE2D_T(array, XAI_S32)

#define XAI_CHECK_TILE2D_EDGE(tile, edge)                                                            \
  XAI_CHECK_ERROR(XAI_TILE2D_GET_EDGE_WIDTH(tile) >= edge && XAI_TILE2D_GET_EDGE_HEIGHT(tile) >= edge, \
                  XAI_ERR_EDGE, "The (" #tile ") tile must have at least " #edge "-pixel edge extension")

#define XAI_CHECK_TILES_ARE_NOT_OVERLAP(tile0, tile1)  XAI_CHECK_ARRAYS_ARE_NOT_OVERLAP(tile0, tile1)

#define XAI_CHECK_TILE2D_IALIGNMENT_NX8(tile)            XAI_CHECK_ARRAY_IALIGNMENT_NX8(tile)
#define XAI_CHECK_TILE2D_IALIGNMENT_2NX8(tile)           XAI_CHECK_ARRAY_IALIGNMENT_2NX8(tile)
#define XAI_CHECK_TILE2D_IALIGNMENT_NX16(tile)           XAI_CHECK_ARRAY_IALIGNMENT_NX16(tile)
#define XAI_CHECK_TILE2D_IALIGNMENT_N_2X32(tile)         XAI_CHECK_ARRAY_IALIGNMENT_N_2X32(tile)
#define XAI_CHECK_TILE2D_OALIGNMENT_NX8(tile)            XAI_CHECK_ARRAY_OALIGNMENT_NX8(tile)
#define XAI_CHECK_TILE2D_OALIGNMENT_2NX8(tile)           XAI_CHECK_ARRAY_OALIGNMENT_2NX8(tile)
#define XAI_CHECK_TILE2D_OALIGNMENT_NX16(tile)           XAI_CHECK_ARRAY_OALIGNMENT_NX16(tile)
#define XAI_CHECK_TILE2D_OALIGNMENT_N_2X32(tile)         XAI_CHECK_ARRAY_OALIGNMENT_N_2X32(tile)

// Checks for confinement of 3D and 4D tiles in single DRAM
#if XAI_EMULATE_LOCAL_RAM && __XTENSA__ && !SYS_MEM_TESTING
#if XCHAL_NUM_DATARAM == 2
#define XAI_ARRAY_START_AND_END_IN_SINGLE_DRAM(t)                                        \
  ((XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) >= ((uint32_t) XCHAL_DATARAM0_VADDR) &&   \
    XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) + XAI_ARRAY_GET_BUFF_SIZE(t)              \
    <= (((uint32_t) XCHAL_DATARAM0_VADDR) + ((uint32_t) XCHAL_DATARAM0_SIZE))) ||        \
   ((XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) >= ((uint32_t) XCHAL_DATARAM1_VADDR)) && \
    (XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) + XAI_ARRAY_GET_BUFF_SIZE(t)             \
     <= (((uint32_t) XCHAL_DATARAM1_VADDR) + ((uint32_t) XCHAL_DATARAM1_SIZE)))))
#elif XCHAL_NUM_DATARAM == 1
#define XAI_ARRAY_START_AND_END_IN_SINGLE_DRAM(t)                                     \
  (XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) >= ((uint32_t) XCHAL_DATARAM0_VADDR) && \
   XAI_PTR_TO_ADDR(XAI_ARRAY_GET_BUFF_PTR(t)) + XAI_ARRAY_GET_BUFF_SIZE(t)            \
   <= (((uint32_t) XCHAL_DATARAM0_VADDR) + ((uint32_t) XCHAL_DATARAM0_SIZE)))
#endif
#define XAI_TILE3D_START_AND_END_IN_SINGLE_DRAM(t)  XAI_ARRAY_START_AND_END_IN_SINGLE_DRAM(t)
#define XAI_TILE4D_START_AND_END_IN_SINGLE_DRAM(t)  XAI_ARRAY_START_AND_END_IN_SINGLE_DRAM(t)
#else
#define XAI_TILE3D_START_AND_END_IN_SINGLE_DRAM(t)  1
#define XAI_TILE4D_START_AND_END_IN_SINGLE_DRAM(t)  1
#define XAI_ARRAY_START_AND_END_IN_SINGLE_DRAM(t)   1
#endif

#define XAI_CHECK_TILE3D_FITS_IN_SINGLE_DRAM(t)                                 \
  XAI_CHECK_ERROR(XAI_TILE3D_START_AND_END_IN_SINGLE_DRAM(t), XAI_ERR_MEMLOCAL, \
                  "Complete data for the argument  (" #t ")  does not fit in single DRAM");

#define XAI_CHECK_TILE4D_FITS_IN_SINGLE_DRAM(t)                                 \
  XAI_CHECK_ERROR(XAI_TILE4D_START_AND_END_IN_SINGLE_DRAM(t), XAI_ERR_MEMLOCAL, \
                  "Complete data for the argument  (" #t ")  does not fit in single DRAM");

#define XAI_CHECK_ARRAY_FITS_IN_SINGLE_DRAM(parray)                                 \
  XAI_CHECK_ERROR(XAI_ARRAY_START_AND_END_IN_SINGLE_DRAM(parray), XAI_ERR_MEMLOCAL, \
                  "Complete data for the argument  (" #parray ")  does not fit in single DRAM");

#define XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(tile)                                                                             \
  XAI_CHECK_ERROR(XAI_TILE2D_STARTS_IN_DRAM(tile), XAI_ERR_MEMLOCAL, "The argument (" #tile ") data does not start in DRAM"); \
  XAI_CHECK_ERROR(XAI_TILE2D_ENDS_IN_DRAM(tile), XAI_ERR_MEMLOCAL, "Complete data for the argument  (" #tile ")  does not lie in DRAM");

#define XAI_CHECK_TILE4D_IN_DRAM_BOUNDARY(tile)  XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(tile)


// Checks for tile consistency
#define XAI_TILE3D_IS_CONSISTENT(t)                                                                                                                                                        \
  ((uint8_t *) XAI_TILE3D_GET_DATA_PTR(t) - (XAI_TILE3D_GET_DIM1_EDGE1(t) + XAI_TILE3D_GET_DIM1_PITCH(t) * XAI_TILE3D_GET_DIM2_EDGE1(t)                                                    \
                                             + XAI_TILE3D_GET_DIM2_PITCH(t) * XAI_TILE3D_GET_DIM3_EDGE1(t)) * XAI_TILE3D_GET_ELEMENT_SIZE(t) >= (uint8_t *) XAI_TILE3D_GET_BUFF_PTR(t)) && \
  ((uint8_t *) XAI_TILE3D_GET_DATA_PTR(t) + (XAI_TILE3D_GET_DIM2_PITCH(t) * (XAI_TILE3D_GET_DIM3(t) + XAI_TILE3D_GET_DIM3_EDGE2(t) - 1)                                                    \
                                             + XAI_TILE3D_GET_DIM1_PITCH(t) * (XAI_TILE3D_GET_DIM2(t) + XAI_TILE3D_GET_DIM2_EDGE2(t) - 1)                                                  \
                                             + XAI_TILE3D_GET_DIM1(t) + XAI_TILE3D_GET_DIM1_EDGE2(t)) * XAI_TILE3D_GET_ELEMENT_SIZE(t)                                                     \
   <= (uint8_t *) XAI_TILE3D_GET_BUFF_PTR(t) + XAI_TILE3D_GET_BUFF_SIZE(t)) &&                                                                                                             \
  (XAI_TILE3D_GET_BUFF_SIZE(t) != 0) &&                                                                                                                                                    \
  (XAI_TILE3D_GET_DIM1(t) > 0) && (XAI_TILE3D_GET_DIM2(t) > 0) && (XAI_TILE3D_GET_DIM3(t) > 0) &&                                                                                          \
  (XAI_TILE3D_GET_DIM1_PITCH(t) >= XAI_TILE3D_GET_DIM1(t) + XAI_TILE3D_GET_DIM1_EDGE1(t) + XAI_TILE3D_GET_DIM1_EDGE2(t))

#define XAI_TILE4D_IS_CONSISTENT(t)                                                                                                                                                        \
  ((uint8_t *) XAI_TILE4D_GET_DATA_PTR(t) - (XAI_TILE4D_GET_DIM1_EDGE1(t) + XAI_TILE4D_GET_DIM1_PITCH(t) * XAI_TILE4D_GET_DIM2_EDGE1(t)                                                    \
                                             + XAI_TILE4D_GET_DIM2_PITCH(t) * XAI_TILE4D_GET_DIM3_EDGE1(t)) * XAI_TILE4D_GET_ELEMENT_SIZE(t) >= (uint8_t *) XAI_TILE4D_GET_BUFF_PTR(t)) && \
  ((uint8_t *) XAI_TILE4D_GET_DATA_PTR(t) + (XAI_TILE4D_GET_DIM3_PITCH(t) * (XAI_TILE4D_GET_DIM4(t) - 1)                                                                                   \
                                             + XAI_TILE4D_GET_DIM2_PITCH(t) * (XAI_TILE4D_GET_DIM3(t) + XAI_TILE4D_GET_DIM3_EDGE2(t) - 1)                                                  \
                                             + XAI_TILE4D_GET_DIM1_PITCH(t) * (XAI_TILE4D_GET_DIM2(t) + XAI_TILE4D_GET_DIM2_EDGE2(t) - 1)                                                  \
                                             + XAI_TILE4D_GET_DIM1(t) + XAI_TILE4D_GET_DIM1_EDGE2(t)) * XAI_TILE4D_GET_ELEMENT_SIZE(t)                                                     \
   <= (uint8_t *) XAI_TILE4D_GET_BUFF_PTR(t) + XAI_TILE4D_GET_BUFF_SIZE(t)) &&                                                                                                             \
  (XAI_TILE4D_GET_BUFF_SIZE(t) != 0) &&                                                                                                                                                    \
  (XAI_TILE4D_GET_DIM1(t) > 0) && (XAI_TILE4D_GET_DIM2(t) > 0) && (XAI_TILE4D_GET_DIM3(t) > 0) && (XAI_TILE4D_GET_DIM4(t) > 0) &&                                                          \
  (XAI_TILE4D_GET_DIM1_PITCH(t) >= XAI_TILE4D_GET_DIM1(t) + XAI_TILE4D_GET_DIM1_EDGE1(t) + XAI_TILE4D_GET_DIM1_EDGE2(t))

#define XAI_TILE3D_SIZE_EQ(t1, t2)                                                                             \
  (XAI_TILE3D_GET_DIM1(t1) == XAI_TILE3D_GET_DIM1(t2) && XAI_TILE3D_GET_DIM2(t1) == XAI_TILE3D_GET_DIM2(t2) && \
   XAI_TILE3D_GET_DIM3(t1) == XAI_TILE3D_GET_DIM3(t2))

#define XAI_TILE3D_PITCH_EQ(t1, t2)                                    \
  (XAI_FRAME3D_GET_DIM1_PITCH(t1) == XAI_FRAME3D_GET_DIM1_PITCH(t2) && \
   XAI_FRAME3D_GET_DIM2_PITCH(t1) == XAI_FRAME3D_GET_DIM2_PITCH(t2))

// common tile error checks
#define XAI_CHECK_TILE3D(tile)                                                                            \
  XAI_CHECK_POINTER(tile);                                                                                \
  XAI_CHECK_ERROR(XAI_TILE3D_IS_CONSISTENT(tile), XAI_ERR_BADARG, "The argument (" #tile ") is invalid"); \
  XAI_CHECK_ERROR(XAI_TYPE_IS_TILE3D(XAI_TILE3D_GET_TYPE(tile)), XAI_ERR_BADARG, "The argument (" #tile ") is not a tile");


#define XAI_TILE3D_CHECK_TYPE(a, type) \
  ((XAI_TYPE_ELEMENT_TYPE(XAI_TILE3D_GET_TYPE(a)) == type) && (XAI_TYPE_IS_TILE3D(XAI_TILE3D_GET_TYPE(a))))

#define XAI_TILE3D_CHECK_ELEMENT_SIZE(a, size)  (XAI_ARRAY_GET_ELEMENT_SIZE(a) == (size))

#define XAI_CHECK_TILE3D_SIZE_EQ(t1, t2)                                                                           \
  XAI_CHECK_ERROR(XAI_TILE3D_SIZE_EQ(t1, t2), XAI_ERR_DATASIZE, "Size of the ("#t1 ") and ("#t2 ") are not same"); \
  if (XAI_TILE3D_GET_DATA_PTR(t1) == XAI_TILE3D_GET_DATA_PTR(t2))                                                  \
  {                                                                                                                \
    XAI_CHECK_ERROR(XAI_TILE3D_PITCH_EQ(t1, t2), XAI_ERR_INPLACE, "Inplace operation not supported when pitch of " \
                    "("#t1 ") and ("#t2 ") are not same");                                                         \
  }

#define XAI_CHECK_TILE3D_I(tile, element_size)                           \
  XAI_CHECK_TILE3D(tile);                                                \
  XAI_CHECK_ERROR(XAI_TILE3D_CHECK_ELEMENT_SIZE(tile, element_size) &&   \
                  !((XAI_TILE3D_GET_TYPE(tile)) & (XAI_TYPE_FLOAT_BIT)), \
                  XAI_ERR_DATATYPE, "The argument (" #tile ") has wrong type")

#define XAI_CHECK_TILE3D_X(tile, element_size)                       \
  XAI_CHECK_TILE3D(tile);                                            \
  XAI_CHECK_ERROR(XAI_TILE3D_CHECK_ELEMENT_SIZE(tile, element_size), \
                  XAI_ERR_DATATYPE, "The argument (" #tile ") has wrong type")

#define XAI_CHECK_TILE3D_T(tile, type) \
  XAI_CHECK_TILE3D(tile);              \
  XAI_CHECK_ERROR(XAI_TILE3D_CHECK_TYPE(tile, type), XAI_ERR_DATATYPE, "The argument (" #tile ") has wrong type")

#define XAI_CHECK_TILE3D_I8(array)   XAI_CHECK_TILE3D_I(array, sizeof(int8_t))
#define XAI_CHECK_TILE3D_I16(array)  XAI_CHECK_TILE3D_I(array, sizeof(int16_t))
#define XAI_CHECK_TILE3D_I32(array)  XAI_CHECK_TILE3D_I(array, sizeof(int32_t))
#define XAI_CHECK_TILE3D_I64(array)  XAI_CHECK_TILE3D_I(array, sizeof(int64_t))

#define XAI_CHECK_TILE3D_X16(array)  XAI_CHECK_TILE3D_X(array, sizeof(int16_t))
#define XAI_CHECK_TILE3D_X32(array)  XAI_CHECK_TILE3D_X(array, sizeof(int32_t))

#define XAI_CHECK_TILE3D_U8(array)   XAI_CHECK_TILE3D_T(array, XAI_U8)
#define XAI_CHECK_TILE3D_S8(array)   XAI_CHECK_TILE3D_T(array, XAI_S8)
#define XAI_CHECK_TILE3D_U16(array)  XAI_CHECK_TILE3D_T(array, XAI_U16)
#define XAI_CHECK_TILE3D_S16(array)  XAI_CHECK_TILE3D_T(array, XAI_S16)
#define XAI_CHECK_TILE3D_U32(array)  XAI_CHECK_TILE3D_T(array, XAI_U32)
#define XAI_CHECK_TILE3D_S32(array)  XAI_CHECK_TILE3D_T(array, XAI_S32)
#define XAI_CHECK_TILE3D_S64(array)  XAI_CHECK_TILE3D_T(array, XAI_S64)
#define XAI_CHECK_TILE3D_F16(array)  XAI_CHECK_TILE3D_T(array, XAI_F16)
#define XAI_CHECK_TILE3D_F32(array)  XAI_CHECK_TILE3D_T(array, XAI_F32)

// checks for 4D tiles
#define XAI_CHECK_TILE4D(tile)                                                                            \
  XAI_CHECK_POINTER(tile);                                                                                \
  XAI_CHECK_ERROR(XAI_TILE4D_IS_CONSISTENT(tile), XAI_ERR_BADARG, "The argument (" #tile ") is invalid"); \
  XAI_CHECK_ERROR(XAI_TYPE_IS_TILE4D(XAI_TILE4D_GET_TYPE(tile)), XAI_ERR_BADARG, "The argument (" #tile ") is not a tile");

#define XAI_TILE4D_SIZE_EQ(t1, t2)                                                                             \
  (XAI_TILE4D_GET_DIM1(t1) == XAI_TILE4D_GET_DIM1(t2) && XAI_TILE4D_GET_DIM2(t1) == XAI_TILE4D_GET_DIM2(t2) && \
   XAI_TILE4D_GET_DIM3(t1) == XAI_TILE4D_GET_DIM3(t2) && XAI_TILE4D_GET_DIM4(t1) == XAI_TILE4D_GET_DIM4(t2))

#define XAI_TILE4D_CHECK_TYPE(a, type) \
  ((XAI_TYPE_ELEMENT_TYPE(XAI_TILE4D_GET_TYPE(a)) == type) && (XAI_TYPE_IS_TILE4D(XAI_TILE4D_GET_TYPE(a))))

#define XAI_TILE4D_CHECK_ELEMENT_SIZE(a, size)  (XAI_ARRAY_GET_ELEMENT_SIZE(a) == (size))

#define XAI_CHECK_TILE4D_I(tile, element_size)                           \
  XAI_CHECK_TILE4D(tile);                                                \
  XAI_CHECK_ERROR(XAI_TILE4D_CHECK_ELEMENT_SIZE(tile, element_size) &&   \
                  !((XAI_TILE4D_GET_TYPE(tile)) & (XAI_TYPE_FLOAT_BIT)), \
                  XAI_ERR_DATATYPE, "The argument (" #tile ") has wrong type")

#define XAI_CHECK_TILE4D_X(tile, element_size)                       \
  XAI_CHECK_TILE4D(tile);                                            \
  XAI_CHECK_ERROR(XAI_TILE4D_CHECK_ELEMENT_SIZE(tile, element_size), \
                  XAI_ERR_DATATYPE, "The argument (" #tile ") has wrong type")

#define XAI_CHECK_TILE4D_T(tile, type) \
  XAI_CHECK_TILE4D(tile);              \
  XAI_CHECK_ERROR(XAI_TILE4D_CHECK_TYPE(tile, type), XAI_ERR_DATATYPE, "The argument (" #tile ") has wrong type")

#define XAI_CHECK_TILE4D_SIZE_EQ(t1, t2) \
  XAI_CHECK_ERROR(XAI_TILE4D_SIZE_EQ(t1, t2), XAI_ERR_DATASIZE, "Size of the ("#t1 ") and ("#t2 ") is not same")

#define XAI_CHECK_TILE4D_I8(array)   XAI_CHECK_TILE4D_I(array, sizeof(int8_t))
#define XAI_CHECK_TILE4D_I16(array)  XAI_CHECK_TILE4D_I(array, sizeof(int16_t))
#define XAI_CHECK_TILE4D_I32(array)  XAI_CHECK_TILE4D_I(array, sizeof(int32_t))

#define XAI_CHECK_TILE4D_X16(array)  XAI_CHECK_TILE4D_X(array, sizeof(int16_t))
#define XAI_CHECK_TILE4D_X32(array)  XAI_CHECK_TILE4D_X(array, sizeof(int32_t))

#define XAI_CHECK_TILE4D_U8(array)   XAI_CHECK_TILE4D_T(array, XAI_U8)
#define XAI_CHECK_TILE4D_S8(array)   XAI_CHECK_TILE4D_T(array, XAI_S8)
#define XAI_CHECK_TILE4D_U16(array)  XAI_CHECK_TILE4D_T(array, XAI_U16)
#define XAI_CHECK_TILE4D_S16(array)  XAI_CHECK_TILE4D_T(array, XAI_S16)
#define XAI_CHECK_TILE4D_F16(array)  XAI_CHECK_TILE4D_T(array, XAI_F16)
#define XAI_CHECK_TILE4D_U32(array)  XAI_CHECK_TILE4D_T(array, XAI_U32)
#define XAI_CHECK_TILE4D_S32(array)  XAI_CHECK_TILE4D_T(array, XAI_S32)
#define XAI_CHECK_TILE4D_F32(array)  XAI_CHECK_TILE4D_T(array, XAI_F32)

// check the minimal alignment requirements for 3D tile
#define XAI_TILE3D_IS_STRIDE_ALIGNED(t)      ((XAI_TILE3D_GET_DIM1_PITCH(t) & (XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE3D_IS_STRIDE_ALIGNED2(t)     ((XAI_TILE3D_GET_DIM1_PITCH(t) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE3D_IS_STRIDE_ALIGNED_2(t)    ((XAI_TILE3D_GET_DIM1_PITCH(t) & (XCHAL_IVPN_SIMD_WIDTH / 2 - 1)) == 0)
#define XAI_TILE3D_IS_STRIDE_ALIGNED_4B(t)   ((XAI_TILE3D_GET_DIM1_PITCH(t) & (3)) == 0)
#define XAI_TILE3D_IS_PTR_ALIGNED_NX8(t)     ((XAI_PTR_TO_ADDR(XAI_TILE3D_GET_DATA_PTR(t)) & (XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE3D_IS_PTR_ALIGNED_2NX8(t)    ((XAI_PTR_TO_ADDR(XAI_TILE3D_GET_DATA_PTR(t)) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE3D_IS_PTR_ALIGNED_NX16(t)    ((XAI_PTR_TO_ADDR(XAI_TILE3D_GET_DATA_PTR(t)) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE3D_IS_PTR_ALIGNED_N_2X32(t)  ((XAI_PTR_TO_ADDR(XAI_TILE3D_GET_DATA_PTR(t)) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE3D_IS_PTR_ALIGNED_4B(t)      ((XAI_PTR_TO_ADDR(XAI_TILE3D_GET_DATA_PTR(t)) & 3) == 0)


#define XAI_TILE3D_IS_ALIGNED_NX8(t)     (XAI_TILE3D_IS_PTR_ALIGNED_NX8(t) && XAI_TILE3D_IS_STRIDE_ALIGNED(t))
#define XAI_TILE3D_IS_ALIGNED_2NX8(t)    (XAI_TILE3D_IS_PTR_ALIGNED_2NX8(t) && XAI_TILE3D_IS_STRIDE_ALIGNED2(t))
#define XAI_TILE3D_IS_ALIGNED_NX16(t)    (XAI_TILE3D_IS_PTR_ALIGNED_NX16(t) && XAI_TILE3D_IS_STRIDE_ALIGNED(t))
#define XAI_TILE3D_IS_ALIGNED_N_2X32(t)  (XAI_TILE3D_IS_PTR_ALIGNED_N_2X32(t) && XAI_TILE3D_IS_STRIDE_ALIGNED_2(t))
#define XAI_TILE3D_IS_ALIGNED_4B(t)      (XAI_TILE3D_IS_PTR_ALIGNED_4B(t) && XAI_TILE3D_IS_STRIDE_ALIGNED_4B(t))

#define XAI_CHECK_TILE3D_ALIGNMENT(array, DEPTH, ERR) \
  XAI_CHECK_ERROR(XAI_TILE3D_IS_ALIGNED_ ## DEPTH(array), XAI_ERR_ ## ERR, "The argument (" #array ") is not fully aligned")

#define XAI_CHECK_TILE3D_IALIGNMENT_NX8(array)     XAI_CHECK_TILE3D_ALIGNMENT(array, NX8, IALIGNMENT)
#define XAI_CHECK_TILE3D_IALIGNMENT_2NX8(array)    XAI_CHECK_TILE3D_ALIGNMENT(array, 2NX8, IALIGNMENT)
#define XAI_CHECK_TILE3D_IALIGNMENT_NX16(array)    XAI_CHECK_TILE3D_ALIGNMENT(array, NX16, IALIGNMENT)
#define XAI_CHECK_TILE3D_IALIGNMENT_N_2X32(array)  XAI_CHECK_TILE3D_ALIGNMENT(array, N_2X32, IALIGNMENT)
#define XAI_CHECK_TILE3D_OALIGNMENT_NX8(array)     XAI_CHECK_TILE3D_ALIGNMENT(array, NX8, OALIGNMENT)
#define XAI_CHECK_TILE3D_OALIGNMENT_2NX8(array)    XAI_CHECK_TILE3D_ALIGNMENT(array, 2NX8, OALIGNMENT)
#define XAI_CHECK_TILE3D_OALIGNMENT_NX16(array)    XAI_CHECK_TILE3D_ALIGNMENT(array, NX16, OALIGNMENT)
#define XAI_CHECK_TILE3D_OALIGNMENT_N_2X32(array)  XAI_CHECK_TILE3D_ALIGNMENT(array, N_2X32, OALIGNMENT)
#define XAI_CHECK_TILE3D_CALIGNMENT_NX8(array)     XAI_CHECK_TILE3D_ALIGNMENT(array, NX8, IALIGNMENT)
#define XAI_CHECK_TILE3D_CALIGNMENT_2NX8(array)    XAI_CHECK_TILE3D_ALIGNMENT(array, 2NX8, IALIGNMENT)
#define XAI_CHECK_TILE3D_CALIGNMENT_NX16(array)    XAI_CHECK_TILE3D_ALIGNMENT(array, NX16, IALIGNMENT)
#define XAI_CHECK_TILE3D_CALIGNMENT_N_2X32(array)  XAI_CHECK_TILE3D_ALIGNMENT(array, N_2X32, IALIGNMENT)

// check the minimal alignment requirements for 4D tile
#define XAI_TILE4D_IS_STRIDE_ALIGNED(t)      ((XAI_TILE4D_GET_DIM1_PITCH(t) & (XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE4D_IS_STRIDE_ALIGNED2(t)     ((XAI_TILE4D_GET_DIM1_PITCH(t) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE4D_IS_STRIDE_ALIGNED_2(t)    ((XAI_TILE4D_GET_DIM1_PITCH(t) & (XCHAL_IVPN_SIMD_WIDTH / 2 - 1)) == 0)
#define XAI_TILE4D_IS_PTR_ALIGNED_NX8(t)     ((XAI_PTR_TO_ADDR(XAI_TILE4D_GET_DATA_PTR(t)) & (XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE4D_IS_PTR_ALIGNED_2NX8(t)    ((XAI_PTR_TO_ADDR(XAI_TILE4D_GET_DATA_PTR(t)) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE4D_IS_PTR_ALIGNED_NX16(t)    ((XAI_PTR_TO_ADDR(XAI_TILE4D_GET_DATA_PTR(t)) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)
#define XAI_TILE4D_IS_PTR_ALIGNED_N_2X32(t)  ((XAI_PTR_TO_ADDR(XAI_TILE4D_GET_DATA_PTR(t)) & (2 * XCHAL_IVPN_SIMD_WIDTH - 1)) == 0)

#define XAI_TILE4D_IS_ALIGNED_NX8(t)         (XAI_TILE4D_IS_PTR_ALIGNED_NX8(t) && XAI_TILE4D_IS_STRIDE_ALIGNED(t))
#define XAI_TILE4D_IS_ALIGNED_2NX8(t)        (XAI_TILE4D_IS_PTR_ALIGNED_2NX8(t) && XAI_TILE4D_IS_STRIDE_ALIGNED2(t))
#define XAI_TILE4D_IS_ALIGNED_NX16(t)        (XAI_TILE4D_IS_PTR_ALIGNED_NX16(t) && XAI_TILE4D_IS_STRIDE_ALIGNED(t))
#define XAI_TILE4D_IS_ALIGNED_N_2X32(t)      (XAI_TILE4D_IS_PTR_ALIGNED_N_2X32(t) && XAI_TILE4D_IS_STRIDE_ALIGNED_2(t))

#define XAI_CHECK_TILE4D_ALIGNMENT(array, DEPTH, ERR) \
  XAI_CHECK_ERROR(XAI_TILE4D_IS_ALIGNED_ ## DEPTH(array), XAI_ERR_ ## ERR, "The argument (" #array ") is not fully aligned")

#define XAI_CHECK_TILE4D_IALIGNMENT_NX8(array)           XAI_CHECK_TILE4D_ALIGNMENT(array, NX8, IALIGNMENT)
#define XAI_CHECK_TILE4D_IALIGNMENT_2NX8(array)          XAI_CHECK_TILE4D_ALIGNMENT(array, 2NX8, IALIGNMENT)
#define XAI_CHECK_TILE4D_IALIGNMENT_NX16(array)          XAI_CHECK_TILE4D_ALIGNMENT(array, NX16, IALIGNMENT)
#define XAI_CHECK_TILE4D_IALIGNMENT_N_2X32(array)        XAI_CHECK_TILE4D_ALIGNMENT(array, N_2X32, IALIGNMENT)
#define XAI_CHECK_TILE4D_OALIGNMENT_NX8(array)           XAI_CHECK_TILE4D_ALIGNMENT(array, NX8, OALIGNMENT)
#define XAI_CHECK_TILE4D_OALIGNMENT_2NX8(array)          XAI_CHECK_TILE4D_ALIGNMENT(array, 2NX8, OALIGNMENT)
#define XAI_CHECK_TILE4D_OALIGNMENT_NX16(array)          XAI_CHECK_TILE4D_ALIGNMENT(array, NX16, OALIGNMENT)
#define XAI_CHECK_TILE4D_OALIGNMENT_N_2X32(array)        XAI_CHECK_TILE4D_ALIGNMENT(array, N_2X32, OALIGNMENT)

#define XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(tile0, tile1)  XAI_CHECK_ARRAYS_ARE_NOT_OVERLAP(tile0, tile1)
#define XAI_CHECK_TILES4D_ARE_NOT_OVERLAP(tile0, tile1)  XAI_CHECK_TILES3D_ARE_NOT_OVERLAP(tile0, tile1)

#define XAI_CHECK_TILE3D_EQUAL(tile1, tile2)                                                  \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_DIM1(tile1) == XAI_TILE3D_GET_DIM1(tile2) &&                 \
                  XAI_TILE3D_GET_DIM2(tile1) == XAI_TILE3D_GET_DIM2(tile2) &&                 \
                  XAI_TILE3D_GET_DIM3(tile1) == XAI_TILE3D_GET_DIM3(tile2), XAI_ERR_DATASIZE, \
                  "Tiles sizes are not equal.");

#define XAI_CHECK_TILE4D_EQUAL(tile1, tile2)                                                  \
  XAI_CHECK_ERROR(XAI_TILE4D_GET_DIM1(tile1) == XAI_TILE4D_GET_DIM1(tile2) &&                 \
                  XAI_TILE4D_GET_DIM2(tile1) == XAI_TILE4D_GET_DIM2(tile2) &&                 \
                  XAI_TILE4D_GET_DIM3(tile1) == XAI_TILE4D_GET_DIM3(tile2) &&                 \
                  XAI_TILE4D_GET_DIM4(tile1) == XAI_TILE4D_GET_DIM4(tile2), XAI_ERR_DATASIZE, \
                  "Tiles sizes are not equal.");

#define XAI_CHECK_TILE3D_ELEMENT_SIZE_EQ(inT, outT)                                      \
  XAI_CHECK_ERROR(XAI_TILE3D_GET_ELEMENT_SIZE(inT) == XAI_TILE3D_GET_ELEMENT_SIZE(outT), \
                  XAI_ERR_DATATYPE, "Input tile element element size must be equal to output tile element size")

#define XAI_CHECK_TILE4D_ELEMENT_SIZE_EQ(inT, outT)                                      \
  XAI_CHECK_ERROR(XAI_TILE4D_GET_ELEMENT_SIZE(inT) == XAI_TILE4D_GET_ELEMENT_SIZE(outT), \
                  XAI_ERR_DATATYPE, "Input tile element element size must be equal to output tile element size")

#ifdef XAI_ERROR_CHECKS_RELAXED_REF
#undef XAI_CHECK_TILE4D_IALIGNMENT_2NX8
#undef XAI_ARRAY_STARTS_IN_DRAM
#undef XAI_ARRAY_ENDS_IN_DRAM
#undef XAI_TILE2D_STARTS_IN_DRAM
#undef XAI_TILE2D_ENDS_IN_DRAM
#undef XAI_TILE3D_START_AND_END_IN_SINGLE_DRAM
#undef XAI_TILE4D_START_AND_END_IN_SINGLE_DRAM
#undef XAI_ARRAY_START_AND_END_IN_SINGLE_DRAM
#undef XAI_ARRAYS_ARE_NOT_OVERLAP

#define XAI_CHECK_TILE4D_IALIGNMENT_2NX8(array)
#define XAI_ARRAY_STARTS_IN_DRAM(t)                 1
#define XAI_ARRAY_ENDS_IN_DRAM(t)                   1
#define XAI_TILE2D_STARTS_IN_DRAM(t)                  1
#define XAI_TILE2D_ENDS_IN_DRAM(t)                    1
#define XAI_TILE3D_START_AND_END_IN_SINGLE_DRAM(t)  1
#define XAI_TILE4D_START_AND_END_IN_SINGLE_DRAM(t)  1
#define XAI_ARRAY_START_AND_END_IN_SINGLE_DRAM(t)   1
#define XAI_ARRAYS_ARE_NOT_OVERLAP(t1, t2)          1
#endif

#if defined SYS_MEM_TESTING || defined XAI_ERROR_CHECKS_RELAXED_REF
#undef XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY
#define XAI_CHECK_TILE3D_IN_DRAM_BOUNDARY(tile)
#endif

// other macros
#define XAI_TO_Q15(val)     ((int16_t) ((val) * (1 << 15) + 0.5))
#define XAI_TO_Q1_14(val)   ((int16_t) ((val) * (1 << 14) + 0.5))
#define XAI_TO_Q2_13(val)   ((int16_t) ((val) * (1 << 13) + 0.5))
#define XAI_TO_Q3_12(val)   ((int16_t) ((val) * (1 << 12) + 0.5))
#define XAI_TO_Q4_11(val)   ((int16_t) ((val) * (1 << 11) + 0.5))
#define XAI_TO_Q5_10(val)   ((int16_t) ((val) * (1 << 10) + 0.5))
#define XAI_TO_Q13_18(val)  ((int) ((val) * (1 << 18) + 0.5))
#define XAI_Q0_16_HALF  0x8000
#endif
