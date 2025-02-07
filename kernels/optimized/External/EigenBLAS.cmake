# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(__EIGEN_BLAS_INCLUDED)
  return()
endif()
set(__EIGEN_BLAS_INCLUDED TRUE)

# ##############################################################################
# Eigen BLAS is built together with Libtorch mobile. By default, it builds code
# from third-party/eigen/blas submodule.
# ##############################################################################

set(EIGEN_BLAS_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/eigen/blas"
    CACHE STRING "Eigen BLAS source directory"
)

set(EigenBlas_SRCS
    ${EIGEN_BLAS_SRC_DIR}/single.cpp
    ${EIGEN_BLAS_SRC_DIR}/double.cpp
    ${EIGEN_BLAS_SRC_DIR}/complex_single.cpp
    ${EIGEN_BLAS_SRC_DIR}/complex_double.cpp
    ${EIGEN_BLAS_SRC_DIR}/xerbla.cpp
    ${EIGEN_BLAS_SRC_DIR}/f2c/srotm.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/srotmg.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/drotm.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/drotmg.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/lsame.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/dspmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/ssbmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/chbmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/sspmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/zhbmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/chpmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/dsbmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/zhpmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/dtbmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/stbmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/ctbmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/ztbmv.c
    ${EIGEN_BLAS_SRC_DIR}/f2c/complexdots.c
)

add_library(eigen_blas STATIC ${EigenBlas_SRCS})

# Dont know what to do with this We build static versions of eigen blas but link
# into a shared library, so they need PIC.
set_property(TARGET eigen_blas PROPERTY POSITION_INDEPENDENT_CODE ON)

install(
  TARGETS eigen_blas
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
