#pragma once

/*
nnlib forces ROW_UNROLL to be 4 when the input and weight are both aligned to 8b
boundary. Through experimentation, we observe that the
xa_nn_matmul_asym8xasym8_asym8 kernel in nnlib performs better when the weight
matrix is uniformly unrolled by a factor of 2 instead of 4 for 8b aligned case.
We add a case for ROW_UNROLL=2 and VEC_UNROLL=2 here. This code is similar to
the ROW_UNROLL=4 and VEC_UNROLL=2 code in
https://www.internalfb.com/code/fbsource/third-party/nnlib-hifi4/xa_nnlib/algo/common/include/xa_nnlib_common_macros.h.
*/

// Unrolling macros that unroll both matrices by a factor of 2.
#if (ROW_UNROLL == 2 && VEC_UNROLL == 2)

#define SETUP_VEC_BATCH UNROLL_SETUP_VEC_BATCH(0) UNROLL_SETUP_VEC_BATCH(1)

#define SETUP_ACC_BATCH         \
  UNROLL_ROW_SETUP_ACC_BATCH(0) \
  UNROLL_ROW_SETUP_ACC_BATCH(1)

#define SETUP_ACC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_SETUP_ACC_BATCH(idx_row, 0)        \
  UNROLL_SETUP_ACC_BATCH(idx_row, 1)

#define SETUP_ACC_BATCH_TAIL   \
  UNROLL_SETUP_ACC_BATCH(0, 0) \
  UNROLL_SETUP_ACC_BATCH(1, 0)

#define LOAD_VEC_BATCH UNROLL_LOAD_VEC_BATCH(0) UNROLL_LOAD_VEC_BATCH(1)

#define LOAD_MAT1         \
  UNROLL_LOAD_ROW_MAT1(0) \
  UNROLL_LOAD_ROW_MAT1(1)

#define KERNEL_MAT1_VEC_BATCH         \
  UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(0) \
  UNROLL_ROW_KERNEL_MAT1_VEC_BATCH(1)

#define KERNEL_MAT1_VEC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row, 0)        \
  UNROLL_KERNEL_MAT1_VEC_BATCH(idx_row, 1)

#define KERNEL_MAT1_VEC_BATCH_TAIL   \
  UNROLL_KERNEL_MAT1_VEC_BATCH(0, 0) \
  UNROLL_KERNEL_MAT1_VEC_BATCH(1, 0)

#define ADD_BIAS_ACC_BATCH   \
  UNROLL_ROW_ADD_BIAS_ACC(0) \
  UNROLL_ROW_ADD_BIAS_ACC(1)

#define ADD_BIAS_BATCH_ACC_VEC_UNROLL(idx_row) \
  UNROLL_ADD_BIAS_ACC_BATCH(idx_row, 0) UNROLL_ADD_BIAS_ACC_BATCH(idx_row, 1)

#define ADD_BIAS_ACC_BATCH_TAIL             \
  LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(0, 0) \
      LOAD_BIAS UNROLL_ADD_BIAS_ACC_BATCH(1, 0)

#define STORE_ACC_BATCH   \
  UNROLL_ROW_STORE_ACC(0) \
  UNROLL_ROW_STORE_ACC(1)

#define STORE_ACC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_STORE_ACC_BATCH(idx_row, 0) UNROLL_STORE_ACC_BATCH(idx_row, 1)

#define STORE_ACC_BATCH_TAIL   \
  UNROLL_STORE_ACC_BATCH(0, 0) \
  UNROLL_STORE_ACC_BATCH(1, 0)

#define ADJUST_ACC_BATCH_TAIL   \
  UNROLL_ADJUST_ACC_BATCH(0, 0) \
  UNROLL_ADJUST_ACC_BATCH(1, 0)

#define ADJUST_ACC_BATCH   \
  UNROLL_ROW_ADJUST_ACC(0) \
  UNROLL_ROW_ADJUST_ACC(1)

#define ADJUST_ACC_BATCH_VEC_UNROLL(idx_row) \
  UNROLL_ADJUST_ACC_BATCH(idx_row, 0) UNROLL_ADJUST_ACC_BATCH(idx_row, 1)

#endif /* (ROW_UNROLL == 2 && VEC_UNROLL == 2)*/
