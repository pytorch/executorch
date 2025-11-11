/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/BinaryLogicalOpTest.h>

namespace torch::executor::testing {

void BinaryLogicalOpTest::test_all_dtypes() {
#define TEST_ENTRY(ctype, dtype) \
  test_op_out<ScalarType::dtype, ScalarType::Double, ScalarType::Double>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#define TEST_ENTRY(ctype, dtype) \
  test_op_out<ScalarType::Double, ScalarType::dtype, ScalarType::Double>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#define TEST_ENTRY(ctype, dtype) \
  test_op_out<ScalarType::Double, ScalarType::Double, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}
} // namespace torch::executor::testing
