/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchdemo.executor;

/** Codes representing tensor data types. */
public enum DType {
  // NOTE: "jniCode" must be kept in sync with scalar_type.h.
  // NOTE: Never serialize "jniCode", because it can change between releases.

  /** Code for dtype torch.uint8. {@link Tensor#dtype()} */
  UINT8(1),
  /** Code for dtype torch.int8. {@link Tensor#dtype()} */
  INT8(2),
  /** Code for dtype torch.int32. {@link Tensor#dtype()} */
  INT32(3),
  /** Code for dtype torch.float32. {@link Tensor#dtype()} */
  FLOAT32(4),
  /** Code for dtype torch.int64. {@link Tensor#dtype()} */
  INT64(5),
  /** Code for dtype torch.float64. {@link Tensor#dtype()} */
  FLOAT64(6),
  ;

  final int jniCode;

  DType(int jniCode) {
    this.jniCode = jniCode;
  }
}
