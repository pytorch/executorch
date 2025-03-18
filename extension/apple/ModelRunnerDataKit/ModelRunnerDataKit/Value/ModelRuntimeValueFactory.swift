/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation

public protocol ModelRuntimeValueFactory {
  func createString(value: String) throws -> ModelRuntimeValue
  func createTensor(value: ModelRuntimeTensorValue) throws -> ModelRuntimeValue
}
