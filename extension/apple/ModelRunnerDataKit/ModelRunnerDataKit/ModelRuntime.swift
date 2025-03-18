/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation

public enum ModelRuntimeError: Error {
  case unsupportedInputType
}

public protocol ModelRuntime {
  func infer(input: [ModelRuntimeValue]) throws -> [ModelRuntimeValue]

  func getModelValueFactory() -> ModelRuntimeValueFactory
  func getModelTensorFactory() -> ModelRuntimeTensorValueFactory
}
