// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

public enum ModelRuntimeError: Error {
  case unsupportedInputType
}

public protocol ModelRuntime {
  func infer(input: [ModelRuntimeValue]) throws -> [ModelRuntimeValue]

  func getModelValueFactory() -> ModelRuntimeValueFactory
  func getModelTensorFactory() -> ModelRuntimeTensorValueFactory
}
