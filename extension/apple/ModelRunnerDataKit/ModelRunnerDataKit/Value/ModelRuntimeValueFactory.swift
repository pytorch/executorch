// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

public protocol ModelRuntimeValueFactory {
  func createString(value: String) throws -> ModelRuntimeValue
  func createTensor(value: ModelRuntimeTensorValue) throws -> ModelRuntimeValue
}
