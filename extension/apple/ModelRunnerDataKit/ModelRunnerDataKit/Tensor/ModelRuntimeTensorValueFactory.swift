// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

public protocol ModelRuntimeTensorValueFactory {
  func createFloatTensor(value: [Float], shape: [Int]) -> ModelRuntimeTensorValue
}
