// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

public class ModelRuntimeTensorValue {
  public let innerValue: ModelRuntimeTensorValueBridging
  public init(innerValue: ModelRuntimeTensorValueBridging) {
    self.innerValue = innerValue
  }

  public func floatRepresentation() throws -> (floatArray: [Float], shape: [Int]) {
    let value = try innerValue.floatRepresentation()
    let data = value.floatArray
    let shape = value.shape
    return (data.compactMap { $0.floatValue }, shape.compactMap { $0.intValue })
  }
}
