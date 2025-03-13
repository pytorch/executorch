// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

public class ModelRuntimeTensorValueBridgingTuple: NSObject {
  @objc public let floatArray: [NSNumber]
  @objc public let shape: [NSNumber]
  @objc public init(floatArray: [NSNumber], shape: [NSNumber]) {
    self.floatArray = floatArray
    self.shape = shape
  }
}

@objc public protocol ModelRuntimeTensorValueBridging {
  func floatRepresentation() throws -> ModelRuntimeTensorValueBridgingTuple
}
