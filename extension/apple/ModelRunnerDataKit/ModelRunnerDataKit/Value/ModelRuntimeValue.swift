// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

public class ModelRuntimeValue {
  public let value: ModelRuntimeValueBridging
  public init(innerValue: ModelRuntimeValueBridging) {
    self.value = innerValue
  }

  public func stringValue() throws -> String {
    return try value.stringValue()
  }

  public func tensorValue() throws -> ModelRuntimeTensorValue {
    return try ModelRuntimeTensorValue(innerValue: value.tensorValue())
  }

  public func arrayValue() throws -> [ModelRuntimeValue] {
    return try value.arrayValue().map { ModelRuntimeValue(innerValue: $0) }
  }
}
