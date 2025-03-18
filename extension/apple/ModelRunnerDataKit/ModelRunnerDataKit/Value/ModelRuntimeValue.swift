/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
