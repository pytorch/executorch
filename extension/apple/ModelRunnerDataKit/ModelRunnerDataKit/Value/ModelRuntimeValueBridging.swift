// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

@objc public protocol ModelRuntimeValueBridging {
  func stringValue() throws -> String
  func tensorValue() throws -> ModelRuntimeTensorValueBridging
  func arrayValue() throws -> [ModelRuntimeValueBridging]
}
