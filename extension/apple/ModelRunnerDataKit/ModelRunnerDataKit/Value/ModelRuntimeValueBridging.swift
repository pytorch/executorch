/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation

@objc public protocol ModelRuntimeValueBridging {
  func stringValue() throws -> String
  func tensorValue() throws -> ModelRuntimeTensorValueBridging
  func arrayValue() throws -> [ModelRuntimeValueBridging]
}
