/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation

public class ModelRuntimeTensorValueBridgingTuple: NSObject {
  @objc public let floatArray: [NSNumber]
  @objc public let shape: [NSNumber]
  @objc public init(floatArray: [NSNumber], shape: [NSNumber]) {
    self.floatArray = floatArray
    self.shape = shape
  }
}
