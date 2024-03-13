/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import UIKit

public struct Classification {
  public let label: String
  public let confidence: Float

  public init(label: String, confidence: Float) {
    self.label = label
    self.confidence = confidence
  }
}

public protocol ImageClassification {
  func classify(image: UIImage) throws -> [Classification]
}
