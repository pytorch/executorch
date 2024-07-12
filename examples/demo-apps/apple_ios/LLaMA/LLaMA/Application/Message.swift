/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import Foundation
import SwiftUI

enum MessageType {
  case prompted
  case generated
  case info
}

struct Message: Identifiable, Equatable {
  let id = UUID()
  let dateCreated = Date()
  var dateUpdated = Date()
  var type: MessageType = .prompted
  var text = ""
  var tokenCount = 0
  var image: UIImage?
}
