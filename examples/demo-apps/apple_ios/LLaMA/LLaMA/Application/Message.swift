// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

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
}
