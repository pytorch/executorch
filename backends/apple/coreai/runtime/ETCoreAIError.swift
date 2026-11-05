// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import Foundation

func bridgeError(
  _ code: ETCoreAIErrorCode, _ message: String, underlying: Error? = nil
) -> NSError {
  var info: [String: Any] = [NSLocalizedDescriptionKey: message]
  if let underlying { info[NSUnderlyingErrorKey] = underlying }
  return NSError(domain: ETCoreAIErrorDomain, code: code.rawValue, userInfo: info)
}

func mappedError(_ error: Error, code: ETCoreAIErrorCode, context: String) -> NSError {
  let error = error as NSError
  return error.domain == ETCoreAIErrorDomain
    ? error : bridgeError(code, "\(context): \(error.localizedDescription)", underlying: error)
}

// Objective-C blocks have no Sendable annotation. Each box is invoked by one detached task only.
final class Completion<Callback>: @unchecked Sendable {
  let call: Callback

  init(_ call: Callback) {
    self.call = call
  }
}
