/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorch
import XCTest

private final class SelfRemovingSink: LogSink {
  private let onDeinit: () -> Void

  init(onDeinit: @escaping () -> Void) {
    self.onDeinit = onDeinit
    Log.shared.add(sink: self)
  }

  deinit {
    Log.shared.remove(sink: self)
    onDeinit()
  }

  func log(level: LogLevel, timestamp: TimeInterval, filename: String, line: UInt, message: String) {}
}

class LogTest: XCTestCase {
  func testSinkRemovesItselfInDeinit() {
    let deinited = DispatchGroup()
    DispatchQueue.concurrentPerform(iterations: 1000) { _ in
      deinited.enter()
      _ = SelfRemovingSink { deinited.leave() }
    }
    var sinks: [SelfRemovingSink] = (0..<1000).map { _ in
      deinited.enter()
      return SelfRemovingSink { deinited.leave() }
    }
    sinks.removeAll()
    XCTAssertEqual(deinited.wait(timeout: .now() + 10), .success)
  }
}
