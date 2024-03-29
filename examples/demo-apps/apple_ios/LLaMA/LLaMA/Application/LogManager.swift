/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI

import ExecuTorch

struct LogEntry: Identifiable, Codable {
  let id: UUID
  let level: Int
  let timestamp: TimeInterval
  let filename: String
  let line: UInt
  let message: String
}

class LogManager: ObservableObject, LogSink {
  @AppStorage("logs") private var data = Data()

  @Published var logs: [LogEntry] = [] {
    didSet {
      data = (try? JSONEncoder().encode(logs)) ?? Data()
    }
  }

  init() {
    logs = (try? JSONDecoder().decode([LogEntry].self, from: data)) ?? []
    Log.shared.add(sink: self)
  }

  deinit {
    Log.shared.remove(sink: self)
  }

  func log(level: LogLevel, timestamp: TimeInterval, filename: String, line: UInt, message: String) {
    let log = LogEntry(id: UUID(), level: level.rawValue, timestamp: timestamp, filename: filename, line: line, message: message)

    DispatchQueue.main.sync {
      self.logs.append(log)
    }
  }

  func clear() {
    logs.removeAll()
  }
}
