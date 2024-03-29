/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI

import ExecuTorch

struct LogView: View {
  @ObservedObject var logManager: LogManager

  var body: some View {
    ScrollView {
      VStack(alignment: .leading) {
        ForEach(logManager.logs) { log in
          Text("\(format(timestamp: log.timestamp)) \(log.filename):\(log.line)")
            .padding(.top)
            .foregroundColor(.secondary)
            .textSelection(.enabled)
          Text(log.message)
            .padding(.bottom)
            .foregroundColor(color(for: log.level))
            .textSelection(.enabled)
        }
      }
    }
    .padding()
    .defaultScrollAnchor(.bottom)
    .navigationBarTitle("Logs", displayMode: .inline)
    .navigationBarItems(trailing:
                          Button(action: { logManager.clear() }) {
                            Image(systemName: "trash")
                          }
    )
  }

  private func format(timestamp: TimeInterval) -> String {
    let totalSeconds = Int(timestamp)
    let hours = (totalSeconds / 3600) % 24
    let minutes = (totalSeconds / 60) % 60
    let seconds = totalSeconds % 60
    let microseconds = Int((timestamp - Double(totalSeconds)) * 1000000)
    return String(format: "%02d:%02d:%02d.%06d", hours, minutes, seconds, microseconds)
  }

  private func color(for level: Int) -> Color {
    switch LogLevel(rawValue: level) {
    case .debug:
      return .blue
    case .info:
      return .primary
    case .error:
      return .red
    case .fatal:
      return .purple
    default:
      return .secondary
    }
  }
}
