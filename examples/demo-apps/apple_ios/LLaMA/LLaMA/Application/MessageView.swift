/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI

struct MessageView: View {
  let message: Message

  var body: some View {
    VStack(alignment: .center) {
      if message.type == .info {
        Text(message.text)
          .font(.caption)
          .foregroundColor(.secondary)
          .padding([.leading, .trailing], 10)
      } else {
        VStack(alignment: message.type == .llamagenerated || message.type == .llavagenerated ? .leading : .trailing) {
          if message.type == .llamagenerated || message.type == .llavagenerated || message.type == .prompted {
            Text(message.type == .llamagenerated ? "Llama" : (message.type == .llavagenerated ? "Llava" : "Prompt"))
              .font(.caption)
              .foregroundColor(.secondary)
              .padding(message.type == .llamagenerated || message.type == .llavagenerated ? .trailing : .leading, 20)
          }
          HStack {
            if message.type != .llamagenerated && message.type != .llavagenerated { Spacer() }
            if message.text.isEmpty {
              if let img = message.image {
                Image(uiImage: img)
                  .resizable()
                  .scaledToFit()
                  .frame(maxWidth: 200, maxHeight: 200)
                  .padding()
                  .background(Color.gray.opacity(0.2))
                  .cornerRadius(8)
                  .padding(.vertical, 2)
              } else {
                ProgressView()
                  .progressViewStyle(CircularProgressViewStyle())
              }
            } else {
              Text(message.text)
                .padding(10)
                .foregroundColor(message.type == .llamagenerated || message.type == .llavagenerated ? .primary : .white)
                .background(message.type == .llamagenerated || message.type == .llavagenerated ? Color(UIColor.secondarySystemBackground) : Color.blue)
                .cornerRadius(20)
                .contextMenu {
                  Button(action: {
                    UIPasteboard.general.string = message.text
                  }) {
                    Text("Copy")
                    Image(systemName: "doc.on.doc")
                  }
                }
            }
            if message.type == .llamagenerated || message.type == .llavagenerated { Spacer() }
          }
          let elapsedTime = message.dateUpdated.timeIntervalSince(message.dateCreated)
          if elapsedTime > 0 && message.type != .info {
            Text(String(format: "%.1f t/s", Double(message.tokenCount) / elapsedTime))
              .font(.caption)
              .foregroundColor(.secondary)
              .padding(message.type == .llamagenerated || message.type == .llavagenerated ? .trailing : .leading, 20)
          }
        }.padding([.leading, .trailing], message.type == .info ? 0 : 10)
      }
    }
  }
}
