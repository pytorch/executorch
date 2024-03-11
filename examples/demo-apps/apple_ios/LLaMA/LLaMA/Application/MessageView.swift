// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
        VStack(alignment: message.type == .generated ? .leading : .trailing) {
          Text(message.type == .generated ? "LLaMA" : "Prompt")
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(message.type == .generated ? .trailing : .leading, 20)
          HStack {
            if message.type != .generated { Spacer() }
            if message.text.isEmpty {
              ProgressView()
                .progressViewStyle(CircularProgressViewStyle())
            } else {
              Text(message.text)
                .padding(10)
                .foregroundColor(message.type == .generated ? .primary : .white)
                .background(message.type == .generated ? Color(UIColor.secondarySystemBackground) : Color.blue)
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
            if message.type == .generated { Spacer() }
          }
          let elapsedTime = message.dateUpdated.timeIntervalSince(message.dateCreated)
          if elapsedTime > 0 && message.type != .info {
            Text(String(format: "%.1f t/s", Double(message.tokenCount) / elapsedTime))
              .font(.caption)
              .foregroundColor(.secondary)
              .padding(message.type == .generated ? .trailing : .leading, 20)
          }
        }.padding([.leading, .trailing], message.type == .info ? 0 : 10)
      }
    }
  }
}
