/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import AVFoundation
import ImageClassification
import SwiftUI

struct TopBar: View {
  let title: String

  var body: some View {
    Text(title)
      .font(.title)
      .foregroundColor(.white)
      .frame(maxWidth: .infinity)
      .background(Color.black.opacity(0.5))
  }
}

struct ClassificationLabelView: View {
  @ObservedObject var controller: ClassificationController

  var body: some View {
    VStack(alignment: .leading) {
      ForEach(controller.classifications.prefix(3), id: \.label) { classification in
        Text("\(classification.label) \(Int(classification.confidence * 100))%")
          .font(.footnote)
          .foregroundColor(.white)
      }
    }
    .padding()
    .frame(maxWidth: .infinity)
    .background(Color.black.opacity(0.5))
  }
}

struct ClassificationTimeView: View {
  @ObservedObject var controller: ClassificationController

  var body: some View {
    VStack {
      if controller.isRunning {
        ProgressView()
          .progressViewStyle(CircularProgressViewStyle(tint: .white))
          .frame(width: nil, height: 34, alignment: .center)
      } else {
        Text("\n\(controller.elapsedTime, specifier: "%.2f") ms")
          .font(.footnote)
          .foregroundColor(.white)
      }
    }
    .frame(maxWidth: .infinity)
    .background(Color.black.opacity(0.5))
  }
}

struct ModeSelector: View {
  @ObservedObject var controller: ClassificationController

  var body: some View {
    HStack {
      ForEach(Mode.allCases, id: \.self) { mode in
        ModeButton(mode: mode, controller: controller)
      }
    }
    .padding()
    .frame(maxWidth: .infinity)
    .background(Color.black.opacity(0.5))
  }
}

struct ModeButton: View {
  var mode: Mode
  @ObservedObject var controller: ClassificationController

  var body: some View {
    Button(action: { controller.mode = mode }) {
      Text(mode.rawValue)
        .fontWeight(.semibold)
        .foregroundColor(.white)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(controller.mode == mode ? Color.red : Color.clear)
        .cornerRadius(15)
    }
  }
}
