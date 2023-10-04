/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import AVFoundation
import SwiftUI

struct ContentView: View {
  @StateObject private var cameraController = CameraController()
  @StateObject private var classificationController = ClassificationController()

  var body: some View {
    ZStack {
      cameraPreview
      controlPanel
    }
  }

  private var cameraPreview: some View {
    CameraPreview(captureSession: cameraController.captureSession)
      .aspectRatio(contentMode: .fill)
      .edgesIgnoringSafeArea(.all)
      .onAppear(perform: startCapturing)
      .onDisappear(perform: stopCapturing)
  }

  private var controlPanel: some View {
    VStack(spacing: 0) {
      TopBar(title: "ExecuTorch Demo")
      ClassificationLabelView(controller: classificationController)
      Spacer()
      ClassificationTimeView(controller: classificationController)
      ModeSelector(controller: classificationController)
    }
  }

  private func startCapturing() {
    UIApplication.shared.isIdleTimerDisabled = true
    cameraController.startCapturing(withTimeInterval: 1.0) { result in
      switch result {
      case .success(let image):
        self.classificationController.classify(image)
      case .failure(let error):
        self.handleError(error)
      }
    }
  }

  private func stopCapturing() {
    UIApplication.shared.isIdleTimerDisabled = false
  }

  private func handleError(_ error: Error) {
    stopCapturing()
    print(error)
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
