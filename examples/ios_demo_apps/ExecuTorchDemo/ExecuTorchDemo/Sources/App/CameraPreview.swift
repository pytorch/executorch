/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import AVFoundation
import SwiftUI

struct CameraPreview: UIViewRepresentable {
  let captureSession: AVCaptureSession

  func makeUIView(context: Context) -> UIView {
    let view = CameraView(frame: UIScreen.main.bounds)
    view.videoPreviewLayer?.session = captureSession
    return view
  }

  func updateUIView(_ uiView: UIView, context: Context) {
    if let view = uiView as? CameraView {
      view.videoPreviewLayer?.frame = uiView.bounds
    }
  }
}

final class CameraView: UIView {
  override class var layerClass: AnyClass {
    return AVCaptureVideoPreviewLayer.self
  }

  var videoPreviewLayer: AVCaptureVideoPreviewLayer? {
    return layer as? AVCaptureVideoPreviewLayer
  }
}
