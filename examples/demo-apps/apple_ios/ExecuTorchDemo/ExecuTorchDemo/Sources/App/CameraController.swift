/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import AVFoundation
import SwiftUI

enum CameraControllerError: Error {
  case authorization(String)
  case capture(String)
  case setup(String)
}

class CameraController: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
  let captureSession = AVCaptureSession()
  private var photoOutput = AVCapturePhotoOutput()
  private var timer: Timer?
  private var callback: ((Result<UIImage, Error>) -> Void)?

  func startCapturing(withTimeInterval interval: TimeInterval,
                      callback: @escaping (Result<UIImage, Error>) -> Void) {
    authorize { error in
      if let error {
        DispatchQueue.main.async {
          callback(.failure(error))
        }
        return
      }
      self.setup { error in
        if let error {
          DispatchQueue.main.async {
            callback(.failure(error))
          }
          return
        }
        self.captureSession.startRunning()
        DispatchQueue.main.async {
          self.callback = callback
          self.timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { _ in
            self.photoOutput.capturePhoto(with: AVCapturePhotoSettings(), delegate: self)
          }
        }
      }
    }
  }

  private func authorize(_ completion: @escaping (Error?) -> Void) {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
      DispatchQueue.global(qos: .userInitiated).async {
        completion(nil)
      }
    case .notDetermined:
      AVCaptureDevice.requestAccess(for: .video) { granted in
        DispatchQueue.global(qos: .userInitiated).async {
          if granted {
            completion(nil)
          } else {
            completion(CameraControllerError.authorization("Camera access denied"))
          }
        }
      }
    default:
      DispatchQueue.global(qos: .userInitiated).async {
        completion(CameraControllerError.authorization("Camera access denied"))
      }
    }
  }

  private func setup(_ callback: (Error?) -> Void) {
    guard let videoCaptureDevice = AVCaptureDevice.default(for: .video)
    else {
      callback(CameraControllerError.setup("Cannot get video capture device"))
      return
    }
    let videoInput: AVCaptureDeviceInput
    do {
      videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
    } catch {
      callback(CameraControllerError.setup("Cannot set up video input: \(error)"))
      return
    }
    if captureSession.canAddInput(videoInput) {
      captureSession.addInput(videoInput)
    } else {
      callback(CameraControllerError.setup("Cannot add video input"))
      return
    }
    if captureSession.canAddOutput(photoOutput) {
      captureSession.addOutput(photoOutput)
    } else {
      callback(CameraControllerError.setup("Cannot add photo output"))
      return
    }
    callback(nil)
  }

  func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
    guard let callback = self.callback else {
      print("No image capturing callback set")
      return
    }
    if let error {
      callback(.failure(CameraControllerError.capture("Image capture error: \(error)")))
    }
    guard let imageData = photo.fileDataRepresentation(),
          let image = UIImage(data: imageData),
          let cgImage = image.cgImage
    else {
      callback(.failure(CameraControllerError.capture("Couldn't get image data")))
      return
    }
    var orientation = UIImage.Orientation.up
    switch UIDevice.current.orientation {
    case .portrait:
      orientation = .right
    case .portraitUpsideDown:
      orientation = .left
    case .landscapeRight:
      orientation = .down
    default:
      break
    }
    callback(.success(UIImage(cgImage: cgImage, scale: image.scale, orientation: orientation)))
  }
}
