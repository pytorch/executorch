/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI
import UIKit

struct ImagePicker: UIViewControllerRepresentable {
  class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    let parent: ImagePicker

    init(parent: ImagePicker) {
      self.parent = parent
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
      if let image = info[.originalImage] as? UIImage {
        parent.selectedImage = image
      }

      parent.presentationMode.wrappedValue.dismiss()
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
      parent.selectedImage = nil
      parent.presentationMode.wrappedValue.dismiss()
    }
  }

  @Environment(\.presentationMode) var presentationMode
  @Binding var selectedImage: UIImage?
  var sourceType: UIImagePickerController.SourceType = .photoLibrary

  func makeCoordinator() -> Coordinator {
    Coordinator(parent: self)
  }

  func makeUIViewController(context: Context) -> UIImagePickerController {
    let picker = UIImagePickerController()
    picker.delegate = context.coordinator
    picker.sourceType = sourceType
    return picker
  }

  func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
}
