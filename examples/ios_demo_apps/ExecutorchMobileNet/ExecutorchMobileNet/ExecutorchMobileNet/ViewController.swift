//
//  ViewController.swift
//  ExecutorchMobileNet
//
//  Created by Chen Lai on 8/10/23.
//

import UIKit

class ViewController: UIViewController {

  @IBOutlet weak var Output1: UITextField!
  @IBOutlet weak var Output2: UITextField!
  @IBOutlet weak var Output3: UITextField!
  @IBOutlet weak var InferenceButton: UIButton!
  private var Inputimage: UIImage?

  private lazy var module: ExecutorchModule = {
    if let filePath = Bundle.main.path(forResource:
                                        "mv2_softmax", ofType: "pte"),
       let module = ExecutorchModule(fileAtPath: filePath) {
      return module
    } else {
      fatalError("Can't find the model file!")
    }
  }()

  override func viewDidLoad() {
    super.viewDidLoad()
    Inputimage = UIImage(named: "dog.jpg")
    let imageView = UIImageView(image: Inputimage)
    imageView.contentMode = .scaleAspectFit
    // Set the position and size of the image view
    imageView.frame = CGRect(x: 50, y: 100, width: 300, height: 300)

    // Add the UIImageView to the view hierarchy
    view.addSubview(imageView)
  }

  @IBAction func changeText(_ sender: Any) {
    print("click")
    if let inputImage = Inputimage {
      let resizedImage = Inputimage?.resized(to: CGSize(width: 224, height: 224))
      guard var pixelBuffer = resizedImage?.normalized() else {
        fatalError("Fail to normalize Image!")
      }
      let w = Int32(resizedImage?.size.width ?? -1)
      let h = Int32(resizedImage?.size.height ?? -1)
      if w == -1 || h == -1 {
        fatalError("Fail to get image width or height!")
      }
      let buffer = self.module.segment(image: UnsafeMutableRawPointer(&pixelBuffer), withWidth: w, withHeight: h)
      let swiftString = String(cString: buffer)
      let substrings = swiftString.split(separator: ",").map { String($0) }
      print(substrings)

      Output1.text = substrings[0]
      Output2.text = substrings[1]
      Output3.text = substrings[2]
    }
  }
}
