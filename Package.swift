// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let url = "https://ossci-ios.s3.amazonaws.com/executorch"
let version = "0.1.0"
let coreml_sha256 = "e8c5000a389bdc98274aa0b359350a47e6d0cccb8af5efc46f814feac6afaf86"
let executorch_sha256 = "e6c5d798b614a03ab8a4891caeaa8a7adf8d58ba29e767079321691ec9f1ffb4"
let mps_sha256 = "3e54e3166b5e739cb3f76b2bc6f7b1982a0401821ab785a93120bacfde4bc1ee"
let optimized_sha256 = "4d353f44badd321cf29fe548db9d66b493b93c6233a7e023988e256f0eefeaa1"
let portable_sha256 = "c501f9b644a3e8a7bab62600b7802e4a9752fb789ba4fd02f46bec47858cec07"
let quantized_sha256 = "4fb5f7216abc0ee16ece91a4bce822b06d67b52ca985c9eecbf9d3f8bd1ea1ba"
let xnnpack_sha256 = "e610904cfd6e96f8f738c25a7bb4f6d7b86995b2cfeb72fc1f30523630dbb285"

struct Framework {
  let name: String
  let checksum: String
  var frameworks: [String] = []
  var libraries: [String] = []

  func target() -> Target {
    .binaryTarget(
      name: name,
      url: "\(url)/\(name)-\(version).zip",
      checksum: checksum
    )
  }

  func dependencies() -> Target {
    .target(
      name: "\(name)_dependencies",
      dependencies: [.target(name: name)],
      path: ".swift/\(name)",
      linkerSettings:
          frameworks.map { .linkedFramework($0) } +
          libraries.map { .linkedLibrary($0) }
    )
  }
}

let frameworks = [
  Framework(
    name: "coreml_backend",
    checksum: coreml_sha256,
    frameworks: [
      "Accelerate",
      "CoreML",
    ],
    libraries: [
      "sqlite3",
    ]
  ),
  Framework(
    name: "executorch",
    checksum: executorch_sha256
  ),
  Framework(
    name: "mps_backend",
    checksum: mps_sha256,
    frameworks: [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ]
  ),
  Framework(
    name: "optimized_backend",
    checksum: optimized_sha256
  ),
  Framework(
    name: "portable_backend",
    checksum: portable_sha256
  ),
  Framework(
    name: "quantized_backend",
    checksum: quantized_sha256
  ),
  Framework(
    name: "xnnpack_backend",
    checksum: xnnpack_sha256
  )
]

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v15),
  ],
  products: frameworks.map { .library(name: $0.name, targets: ["\($0.name)_dependencies"]) },
  targets: frameworks.flatMap { [$0.target(), $0.dependencies()] }
)
